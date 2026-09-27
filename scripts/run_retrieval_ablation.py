#!/usr/bin/env python3
"""Execute the 4-way information retrieval ablation benchmark across the 100-query suite.

Architectures evaluated:
1. Baseline 1: Raw-Data LLM-RAG (Raw telemetry vectors + LLM search)
2. Baseline 2: Rule-Based SQL (Relational metadata filtering)
3. Baseline 3: Pure Physics-RAG (14D Invariant Manifold Latent Ranking)
4. Proposed: Two-Tier Hybrid Cascade (DuckDB Relational Slicing + 14D Latent Ranking)

Deterministic Reproduction:
By default, uses PARSER_MODE='deterministic' (deterministic regex parser & pre-warmed cache)
allowing anyone to reproduce all peer-reviewed benchmark results end-to-end in minutes
without requiring a Google Gemini API key.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Ensure local src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings import build_physics_anchored_feature_matrix
from src.physics_rag import (
    RawDataLLMRAG,
    RuleBasedSQLRAG,
    PhysicsRAG,
    HybridPhysicsRAG,
    ALL_PHYSICS_BENCHMARK_QUERIES_100,
    evaluate_rag_benchmark_suite,
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SITES = ["duck_nc", "narrabeen_aus", "torrey_pines_ca", "truc_vert_france"]


def load_unified_5year_dataset() -> pd.DataFrame:
    """Ingest the 4-site multi-channel telemetry (175,392 contiguous hourly records)."""
    site_dfs = []
    for s in SITES:
        proc_path = os.path.join(PROCESSED_DIR, f"{s}_2012_2016_sensors.parquet")
        root_path = os.path.join(DATA_DIR, f"{s}_sensors.parquet")
        path = proc_path if os.path.exists(proc_path) else root_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing parquet for site '{s}'. Run 'python scripts/bootstrap_data.py'.")

        df_s = pd.read_parquet(path)
        if "timestamp" in df_s.columns:
            df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], utc=True)
        df_s["site_id"] = s
        df_s["site"] = s
        if "breaker_regime" not in df_s.columns and "xi_iribarren" in df_s.columns:
            xi = df_s["xi_iribarren"].values
            df_s["breaker_regime"] = np.where(xi < 0.5, "Spilling", np.where(xi < 1.0, "Plunging", "Surging"))
        site_dfs.append(df_s)

    df_global = pd.concat(site_dfs, ignore_index=True)
    return df_global


def main():
    parser = argparse.ArgumentParser(description="Run 4-way information retrieval ablation benchmark.")
    parser.add_argument("--parser-mode", type=str, default="pure_gemini", choices=["pure_gemini", "gemini", "hybrid"],
                        help="LLM Query parser mode (default: 'pure_gemini' using authentic Gemini 2.5 Flash query traces/API).")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of benchmark queries to evaluate (1 to 100).")
    parser.add_argument("--k-radii", type=str, default="1,5,10,20", help="Comma-separated rank radii cutoffs.")
    parser.add_argument("--live-api", action="store_true", help="Force live API calls to Google Gemini instead of loading verified query traces.")
    parser.add_argument("--output-dir", type=str, default="data/validation_results", help="Directory to save output CSVs.")
    args = parser.parse_args()

    k_radii = [int(k.strip()) for k in args.k_radii.split(",")]
    n_queries = min(100, max(1, args.n_queries))
    use_live_gemini = args.live_api

    print("=" * 95)
    print(" 4-WAY INFORMATION RETRIEVAL ABLATION BENCHMARK (100-QUERY SUITE)")
    print(f" LLM Engine: Google Gemini 2.5 Flash | Mode: '{args.parser_mode}' | Live API: {use_live_gemini}")
    print(f" (Deterministic heuristic regex shortcuts have been completely eliminated)")
    print(f" Evaluating N = {n_queries} queries across cutoffs k = {k_radii}")
    print("=" * 95)

    # 1. Ingest Data
    t0 = time.perf_counter()
    print("\n[1/4] Ingesting 5-year multi-site telemetry dataset...")
    df_global = load_unified_5year_dataset()
    print(f"  ✓ Ingested {len(df_global):,} contiguous hourly records across {len(SITES)} global archetypes.")

    # 2. Extract 14D Invariant Feature Matrix
    print("\n[2/4] Extracting 14D physics-anchored invariant feature matrix...")
    X_physics, physics_feat_names = build_physics_anchored_feature_matrix(df_global)
    robust_scaler = RobustScaler().fit(X_physics)
    X_physics_scaled = robust_scaler.transform(X_physics)
    print(f"  ✓ 14D Invariant Matrix: {X_physics_scaled.shape[0]:,} records x {X_physics_scaled.shape[1]} dimensions.")

    # 3. Instantiate RAG Engines
    print("\n[3/4] Initializing candidate retrieval architectures...")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    llm_raw_rag = RawDataLLMRAG(df=df_global, api_key=gemini_key if use_live_gemini else None, use_live_api=use_live_gemini)
    sql_rag = RuleBasedSQLRAG(df=df_global)
    phys_rag = PhysicsRAG(
        df=df_global,
        api_key=gemini_key if use_live_gemini else None,
        embeddings=X_physics_scaled,
        feature_names=physics_feat_names,
        scaler_meta={"scaler": robust_scaler, "scaler_type": "robust"},
        top_k=5,
    )
    hybrid_rag = HybridPhysicsRAG.from_physics_rag(phys_rag, alpha=0.35, regime_boost=0.10)
    print("  ✓ All 4 retrieval engines initialized successfully.")

    # 4. Evaluate Benchmark Suite
    benchmark_suite = ALL_PHYSICS_BENCHMARK_QUERIES_100[:n_queries]

    print("\n[4/4] Executing multi-architecture benchmark evaluations with authentic Gemini LLM...")
    print("  • Evaluating Baseline 1: Raw-Data LLM-RAG (Gemini 2.5 Flash on raw column key-value pairs)...")
    df_llm, summary_llm = evaluate_rag_benchmark_suite(
        rag=llm_raw_rag,
        benchmark_queries=benchmark_suite,
        k_list=k_radii,
        use_llm_parser=True,
        parser_mode=args.parser_mode,
        allow_synthetic_probes=True,
    )

    print("  • Evaluating Baseline 2: Rule-Based SQL...")
    df_sql, summary_sql = evaluate_rag_benchmark_suite(
        rag=sql_rag,
        benchmark_queries=benchmark_suite,
        k_list=k_radii,
        use_llm_parser=False,
        parser_mode=args.parser_mode,
        allow_synthetic_probes=True,
    )

    print("  • Evaluating Baseline 3: Pure Physics-RAG (14D Invariant Latent Embedding)...")
    df_phys, summary_phys = evaluate_rag_benchmark_suite(
        rag=phys_rag,
        benchmark_queries=benchmark_suite,
        k_list=k_radii,
        use_llm_parser=True,
        parser_mode=args.parser_mode,
        allow_synthetic_probes=True,
    )

    print("  • Evaluating Proposed: Two-Tier Hybrid Cascade (DuckDB Slicing + 14D Invariant Ranking)...")
    df_hybrid, summary_hybrid = evaluate_rag_benchmark_suite(
        rag=hybrid_rag,
        benchmark_queries=benchmark_suite,
        k_list=k_radii,
        use_llm_parser=True,
        parser_mode=args.parser_mode,
        allow_synthetic_probes=True,
    )

    # Master Table (Grand Comparison)
    grand_rows = [
        {
            "Architecture": "Baseline 1: Raw-Data LLM-RAG",
            "Precision@1": round(summary_llm.get("mean_p_at_1", 0.0), 4),
            "Precision@5": round(summary_llm.get("mean_p_at_5", 0.0), 4),
            "Precision@10": round(summary_llm.get("mean_p_at_10", 0.0), 4),
            "Precision@20": round(summary_llm.get("mean_p_at_20", 0.0), 4),
            "Mean NDCG@5": round(summary_llm.get("mean_ndcg_at_5", 0.0), 4),
            "MRR": round(summary_llm.get("mean_mrr", 0.0), 4),
            "Regime Purity@5": round(summary_llm.get("regime_purity_at_5", 0.0), 4),
        },
        {
            "Architecture": "Baseline 2: Rule-Based SQL",
            "Precision@1": round(summary_sql.get("mean_p_at_1", 0.0), 4),
            "Precision@5": round(summary_sql.get("mean_p_at_5", 0.0), 4),
            "Precision@10": round(summary_sql.get("mean_p_at_10", 0.0), 4),
            "Precision@20": round(summary_sql.get("mean_p_at_20", 0.0), 4),
            "Mean NDCG@5": round(summary_sql.get("mean_ndcg_at_5", 0.0), 4),
            "MRR": round(summary_sql.get("mean_mrr", 0.0), 4),
            "Regime Purity@5": round(summary_sql.get("regime_purity_at_5", 0.0), 4),
        },
        {
            "Architecture": "Baseline 3: Pure Physics-RAG",
            "Precision@1": round(summary_phys.get("mean_p_at_1", 0.0), 4),
            "Precision@5": round(summary_phys.get("mean_p_at_5", 0.0), 4),
            "Precision@10": round(summary_phys.get("mean_p_at_10", 0.0), 4),
            "Precision@20": round(summary_phys.get("mean_p_at_20", 0.0), 4),
            "Mean NDCG@5": round(summary_phys.get("mean_ndcg_at_5", 0.0), 4),
            "MRR": round(summary_phys.get("mean_mrr", 0.0), 4),
            "Regime Purity@5": round(summary_phys.get("regime_purity_at_5", 0.0), 4),
        },
        {
            "Architecture": "Proposed: Two-Tier Hybrid Cascade",
            "Precision@1": round(summary_hybrid.get("mean_p_at_1", 0.0), 4),
            "Precision@5": round(summary_hybrid.get("mean_p_at_5", 0.0), 4),
            "Precision@10": round(summary_hybrid.get("mean_p_at_10", 0.0), 4),
            "Precision@20": round(summary_hybrid.get("mean_p_at_20", 0.0), 4),
            "Mean NDCG@5": round(summary_hybrid.get("mean_ndcg_at_5", 0.0), 4),
            "MRR": round(summary_hybrid.get("mean_mrr", 0.0), 4),
            "Regime Purity@5": round(summary_hybrid.get("regime_purity_at_5", 0.0), 4),
        },
    ]
    df_grand = pd.DataFrame(grand_rows)

    print("\n" + "=" * 95)
    print(" TABLE I: GRAND 4-WAY INFORMATION RETRIEVAL BENCHMARK ACROSS 100 QUERIES")
    print("=" * 95)
    print(df_grand.to_string(index=False))
    print("=" * 95)

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "benchmark_retrieval_ablation_results.csv")
    df_grand.to_csv(out_csv, index=False)
    print(f"\n✓ Saved benchmark comparison results to '{out_csv}'.")

    # Also save to reports/tables
    tables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "tables"))
    os.makedirs(tables_dir, exist_ok=True)
    tables_csv = os.path.join(tables_dir, "grand_4way_rag_benchmark.csv")
    df_grand.to_csv(tables_csv, index=False)
    print(f"✓ Updated manuscript master table at '{tables_csv}'.")

    # Save per-query dataframes for paired hypothesis testing
    df_hybrid.to_csv(os.path.join(args.output_dir, "hybrid_cascade_100_results.csv"), index=False)
    df_llm.to_csv(os.path.join(args.output_dir, "raw_llm_rag_100_results.csv"), index=False)
    df_sql.to_csv(os.path.join(args.output_dir, "rule_sql_100_results.csv"), index=False)
    df_phys.to_csv(os.path.join(args.output_dir, "pure_physics_100_results.csv"), index=False)
    print(f"✓ Saved per-query results for all 4 architectures to '{args.output_dir}/'.")

    total_time = time.perf_counter() - t0
    print(f"✓ Total ablation execution time: {total_time:.2f}s.\n")


if __name__ == "__main__":
    main()
