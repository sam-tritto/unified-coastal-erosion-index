import os
import sys
import time
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath("."))
load_dotenv(os.path.join(os.getcwd(), ".env"))

import src.physics_rag as prag
from src.physics_rag import (
    PhysicsQuerySpec,
    GeminiPhysicsQueryParser,
    DEFAULT_PHYSICS_BENCHMARK_QUERIES,
    evaluate_rag_benchmark_suite,
    PhysicsRAG,
    HybridPhysicsRAG,
    build_physics_anchored_feature_matrix,
)
from sklearn.preprocessing import RobustScaler

print("=" * 90)
print(" EXECUTING TRANSPARENT QUERY PARSER ABLATION: PURE GEMINI vs PURE REGEX vs HYBRID")
print("=" * 90)

queries = DEFAULT_PHYSICS_BENCHMARK_QUERIES
total_q = len(queries)
print(f"Total benchmark queries: {total_q}")

FIELDS_TO_MONITOR = [
    "site_id",
    "date_start",
    "date_end",
    "breaker_regime",
    "hs_min",
    "hs_max",
    "tp_min",
    "tp_max",
    "surge_min",
    "wind_normal_min",
    "target_synthetic_values",
    "focus",
    "is_failure_mode",
]

parsed_records = []

for idx, q_item in enumerate(queries, 1):
    q_text = q_item["query"]
    cat = q_item.get("category", "General")
    mode = q_item.get("mode", "Unspecified")

    # 1. Pure Deterministic Regex
    t0_re = time.perf_counter()
    spec_regex = GeminiPhysicsQueryParser.parse(q_text, parser_mode="pure_regex")
    dt_regex = (time.perf_counter() - t0_re) * 1000.0

    # 2. Pure Live Gemini Flash (no regex backfill)
    t0_gem = time.perf_counter()
    spec_gemini = GeminiPhysicsQueryParser.parse(q_text, parser_mode="pure_gemini")
    dt_gemini = (time.perf_counter() - t0_gem) * 1000.0

    # 3. Hybrid (Gemini + Regex Backfill)
    t0_hyb = time.perf_counter()
    # In hybrid mode, pass pre-parsed gemini or parse with hybrid
    spec_hybrid = GeminiPhysicsQueryParser.parse(q_text, parser_mode="hybrid")
    dt_hybrid = (time.perf_counter() - t0_hyb) * 1000.0

    # Analyze field-by-field extraction
    gemini_extracted_fields = []
    regex_extracted_fields = []
    hybrid_extracted_fields = []
    backfilled_fields = []

    for f in FIELDS_TO_MONITOR:
        val_gem = getattr(spec_gemini, f, None)
        val_reg = getattr(spec_regex, f, None)
        val_hyb = getattr(spec_hybrid, f, None)

        if val_gem is not None:
            gemini_extracted_fields.append(f)
        if val_reg is not None:
            regex_extracted_fields.append(f)
        if val_hyb is not None:
            hybrid_extracted_fields.append(f)

        if val_gem is None and val_hyb is not None and val_reg is not None:
            backfilled_fields.append(f)

    is_pure_zero_shot = (len(backfilled_fields) == 0) and not spec_gemini.intent_summary.startswith("Gemini zero-shot parse failed")
    gemini_failed = spec_gemini.intent_summary.startswith("Gemini zero-shot parse failed")

    status = "Pure Zero-Shot Gemini" if is_pure_zero_shot else ("Gemini Failed (Fallback)" if gemini_failed else "Hybrid (Regex Backfilled)")

    rec = {
        "query_index": idx,
        "query": q_text,
        "category": cat,
        "mode": mode,
        "status": status,
        "is_pure_zero_shot": is_pure_zero_shot,
        "n_backfilled": len(backfilled_fields),
        "backfilled_fields": backfilled_fields,
        "gemini_extracted_fields": gemini_extracted_fields,
        "regex_extracted_fields": regex_extracted_fields,
        "latency_regex_ms": round(dt_regex, 2),
        "latency_gemini_ms": round(dt_gemini, 2),
        "latency_hybrid_ms": round(dt_hybrid, 2),
        "gemini_site_id": spec_gemini.site_id,
        "regex_site_id": spec_regex.site_id,
        "hybrid_site_id": spec_hybrid.site_id,
        "gemini_date_range": f"{spec_gemini.date_start} to {spec_gemini.date_end}" if spec_gemini.date_start else None,
        "regex_date_range": f"{spec_regex.date_start} to {spec_regex.date_end}" if spec_regex.date_start else None,
        "hybrid_date_range": f"{spec_hybrid.date_start} to {spec_hybrid.date_end}" if spec_hybrid.date_start else None,
        "gemini_breaker_regime": spec_gemini.breaker_regime,
        "regex_breaker_regime": spec_regex.breaker_regime,
        "hybrid_breaker_regime": spec_hybrid.breaker_regime,
        "gemini_target_synth": bool(spec_gemini.target_synthetic_values),
        "regex_target_synth": bool(spec_regex.target_synthetic_values),
        "hybrid_target_synth": bool(spec_hybrid.target_synthetic_values),
        "spec_gemini_dict": spec_gemini.model_dump(),
        "spec_regex_dict": spec_regex.model_dump(),
        "spec_hybrid_dict": spec_hybrid.model_dump(),
    }
    parsed_records.append(rec)

    bf_str = f" [Backfilled: {', '.join(backfilled_fields)}]" if backfilled_fields else " [100% Pure Zero-Shot]"
    print(f"[{idx:02d}/{total_q}] ({dt_gemini/1000.0:.2f}s) {q_text[:45]}... -> {status}{bf_str}")
    time.sleep(0.15)  # Safe spacing for API RPM limits

# --- 2. Macro Extraction Summary Table ---
df_parsed = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_dict")} for r in parsed_records])

total_queries = len(df_parsed)
pure_zero_shot_cnt = int(df_parsed["is_pure_zero_shot"].sum())
backfilled_cnt = total_queries - pure_zero_shot_cnt

# Field coverage rates
def get_field_coverage(field_name, prefix):
    col = f"{prefix}_{field_name}"
    if col in df_parsed.columns:
        return float((df_parsed[col].notna() & (df_parsed[col] != False)).mean() * 100.0)
    # Check in extracted fields list
    extracted_col = f"{prefix}_extracted_fields"
    if extracted_col in df_parsed.columns:
        return float(df_parsed[extracted_col].apply(lambda lst: field_name in lst).mean() * 100.0)
    return 0.0

macro_rows = [
    {
        "Parser Configuration": "1. Pure Deterministic Regex",
        "LLM API Required": "No (Offline)",
        "Mean Latency (ms)": f"{df_parsed['latency_regex_ms'].mean():.2f}",
        "Zero-Shot Purity (%)": "N/A (Rule-based)",
        "Site Extraction (%)": f"{get_field_coverage('site_id', 'regex'):.1f}%",
        "Date Window (%)": f"{get_field_coverage('date_range', 'regex'):.1f}%",
        "Breaker Regime (%)": f"{get_field_coverage('breaker_regime', 'regex'):.1f}%",
        "Target Synth (%)": f"{get_field_coverage('target_synth', 'regex'):.1f}%",
        "Mean Fields Extracted": f"{df_parsed['regex_extracted_fields'].apply(len).mean():.2f}",
    },
    {
        "Parser Configuration": "2. Pure Gemini Flash (No Backfill)",
        "LLM API Required": "Yes (gemini-2.5-flash)",
        "Mean Latency (ms)": f"{df_parsed['latency_gemini_ms'].mean():.2f}",
        "Zero-Shot Purity (%)": f"100.0%",
        "Site Extraction (%)": f"{get_field_coverage('site_id', 'gemini'):.1f}%",
        "Date Window (%)": f"{get_field_coverage('date_range', 'gemini'):.1f}%",
        "Breaker Regime (%)": f"{get_field_coverage('breaker_regime', 'gemini'):.1f}%",
        "Target Synth (%)": f"{get_field_coverage('target_synth', 'gemini'):.1f}%",
        "Mean Fields Extracted": f"{df_parsed['gemini_extracted_fields'].apply(len).mean():.2f}",
    },
    {
        "Parser Configuration": "3. Hybrid (Gemini + Regex Backfill)",
        "LLM API Required": "Yes + Regex Ensemble",
        "Mean Latency (ms)": f"{df_parsed['latency_hybrid_ms'].mean():.2f}",
        "Zero-Shot Purity (%)": f"{pure_zero_shot_cnt / total_queries * 100.0:.1f}% ({pure_zero_shot_cnt}/{total_queries})",
        "Site Extraction (%)": f"{get_field_coverage('site_id', 'hybrid'):.1f}%",
        "Date Window (%)": f"{get_field_coverage('date_range', 'hybrid'):.1f}%",
        "Breaker Regime (%)": f"{get_field_coverage('breaker_regime', 'hybrid'):.1f}%",
        "Target Synth (%)": f"{get_field_coverage('target_synth', 'hybrid'):.1f}%",
        "Mean Fields Extracted": f"{df_parsed['hybrid_extracted_fields'].apply(len).mean():.2f}",
    },
]

df_macro_summary = pd.DataFrame(macro_rows)

print("\n" + "=" * 95)
print("               TRANSPARENT QUERY PARSER ABLATION: MACRO-LEVEL SUMMARY              ")
print("=" * 95)
print(df_macro_summary.to_string(index=False))

# --- 3. Attribute Backfill Frequency Breakdown ---
all_backfilled = [f for r in parsed_records for f in r["backfilled_fields"]]
bf_series = pd.Series(all_backfilled).value_counts()
print("\n--- Attributes Most Frequently Backfilled by Deterministic Regex ---")
for attr, cnt in bf_series.items():
    print(f"  • {attr:25s}: {cnt:2d} / {total_queries} queries ({cnt/total_queries*100:.1f}%)")

# Save tables
os.makedirs("reports/tables", exist_ok=True)
json_out = "reports/tables/gemini_query_parser_ablation.json"
csv_out = "reports/tables/gemini_query_parser_ablation.csv"
macro_out = "reports/tables/parser_mode_macro_summary.csv"

with open(json_out, "w") as f:
    json.dump(parsed_records, f, indent=2, default=str)
df_parsed.to_csv(csv_out, index=False)
df_macro_summary.to_csv(macro_out, index=False)
print(f"\n✓ Saved ablation logs to:\n  - {json_out}\n  - {csv_out}\n  - {macro_out}")

# --- 4. Downstream Retrieval Evaluation across Parser Modes ---
print("\n" + "=" * 95)
print(" DOWNSTREAM RETRIEVAL IMPACT ON HYBRID CASCADE PHYSICS-RAG (N=30 BENCHMARK QUERIES)")
print("=" * 95)

# Load global dataset
sites = ["duck_nc", "narrabeen_aus", "torrey_pines_ca", "truc_vert_france"]
parquet_paths = {s: f"data/processed/{s}_2012_2016_sensors.parquet" for s in sites}
data_frames = []
for site_key, ppath in parquet_paths.items():
    if os.path.exists(ppath):
        sub_df = pd.read_parquet(ppath)
        if "site" not in sub_df.columns:
            sub_df["site"] = site_key
        if "site_id" not in sub_df.columns:
            sub_df["site_id"] = site_key
        data_frames.append(sub_df)

if data_frames:
    df_global = pd.concat(data_frames, ignore_index=True)
    X_physics, physics_feat_names = build_physics_anchored_feature_matrix(df_global)
    robust_scaler = RobustScaler().fit(X_physics)
    X_scaled = robust_scaler.transform(X_physics)

    phys_rag = PhysicsRAG(
        df=df_global,
        embeddings=X_scaled,
        feature_names=physics_feat_names,
        scaler_meta={"scaler": robust_scaler, "scaler_type": "robust"},
        top_k=5,
    )
    hybrid_rag = HybridPhysicsRAG.from_physics_rag(phys_rag, alpha=0.35, regime_boost=0.10)

    eval_modes = [
        ("1. Pure Deterministic Regex", "pure_regex"),
        ("2. Pure Gemini Flash (No Backfill)", "pure_gemini"),
        ("3. Hybrid (Gemini + Regex Backfill)", "hybrid"),
    ]

    retrieval_summaries = []
    for mode_label, mode_key in eval_modes:
        print(f"\nEvaluating Hybrid Cascade with Parser: '{mode_label}' (N=30)...")
        # Build pre-parsed specs to guarantee 100% exact parser adherence
        pre_specs = []
        for q_item in queries[:30]:
            if mode_key == "pure_regex":
                sp = GeminiPhysicsQueryParser.parse(q_item["query"], parser_mode="pure_regex")
            elif mode_key == "pure_gemini":
                sp = GeminiPhysicsQueryParser.parse(q_item["query"], parser_mode="pure_gemini")
            else:
                sp = GeminiPhysicsQueryParser.parse(q_item["query"], parser_mode="hybrid")
            pre_specs.append(sp)

        # Evaluate queries using these specs
        p1_list, p5_list, ndcg5_list, mrr_list = [], [], [], []
        for i, q_item in enumerate(queries[:30]):
            res = prag.evaluate_retrieval_query(
                rag=hybrid_rag,
                query_spec_or_text=q_item,
                k=5,
                use_llm_parser=True,
                parser_mode=mode_key,
                spec=pre_specs[i],
            )
            p1_list.append(res["precision_at_1"])
            p5_list.append(res["precision_at_5"])
            ndcg5_list.append(res["ndcg_at_5"])
            mrr_list.append(res["mrr"])

        retrieval_summaries.append({
            "Parser Mode": mode_label,
            "Mean P@1": np.mean(p1_list),
            "Mean P@5": np.mean(p5_list),
            "Mean NDCG@5": np.mean(ndcg5_list),
            "Mean MRR": np.mean(mrr_list),
        })

    df_retrieval_impact = pd.DataFrame(retrieval_summaries)
    print("\n" + "=" * 80)
    print("      DOWNSTREAM RETRIEVAL PERFORMANCE BY PARSER CONFIGURATION (N=30)     ")
    print("=" * 80)
    print(df_retrieval_impact.to_string(index=False))

    retrieval_impact_path = "reports/tables/parser_mode_downstream_retrieval.csv"
    df_retrieval_impact.to_csv(retrieval_impact_path, index=False)
    print(f"\n✓ Saved downstream retrieval impact to {retrieval_impact_path}")
