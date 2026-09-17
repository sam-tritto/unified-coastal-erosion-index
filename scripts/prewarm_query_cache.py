#!/usr/bin/env python3
"""Pre-warm persistent query cache for all N=100 benchmark queries.

Ensures:
1. All 100 queries are parsed and verified with live Gemini Flash.
2. Results are permanently cached to disk (data/cache/gemini_query_cache.json and data/cache/raw_llm_query_cache.json).
3. Transient Google 504/503 errors automatically retry with exponential backoff.
4. Notebook execution can subsequently run in <15 seconds with zero token burn and zero network dependencies.
"""

import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.physics_rag import (
    ALL_PHYSICS_BENCHMARK_QUERIES_100,
    GeminiPhysicsQueryParser,
    RawDataLLMRAG,
    _load_query_cache,
    _save_query_cache,
    _load_raw_llm_cache,
    _save_raw_llm_cache,
    _GEMINI_PARSE_CACHE,
    _RAW_DATA_LLM_CACHE,
)

def main():
    print("=" * 80)
    print(" PRE-WARMING PERSISTENT CACHE FOR N=100 BENCHMARK QUERIES")
    print("=" * 80)

    _load_query_cache()
    _load_raw_llm_cache()

    # Minimal dataframe for RawDataLLMRAG schema prompting
    mock_df = pd.DataFrame([{
        "site_id": "duck_nc",
        "timestamp": "2020-01-01 00:00:00",
        "hs_m": 2.5,
        "tp_s": 10.0,
        "total_water_level_m": 1.5,
        "surge_m": 0.5,
        "wind_speed_ms": 12.0,
        "tan_beta": 0.05,
    }])
    raw_rag = RawDataLLMRAG(df=mock_df, use_live_api=True)

    queries = ALL_PHYSICS_BENCHMARK_QUERIES_100
    n_total = len(queries)

    print(f"Total queries to verify/cache: {n_total}")
    print(f"Initially cached in GeminiQueryParser: {len(_GEMINI_PARSE_CACHE)}")
    print(f"Initially cached in RawDataLLM: {len(_RAW_DATA_LLM_CACHE)}\n")

    for i, q_entry in enumerate(queries, start=1):
        q_text = q_entry["query"]
        parser_cache_key = f"gemini-2.5-flash:pure_gemini:{q_text.strip()}"
        raw_cache_key = f"gemini-2.5-flash:{q_text.strip()}"

        t0 = time.time()
        # 1. Warm GeminiPhysicsQueryParser
        p_cached = parser_cache_key in _GEMINI_PARSE_CACHE
        spec = GeminiPhysicsQueryParser.parse(q_text, parser_mode="pure_gemini")

        # 2. Warm RawDataLLMRAG
        r_cached = raw_cache_key in _RAW_DATA_LLM_CACHE
        raw_res = raw_rag.query(q_text)

        elapsed = time.time() - t0
        status_p = "CACHE" if p_cached else "LIVE"
        status_r = "CACHE" if r_cached else "LIVE"

        print(f"[{i:03d}/{n_total:03d}] ({elapsed:5.2f}s) Parser:[{status_p}] RawLLM:[{status_r}] | {q_text[:55]}...")

    print("\n" + "=" * 80)
    print(" PRE-WARMING COMPLETE!")
    print(f"Total entries in GeminiQueryParser cache: {len(_GEMINI_PARSE_CACHE)}")
    print(f"Total entries in RawDataLLM cache: {len(_RAW_DATA_LLM_CACHE)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
