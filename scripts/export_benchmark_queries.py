#!/usr/bin/env python3
"""Export the standardized 100-query benchmark suite to JSON and YAML schemas.

Standardizes all 100 queries across:
- 3 Operational Tiers:
  * Tier 1: Baseline Operational (Categories 1–5, N = 30)
  * Tier 2: Advanced Morphodynamics (Categories 6–7, N = 30)
  * Tier 3: Adversarial Stress-Tests & 3D Pathologies (Categories 8–9, N = 40)
- 4 Operational Modes:
  * Mode A: Pure Phenomenological Latent Search (N = 30)
  * Mode B: Full Operational Historical Named Storms (N = 30)
  * Mode C: Deliberate Ambiguity & Intent Stress Tests (N = 20)
  * Mode D: 3D Physical Morphodynamic Pathologies (N = 20)
"""

from __future__ import annotations

import os
import re
import json
import yaml
from typing import Dict, List, Any

import src.physics_rag as prag
from src.physics_rag import ALL_PHYSICS_BENCHMARK_QUERIES_100

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
JSON_OUT = os.path.join(DATA_DIR, "benchmark_100_queries.json")
YAML_OUT = os.path.join(DATA_DIR, "benchmark_100_queries.yaml")


def assign_tier(category: str) -> str:
    """Assign operational evaluation tier based on benchmark category."""
    cat_lower = category.lower()
    if any(k in cat_lower for k in ["1.", "2.", "3.", "4.", "5.", "historical storm", "profile changes", "thresholds", "complex physical", "multi-criteria"]):
        return "Tier 1: Baseline Operational"
    elif any(k in cat_lower for k in ["6.", "7.", "advanced morphodynamics", "extended historical"]):
        return "Tier 2: Advanced Morphodynamics"
    elif any(k in cat_lower for k in ["8.", "9.", "ambiguity", "patholog"]):
        return "Tier 3: Adversarial Stress-Tests & 3D Pathologies"
    return "Tier 1: Baseline Operational"


def extract_temporal_window(query_text: str, description: str) -> str:
    """Extract expected temporal window or mark as phenomenological/unbounded."""
    # Look for dates like 2012, 2013, 2014, 2015, 2016 with month names
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
              "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pattern = rf"(?:{'|'.join(months)})\s*(?:\d{{1,2}}(?:-\d{{1,2}})?,\s*)?201[2-6]"
    match = re.search(pattern, query_text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    # Check description
    desc_match = re.search(r"\(.*201[2-6].*\)", description)
    if desc_match:
        return desc_match.group(0).strip("()")

    return "Continuous Multi-Site Invariance (Unbounded Temporal Window)"


def export_benchmark_suite() -> List[Dict[str, Any]]:
    """Serialize all 100 queries into a standardized schema."""
    os.makedirs(DATA_DIR, exist_ok=True)
    serialized_queries: List[Dict[str, Any]] = []

    for idx, q in enumerate(ALL_PHYSICS_BENCHMARK_QUERIES_100, 1):
        cat = q.get("category", "Uncategorized")
        mode = q.get("mode", "Mode A: Pure Phenomenological")
        tier = assign_tier(cat)
        q_text = q["query"]
        desc = q.get("description", "")
        synth = q.get("target_synthetic_values", {}) or {}
        regime = q.get("target_regime", None)
        temp_window = extract_temporal_window(q_text, desc)

        is_ambiguity = "ambiguity" in cat.lower() or "mode c" in mode.lower()
        is_pathology = "patholog" in cat.lower() or "mode d" in mode.lower()

        # Construct ground truth criteria summary
        apex_criteria = {}
        for k, v in synth.items():
            if isinstance(v, (int, float)):
                apex_criteria[k] = round(float(v), 4)
            else:
                apex_criteria[k] = v

        entry = {
            "query_id": f"QUERY_{idx:03d}",
            "query_index": idx,
            "query": q_text,
            "tier": tier,
            "category": cat,
            "mode": mode,
            "description": desc,
            "target_regime": regime,
            "target_synthetic_values": apex_criteria,
            "expected_temporal_window": temp_window,
            "is_adversarial_ambiguity": is_ambiguity,
            "is_3d_morphodynamic_pathology": is_pathology,
        }
        serialized_queries.append(entry)

    # Save JSON
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "1.0.0",
                "total_queries": len(serialized_queries),
                "tiers": {
                    "Tier 1: Baseline Operational": sum(1 for q in serialized_queries if q["tier"].startswith("Tier 1")),
                    "Tier 2: Advanced Morphodynamics": sum(1 for q in serialized_queries if q["tier"].startswith("Tier 2")),
                    "Tier 3: Adversarial Stress-Tests & 3D Pathologies": sum(1 for q in serialized_queries if q["tier"].startswith("Tier 3")),
                },
                "modes": {
                    "Mode A: Pure Phenomenological": sum(1 for q in serialized_queries if "Mode A" in q["mode"]),
                    "Mode B: Full Operational": sum(1 for q in serialized_queries if "Mode B" in q["mode"]),
                    "Mode C: Deliberate Ambiguity": sum(1 for q in serialized_queries if "Mode C" in q["mode"]),
                    "Mode D: 3D Physical Pathology": sum(1 for q in serialized_queries if "Mode D" in q["mode"]),
                },
                "queries": serialized_queries,
            },
            f,
            indent=2,
        )

    # Save YAML
    with open(YAML_OUT, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "schema_version": "1.0.0",
                "total_queries": len(serialized_queries),
                "queries": serialized_queries,
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"✓ Exported {len(serialized_queries)} standardized benchmark queries:")
    print(f"  • JSON: {JSON_OUT} ({os.path.getsize(JSON_OUT):,} bytes)")
    print(f"  • YAML: {YAML_OUT} ({os.path.getsize(YAML_OUT):,} bytes)")
    return serialized_queries


def main():
    print("=" * 80)
    print(" STANDARDIZED 100-QUERY ADVERSARIAL BENCHMARK SUITE EXPORTER")
    print("=" * 80)
    export_benchmark_suite()


if __name__ == "__main__":
    main()
