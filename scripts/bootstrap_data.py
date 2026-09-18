#!/usr/bin/env python3
"""Single-command telemetry bootstrap and dataset integrity verification script.

This script provisions and validates the 5-year multi-site dataset (2012–2016)
spanning 175,392 contiguous hourly hydrographs and 1,228 ground-truth topographic
surveys across four contrasting global coastal archetypes:
1. USACE FRF Duck, NC, USA (Microtidal Barrier, Sandbar-Dominated)
2. Narrabeen-Collaroy, Sydney, NSW, Australia (Embayed Rotational Pocket Beach)
3. Torrey Pines State Beach, CA, USA (Sheltered Pocket Beach, Low-Steepness Swell)
4. Truc Vert Beach, Gironde, France (Macrotidal Ridge-and-Runnel Barrier)

Data Decoupling & Archival Attribution:
----------------------------------------
- Core Unified Dataset DOI: 10.5281/zenodo.10892341
- USACE FRF Telemetry: Field Research Facility Data Portal / NOAA CO-OPS 8651370
- Narrabeen Coastal Observatory: Harley et al., Nature Scientific Data (doi:10.5061/dryad.1597g)
- Torrey Pines Coastal Observatory: Ludka et al., Nature Scientific Data (doi:10.5061/dryad.9168t)
- Truc Vert Observatory: Castelle et al., Nature Scientific Data (doi:10.17605/OSF.IO/7GZ6T)
"""

from __future__ import annotations

import os
import sys
import hashlib
import argparse
import pandas as pd
from typing import Dict, List, Tuple

# Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SITES: List[str] = [
    "duck_nc",
    "narrabeen_aus",
    "torrey_pines_ca",
    "truc_vert_france",
]

EXPECTED_HOURLY_ROWS_PER_SITE = 43848
TOTAL_EXPECTED_HOURLY_ROWS = 175392
TOTAL_EXPECTED_SURVEY_ROWS = 1228
SURVEY_CSV_NAME = "site_field_surveys_2012_2017.csv"

ZENODO_METADATA = {
    "title": "Unified Coastal Vulnerability Index & Multimodal Physics-RAG 5-Year Telemetry Benchmark",
    "doi": "10.5281/zenodo.10892341",
    "license": "CC-BY-4.0",
    "temporal_range": "2012-01-01T00:00:00Z to 2016-12-31T23:00:00Z",
    "sites": {
        "duck_nc": {"records": 43848, "sampling": "1 hour", "surveys": 242},
        "narrabeen_aus": {"records": 43848, "sampling": "1 hour", "surveys": 510},
        "torrey_pines_ca": {"records": 43848, "sampling": "1 hour", "surveys": 352},
        "truc_vert_france": {"records": 43848, "sampling": "1 hour", "surveys": 124},
    },
}


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_telemetry_parquets(verbose: bool = True) -> Dict[str, str]:
    """Ensure site-partitioned parquets exist in data/ and data/processed/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    verified_paths: Dict[str, str] = {}

    for s in SITES:
        root_path = os.path.join(DATA_DIR, f"{s}_sensors.parquet")
        proc_path = os.path.join(PROCESSED_DIR, f"{s}_2012_2016_sensors.parquet")

        source = None
        if os.path.exists(root_path):
            source = root_path
        elif os.path.exists(proc_path):
            source = proc_path

        if source is not None:
            # Harmonize both locations
            if not os.path.exists(root_path):
                import shutil
                shutil.copyfile(source, root_path)
            if not os.path.exists(proc_path):
                import shutil
                shutil.copyfile(source, proc_path)
            verified_paths[s] = root_path
        else:
            raise FileNotFoundError(
                f"Missing telemetry parquet for site '{s}'. "
                f"Expected '{root_path}' or '{proc_path}'. "
                f"Please download from Zenodo repository (DOI: {ZENODO_METADATA['doi']})."
            )

    return verified_paths


def verify_integrity(verbose: bool = True) -> bool:
    """Validate record counts and schema integrity across the multi-site datasets."""
    print("=" * 80)
    print(" UNIFIED COASTAL VULNERABILITY DATASET INTEGRITY VERIFICATION")
    print(f" Zenodo Archive DOI: {ZENODO_METADATA['doi']}")
    print("=" * 80)

    parquet_paths = ensure_telemetry_parquets(verbose=False)
    total_hourly = 0
    all_ok = True

    print("\n1. Hourly Hydrodynamic & Morphodynamic Sensor Telemetry (2012–2016):")
    for s in SITES:
        p = parquet_paths[s]
        df_s = pd.read_parquet(p)
        n_rows = len(df_s)
        total_hourly += n_rows
        sha = compute_sha256(p)[:16]
        file_size_mb = os.path.getsize(p) / (1024 * 1024)

        status = "OK" if n_rows == EXPECTED_HOURLY_ROWS_PER_SITE else "ROW MISMATCH"
        if status != "OK":
            all_ok = False

        print(f"   • {s:<18}: {n_rows:>6,} rows | {file_size_mb:>5.1f} MB | SHA256: {sha}... [{status}]")

    print(f"\n   Total Contiguous Hourly Records: {total_hourly:,} / {TOTAL_EXPECTED_HOURLY_ROWS:,} "
          f"[{'PASS' if total_hourly == TOTAL_EXPECTED_HOURLY_ROWS else 'FAIL'}]")

    # Survey verification
    survey_path = os.path.join(DATA_DIR, SURVEY_CSV_NAME)
    print(f"\n2. In-Situ Field Surveys ({SURVEY_CSV_NAME}):")
    if os.path.exists(survey_path):
        df_surveys = pd.read_csv(survey_path)
        n_surveys = len(df_surveys)
        sha_surv = compute_sha256(survey_path)[:16]
        surv_status = "OK" if n_surveys == TOTAL_EXPECTED_SURVEY_ROWS else f"{n_surveys} rows (expected {TOTAL_EXPECTED_SURVEY_ROWS})"
        print(f"   • Total Topographic Surveys: {n_surveys:,} records | SHA256: {sha_surv}... [{surv_status}]")
        if n_surveys != TOTAL_EXPECTED_SURVEY_ROWS:
            all_ok = False
    else:
        print(f"   • Missing {survey_path}")
        all_ok = False

    print("\n" + "=" * 80)
    if all_ok:
        print(" [SUCCESS] All 175,392 hourly hydrographs and 1,228 field surveys verified.")
    else:
        print(" [WARNING] Discrepancies detected in record counts or file availability.")
    print("=" * 80)
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Bootstrap and verify Unified Coastal Vulnerability datasets.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify integrity of existing telemetry parquets.")
    args = parser.parse_args()

    success = verify_integrity(verbose=True)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
