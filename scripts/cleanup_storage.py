#!/usr/bin/env python3
"""
Storage Cleanup Script for Solar Flare Forecasting Project.
Supports --dry-run (default/safe inspection) and --execute (actual file removal).

Usage:
    python scripts/cleanup_storage.py --dry-run
    python scripts/cleanup_storage.py --execute
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Base workspace directory
WORKSPACE_DIR = Path(__file__).resolve().parent.parent

# Defined list of files and directories safe for deletion
TARGETS = [
    # Duplicate ML dataset arrays (v3 is active)
    WORKSPACE_DIR / "data" / "ml" / "X_sequences_expanded.npy",
    WORKSPACE_DIR / "data" / "ml" / "X_sequences_memmap.dat",
    WORKSPACE_DIR / "data" / "ml" / "y_labels_expanded.npy",
    WORKSPACE_DIR / "data" / "ml" / "sequence_metadata_expanded.csv",

    # Redundant root upload package (regenerable on demand via prepare_kaggle_dataset.py)
    WORKSPACE_DIR / "solar_flare_dataset_v3.zip",

    # Incomplete HTTP stream download file
    WORKSPACE_DIR / "pradan1.issdc.gov.in" / "al1" / "protected" / "downloadData" / "hel1os" / "level1" / "2026" / "09" / "17" / "N00_0000" / "HLS_20260917_000005_43185sec_lev1_V111.zip.part",

    # Duplicate export folder for Kaggle (weights exist in results/models/)
    WORKSPACE_DIR / "results" / "kaggle_export",

    # Deprecated baseline DNN model weights
    WORKSPACE_DIR / "results" / "deep_learning_baseline" / "best_dnn_model.pt",

    # Superseded initial 1D CNN baseline model folds
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_1.pt",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_1_history.pkl",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_2.pt",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_2_history.pkl",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_3.pt",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_3_history.pkl",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_4.pt",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_4_history.pkl",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_5.pt",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_fold_5_history.pkl",
    WORKSPACE_DIR / "results" / "models" / "1d_cnn_baseline_summary.pkl",
]

def get_item_size(path: Path) -> int:
    """Calculate file or directory size in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return 0

def find_pycache_dirs(root: Path):
    """Recursively find all __pycache__ directories."""
    pycaches = []
    for p in root.rglob('__pycache__'):
        if p.is_dir() and '.git' not in p.parts:
            pycaches.append(p)
    return pycaches

def main():
    parser = argparse.ArgumentParser(description="Solar Flare Project Storage Cleanup Utility")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--dry-run", action="store_true", default=True, help="Simulate cleanup without deleting any files (default)")
    group.add_argument("--execute", action="store_true", help="Execute file deletion permanently")
    args = parser.parse_args()

    mode_str = "EXECUTION MODE (Permanent Deletion)" if args.execute else "DRY-RUN MODE (Simulation)"
    print(f"==================================================")
    print(f" Solar Flare Project Cleanup Utility")
    print(f" Mode: {mode_str}")
    print(f" Workspace: {WORKSPACE_DIR}")
    print(f"==================================================\n")

    total_bytes = 0
    items_to_remove = []

    # Check explicit target files/directories
    for target in TARGETS:
        if target.exists():
            sz = get_item_size(target)
            total_bytes += sz
            items_to_remove.append((target, sz))
        else:
            print(f"[SKIP - NOT FOUND] {target.relative_to(WORKSPACE_DIR)}")

    # Check __pycache__ directories
    pycaches = find_pycache_dirs(WORKSPACE_DIR)
    for pyc in pycaches:
        sz = get_item_size(pyc)
        total_bytes += sz
        items_to_remove.append((pyc, sz))

    print(f"\n--- TARGETED ITEMS FOR CLEANUP ({len(items_to_remove)} items) ---")
    for item, sz in items_to_remove:
        rel = item.relative_to(WORKSPACE_DIR)
        kind = "DIR " if item.is_dir() else "FILE"
        print(f"  [{kind}] {rel} ({sz / (1024*1024):.4f} MB)")

    total_mb = total_bytes / (1024 * 1024)
    print(f"\n--------------------------------------------------")
    print(f" TOTAL RECLAIMABLE STORAGE: {total_mb:.2f} MB ({total_bytes} bytes)")
    print(f"--------------------------------------------------\n")

    if args.execute:
        confirm = input("Are you sure you want to permanently delete these items? Type 'yes' to proceed: ")
        if confirm.strip().lower() == 'yes':
            print("\nDeleting items...")
            deleted_count = 0
            for item, _ in items_to_remove:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    print(f"  [DELETED] {item.relative_to(WORKSPACE_DIR)}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  [ERROR] Could not delete {item.relative_to(WORKSPACE_DIR)}: {e}")
            print(f"\nCleanup complete. Successfully removed {deleted_count} items ({total_mb:.2f} MB saved).")
        else:
            print("Operation aborted. No files were deleted.")
    else:
        print("[INFO] This was a dry-run. No files were deleted.")
        print("To permanently execute cleanup, run: python scripts/cleanup_storage.py --execute")

if __name__ == "__main__":
    main()
