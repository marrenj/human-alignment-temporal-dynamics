"""Build the subject-averaged NOD reference RDMs and stimulus index in pipeline/data/nod.

Two outputs:
1. nod_images.csv — category index mapping (category, image) for NodDataset
2. Subject-averaged reference RDM .npy files per ROI

The input is the per-subject NOD ROI RDM tree (sub-01/, sub-02/, ..., each with
`<ROI>_roi_rdm_vector.npy` and `filenames.txt`), configured as `nod_rdms` in
pipeline/paths.yaml or passed via --nod_rdm_dir.

Usage:
    python pipeline/scripts/preprocessing/prepare_nod_rdms.py
    python pipeline/scripts/preprocessing/prepare_nod_rdms.py \\
        --nod_rdm_dir /path/to/NOD_rdms --out_dir pipeline/data/nod
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from temporal_perturbations import paths  # noqa: E402

ROIS = [
    "V1", "V2", "V3", "V4", "V8",
    "VMV1", "VMV2", "VMV3", "VVC",
    "LO1", "LO2", "LO3",
    "PIT", "FFC",
]


def build_category_index(nod_rdm_dir: Path):
    """Build nod_images.csv from filenames.txt and actual image directory."""
    filenames_txt = nod_rdm_dir / "sub-01" / "filenames.txt"
    with open(filenames_txt) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    rows = []
    for ln in lines:
        ln = ln.replace("\\", "/")
        parts = ln.split("/")
        category = parts[0]           # e.g. "n01440764"
        image_name = parts[1]         # e.g. "n01440764_26515.JPEG"
        rows.append({"category": category, "image": image_name})

    # filenames.txt has exactly 1 image per category (the fMRI stimulus).
    # Use max_images_per_category=1 in the config to match the 1000x1000 RDM.
    df = pd.DataFrame(rows).sort_values(["category", "image"]).reset_index(drop=True)
    print(f"  Category index: {len(df)} images across {df['category'].nunique()} categories")
    return df


def average_subject_rdms(nod_rdm_dir: Path):
    """Average per-subject ROI RDM vectors across all 30 subjects."""
    subject_dirs = sorted([d for d in nod_rdm_dir.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])
    print(f"  Found {len(subject_dirs)} subjects")

    avg_rdms = {}
    for roi in ROIS:
        vectors = []
        for sub_dir in subject_dirs:
            npy_path = sub_dir / f"{roi}_roi_rdm_vector.npy"
            if npy_path.exists():
                v = np.load(npy_path).astype(np.float64)
                vectors.append(v)
        if vectors:
            avg = np.mean(vectors, axis=0)
            avg_rdms[roi] = avg
            print(f"    {roi}: averaged {len(vectors)} subjects, vector length {len(avg)}")
        else:
            print(f"    {roi}: NO DATA FOUND")

    return avg_rdms


def _resolve(p: str | Path) -> Path:
    """Resolve relative paths against the repo root, leave absolute paths alone."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nod_rdm_dir", default=None,
                   help="Per-subject NOD ROI RDM tree (sub-01/, sub-02/, ...). Default: paths key nod_rdms.")
    p.add_argument("--out_dir", default="pipeline/data/nod",
                   help="Where to write nod_images.csv, reference_rdms/, filenames.txt. "
                        "Relative paths resolve against the repo root.")
    return p.parse_args()


def main():
    args = parse_args()
    nod_rdm_dir = _resolve(args.nod_rdm_dir or paths.path("nod_rdms"))
    out_dir = _resolve(args.out_dir)

    if not nod_rdm_dir.exists():
        raise FileNotFoundError(
            f"NOD RDM dir does not exist: {nod_rdm_dir}\n"
            "Set nod_rdms in pipeline/paths.yaml or pass --nod_rdm_dir."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"NOD RDM dir : {nod_rdm_dir}")
    print(f"Out dir     : {out_dir}\n")

    print("Step 1: Building category index...")
    cat_df = build_category_index(nod_rdm_dir)
    cat_csv = out_dir / "nod_images.csv"
    cat_df.to_csv(cat_csv, index=False)
    print(f"  Saved to {cat_csv}")

    print("\nStep 2: Averaging subject RDMs...")
    avg_rdms = average_subject_rdms(nod_rdm_dir)

    rdm_dir = out_dir / "reference_rdms"
    rdm_dir.mkdir(exist_ok=True)
    for roi, vec in avg_rdms.items():
        out_path = rdm_dir / f"{roi}_avg_rdm_vector.npy"
        np.save(out_path, vec)
    print(f"  Saved {len(avg_rdms)} ROI RDMs to {rdm_dir}")

    filenames_src = nod_rdm_dir / "sub-01" / "filenames.txt"
    filenames_dst = out_dir / "filenames.txt"
    filenames_dst.write_text(filenames_src.read_text())
    print(f"  Copied filenames.txt to {filenames_dst}")


if __name__ == "__main__":
    main()
