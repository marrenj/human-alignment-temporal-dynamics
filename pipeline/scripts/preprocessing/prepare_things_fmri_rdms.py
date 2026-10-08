"""Prepare THINGS-fMRI brain-RSA reference data for the neural alignment pipeline.

Mirror of ``prepare_nod_rdms.py`` for the THINGS-fMRI (Hebart et al. 2023)
neural targets. Three outputs, all under ``pipeline/data/things_fmri/``:

1. ``reference_rdms/<ROI>_avg_rdm_vector.npy`` — one subject-averaged reference
   RDM per ROI, stored as a 720x720 upper-triangular vector (length 258840).
2. ``things_fmri_annotations.csv`` — the (concept, stimulus) index consumed by
   ``ThingsFMRIDataset`` to run the model over the fMRI stimuli.
3. ``things_fmri_concept_index.npy`` — a copy of the 720-concept order the RDMs
   and the model RDM are both aligned to (alphabetical).

Provenance: the per-subject 720x720 concept RDMs are decoding RDMs built by
``THINGS-fMRI_Complete/rdm.py`` via ``metadata_df.groupby('concept')`` (pandas
sorts alphabetically), so every subject/ROI shares the same alphabetical
720-concept order — identical to ``pipeline/data/things_fmri_concept_index.npy``.
Averaging across the 3 core subjects reproduces published group-level vectors,
which can optionally be cross-checked via --prebuilt_dir.

Usage:
    python pipeline/scripts/preprocessing/prepare_things_fmri_rdms.py
    python pipeline/scripts/preprocessing/prepare_things_fmri_rdms.py \\
        --things_rdm_dir /path/to/THINGS-fMRI_Complete/all_rdm \\
        --stim_metadata  /path/to/THINGS_fmri_image_metadata.csv \\
        --img_dir        /path/to/THINGS_images \\
        --out_dir        pipeline/data/things_fmri
"""

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from temporal_perturbations import paths  # noqa: E402

N_CONCEPTS = 720
UPPER_TRI_LEN = N_CONCEPTS * (N_CONCEPTS - 1) // 2  # 258840

# The 27 THINGS-fMRI ROIs, keyed by clean name -> filename tag in all_rdm/.
# Category-selective ROIs keep their name; pRF visual areas drop the
# `resampled_..._interp-nn` decoration.
CATEGORY_ROIS = [
    "lEBA", "rEBA", "lFFA", "rFFA", "lOFA", "rOFA",
    "lLOC", "rLOC", "LOC", "lPPA", "rPPA",
    "lRSC", "rRSC", "lTOS", "rTOS",
]
PRF_ROIS = [f"va-{i}" for i in range(1, 13)]
ROIS = CATEGORY_ROIS + PRF_ROIS


def _file_tag(roi: str) -> str:
    """Map a clean ROI name to its filename tag under all_rdm/."""
    if roi.startswith("va-"):
        return f"resampled_{roi}_interp-nn"
    return roi


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _upper_tri(square: np.ndarray) -> np.ndarray:
    return square[np.triu_indices_from(square, k=1)]


def average_subject_rdms(rdm_dir: Path, subjects):
    """Average each ROI's per-subject 720x720 RDM into an upper-tri vector."""
    avg = {}
    for roi in ROIS:
        tag = _file_tag(roi)
        vectors = []
        for sub in subjects:
            path = rdm_dir / f"{sub}_{tag}_concept_rdm_4_fold.npy"
            if not path.exists():
                print(f"    {roi}: MISSING {path.name}")
                continue
            m = np.load(path).astype(np.float64)
            if m.shape != (N_CONCEPTS, N_CONCEPTS):
                raise ValueError(f"{path} has shape {m.shape}, expected {(N_CONCEPTS, N_CONCEPTS)}")
            vectors.append(_upper_tri(m))
        if len(vectors) != len(subjects):
            raise ValueError(f"ROI {roi}: found {len(vectors)}/{len(subjects)} subjects")
        avg[roi] = np.mean(vectors, axis=0)
        print(f"    {roi:6}: averaged {len(vectors)} subjects -> vector length {len(avg[roi])}")
    return avg


def crosscheck_prebuilt(avg: dict, prebuilt_dir: Path):
    """Compare against the pre-averaged group vectors, if present."""
    if not prebuilt_dir.exists():
        print(f"  [crosscheck] {prebuilt_dir} absent — skipping")
        return
    n_ok = 0
    for roi, vec in avg.items():
        pre_path = prebuilt_dir / f"things_fmri_{roi}_rdm_uppertri_vector.npy"
        if not pre_path.exists():
            print(f"  [crosscheck] {roi}: no prebuilt file")
            continue
        pre = np.load(pre_path)
        if not np.allclose(pre, vec):
            print(f"  [crosscheck] {roi}: MISMATCH max|diff|={np.max(np.abs(pre - vec)):.3e}")
        else:
            n_ok += 1
    print(f"  [crosscheck] {n_ok}/{len(avg)} ROIs match the group vectors at {prebuilt_dir}")


def build_annotations(stim_metadata: Path, img_dir: Path, concepts_720):
    """Filter the fMRI stimulus metadata to the 720 concepts with images on disk."""
    df = pd.read_csv(stim_metadata)
    if not {"concept", "stimulus"}.issubset(df.columns):
        raise ValueError(f"{stim_metadata} needs 'concept' and 'stimulus' columns; has {list(df.columns)}")

    concept_set = set(concepts_720)
    df = df[df["concept"].isin(concept_set)].copy()

    exists = df.apply(lambda r: (img_dir / str(r["concept"]) / str(r["stimulus"])).exists(), axis=1)
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"    dropping {n_missing} rows whose image is absent under {img_dir}")
    df = df[exists].reset_index(drop=True)

    covered = set(df["concept"].unique())
    missing_concepts = concept_set - covered
    if missing_concepts:
        raise ValueError(f"{len(missing_concepts)} concepts have no usable image: {sorted(missing_concepts)[:10]}")

    per_concept = df["concept"].value_counts()
    print(f"    annotations: {len(df)} stimuli across {df['concept'].nunique()} concepts "
          f"({per_concept.min()}-{per_concept.max()} imgs/concept)")
    return df


def main():
    args = parse_args()
    rdm_dir = _resolve(args.things_rdm_dir or paths.path("things_fmri_rdms"))
    stim_metadata = _resolve(args.stim_metadata or paths.path("things_fmri_metadata"))
    img_dir = _resolve(args.img_dir or paths.path("things_fmri_images"))
    concept_index_path = _resolve(args.concept_index)
    out_dir = _resolve(args.out_dir)

    if not rdm_dir.exists():
        raise SystemExit(f"THINGS-fMRI RDM dir not found: {rdm_dir}")

    concepts_720 = [str(c) for c in np.load(concept_index_path, allow_pickle=True).reshape(-1)]
    if len(concepts_720) != N_CONCEPTS:
        raise ValueError(f"concept index has {len(concepts_720)} concepts, expected {N_CONCEPTS}")
    if concepts_720 != sorted(concepts_720):
        raise ValueError("concept index is not alphabetical; RDM row order assumption is violated")

    subjects = sorted({m.group(1) for f in rdm_dir.glob("*_concept_rdm_4_fold.npy")
                       if (m := re.match(r"(sub-\d+)_", f.name))})
    print(f"Subjects: {subjects}")

    print("Averaging per-subject ROI RDMs:")
    avg = average_subject_rdms(rdm_dir, subjects)
    for roi, vec in avg.items():
        if vec.shape != (UPPER_TRI_LEN,):
            raise ValueError(f"ROI {roi}: vector length {vec.shape} != {UPPER_TRI_LEN}")

    if args.prebuilt_dir:
        crosscheck_prebuilt(avg, _resolve(args.prebuilt_dir))

    ref_dir = out_dir / "reference_rdms"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for roi, vec in avg.items():
        np.save(ref_dir / f"{roi}_avg_rdm_vector.npy", vec)
    print(f"Wrote {len(avg)} reference RDM vectors -> {ref_dir}")

    print("Building annotations CSV:")
    ann = build_annotations(stim_metadata, img_dir, concepts_720)
    ann_path = out_dir / "things_fmri_annotations.csv"
    ann.to_csv(ann_path)
    print(f"Wrote annotations -> {ann_path}")

    shutil.copy(concept_index_path, out_dir / "things_fmri_concept_index.npy")
    print(f"Copied concept index -> {out_dir / 'things_fmri_concept_index.npy'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--things_rdm_dir", default=None,
                   help="Per-subject 720x720 ROI RDM tree (sub-0X_<ROI>_concept_rdm_4_fold.npy). "
                        "Default: paths key things_fmri_rdms.")
    p.add_argument("--stim_metadata", default=None,
                   help="fMRI stimulus metadata CSV with 'concept' and 'stimulus' columns. "
                        "Default: paths key things_fmri_metadata.")
    p.add_argument("--img_dir", default=None,
                   help="Concept-organized THINGS-fMRI image dir (img_dir/<concept>/<stimulus>). "
                        "Default: paths key things_fmri_images.")
    p.add_argument(
        "--concept_index",
        default="pipeline/data/things_fmri_concept_index.npy",
        help="720-concept alphabetical order the RDMs align to.",
    )
    p.add_argument(
        "--prebuilt_dir",
        default=None,
        help="Optional pre-averaged group vectors for cross-checking.",
    )
    p.add_argument(
        "--out_dir",
        default="pipeline/data/things_fmri",
        help="Output dir for reference_rdms/, annotations, concept index.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
