"""Materialize VTAB datasets to ImageFolder trees so the PyTorch probe can read them.

VTAB's canonical splits and task definitions live in Google's `task_adaptation`
package, which requires TensorFlow. TensorFlow cannot be installed alongside the
probe: it forces numpy 2.x, and this project's env pins numpy 1.26 (torch 2.2.1,
composer <1.27, brain-score <2 all depend on it). So the two live in separate
conda envs and communicate through image files on disk.

    Run this with the TensorFlow (vtab-prep) env's interpreter, not the main env:
        /path/to/envs/vtab-prep/bin/python \
            pipeline/scripts/downloads/export_vtab_datasets.py --datasets caltech101

Images are written at ORIGINAL resolution. VTAB's own pipeline would resize to 224
with its recipe; letting it do that and then resizing again in the probe would be a
silent double-resize, and each architecture needs its own recipe (`in1k_bilinear` /
`in1k_bicubic` / `clip`) applied to the original pixels.

Class directories are ZERO-PADDED integers. VTAB emits integer labels, while
`torchvision.datasets.ImageFolder` assigns labels by sorted directory name -- with
unpadded names "10" sorts before "2" and every label above 9 silently permutes.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from temporal_perturbations import paths  # noqa: E402

DEFAULT_SPLITS = ["train", "test"]


def _builder(name, data_dir):
    """Construct the task_adaptation dataset for a VTAB name.

    Mirrors clip_benchmark.datasets.builder.build_vtab_dataset so the splits are
    identical to what `build_dataset("vtab/<name>")` would have produced.
    """
    if name == "caltech101":
        from task_adaptation.data.caltech import Caltech101
        return Caltech101(data_dir=data_dir)
    if name == "cars":
        from task_adaptation.data.cars import CarsData
        return CarsData(data_dir=data_dir)
    if name in ("cifar10", "cifar100"):
        from task_adaptation.data.cifar import CifarData
        return CifarData(data_dir=data_dir, num_classes=10 if name == "cifar10" else 100)
    if name.startswith("clevr_"):
        from task_adaptation.data.clevr import CLEVRData
        return CLEVRData(task=name[len("clevr_"):], data_dir=data_dir)
    if name == "dtd":
        from task_adaptation.data.dtd import DTDData
        return DTDData(data_dir=data_dir)
    if name == "eurosat":
        from task_adaptation.data.eurosat import EurosatData
        return EurosatData(subset="rgb", data_key="image", data_dir=data_dir)
    if name == "food101":
        from task_adaptation.data.food101 import Food101Data
        return Food101Data(data_dir=data_dir)
    if name.startswith("kitti_"):
        from task_adaptation.data.kitti import KittiData
        return KittiData(task=name[len("kitti_"):], data_dir=data_dir)
    if name == "flowers":
        from task_adaptation.data.oxford_flowers102 import OxfordFlowers102Data
        return OxfordFlowers102Data(data_dir=data_dir)
    if name == "pets":
        from task_adaptation.data.oxford_iiit_pet import OxfordIIITPetData
        return OxfordIIITPetData(data_dir=data_dir)
    if name == "pcam":
        from task_adaptation.data.patch_camelyon import PatchCamelyonData
        return PatchCamelyonData(data_dir=data_dir)
    if name == "resisc45":
        from task_adaptation.data.resisc45 import Resisc45Data
        return Resisc45Data(data_dir=data_dir)
    if name == "svhn":
        from task_adaptation.data.svhn import SvhnData
        return SvhnData(data_dir=data_dir)
    raise ValueError(f"No VTAB builder mapped for {name!r}")


def export_split(ds, name, split, out_root, limit=None, force=False):
    import numpy as np
    from PIL import Image

    expected = ds.get_num_samples(split)
    target = out_root / name / split
    manifest = out_root / name / f"_manifest_{split}.json"

    if manifest.exists() and not force:
        got = json.loads(manifest.read_text())
        if got.get("n_written") == got.get("n_expected"):
            print(f"  [skip] {name}/{split}: {got['n_written']} images already exported")
            return got

    target.mkdir(parents=True, exist_ok=True)
    # batch_size=1: VTAB images are variable-sized for several datasets, so TF cannot
    # batch them without a resize -- and resizing here is exactly what we must not do.
    it = ds.get_tf_data(split, batch_size=1, epochs=1, for_eval=True)

    n, labels_seen = 0, set()
    pad = None
    for rec in it:
        img = np.asarray(rec["image"])[0]
        lab = int(np.asarray(rec["label"]).ravel()[0])
        if pad is None:
            pad = 3          # 3 digits covers every VTAB task (max 1000 classes)
        d = target / f"{lab:0{pad}d}"
        d.mkdir(exist_ok=True)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        Image.fromarray(img.astype(np.uint8)).save(d / f"{n:06d}.png")
        labels_seen.add(lab)
        n += 1
        if n % 2000 == 0:
            print(f"    {name}/{split}: {n}/{expected}", flush=True)
        if limit and n >= limit:
            break

    info = {"dataset": name, "split": split, "n_written": n, "n_expected": expected,
            "n_classes_seen": len(labels_seen), "complete": (limit is None and n == expected)}
    manifest.write_text(json.dumps(info, indent=2))
    print(f"  [done] {name}/{split}: {n} images, {len(labels_seen)} classes "
          f"(expected {expected})")
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", required=True,
                   help="Comma-separated VTAB names WITHOUT the vtab/ prefix (e.g. caltech101,dtd).")
    p.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    p.add_argument("--tfds_dir", default=None, help="Default: <probe_datasets>/tfds.")
    p.add_argument("--out_root", default=None, help="Default: <probe_datasets>/vtab_export.")
    p.add_argument("--limit", type=int, default=None, help="Cap images per split (smoke tests).")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = p.parse_args()
    args.tfds_dir = str(args.tfds_dir or paths.path("probe_datasets") / "tfds")
    args.out_root = args.out_root or paths.path("probe_datasets") / "vtab_export"

    os.environ.setdefault("TFDS_DATA_DIR", args.tfds_dir)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # keep TF off the GPUs

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    names = [d.strip().removeprefix("vtab/") for d in args.datasets.split(",") if d.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    results = {}
    for name in names:
        print(f"[{name}]")
        try:
            ds = _builder(name, args.tfds_dir)
        except Exception as exc:
            print(f"  [fail] build: {type(exc).__name__}: {exc}")
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        avail = sorted(ds.splits)
        for split in splits:
            if split not in avail:
                print(f"  [skip] {name}/{split}: not in {avail}")
                continue
            if args.dry_run:
                print(f"  [plan] {name}/{split}: {ds.get_num_samples(split)} images")
                continue
            try:
                results.setdefault(name, {})[split] = export_split(
                    ds, name, split, out_root, limit=args.limit, force=args.force)
            except Exception as exc:
                print(f"  [fail] {name}/{split}: {type(exc).__name__}: {exc}")
                results.setdefault(name, {})[split] = {"error": f"{type(exc).__name__}: {exc}"}

    if not args.dry_run:
        (out_root / "_export_summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nexport root: {out_root}")

    # Per-split exceptions are caught so one bad dataset cannot sink a batch; still exit
    # non-zero so a total failure is not reported as success.
    failed = [n for n, v in results.items()
              if "error" in v or any("error" in s for s in v.values() if isinstance(s, dict))]
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
