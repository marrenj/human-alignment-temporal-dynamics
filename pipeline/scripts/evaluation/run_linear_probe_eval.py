"""Frozen linear-probe benchmark for a single `runs_root` checkpoint: top-1 and top-5
on VTAB-split CIFAR-10/100 (`vtab/cifar10`, `vtab/cifar100`) by default.

Motivation
----------
`run_linear_probe_generalization.py` answers a *relative* question -- does a perturbed
run's alignment advantage cost it generic visual transfer, measured as a delta against
its baseline at a matched epoch. This script answers the *absolute* one: how good is
this backbone's representation on its own, as one row of a per-architecture table.

The two differ in more than framing, which is why they are separate drivers:

  =================  ==============================  ==========================
                     generalization                  eval (this script)
  =================  ==============================  ==========================
  unit               a perturbed/baseline PAIR       ONE checkpoint
  default epoch      the run's ONSET epoch           the run's FINAL epoch
  baseline run       the comparison partner          a first-class target row
  reported           baseline, perturbed, delta      top-1 and top-5
  =================  ==============================  ==========================

Shared infrastructure (nothing reimplemented here)
--------------------------------------------------
  * Model construction and per-epoch weight loading: `build_model` from
    `run_model_to_model_rsa`, covering all four architectures. Each
    returns a model whose forward() is already the feature to probe -- the pooled
    penultimate (pre-logits) vector for the three timm families, the 768-d
    pre-projection CLS token for CLIP-ViT-B/32.
  * Run naming: `ARCH_PREFIX`; each family's baseline is `{prefix}_baseline`.
  * Checkpoint discovery: `scratch_utils.available_scratch_epochs` /
    `scratch_checkpoint_at_epoch`.
  * Eval preprocessing: `eval_preprocessing(architecture)` -- the normalization and crop
    geometry the checkpoint was TRAINED under, which differs per architecture.
  * The probe itself: `linear_probe_eval.run_linear_probe`, with its own per-dataset
    result cache and feature cache.

Feature space (`--clip_feature`)
--------------------------------
CLIP-ViT-B/32 only; a no-op for the timm architectures. `cls` (default) probes the
vision encoder's own 768-d CLS token PRE-projection -- the analog of the timm models'
penultimate feature. `projected` probes the 512-d L2-normalized joint image-text
embedding, which is the retrieval space, not a visual-similarity one.

The two spaces are kept apart by OUTPUT ROOT, not by a filename tag: the consumer glob
is `<root>/*/linear_probe_eval_epoch*.json`, which would match any trailing tag and
silently pool both variants into one table. `cls` keeps the plain root
(`<results_root>/linear_probe_eval_scratch_std/`); `projected` gets the sibling root.

Usage
-----
    # Every run of all four architectures at each run's final epoch (one subprocess each):
    python pipeline/scripts/evaluation/run_linear_probe_eval.py

    # The four baselines only (the headline table):
    for A in RN50 ViT-B/16 ConvNeXt-B "CLIP ViT-B/32"; do
        python pipeline/scripts/evaluation/run_linear_probe_eval.py \\
            --architecture "$A" --runs baseline --device cuda:0
    done

    # Every run for one architecture, at each run's final epoch:
    python pipeline/scripts/evaluation/run_linear_probe_eval.py \\
        --architecture RN50 --device cuda:0

    # Specific epochs along a trajectory:
    python pipeline/scripts/evaluation/run_linear_probe_eval.py \\
        --architecture ViT-B/16 --perturb_type label_shuffle --phase early \\
        --epochs 5,20,89 --device cuda:0

    # An explicit checkpoint, bypassing runs_root discovery:
    python pipeline/scripts/evaluation/run_linear_probe_eval.py \\
        --architecture RN50 --checkpoint /path/to/epoch_089.ckpt \\
        --epoch 89 --run_name my_run --device cuda:0

    # Plan only, no compute:
    python pipeline/scripts/evaluation/run_linear_probe_eval.py \\
        --architecture ConvNeXt-B --dry-run
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP_DIR))

from run_model_to_model_rsa import (  # noqa: E402
    ARCH_PREFIX,
    CLIP_ARCH,
    CLIP_FEATURES,
    DEFAULT_CLIP_FEATURE,
    PERTURB_TYPES,
    PHASES,
    build_model as build_scratch_model,
)
from temporal_perturbations import paths  # noqa: E402
from temporal_perturbations.data.eval_transforms import eval_preprocessing  # noqa: E402
from temporal_perturbations.inference.linear_probe_eval import (  # noqa: E402
    DEFAULT_PROBE_SOLVER,
    PROBE_SOLVERS,
    PROBE_SWEEP_MAX_ITER,
    is_current_recipe,
    run_linear_probe,
)
from temporal_perturbations.models.scratch.scratch_utils import (  # noqa: E402
    available_scratch_epochs,
    scratch_checkpoint_at_epoch,
)
from temporal_perturbations.utils.run_hparams import onset_epoch as _onset_epoch  # noqa: E402

# VTAB's canonical splits, matching PAPER_PROBE_DATASETS. NOTE this is NOT the same
# data as the plain "cifar10"/"cifar100" builders: VTAB trains on 90% of the official
# train split (45000, not 50000), so numbers produced under the plain names are not
# comparable to these. Requires an export from export_vtab_datasets.py.
DEFAULT_DATASETS = ["vtab/cifar10", "vtab/cifar100"]
DEFAULT_OUTPUT_DIR = paths.results_root() / "linear_probe_eval_scratch_std"
# Feature caches are large and regenerable, so they live apart from the results.
DEFAULT_FEATURE_CACHE = paths.path("feature_cache") / "probe_features"


def resolve_output_dir(output_dir, architecture, clip_feature):
    """Where records land, keeping CLIP's two feature spaces in separate trees.

    A filename tag would not work: consumers glob `*/linear_probe_eval_epoch*.json`,
    which matches any trailing tag, so both variants would pool into one dataframe.
    The default `cls` keeps the plain root -- that is where the cross-architecture
    table is assembled, and the timm architectures only ever write there.
    """
    if output_dir is not None:
        return Path(output_dir)
    if architecture == CLIP_ARCH and clip_feature != DEFAULT_CLIP_FEATURE:
        return DEFAULT_OUTPUT_DIR.parent / f"{DEFAULT_OUTPUT_DIR.name}_clip_{clip_feature}"
    return DEFAULT_OUTPUT_DIR


def _cache_variant(args):
    """Path tag keeping CLIP's two feature spaces out of one another's cache."""
    if args.architecture == CLIP_ARCH and args.clip_feature != DEFAULT_CLIP_FEATURE:
        return f"_{args.clip_feature}"
    return ""


def probe_checkpoint(load_weights, ckpt_path, *, label, probe_dir, epoch, datasets,
                     preprocessing, image_size, batch_size, num_workers, device, force,
                     data_root, download, probe_solver, probe_max_iter, logger,
                     feature_cache_dir=None, max_train_samples=None):
    """Load `ckpt_path` and run the linear probe on it; return {dataset: result_block}.

    Returns each dataset's block verbatim rather than projecting it to a scalar, so a
    caller wanting only top-1 reads `block["score"]` and one wanting both metrics reads
    `block["top5_accuracy"]` too -- no second vocabulary, and no metric can be dropped
    on the way out.

    `load_weights` returns the model to probe -- the same object mutated in place
    for the timm architectures, a freshly-built one for CLIP.

    `probe_dir` is where run_linear_probe caches features and per-dataset results.
    It must uniquely identify the CHECKPOINT, since that cache is keyed only on
    dataset + epoch + preprocessing tag -- two different checkpoints sharing a
    probe_dir at the same epoch would silently reuse each other's features.

    Raises if NO dataset scored. `run_linear_probe` catches per-dataset failures so
    one bad dataset can't sink a sweep, but when every dataset fails the cause is
    systemic (a missing dependency, an unreachable data root, OOM) and will fail
    identically for every remaining checkpoint. Continuing there just burns
    checkpoint loads and writes result JSONs with empty accuracies.
    """
    logger.info("  [load] %s <- %s", label, ckpt_path)
    model = load_weights(ckpt_path)
    results = run_linear_probe(
        model,
        output_dir=probe_dir,
        epoch=epoch,
        preprocessing=preprocessing,
        datasets=datasets,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        force=force,
        data_root=data_root,
        download=download,
        probe_solver=probe_solver,
        probe_max_iter=probe_max_iter,
        max_train_samples=max_train_samples,
        feature_cache_dir=feature_cache_dir,
        logger=logger,
    )
    scored = {
        ds: results[ds] for ds in datasets
        if ds in results and results[ds].get("score") is not None
    }
    if not scored:
        raise RuntimeError(
            f"Linear probe scored 0/{len(datasets)} datasets for {label} at "
            f"{ckpt_path}. This is systemic, not dataset-specific -- every remaining "
            f"checkpoint would fail the same way. See the per-dataset *.error.txt "
            f"under {probe_dir} for the underlying traceback."
        )
    return scored


def discover_runs(runs_root, architecture, include, perturb_type, phase, epochs_arg, logger):
    """Every run to probe for one architecture, each with its epoch list.

    The baseline is a target row here, not a comparison partner.

    Default epoch is the run's LAST checkpoint: the fully-trained model, which is what
    a per-architecture benchmark row means. The onset default belongs to the pair
    driver, and would be actively wrong here -- `run_hparams.onset_epoch` returns 0
    rather than None for a baseline, so an onset field would read as "onset at 0".
    """
    prefix = ARCH_PREFIX[architecture]
    root = Path(runs_root)

    names = []
    if include in ("all", "baseline"):
        names.append((f"{prefix}_baseline", None, None))
    if include in ("all", "perturbed"):
        for pt in ([perturb_type] if perturb_type else PERTURB_TYPES):
            for ph in ([phase] if phase else PHASES):
                names.append((f"{prefix}_{pt}_{ph}", pt, ph))

    entries = []
    for name, pt, ph in names:
        run_dir = root / name
        if not run_dir.is_dir():
            logger.warning("[warn] missing run dir, skipping: %s", name)
            continue
        available = available_scratch_epochs(run_dir)
        if not available:
            logger.warning("[warn] no checkpoints, skipping: %s", name)
            continue
        if epochs_arg == "onset":
            # Each run's own perturbation onset, read from its hparams.yaml. NOT usable for
            # baselines: run_hparams.onset_epoch returns 0 (not None) for them, which would
            # silently probe epoch 0 instead of erroring.
            if pt is None:
                logger.warning("[warn] --epochs onset is meaningless for the baseline run "
                               "%s (onset_epoch returns 0); skipping", name)
                continue
            onset = _onset_epoch(run_dir)
            epochs = [onset] if onset in available else []
            if not epochs:
                logger.warning("[warn] onset epoch %s not available for %s (have %d..%d)",
                               onset, name, min(available), max(available))
        elif epochs_arg:
            epochs = sorted(epochs_arg & available)
        else:
            epochs = [max(available)]
        if not epochs:
            logger.warning("[warn] none of --epochs exist for %s (available %d..%d), skipping",
                           name, min(available), max(available))
            continue
        entries.append({"run_dir": run_dir, "name": name, "type": pt, "phase": ph,
                        "is_baseline": pt is None, "epochs": epochs})
    return entries


def _is_complete(out_json, datasets, solver, preprocessing_tag):
    """Whether an existing record already covers every requested dataset with both metrics.

    A change of solver or of preprocessing invalidates the whole record, for the same
    reason the pair driver rejects them: mixing fits or mixing input normalizations
    across an architecture's rows makes the table incomparable. A record written before
    top-5 existed lacks the key and is likewise rejected -- re-fitting is cheap because
    `run_linear_probe` caches the extracted features separately.
    """
    if not out_json.exists():
        return False
    try:
        with open(out_json) as f:
            cached = json.load(f)
    except (OSError, ValueError):
        return False
    if cached.get("solver") != solver or cached.get("preprocessing") != preprocessing_tag:
        return False
    blocks = cached.get("results", {})
    # Per-block, not record-level: a merged record can hold blocks from before the
    # C-sweep existed next to swept ones, and only the former need re-fitting.
    return all(
        ds in blocks and "top5_accuracy" in blocks[ds]
        and is_current_recipe(blocks[ds])
        for ds in datasets
    )


def main():
    args = parse_args()
    if args.architecture is None:
        if args.checkpoint:
            raise SystemExit("--checkpoint requires --architecture.")
        failed = [a for a in sorted(ARCH_PREFIX)
                  if subprocess.run([sys.executable, __file__, "--architecture", a,
                                     *sys.argv[1:]]).returncode]
        if failed:
            sys.exit(f"failed: {failed}")
        return
    args.runs_root = args.runs_root or paths.path("runs_root")
    args.data_root = args.data_root or str(paths.path("probe_datasets"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("linear_probe_eval")
    device = torch.device(args.device)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    epochs_arg = ("onset" if args.epochs == "onset"
                  else {int(e) for e in args.epochs.split(",")} if args.epochs else None)

    if args.checkpoint:
        if args.epoch is None or not args.run_name:
            raise SystemExit("--checkpoint requires --epoch and --run_name.")
        entries = [{"name": args.run_name, "type": None, "phase": None, "is_baseline": None,
                    "checkpoint": Path(args.checkpoint), "epochs": [args.epoch]}]
    else:
        if args.epoch is not None:
            raise SystemExit("--epoch belongs to --checkpoint; use --epochs for discovery.")
        entries = discover_runs(args.runs_root, args.architecture, args.runs,
                                args.perturb_type, args.phase, epochs_arg, logger)

    logger.info("[sweep] architecture=%s runs=%d", args.architecture, len(entries))

    if args.dry_run:
        for e in entries:
            print(f"{e['name']}  baseline={e['is_baseline']}  epochs={e['epochs']}")
        return

    out_root = resolve_output_dir(args.output_dir, args.architecture, args.clip_feature)
    _, load_weights = build_scratch_model(args.architecture, device, args.clip_feature)
    preprocessing = eval_preprocessing(args.architecture)
    logger.info("[sweep] preprocessing=%s (mean=%s std=%s, resize x%.4f, %s)",
                preprocessing.tag, preprocessing.mean, preprocessing.std,
                preprocessing.resize_ratio, preprocessing.interpolation)
    logger.info("[sweep] output root: %s", out_root)

    n_written = n_skipped = 0
    for e in entries:
        out_dir = out_root / e["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in e["epochs"]:
            out_json = out_dir / f"linear_probe_eval_epoch{epoch}.json"
            if not args.force and _is_complete(out_json, datasets, args.probe_solver,
                                               preprocessing.tag):
                logger.info("[skip] %s epoch%d (cached)", e["name"], epoch)
                n_skipped += 1
                continue

            ckpt = e.get("checkpoint") or scratch_checkpoint_at_epoch(e["run_dir"], epoch)
            blocks = probe_checkpoint(
                load_weights, ckpt, label=e["name"], probe_dir=out_dir / "probe",
                epoch=epoch, datasets=datasets, preprocessing=preprocessing,
                image_size=args.image_size, batch_size=args.batch_size,
                num_workers=args.num_workers, device=device, force=args.force,
                data_root=args.data_root, download=not args.no_download,
                probe_solver=args.probe_solver, probe_max_iter=args.probe_max_iter,
                max_train_samples=args.max_train_samples,
                # The variant MUST be in the cache path: run_linear_probe keys cached
                # features on dataset + epoch + preprocessing tag only, so without it a
                # `projected` run silently reuses `cls` 768-d features. Same inversion as
                # resolve_output_dir -- the default `cls` keeps the untagged path.
                feature_cache_dir=(Path(args.feature_cache) / e["name"]
                                   / f"epoch{epoch}{_cache_variant(args)}"),
                logger=logger,
            )

            # Merge rather than replace: this record is keyed on (run, epoch), but a given
            # invocation only knows the datasets IT was asked for, so writing it wholesale
            # would drop datasets measured by an earlier run with a different --datasets.
            prior = {}
            if out_json.exists() and not args.force:
                try:
                    cached = json.loads(out_json.read_text())
                    if (cached.get("solver") == args.probe_solver
                            and cached.get("preprocessing") == preprocessing.tag):
                        prior = cached.get("results", {})
                except (OSError, ValueError):
                    pass
            merged = {**prior, **blocks}

            result = {
                "architecture": args.architecture,
                "run_name": e["name"],
                "is_baseline": e["is_baseline"],
                "perturb_type": e["type"], "phase": e["phase"],
                "epoch": epoch,
                "checkpoint": Path(ckpt).name,
                "clip_feature": args.clip_feature if args.architecture == CLIP_ARCH else None,
                "datasets": sorted(merged),
                "datasets_this_run": datasets,
                "solver": args.probe_solver,
                "preprocessing": preprocessing.tag,
                "results": merged,
            }
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
            n_written += 1
            for ds, blk in blocks.items():
                top5 = blk.get("top5_accuracy")
                logger.info("[done] %s epoch%d %s: top1=%.4f top5=%s%s",
                            e["name"], epoch, ds, blk["score"],
                            "n/a" if top5 is None else f"{top5:.4f}",
                            "" if blk.get("converged", True) else "  [not-converged]")

        gc.collect()

    logger.info("[sweep] done: %d written, %d cached -> %s", n_written, n_skipped, out_root)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--architecture", default=None, choices=sorted(ARCH_PREFIX),
                   help="Which runs_root architecture family (note the CLIP entry has a "
                        "space, not a hyphen); default: all four, one subprocess each.")
    p.add_argument("--runs_root", default=None, help="Default: paths key runs_root.")
    p.add_argument("--runs", default="all", choices=["all", "baseline", "perturbed"],
                   help="Which runs to probe. 'baseline' alone gives the one-row-per-"
                        "architecture headline table.")
    p.add_argument("--perturb_type", default=None, choices=PERTURB_TYPES,
                   help="Restrict the perturbed runs to one type. Default: all three.")
    p.add_argument("--phase", default=None, choices=PHASES,
                   help="Restrict the perturbed runs to one onset phase. Default: all three.")
    p.add_argument("--epochs", default=None,
                   help="Comma-separated epochs to probe, or the literal 'onset' to use each "
                        "perturbed run's own onset epoch (read from its hparams.yaml). "
                        "Default: each run's FINAL epoch.")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint path; bypasses runs_root discovery. "
                        "Requires --epoch and --run_name.")
    p.add_argument("--epoch", type=int, default=None, help="Epoch label for --checkpoint.")
    p.add_argument("--run_name", default=None, help="Output subdir name for --checkpoint.")
    p.add_argument("--clip_feature", default=DEFAULT_CLIP_FEATURE, choices=sorted(CLIP_FEATURES),
                   help="CLIP-ViT-B/32 only (no-op for the timm architectures): which visual "
                        "representation to probe. "
                        f"'cls' (default) = {CLIP_FEATURES['cls']}; "
                        f"'projected' = {CLIP_FEATURES['projected']}. "
                        "Selects the output root so the two cannot pool into one table.")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--data_root", default=None,
                   help="Probe dataset root (holds vtab_export/). Default: paths key "
                        "probe_datasets.")
    p.add_argument("--feature_cache", default=str(DEFAULT_FEATURE_CACHE),
                   help="Where extracted features are cached, separate from --output_dir.")
    p.add_argument("--no_download", action="store_true",
                   help="Fail instead of downloading if the datasets aren't already "
                        "under --data_root.")
    p.add_argument("--probe_solver", default=DEFAULT_PROBE_SOLVER, choices=sorted(PROBE_SOLVERS),
                   help="Where the logistic regression runs. 'torch' (default) fits it with "
                        "full-batch L-BFGS on --device; 'sklearn' is the CPU reference. Both "
                        "minimize the same objective under the same stopping rule. Changing "
                        "this invalidates cached scores; extracted features are reused.")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Subsample train set to at most this many images (stratified). "
                        "Speeds up feature extraction for large datasets like ImageNet.")
    p.add_argument("--probe_max_iter", type=int, default=None,
                   help="L-BFGS iteration cap for the FINAL probe fit. Default: the solver's "
                        "own (20000 torch / 1000 sklearn). The C-sweep's selection fits are "
                        f"always capped at {PROBE_SWEEP_MAX_ITER} -- selection needs a ranking, "
                        "not convergence. "
                        "A capped final fit understates BOTH metrics, so watch for "
                        "[not-converged].")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_dir", default=None,
                   help=f"Default: {DEFAULT_OUTPUT_DIR} (a sibling root for "
                        "--clip_feature projected).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
