"""Post-hoc behavioral alignment over ALL 1854 THINGS images vs human SPoSE.

Same per-epoch sweep as the 48-image post-hoc evals, but over the full 1854
THINGS image set. Two complementary alignment metrics are computed per
checkpoint from a single forward pass:

  * RSA  — the target ("human") RDM is built from the 66-d human SPoSE
    embeddings (spose_embedding66d_rescaled.csv) using the same distance metric
    as the model RDM (exactly trainer.compute_behavioral_rsa's in-loop target);
    RSA is the Spearman correlation over the ~1.7M upper-triangular pairs of the
    1854x1854 RDMs.
  * CKA  — linear Centered Kernel Alignment (Kornblith et al. 2019) directly
    between the model embeddings and the 66-d SPoSE features, in the same image
    order. Unlike RSA, CKA compares feature spaces rather than rank-order of
    pairwise dissimilarities, so it is a useful cross-check on the RSA signal.

Both scores land in the same aggregate JSON (rsa_results / cka_results per
epoch). Re-running an already-RSA'd run adds CKA while preserving the cached RSA
number (fp16 forwards aren't bit-reproducible).

One script covers both model families, dispatched on the run-dir prefix:
  * timm scratch  (resnet50_imagenet / convnext_b_imagenet / vitb16_imagenet)
                  -- pooled penultimate feature
  * CLIP ViT-B/32 (clip_vitb32_*)
                  -- loaded via the ported `clip_yfcc` package, with
                  `--clip_feature` selecting the representation

CLIP feature space
------------------
`--clip_feature cls` (DEFAULT) scores the vision encoder's own 768-d CLS token
(post-`ln_post`, PRE-projection) -- the analog of the timm models' pooled
penultimate feature, and what the FR-RSA eval uses for this checkpoint family.
`--clip_feature projected` scores the 512-d L2-normalized joint image-text
embedding instead (the contrastive/retrieval space). `projected` keeps the
untagged aggregate filename and `cls` writes a `..._rsa_summary_cls.json`
sibling; see `_variant_suffix`.

Eval preprocessing
------------------
`--preprocessing auto` (DEFAULT) feeds each checkpoint the transform it was
TRAINED under, from `data.eval_transforms`: ImageNet stats with
Resize(256)->CenterCrop(224) for the timm families, CLIP stats with Resize(224)
for clip_vitb32_*. `--preprocessing things_legacy` instead uses
ThingsBehavioralDataset's built-in transform (THINGS image-set statistics with an
aspect-squashing Resize((224,224)), the CLIP-HBA recipe). `things_legacy` stays
untagged and every other recipe gets a trailing filename tag, composed after the
CLIP one; see `_preprocessing_suffix`.

Writes one aggregate JSON per run, schema-compatible with the 48-image sweeps so
the plotting code is shared, to `<results_root>/posthoc_rdm1854/<run>/`.

Usage
-----
    python pipeline/scripts/evaluation/run_behavioral_rsa.py                  # every run under runs_root
    python pipeline/scripts/evaluation/run_behavioral_rsa.py --run_name resnet50_imagenet_baseline
    python pipeline/scripts/evaluation/run_behavioral_rsa.py \\
        --run_name vitb16_imagenet_image_noise_late_recovery_dynamics --sub_epoch
"""

from pathlib import Path
import argparse
import gc
import json
import subprocess
import sys

EXP_DIR    = Path(__file__).resolve().parent
PIPELINE   = EXP_DIR.parent.parent
REPO_ROOT  = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from temporal_perturbations import paths
from temporal_perturbations.utils.load_yaml_config import load_yaml_config
from temporal_perturbations.utils.logging import setup_logger
from temporal_perturbations.data.things_dataset import ThingsBehavioralDataset
from temporal_perturbations.models.scratch.scratch_utils import (
    architecture_for_run_dir,
    build_cross_arch_model,
    embed_batched,
    sorted_scratch_checkpoints,
    sorted_subepoch_checkpoints,
    subepoch_parent_epoch,
)
from temporal_perturbations.inference.inference_core import (
    compute_model_rdm,
    compute_rdm_similarity,
    build_spose_target_rdm,
    compute_cka,
)
from temporal_perturbations.models.clip_yfcc.feature_space import (
    CLIP_FEATURES,
    DEFAULT_CLIP_FEATURE,
)
from temporal_perturbations.data.eval_transforms import (
    PREPROCESSING_CHOICES,
    THINGS_LEGACY,
    resolve_preprocessing,
)
import pandas as pd

DEFAULT_CONFIG = PIPELINE / "configs" / "inference_config_posthoc_1854.yaml"
RESULTS_SUBDIR = "posthoc_rdm1854"

INFERENCE_INT_KEYS = [
    "batch_size", "num_classes", "rank", "vision_layers",
    "transformer_layers", "random_seed", "cuda", "num_workers",
    "max_images_per_category",
]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Behavioral RSA sweep over the full 1854 THINGS images (SPoSE target)."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--run_name", type=str, default=None,
                   help="Run directory name under runs_root. Omit (with no --checkpoint_dir) "
                        "to score every run under runs_root, one subprocess per run.")
    p.add_argument("--runs_root", type=Path, default=None, help="Default: paths key runs_root.")
    p.add_argument("--checkpoint_dir", type=Path, default=None,
                   help="Default: <runs_root>/<run_name>/checkpoints/all_epochs "
                        "(checkpoints/sub_epoch with --sub_epoch).")
    p.add_argument("--save_path", type=Path, default=None,
                   help="Default: <results_root>/posthoc_rdm1854/<run_name>.")
    p.add_argument("--architecture", type=str, default=None,
                   help="Override the architecture inferred from the run dir.")
    p.add_argument("--clip_feature", default=DEFAULT_CLIP_FEATURE, choices=sorted(CLIP_FEATURES),
                   help="clip_vitb32_* runs only (ignored for the timm families): which "
                        "visual representation to score against SPoSE. "
                        f"'cls' (default) = {CLIP_FEATURES['cls']}; "
                        f"'projected' = {CLIP_FEATURES['projected']}. "
                        "See _variant_suffix for how each is kept on disk.")
    p.add_argument("--preprocessing", default="auto", choices=PREPROCESSING_CHOICES,
                   help="Eval-time image preprocessing. 'auto' (default) uses the recipe "
                        "this checkpoint was TRAINED under -- ImageNet stats + "
                        "Resize(256)->CenterCrop(224) for the timm families, CLIP stats + "
                        "Resize(224) for clip_vitb32_*. 'things_legacy' uses THINGS image-set "
                        "stats with an aspect-squashing resize (the CLIP-HBA recipe). "
                        "Non-legacy recipes get a trailing filename tag; see "
                        "_preprocessing_suffix.")
    p.add_argument("--img_dir", type=str, default=None,
                   help="Override the config's img_dir (the THINGS image root).")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Override the config's DataLoader worker count.")
    p.add_argument("--epochs", type=int, nargs="+", default=None,
                   help="Score only these epochs (default: every checkpoint under "
                        "--checkpoint_dir). Writes into the same aggregate JSON a full "
                        "sweep uses, so a later full pass fills in the rest without "
                        "rescoring these.")
    p.add_argument("--sub_epoch", action="store_true", default=False,
                   help="Evaluate sub-epoch checkpoints (epoch_NNN_step_SSSSS.ckpt). "
                        "--epochs filters by step number in this mode.")
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=False)
    p.add_argument("--n_permutations", type=int, default=10000,
                   help="Condition-label permutations for RSA p-values (Kriegeskorte et al., 2008).")
    return p.parse_args(argv)


def _variant_suffix(clip_feature):
    """Filename tag keeping the two CLIP feature spaces apart on disk.

    `projected` keeps the untagged name and `cls`, though the default, is tagged.
    The tag is TRAILING so a glob anchored on `rsa_summary.json` cannot pool the
    two spaces. Callers apply this ONLY to clip_vitb32_* runs; the timm families
    have one feature space and keep the untagged name.
    """
    return "" if clip_feature == "projected" else f"_{clip_feature}"


def _preprocessing_suffix(preprocessing):
    """Filename tag keeping eval-preprocessing variants apart on disk.

    `things_legacy` stays UNTAGGED and every other recipe -- including the default
    architecture-correct one -- gets a trailing tag, composed after the CLIP tag.
    Unlike `_variant_suffix` this applies to every family.
    """
    return "" if preprocessing.tag == THINGS_LEGACY.tag else f"_{preprocessing.tag}"


def _device_from_config(config):
    cuda = config.get("cuda", 0)
    return torch.device("cuda") if cuda == -1 else torch.device(f"cuda:{cuda}")


def _build_model(run_dir, architecture, device, logger, clip_feature=DEFAULT_CLIP_FEATURE):
    """Return (model, load_state_dict_fn, architecture_label) for the run's family."""
    arch_label = architecture or (
        "CLIP ViT-B/32" if run_dir.name.startswith("clip_vitb32")
        else architecture_for_run_dir(run_dir)
    )
    if arch_label == "CLIP ViT-B/32":
        logger.info("CLIP ViT-B/32: scoring the %s representation (%s).",
                    clip_feature, CLIP_FEATURES[clip_feature])
    else:
        logger.info("Building %s feature model...", arch_label)
    model, load_weights = build_cross_arch_model(arch_label, device, clip_feature)
    return model, load_weights, arch_label


def _preload_things_images(img_dir, annotations_file, batch_size, num_workers, preprocessing,
                           logger):
    """Decode all 1854 THINGS images once and keep them resident in RAM.

    Re-reading/decoding the images per checkpoint (1854 JPEGs x ~hundreds of
    checkpoints) is the dominant cost and is disk/CPU-bound; caching the
    preprocessed tensor turns each checkpoint into a pure GPU forward pass.

    `preprocessing` overrides ThingsBehavioralDataset's THINGS-statistics
    squash-resize (the CLIP-HBA recipe) with the checkpoint's own training recipe.
    """
    ds = ThingsBehavioralDataset(img_annotations_file=annotations_file, img_dir=img_dir,
                                 transform=preprocessing.build(224))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    names, chunks = [], []
    for image_names, images, _ in loader:
        names.extend(list(image_names))
        chunks.append(images)
    images = torch.cat(chunks, dim=0)
    gb = images.element_size() * images.nelement() / 1e9
    logger.info("  Preloaded %d THINGS images once -> %s (%.2f GB cached in RAM)",
                len(names), tuple(images.shape), gb)
    return images, names


def _embed_cached(model, images_cpu, batch_size, device):
    """Forward the cached image tensor through the model in batches."""
    arr = embed_batched(model, images_cpu, batch_size, device)
    return torch.from_numpy(arr)


def _build_spose_matrix(annotations_file, image_order):
    """Return the raw [N, 66] SPoSE embedding matrix in `image_order`.

    Same CSV convention as build_spose_target_rdm (index_col=0; first column is
    the image name, remaining columns the 66-d embedding), but returns the
    feature matrix itself rather than its RDM — CKA operates on features, not on
    pairwise dissimilarities.
    """
    df = pd.read_csv(annotations_file, index_col=0)
    name_col = df.columns[0]
    df = df.set_index(name_col).loc[list(image_order)].reset_index()
    return df.iloc[:, 1:].to_numpy(dtype=np.float64)


def _ckpt_subdir(sub_epoch):
    return "sub_epoch" if sub_epoch else "all_epochs"


def _all_runs(runs_root, sub_epoch):
    return sorted(d.name for d in runs_root.iterdir()
                  if (d / "checkpoints" / _ckpt_subdir(sub_epoch)).is_dir())


def main():
    args = _parse_args()
    runs_root = args.runs_root or paths.path("runs_root")
    if args.run_name is None and args.checkpoint_dir is None:
        failed = []
        for name in _all_runs(runs_root, args.sub_epoch):
            rc = subprocess.run([sys.executable, __file__, "--run_name", name,
                                 *sys.argv[1:]]).returncode
            if rc:
                failed.append(name)
        if failed:
            sys.exit(f"failed: {failed}")
        return

    if args.checkpoint_dir is None:
        args.checkpoint_dir = (runs_root / args.run_name / "checkpoints"
                               / _ckpt_subdir(args.sub_epoch))
    cwd = Path.cwd()

    config = load_yaml_config(args.config, numeric_keys=INFERENCE_INT_KEYS)

    def _resolve(p):
        if p is None:
            return None
        pth = Path(p)
        return pth if pth.is_absolute() else cwd / pth

    # CLI overrides land before path resolution so a relative override still works.
    if args.img_dir is not None:
        config["img_dir"] = args.img_dir
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    for key in ("img_dir", "annotations_file"):
        if config.get(key):
            config[key] = str(_resolve(config[key]))

    ckpt_dir = Path(args.checkpoint_dir)
    run_dir = ckpt_dir.parent.parent           # all_epochs|sub_epoch -> checkpoints -> <run_dir>
    run_name = args.run_name or run_dir.name

    base_save_dir = Path(args.save_path or paths.results_root() / RESULTS_SUBDIR / run_name)
    dataset = config.get("dataset", "things")
    eval_type = config["evaluation_type"]

    # Only clip_vitb32_* runs have two feature spaces (see _variant_suffix).
    is_clip_run = run_dir.name.startswith("clip_vitb32")
    variant = _variant_suffix(args.clip_feature) if is_clip_run else ""

    # The recipe this checkpoint was TRAINED under. `is_clip_run` decides the family
    # rather than the architecture label, since --architecture may override the label
    # but cannot change which pipeline produced the weights.
    arch_for_pp = None if is_clip_run else (args.architecture or architecture_for_run_dir(run_dir))
    preprocessing = resolve_preprocessing(
        args.preprocessing, arch_for_pp,
        "clip_contrastive" if is_clip_run else "scratch_lightning",
    )
    variant += _preprocessing_suffix(preprocessing)

    agg_json = base_save_dir / f"test_{dataset}_{eval_type}_full1854_rsa_summary{variant}.json"

    base_save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(base_save_dir / f"sweep_log_{dataset}_{eval_type}_full1854{variant}.txt"))
    logger.info("=== Full-1854 behavioral RSA sweep started ===")
    logger.info("  run_name       : %s", run_name)
    logger.info("  checkpoint_dir : %s", ckpt_dir)
    logger.info("  annotations    : %s", config["annotations_file"])

    device = _device_from_config(config)
    logger.info("  device         : %s", device)

    model, load_state_dict_fn, architecture = _build_model(
        run_dir, args.architecture, device, logger, args.clip_feature,
    )
    logger.info("  architecture   : %s", architecture)
    logger.info("  preprocessing  : %s (%s resize, mean=%s, %s)",
                preprocessing.tag, preprocessing.resize_mode, preprocessing.mean,
                preprocessing.interpolation)
    if is_clip_run:
        logger.info("  clip_feature   : %s", args.clip_feature)
    logger.info("  aggregate JSON : %s", agg_json.name)

    rsa_similarity_metric = config["rsa_similarity_metric"]
    model_rdm_distance_metric = config["model_rdm_distance_metric"]
    reference_rdm_distance_metric = config["reference_rdm_distance_metric"]

    if args.sub_epoch:
        checkpoints = sorted_subepoch_checkpoints(ckpt_dir)
        _parent_epoch = subepoch_parent_epoch(ckpt_dir)
        _total_steps = max(s for s, _ in checkpoints) if checkpoints else 0
        logger.info("Sub-epoch mode: parent_epoch=%d, %d checkpoints, steps %d..%d",
                    _parent_epoch, len(checkpoints),
                    checkpoints[0][0] if checkpoints else 0,
                    checkpoints[-1][0] if checkpoints else 0)
    else:
        checkpoints = sorted_scratch_checkpoints(ckpt_dir)
        _parent_epoch = None
        _total_steps = None

    if args.epochs:
        wanted = set(args.epochs)
        checkpoints = [(e, c) for e, c in checkpoints if e in wanted]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    logger.info("Found %d checkpoints (%s %d..%d).",
                len(checkpoints),
                "steps" if args.sub_epoch else "epochs",
                checkpoints[0][0], checkpoints[-1][0])

    if args.dry_run:
        label = "step" if args.sub_epoch else "epoch"
        for epoch, ckpt_path in checkpoints:
            logger.info("  %s %5d  %s", label, epoch, ckpt_path.name)
        logger.info("--- aggregate JSON would be: %s ---", agg_json)
        return

    cka_unbiased = bool(config.get("cka_unbiased", False))

    agg = {
        "dataset": dataset,
        "evaluation_type": eval_type,
        "run_name": run_name,
        "architecture": architecture,
        # Which CLIP representation these numbers come from; None for the timm
        # families. Records predating this field were produced with 'projected'.
        "clip_feature": args.clip_feature if is_clip_run else None,
        # Eval preprocessing these numbers came from. Records predating this field
        # were produced under 'things_legacy'.
        "preprocessing": preprocessing.tag,
        "n_images": 1854,
        "description": "Post-hoc behavioral alignment over all 1854 THINGS images. "
                       "RSA: Spearman over upper-tri of model vs SPoSE RDMs. "
                       "CKA: linear Centered Kernel Alignment between model "
                       "embeddings and the 66-d SPoSE features (feature-space, "
                       "no RDM).",
        "results": [],
    }
    if args.sub_epoch:
        agg["sub_epoch"] = True
        agg["parent_epoch"] = _parent_epoch
        agg["total_steps"] = _total_steps
    if agg_json.exists():
        with open(agg_json) as f:
            agg = json.load(f)
        agg["preprocessing"] = preprocessing.tag
        if args.sub_epoch:
            agg["sub_epoch"] = True
            agg["parent_epoch"] = _parent_epoch
            agg["total_steps"] = _total_steps

    # Existing records, indexed by epoch, so we can add whichever metric is
    # missing without recomputing (and without perturbing) the other. RSA was
    # computed on an earlier pass for most runs; only CKA is new there.
    existing = {str(r.get("epoch")): r for r in agg.get("results", [])
                if r.get("run_name") == run_name}

    def _has_rsa(ep):
        return bool(existing.get(str(ep), {}).get("rsa_results"))

    def _has_cka(ep):
        return bool(existing.get(str(ep), {}).get("cka_results"))

    todo = [
        (epoch, ckpt_path) for epoch, ckpt_path in checkpoints
        if args.force or not (_has_rsa(epoch) and _has_cka(epoch))
    ]
    if not todo:
        logger.info("All %d checkpoints already scored (RSA+CKA) for %s — nothing to do.",
                    len(checkpoints), run_name)
        return

    # Decode the 1854 images once; every checkpoint reuses the cached tensor so
    # the per-checkpoint cost is a pure GPU forward pass rather than 1854 JPEG reads.
    images_cpu, image_names = _preload_things_images(
        config["img_dir"], config["annotations_file"],
        config["batch_size"], config["num_workers"], preprocessing, logger,
    )
    reference_vec = build_spose_target_rdm(
        config["annotations_file"], reference_rdm_distance_metric, image_order=image_names,
    )
    n_images = len(image_names)
    reference_rdm_full = np.zeros((n_images, n_images), dtype=np.float64)
    reference_rdm_full[np.triu_indices(n_images, k=1)] = reference_vec
    reference_rdm_full += reference_rdm_full.T
    spose_matrix = _build_spose_matrix(config["annotations_file"], image_names)
    logger.info("  SPoSE target RDM upper-tri length: %d  (metric=%s); "
                "SPoSE feature matrix %s for CKA (unbiased=%s)",
                len(reference_vec), reference_rdm_distance_metric,
                tuple(spose_matrix.shape), cka_unbiased)

    skipped = []
    for epoch, ckpt_path in todo:
        logger.info("\n=== %s %s  (%s) ===",
                    "Step" if args.sub_epoch else "Epoch", epoch, ckpt_path.name)
        try:
            model = load_state_dict_fn(str(ckpt_path))
            embeddings = _embed_cached(model, images_cpu, config["batch_size"], device)
        except Exception as exc:  # corrupt/truncated ckpt, OOM, etc. — skip, keep the sweep going
            logger.warning("  SKIP epoch %s (%s): %s: %s",
                           epoch, ckpt_path.name, type(exc).__name__, exc)
            skipped.append(epoch)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        logger.info("  Embeddings shape: %s", tuple(embeddings.shape))

        record = existing.get(str(epoch), {
            "epoch": str(epoch),
            "run_name": run_name,
            "dataset": dataset,
            "evaluation_type": eval_type,
            "architecture": architecture,
        })
        if args.sub_epoch and "sub_epoch" not in record:
            record["sub_epoch"] = True
            record["parent_epoch"] = _parent_epoch
            record["step"] = epoch
            record["total_steps"] = _total_steps

        # RSA — preserve any already-cached score (fp16 forwards aren't
        # bit-reproducible; don't silently shift published RSA numbers).
        if args.force or not _has_rsa(epoch):
            model_rdm = compute_model_rdm(
                embeddings, dataset_name=dataset, annotations_file=config["annotations_file"],
                categories=None, distance_metric=model_rdm_distance_metric,
            )

            ref_vec_epoch = reference_vec
            n_dropped = 0
            if np.isnan(model_rdm).any():
                n = model_rdm.shape[0]
                off_diag_nan = np.array([
                    np.isnan(model_rdm[i, np.arange(n) != i]).all() for i in range(n)
                ])
                nan_rows = np.where(off_diag_nan)[0]
                if len(nan_rows) > 0:
                    logger.warning("  Dropping %d degenerate (zero-variance) images "
                                   "from RDM: indices %s", len(nan_rows), nan_rows.tolist())
                    keep = np.ones(model_rdm.shape[0], dtype=bool)
                    keep[nan_rows] = False
                    model_rdm = model_rdm[np.ix_(keep, keep)]
                    ref_sub = reference_rdm_full[np.ix_(keep, keep)]
                    ref_vec_epoch = ref_sub[np.triu_indices_from(ref_sub, k=1)]
                    n_dropped = len(nan_rows)

            model_rdm_vec = model_rdm[np.triu_indices_from(model_rdm, k=1)]
            if model_rdm_vec.shape != ref_vec_epoch.shape:
                raise ValueError(
                    f"Shape mismatch: model {model_rdm_vec.shape} vs reference {ref_vec_epoch.shape}"
                )
            n_permutations = args.n_permutations
            rho, p_value = compute_rdm_similarity(
                model_rdm_vec, ref_vec_epoch, similarity_metric=rsa_similarity_metric,
                n_permutations=n_permutations,
            )
            rsa_block = {
                    "epoch": str(epoch),
                    "evaluation_type": eval_type,
                    "dataset": dataset,
                    "reference_rdm_name": "spose_1854",
                    "score": float(rho),
                    "p_value": float(p_value),
                    "p_method": "permutation",
                    "n_permutations": n_permutations,
                    "rsa_similarity_metric": rsa_similarity_metric,
                    "model_rdm_distance_metric": model_rdm_distance_metric,
                    "reference_rdm_distance_metric": reference_rdm_distance_metric,
            }
            if n_dropped > 0:
                rsa_block["n_degenerate_images_dropped"] = n_dropped
            record["rsa_results"] = {"behavioral_rdm": rsa_block}
            del model_rdm, model_rdm_vec
            logger.info("  RSA rho=%.4f  p=%.4g", rho, p_value)
        else:
            logger.info("  RSA cached (%.4f) — preserved.",
                        record["rsa_results"]["behavioral_rdm"]["score"])

        # CKA — feature-space alignment of model embeddings vs SPoSE features.
        emb_np = embeddings.detach().cpu().numpy().astype(np.float64)
        cka = compute_cka(emb_np, spose_matrix, unbiased=cka_unbiased)
        record["cka_results"] = {
            "behavioral_cka": {
                "epoch": str(epoch),
                "evaluation_type": eval_type,
                "dataset": dataset,
                "reference_name": "spose_1854",
                "score": float(cka),
                "kernel": "linear",
                "unbiased": cka_unbiased,
            }
        }
        logger.info("  CKA=%.4f", cka)

        agg.setdefault("results", [])
        agg["results"] = [
            r for r in agg["results"]
            if not (str(r.get("epoch")) == str(epoch) and r.get("run_name") == run_name)
        ]
        agg["results"].append(record)
        existing[str(epoch)] = record
        agg["results"].sort(key=lambda r: (
            int(r["epoch"]) if str(r.get("epoch", "")).isdigit() else 10 ** 9,
            str(r.get("run_name", "")),
        ))
        with open(agg_json, "w") as f:
            json.dump(agg, f, indent=2)

        del embeddings, emb_np
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if skipped:
        logger.warning("Skipped %d unreadable checkpoint(s): epochs %s",
                       len(skipped), skipped)
    logger.info("\n=== Sweep complete. Aggregate results: %s ===", agg_json)


if __name__ == "__main__":
    main()
