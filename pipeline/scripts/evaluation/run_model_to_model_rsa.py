"""Model-to-model RSA/CKA sweep for the from-scratch (`runs_root`) models.

For ResNet50 / ViT-B/16 / ConvNeXt-B / CLIP-ViT-B/32: for every perturbed run,
compare that run's representations against its architecture's baseline run AT
MATCHED TRAINING EPOCHS, producing the divergence trajectory (onset spike +
recovery). Each architecture has three onsets per perturbation type and a single
seed, so the informative axis is TRAINING EPOCH: one divergence trajectory per
onset condition.

Representation compared
-----------------------
The pooled penultimate (pre-logits) feature -- RN50 2048-d, ConvNeXt-B 1024-d,
ViT-B/16 768-d -- the natural analog of CLIP-HBA's 66-d readout.

For CLIP-ViT-B/32 the analogous choice is the vision encoder's own 768-d CLS
token (post-`ln_post`, PRE-projection, unnormalized) -- `--clip_feature cls`,
the default. The contrastive `proj` head is trained to align images with
captions rather than to preserve visual similarity structure, so scoring after
it measures the caption-alignment geometry instead of the encoder's visual
representation; `CLIPVisionCLSExtractor` is also what the FR-RSA eval uses.

`--clip_feature projected` scores the 512-d L2-normalized joint image-text
embedding (the zero-shot/retrieval space) instead. The two write to separate
output roots -- see `resolve_output_dir`.

No frozen-prefix shortcut is possible: every layer of these models changes each
epoch. What IS cached is the *preprocessed image tensor* for each stimulus set,
built once and reused for every checkpoint, so the 8640-image THINGS-fMRI set
is decoded exactly once instead of once per checkpoint.

All three measures (RSA-Pearson, RSA-Spearman, CKA) are computed per comparison
via the shared `compare_embeddings`. Results land under
`<results_root>/model_to_model_rsa_scratch[_clip_<feature>][_pp-<tag>][_bnrecal]/<run>/`.

Usage:
    python pipeline/scripts/evaluation/run_model_to_model_rsa.py      # all four architectures
    python pipeline/scripts/evaluation/run_model_to_model_rsa.py --architecture RN50 --recalibrate_bn

    # Plan only, no compute:
    python pipeline/scripts/evaluation/run_model_to_model_rsa.py --architecture ViT-B/16 --dry-run
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths
from temporal_perturbations.data.eval_transforms import (
    PREPROCESSING_CHOICES,
    THINGS_LEGACY,
    resolve_preprocessing,
)
from temporal_perturbations.data.things_dataset import ThingsBehavioralDataset, ThingsFMRIDataset
from temporal_perturbations.inference.model_to_model_comparison import (
    compare_embeddings,
    load_stimulus_configs,
)
from temporal_perturbations.models.scratch.scratch_utils import (
    ARCH_PREFIX,
    CLIP_ARCH,
    available_scratch_epochs,
    build_cross_arch_model,
    embed_batched,
    sorted_scratch_checkpoints,
)
from temporal_perturbations.models.clip_yfcc.feature_space import (
    CLIP_FEATURES,
    DEFAULT_CLIP_FEATURE,
)
from temporal_perturbations.utils.run_hparams import onset_epoch as _onset_epoch

PERTURB_TYPES = ["image_noise", "label_shuffle", "uniform_images"]
PHASES = ["early", "mid", "late"]

# things_fmri_720 only; the 48-image behavioral set is a thin basis for an RDM
# (1128 pairs against 258,840) and no figure plots it. Pass it via --config to add it.
DEFAULT_CONFIGS = [
    "pipeline/configs/model_to_model_rsa.yaml",
]
DEFAULT_OUTPUT_DIR = paths.results_root() / "model_to_model_rsa_scratch"

# Epochs sampled relative to onset, where the divergence spike and the fast part
# of recovery live. Combined with a coarse global stride for the long tail.
ONSET_OFFSETS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]

BN_CALIB_IMAGES = 8192
BN_CALIB_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.PILToTensor(),
])
BN_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
BN_IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_bn_calib_set(imagenet_train, n_images, logger):
    per_class = max(1, n_images // 1000)
    xs = []
    class_dirs = sorted(p for p in imagenet_train.iterdir() if p.is_dir())
    for cdir in class_dirs:
        for f in sorted(cdir.iterdir())[:per_class]:
            try:
                xs.append(BN_CALIB_TF(Image.open(f).convert('RGB')))
            except OSError:
                continue
        if len(xs) >= n_images:
            break
    x = torch.stack(xs[:n_images])
    logger.info("[bn] calibration set: %s (%d images)", tuple(x.shape), len(x))
    return x


@torch.no_grad()
def recalibrate_bn(model, calib_x, device, batch_size, logger):
    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for i in range(0, len(calib_x), batch_size):
        batch = calib_x[i:i + batch_size].to(device, non_blocking=True).float().div_(255)
        batch = (batch - BN_IN_MEAN.to(device)) / BN_IN_STD.to(device)
        model(batch)
    for m in bns:
        m.eval()
    logger.info("[bn] recalibrated %d BatchNorm layers on %d images", len(bns), len(calib_x))
    return len(bns)


def resolve_output_dir(output_dir, architecture, clip_feature, preprocessing=THINGS_LEGACY,
                       bn_recalibrated=False):
    """Where this (architecture, clip_feature, preprocessing) combination writes.

    An explicit --output_dir is honored verbatim. Otherwise each variant that yields
    non-interchangeable numbers gets its OWN ROOT, because they must never be pooled:

      * CLIP's two feature spaces: `projected` keeps the plain root and `cls` -- the
        default -- gets a `_clip_cls` root.
      * The eval preprocessing: `things_legacy` keeps the untagged root and any other
        recipe gets a `_pp-<tag>` sibling.

    A filename suffix would NOT achieve either separation: the figure scripts glob
    `*/model_to_model_rsa_*_epoch*.json`, which matches any tag placed after `_epoch`,
    so both variants would load into one dataframe. A distinct root is the only
    separation that glob respects.
    """
    if output_dir is not None:
        return Path(output_dir)
    name = DEFAULT_OUTPUT_DIR.name
    if architecture == CLIP_ARCH and clip_feature != "projected":
        name = f"{name}_clip_{clip_feature}"
    if preprocessing.tag != THINGS_LEGACY.tag:
        name = f"{name}_pp-{preprocessing.tag}"
    if bn_recalibrated:
        name = f"{name}_bnrecal"
    return DEFAULT_OUTPUT_DIR.parent / name


def _epoch_grid(onset, available, stride):
    dense = {onset + d for d in ONSET_OFFSETS}
    coarse = set(range(0, max(available) + 1, stride))
    return sorted((dense | coarse | {max(available)}) & available)


def discover_runs(runs_root, architecture, stride, logger, seed_suffix=""):
    """Baseline dir + one entry per perturbed run, each with its epoch grid.

    `seed_suffix` (e.g. "_seed43") selects a replicate family; each of its perturbed
    runs is paired with that family's own baseline, the run it forked from.
    """
    prefix = ARCH_PREFIX[architecture]
    root = Path(runs_root)

    baseline_dir = root / f"{prefix}_baseline{seed_suffix}"
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"No baseline run at {baseline_dir}")
    baseline_avail = available_scratch_epochs(baseline_dir)

    runs = []
    for ptype in PERTURB_TYPES:
        for phase in PHASES:
            name = f"{prefix}_{ptype}_{phase}{seed_suffix}"
            run_dir = root / name
            if not run_dir.is_dir():
                logger.warning("[warn] missing run dir, skipping: %s", run_dir.name)
                continue
            available = available_scratch_epochs(run_dir)
            if not available:
                logger.warning("[warn] no checkpoints, skipping: %s", run_dir.name)
                continue
            onset = _onset_epoch(run_dir)
            # Only epochs present in BOTH runs can be compared.
            epochs = [e for e in _epoch_grid(onset, available, stride) if e in baseline_avail]
            runs.append({
                "run_dir": run_dir, "name": name, "type": ptype,
                "phase": phase, "onset": onset, "epochs": epochs,
            })
    return baseline_dir, runs


def pair_runs(run_a, run_b, stride, epochs_arg, logger):
    """Single explicit run-pair instead of the perturbed-vs-baseline discovery.

    Used for the seed control: two *unperturbed* baseline runs compared at
    matched epochs, which bounds how far apart two training trajectories drift
    with no perturbation at all. Note this is a CEILING, not a matched control --
    different-seed baselines are independent from epoch 0, whereas a perturbed
    run forks from the baseline at its onset epoch and shares all history before
    that.
    """
    run_a, run_b = Path(run_a), Path(run_b)
    for d in (run_a, run_b):
        if not d.is_dir():
            raise FileNotFoundError(f"No run dir at {d}")
    shared = available_scratch_epochs(run_a) & available_scratch_epochs(run_b)
    if not shared:
        raise RuntimeError(f"No common checkpoint epochs between {run_a.name} and {run_b.name}")

    if epochs_arg:
        epochs = sorted({int(e) for e in epochs_arg.split(",") if e.strip()} & shared)
    else:
        epochs = sorted(set(range(0, max(shared) + 1, stride)) | {min(shared), max(shared)})
        epochs = [e for e in epochs if e in shared]
    logger.info("[pair] %s vs %s: %d epochs", run_a.name, run_b.name, len(epochs))
    return run_b, [{
        "run_dir": run_a, "name": f"{run_a.name}_vs_{run_b.name}",
        "type": "none", "phase": "seed_control", "onset": None, "epochs": epochs,
    }]


def build_stimulus_cache(cfg, preprocessing, logger):
    """Decode + preprocess every stimulus ONCE; reused for every checkpoint.

    `preprocessing` overrides the dataset classes' THINGS-statistics squash-resize
    (the CLIP-HBA recipe) with the checkpoint's own training recipe.
    """
    import pandas as pd
    transform = preprocessing.build(224)
    has_concepts = "concept" in pd.read_csv(cfg["annotations_file"], nrows=1).columns
    if has_concepts:
        dataset = ThingsFMRIDataset(img_annotations_file=cfg["annotations_file"],
                                    img_dir=cfg["img_dir"], transform=transform)
    else:
        dataset = ThingsBehavioralDataset(img_annotations_file=cfg["annotations_file"],
                                          img_dir=cfg["img_dir"], transform=transform)

    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=False)
    names, categories, chunks = [], [], []
    for batch in loader:
        if has_concepts:
            image_names, images, cats = batch
            categories.extend(list(cats))
        else:
            image_names, images, _ = batch
        names.extend(list(image_names))
        chunks.append(images)

    images = torch.cat(chunks, dim=0)
    logger.info("[cache] %s: images %s (%.2f GB), %d stimuli, %d concepts",
                cfg["stimulus_set"], tuple(images.shape),
                images.element_size() * images.nelement() / 1e9,
                len(names), len(set(categories)) if categories else len(names))
    return {"images": images, "names": names, "categories": categories}


def build_model(architecture, device, clip_feature=DEFAULT_CLIP_FEATURE):
    return build_cross_arch_model(architecture, device, clip_feature)


def embed(model, cache, device, batch_size):
    return embed_batched(model, cache["images"], batch_size, device)


def main():
    args = parse_args()
    if args.architecture is None:
        if args.run_a or args.run_b:
            raise SystemExit("--run_a/--run_b require --architecture.")
        failed = [a for a in sorted(ARCH_PREFIX)
                  if subprocess.run([sys.executable, __file__, "--architecture", a,
                                     *sys.argv[1:]]).returncode]
        if failed:
            sys.exit(f"failed: {failed}")
        return
    args.runs_root = args.runs_root or paths.path("runs_root")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("scratch_model_to_model_rsa")
    device = torch.device(args.device)

    if args.run_a or args.run_b:
        if not (args.run_a and args.run_b):
            raise SystemExit("--run_a and --run_b must be given together.")
        baseline_dir, runs = pair_runs(args.run_a, args.run_b, args.epoch_stride, args.epochs, logger)
    else:
        baseline_dir, runs = discover_runs(args.runs_root, args.architecture, args.epoch_stride, logger,
                                           args.seed_suffix)
    logger.info("[sweep] architecture=%s reference=%s", args.architecture, baseline_dir.name)
    total_pairs = sum(len(r["epochs"]) for r in runs)
    baseline_epochs = sorted({e for r in runs for e in r["epochs"]})
    logger.info("[sweep] %d perturbed runs, %d (run, epoch) pairs, %d distinct baseline epochs",
                len(runs), total_pairs, len(baseline_epochs))

    if args.dry_run:
        for r in runs:
            print(f"{r['name']}  onset={r['onset']}  {len(r['epochs'])} epochs: {r['epochs']}")
        print(f"\nbaseline epochs ({len(baseline_epochs)}): {baseline_epochs}")
        print(f"total checkpoint loads: {total_pairs + len(baseline_epochs)}")
        return

    preprocessing = resolve_preprocessing(args.preprocessing, args.architecture)
    logger.info("[sweep] preprocessing=%s (%s resize, mean=%s std=%s, %s)",
                preprocessing.tag, preprocessing.resize_mode, preprocessing.mean,
                preprocessing.std, preprocessing.interpolation)

    stimulus_configs = load_stimulus_configs(args.config, REPO_ROOT)
    if args.img_dir:
        # Applied to EVERY stimulus config, bypassing load_stimulus_configs'
        # single-config img_dir_override.
        for cfg in stimulus_configs:
            cfg["img_dir"] = str(args.img_dir)
        logger.info("[sweep] img_dir overridden for all %d stimulus set(s): %s",
                    len(stimulus_configs), args.img_dir)
    caches = {cfg["stimulus_set"]: build_stimulus_cache(cfg, preprocessing, logger)
              for cfg in stimulus_configs}

    model, load_weights = build_model(args.architecture, device, args.clip_feature)
    baseline_ckpts = dict(sorted_scratch_checkpoints(baseline_dir / "checkpoints" / "all_epochs"))

    calib_x = None
    if args.recalibrate_bn:
        calib_x = build_bn_calib_set(args.imagenet_train or paths.path("imagenet_root") / "train",
                                     BN_CALIB_IMAGES, logger)

    baseline_emb = {}  # epoch -> {stimulus_set: embeddings}

    def get_baseline(epoch):
        nonlocal model
        if epoch not in baseline_emb:
            model = load_weights(baseline_ckpts[epoch])
            if calib_x is not None:
                recalibrate_bn(model, calib_x, device, 128, logger)
            baseline_emb[epoch] = {
                cfg["stimulus_set"]: embed(model, caches[cfg["stimulus_set"]], device, cfg["batch_size"])
                for cfg in stimulus_configs
            }
        return baseline_emb[epoch]

    out_root = resolve_output_dir(args.output_dir, args.architecture, args.clip_feature,
                                  preprocessing, bn_recalibrated=args.recalibrate_bn)
    logger.info("[sweep] output_dir=%s", out_root)
    n_written = n_skipped = n_done = 0

    for r in runs:
        if args.limit and n_done >= args.limit:
            break
        out_dir = out_root / r["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        run_ckpts = dict(sorted_scratch_checkpoints(r["run_dir"] / "checkpoints" / "all_epochs"))

        for epoch in r["epochs"]:
            if args.limit and n_done >= args.limit:
                logger.info("[sweep] --limit %d reached; stopping.", args.limit)
                break
            pending = []
            for cfg in stimulus_configs:
                out_json = out_dir / f"model_to_model_rsa_{cfg['stimulus_set']}_epoch{epoch}.json"
                if out_json.exists() and not args.force:
                    n_skipped += 1
                else:
                    pending.append((cfg, out_json))
            if not pending:
                continue

            emb_b_all = get_baseline(epoch)
            model = load_weights(run_ckpts[epoch])
            if calib_x is not None:
                recalibrate_bn(model, calib_x, device, 128, logger)
            emb_a = {
                cfg["stimulus_set"]: embed(model, caches[cfg["stimulus_set"]], device, cfg["batch_size"])
                for cfg, _ in pending
            }

            for cfg, out_json in pending:
                name = cfg["stimulus_set"]
                cache = caches[name]
                res = compare_embeddings(cfg, emb_a[name], cache["categories"], cache["names"],
                                         emb_b_all[name], cache["categories"], cache["names"], epoch)
                result = {
                    "epoch": epoch, "stimulus_set": name,
                    "architecture": args.architecture,
                    # Only varies for CLIP; recorded always so a record is self-describing.
                    "clip_feature": args.clip_feature if args.architecture == CLIP_ARCH else None,
                    "preprocessing": preprocessing.tag,
                    "bn_recalibrated": args.recalibrate_bn,
                    "perturb_type": r["type"], "phase": r["phase"], "onset_epoch": r["onset"],
                    "run_a": r["run_dir"].name, "label_a": r["name"],
                    "run_b": baseline_dir.name, "label_b": baseline_dir.name,
                    **res,
                }
                with open(out_json, "w") as f:
                    json.dump(result, f, indent=2)
                n_written += 1
                logger.info("[done] %s %s epoch%d: RSA(p)=%.4f RSA(s)=%.4f CKA=%.4f",
                            r["name"], name, epoch,
                            res["rsa"]["pearson"]["rho"], res["rsa"]["spearman"]["rho"],
                            res["cka"]["score"])
            # Counted per (run, epoch), i.e. per checkpoint load -- the unit --limit
            # bounds, and the unit that actually costs time here.
            n_done += 1
            del emb_a

        logger.info("[progress] %s complete (%d written, %d cached so far)", r["name"], n_written, n_skipped)

    logger.info("[sweep] done: %d JSON written, %d already existed -> %s", n_written, n_skipped, out_root)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--architecture", default=None, choices=sorted(ARCH_PREFIX),
                   help="Which runs_root architecture family to sweep. Default: all four, "
                        "one subprocess each.")
    p.add_argument("--runs_root", default=None, help="Default: paths key runs_root.")
    p.add_argument("--run_a", default=None,
                   help="Compare this explicit run dir against --run_b instead of running the "
                        "perturbed-vs-baseline discovery. Used for the two-baseline seed control.")
    p.add_argument("--run_b", default=None, help="Reference run dir for --run_a.")
    p.add_argument("--seed_suffix", default="",
                   help='Replicate family to discover (e.g. "_seed43"); perturbed runs and '
                        'their baseline both carry it.')
    p.add_argument("--epochs", default=None,
                   help="Comma-separated epoch list for --run_a/--run_b mode. Defaults to "
                        "--epoch_stride over the epochs both runs have.")
    p.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--img_dir", default=None,
                   help="Override the stimulus img_dir in EVERY --config.")
    p.add_argument("--epoch_stride", type=int, default=10,
                   help="Coarse epoch stride for the post-recovery tail; the epochs "
                        "just after onset are always sampled densely on top of this.")
    p.add_argument("--clip_feature", default=DEFAULT_CLIP_FEATURE, choices=sorted(CLIP_FEATURES),
                   help="CLIP-ViT-B/32 only (no-op for the timm architectures): which visual "
                        "representation to score. "
                        f"'cls' (default) = {CLIP_FEATURES['cls']}; "
                        f"'projected' = {CLIP_FEATURES['projected']}. "
                        "Each writes to its own output root unless --output_dir is given.")
    p.add_argument("--preprocessing", default="auto", choices=PREPROCESSING_CHOICES,
                   help="Eval-time image preprocessing. 'auto' (default) uses the recipe "
                        "this architecture's checkpoints were TRAINED under -- ImageNet "
                        "stats + Resize(256)->Crop(224) for the supervised timm families, "
                        "CLIP stats + Resize(224) for the contrastive one. "
                        "'things_legacy' uses THINGS image-set stats with an aspect-squashing "
                        "resize (the CLIP-HBA recipe). Non-legacy recipes write to a "
                        "'_pp-<tag>' root.")
    p.add_argument("--output_dir", default=None,
                   help=f"Defaults to {DEFAULT_OUTPUT_DIR}, plus a '_clip_<feature>' and/or "
                        "'_pp-<tag>' suffix for non-default variants (see resolve_output_dir).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many (run, epoch) pairs. For smoke-testing: "
                        "checkpoint IO dominates this sweep, so limiting the stimulus set "
                        "does not bound the cost -- only limiting the pairs does.")
    p.add_argument("--recalibrate_bn", action="store_true",
                   help="Recalibrate BatchNorm running statistics on clean ImageNet train "
                        "images before extracting embeddings. Only affects RN50 (the other "
                        "architectures use LayerNorm). Writes to a '_bnrecal' output root.")
    p.add_argument("--imagenet_train", type=Path, default=None,
                   help="BN calibration images; default <imagenet_root>/train.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
