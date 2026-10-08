"""Run layer-reweighted RSA (FR-RSA) against NOD / THINGS-fMRI for the from-scratch runs.

By default every perturbed run is evaluated at its final checkpoint, plus the
epoch-matched final checkpoint of its baseline. `--include_onset` adds each
perturbed run's onset epoch (and the matching baseline epoch); `--baseline_step N`
adds every Nth baseline epoch for a baseline trajectory.

Results go to `<results_root>/frrsa/<run>/inference_results_{dataset}_frrsa[_{tag}]_epoch{E}.json`.
Append-only: an existing result is left alone unless --force is passed.

    python pipeline/scripts/evaluation/run_frrsa.py --dry-run          # show the job list
    python pipeline/scripts/evaluation/run_frrsa.py --baseline_step 5 --include_onset
    python pipeline/scripts/evaluation/run_frrsa.py --arch resnet50 --recalibrate_bn
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

EXP_DIR   = Path(__file__).resolve().parent
PIPELINE  = EXP_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths
from temporal_perturbations.data.eval_transforms import (
    PREPROCESSING_CHOICES,
    frrsa_result_filename,
    resolve_preprocessing,
)
from temporal_perturbations.utils.run_hparams import onset_epoch

ARCH_MAP = [
    ("resnet50_imagenet",   "RN50",       "scratch_lightning"),
    ("vitb16_imagenet",     "ViT-B/16",   "scratch_lightning"),
    ("convnext_b_imagenet", "ConvNeXt-B", "scratch_lightning"),
    ("clip_vitb32",         "ViT-B/32",   "clip_contrastive"),
]


SKIP_SUBSTRINGS = ("declip",)

BN_CALIB_IMAGES = 8192


def build_bn_calib_set(imagenet_train, n_images, preprocessing, logger):
    """Load ``n_images`` clean ImageNet train images, preprocessed with ``preprocessing``."""
    import torch
    from PIL import Image

    transform = preprocessing.build(224)
    per_class = max(1, n_images // 1000)
    xs = []
    class_dirs = sorted(p for p in imagenet_train.iterdir() if p.is_dir())
    for cdir in class_dirs:
        for f in sorted(cdir.iterdir())[:per_class]:
            try:
                xs.append(transform(Image.open(f).convert('RGB')))
            except OSError:
                continue
        if len(xs) >= n_images:
            break
    x = torch.stack(xs[:n_images])
    logger.info("[bn] calibration set: %s (%d images, preprocessing=%s)",
                tuple(x.shape), len(x), preprocessing.tag)
    return x


def recalibrate_bn(model, calib_x, device, batch_size, logger):
    """Reset BN running buffers and re-estimate from clean data."""
    import torch
    from torch.nn.modules.batchnorm import _BatchNorm

    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not bns:
        return 0
    model.eval()
    with torch.no_grad():
        for m in bns:
            m.reset_running_stats()
            m.momentum = None
            m.train()
        for i in range(0, len(calib_x), batch_size):
            model(calib_x[i:i + batch_size].to(device, non_blocking=True))
        for m in bns:
            m.eval()
    logger.info("[bn] recalibrated %d BatchNorm layers on %d images", len(bns), len(calib_x))
    return len(bns)


# Config keys coerced to int on load -- same list run_inference.py uses, so an
# in-process config matches what a subprocess would have built.
INFERENCE_INT_KEYS = [
    "batch_size", "num_classes", "rank", "vision_layers", "transformer_layers",
    "random_seed", "cuda", "num_workers", "max_images_per_category",
]

EPOCH_RE = re.compile(r"epoch(?:_epoch=|_)(\d+)\.ckpt$")


def arch_of(run_name):
    for prefix, arch, tmode in ARCH_MAP:
        if run_name.startswith(prefix):
            return prefix, arch, tmode
    return None, None, None


def available_epochs(run_dir):
    ae = run_dir / "checkpoints" / "all_epochs"
    out = {}
    if ae.exists():
        for f in sorted(ae.glob("*.ckpt")):
            m = EPOCH_RE.search(f.name)
            if m:
                out.setdefault(int(m.group(1)), f)
    return dict(sorted(out.items()))


def baseline_for(run_name, prefix):
    """The baseline a perturbed run must be compared against.

    Seed-43 forks have their own seed-43 baseline; pairing them with the seed-42
    baseline would fold a seed difference into the perturbation delta.
    """
    if run_name.endswith("_seed43"):
        return f"{prefix}_baseline_seed43"
    return f"{prefix}_baseline"


def _excl(spec):
    """--exclude "a,b" -> ("a", "b"); empty/None -> ()."""
    return tuple(s.strip() for s in (spec or "").split(",") if s.strip())


def build_jobs(runs_root, arch_filter=None, include_seed43=True,
               baseline_step=None, include_onset=False, exclude=(),
               seed43_only=False, baselines_only=False):
    """[(run_name, arch, tmode, epoch, ckpt)] — perturbed finals + matched baseline finals.

    `baseline_step` additionally schedules every Nth baseline epoch, giving the
    baseline a trajectory rather than a single endpoint; `include_onset` adds
    each perturbed run's onset epoch (and the matching baseline epoch), which is
    where the immediate hit lives. `exclude` drops any run whose name contains one
    of the given substrings, on top of the built-in SKIP_SUBSTRINGS. `baselines_only`
    keeps the baseline epochs the perturbed runs imply but drops the perturbed jobs.
    """
    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())

    baseline_epochs = {}   # baseline run name -> set of epochs needed
    perturbed = []

    for d in run_dirs:
        name = d.name
        if any(s in name for s in (*SKIP_SUBSTRINGS, *exclude)):
            continue
        prefix, arch, tmode = arch_of(name)
        if prefix is None:
            continue
        if arch_filter and not name.startswith(arch_filter):
            continue
        if "baseline" in name:
            continue
        if not include_seed43 and name.endswith("_seed43"):
            continue
        if seed43_only and not name.endswith("_seed43"):
            continue
        eps = available_epochs(d)
        if not eps:
            continue
        bname = baseline_for(name, prefix)
        wanted = {max(eps)}
        if include_onset:
            onset = onset_epoch(d)
            if onset is not None and onset in eps:
                wanted.add(onset)
        for e in sorted(wanted):
            perturbed.append((name, arch, tmode, e, eps[e]))
        baseline_epochs.setdefault(bname, set()).update(wanted)

    if baseline_step:
        for bname in list(baseline_epochs):
            beps = available_epochs(runs_root / bname)
            baseline_epochs[bname].update(e for e in beps if e % baseline_step == 0)

    jobs = []
    for bname, epochs in sorted(baseline_epochs.items()):
        bdir = runs_root / bname
        beps = available_epochs(bdir)
        _, barch, btmode = arch_of(bname)
        for e in sorted(epochs):
            if e in beps:
                jobs.append((bname, barch, btmode, e, beps[e]))
            else:
                print(f"  [warn] {bname} has no checkpoint at epoch {e}; "
                      f"available max = {max(beps) if beps else 'none'}")
    if not baselines_only:
        jobs.extend(sorted(perturbed))
    return jobs


def _load_eval_config(cfg_path, img_dir, out_dir, job, preprocessing_name, cuda):
    """The config `run_inference.py` would have built for this job.

    Mirrors run_inference's path resolution (relative paths against the repo root,
    which is that script's cwd) and its --set overrides, so an in-process run and a
    subprocess run see the same dict.
    """
    from temporal_perturbations.utils.load_yaml_config import load_yaml_config

    cfg = load_yaml_config(cfg_path, numeric_keys=INFERENCE_INT_KEYS)
    cfg.update({
        "training_mode": job["tmode"], "architecture": job["arch"],
        "model_weights_path": str(job["ckpt"]), "epoch": job["epoch"],
        "cuda": cuda, "preprocessing": preprocessing_name,
        "img_dir": str(img_dir), "inference_save_dir": str(out_dir),
    })
    for key in ("model_weights_path", "annotations_file"):
        if cfg.get(key):
            pth = Path(cfg[key])
            cfg[key] = str(pth if pth.is_absolute() else (REPO_ROOT / pth))
    if cfg.get("reference_rdm_paths"):
        cfg["reference_rdm_paths"] = {
            roi: str(Path(pp) if Path(pp).is_absolute() else (REPO_ROOT / pp))
            for roi, pp in cfg["reference_rdm_paths"].items()
        }
    return cfg


def _build_eval_model(cfg, device, logger):
    """Build + load one checkpoint, mirroring inference_core's model construction.

    Deliberately narrow: FR-RSA's suites declare `applies_to: scratch,
    scratch_lightning, clip_contrastive`, and only the latter two appear in the
    runs_root sweep, so only those two branches are reproduced here.

    It must be the RAW model, not a feature wrapper. `frrsa_core.select_layers`
    picks extraction points by structural inspection (`layer1`/`layer4` for timm
    ResNet, `blocks.0..` for ViT, `visual.transformer` for CLIP), so the pooled-
    feature wrappers the model-to-model sweep uses would not resolve.

    Kept in step with inference_core by the equivalence test in tests/.
    """
    from temporal_perturbations.models.factory import build_model
    from temporal_perturbations.models.lightning_ckpt import load_lightning_state_dict

    tmode = cfg["training_mode"]
    ckpt_path = Path(cfg["model_weights_path"])

    if tmode == "clip_contrastive":
        from temporal_perturbations.models.clip_yfcc import load_clip_lightning
        model = load_clip_lightning(
            ckpt_path=str(ckpt_path),
            config_preset=cfg.get("clip_yfcc_config_preset", "vit_b_32"),
            map_location=device,
        )
        return model.to(device)

    if tmode != "scratch_lightning":
        raise ValueError(
            f"In-process FR-RSA covers scratch_lightning and clip_contrastive; "
            f"got training_mode={tmode!r}. Run it through run_inference.py instead."
        )

    model = build_model(
        architecture=cfg.get("architecture"), pretrained=False, clip_hba_backbone=None,
        vision_layers=None, transformer_layers=None, rank=None,
        cuda=cfg.get("cuda", 0), device=device, wandb_watch_model=False,
        wandb_log_freq=None, num_classes=cfg.get("num_classes", 1000),
    )
    state_dict = load_lightning_state_dict(ckpt_path, map_location="cpu")
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state_dict, strict=True)
    logger.info("  loaded %s epoch %s from %s",
                cfg.get("architecture"), cfg.get("epoch"), ckpt_path.name)
    return model.to(device)


def run_in_process(todo, args):
    """Evaluate every job in ONE process, decoding each stimulus set once.

    A subprocess per (run, epoch, dataset) would re-decode the whole stimulus set each
    time (8,640 THINGS-fMRI JPEGs take ~10 minutes). Here the images are decoded once
    per (dataset, preprocessing recipe); only the weights change per checkpoint.

    Jobs are ordered by (architecture, dataset) so each cache and each model is built
    once rather than thrashing between them.
    """
    import torch
    from temporal_perturbations.inference.extract_layer_embeddings import (
        build_neural_stimulus_cache,
    )
    from temporal_perturbations.inference.frrsa_eval import run_frrsa_eval
    from temporal_perturbations.inference.inference_core import prepare_reference_rdms
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("frrsa_sweep")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    do_bn_recal = args.recalibrate_bn
    imagenet_train = args.imagenet_train

    caches, refs, bn_caches, failures = {}, {}, {}, []
    t_start = time.time()
    ordered = sorted(todo, key=lambda j: (j["arch"], j["dataset"]))

    for i, job in enumerate(ordered, 1):
        label = job["label"]
        print(f"\n[{i}/{len(ordered)}] {label}", flush=True)
        t0 = time.time()
        try:
            cfg = _load_eval_config(job["cfg_path"], job["img_dir"], job["out_dir"],
                                    job, args.preprocessing, args.cuda)
            cfg["bn_recalibrated"] = do_bn_recal
            cfg["frrsa_save_dir"] = str(job["out_dir"])

            # One cache per (dataset, recipe): the recipes differ per architecture, so
            # this is reused across every checkpoint that shares one.
            ckey = (job["dataset"], job["pp"].tag)
            if ckey not in caches:
                caches[ckey] = build_neural_stimulus_cache(
                    job["dataset"], cfg["img_dir"], cfg["annotations_file"],
                    cfg["max_images_per_category"], job["pp"],
                    cfg["batch_size"], cfg["num_workers"], logger=logger,
                )
            if job["dataset"] not in refs:
                refs[job["dataset"]] = prepare_reference_rdms(cfg)

            model = _build_eval_model(cfg, device, logger)

            if do_bn_recal:
                bn_pp = resolve_preprocessing("auto", job["arch"], job["tmode"])
                bkey = bn_pp.tag
                if bkey not in bn_caches:
                    bn_caches[bkey] = build_bn_calib_set(
                        imagenet_train, BN_CALIB_IMAGES, bn_pp, logger,
                    )
                recalibrate_bn(model, bn_caches[bkey], device,
                               cfg["batch_size"], logger)

            run_frrsa_eval(
                model=model, config=cfg, device=device, training_mode=job["tmode"],
                epoch=job["epoch"], logger=logger,
                stimulus_cache=caches[ckey], reference_rdms=refs[job["dataset"]],
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ok = job["out_json"].exists()
            print(f"    {(time.time()-t0)/60:.1f} min {'OK' if ok else 'NO OUTPUT'}",
                  flush=True)
            if not ok:
                failures.append(label)
        except Exception as exc:  # noqa: BLE001 -- one bad checkpoint must not sink the sweep
            logger.exception("  [fail] %s", label)
            print(f"    FAILED after {(time.time()-t0)/60:.1f} min: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            failures.append(label)

    print(f"\n=== done in {(time.time()-t_start)/60:.1f} min "
          f"({len(caches)} stimulus cache(s) built, not {len(ordered)}) ===")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_root", type=Path, default=None, help="Default: paths key runs_root.")
    ap.add_argument("--nod_dir", type=Path, default=None, help="Default: paths key nod_images.")
    ap.add_argument("--things_dir", type=Path, default=None,
                    help="Default: paths key things_fmri_images.")
    ap.add_argument("--output_dir", type=Path, default=paths.results_root() / "frrsa",
                    help="Results go to <output_dir>/<run>/.")
    ap.add_argument("--datasets", nargs="+", default=["nod", "things"])
    ap.add_argument("--arch", default=None, help="only runs whose name starts with this")
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--preprocessing", default="auto", choices=PREPROCESSING_CHOICES,
                    help="Eval-time image preprocessing, also used to name the output. "
                         "'auto' (default) uses the recipe each checkpoint was trained under.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--baseline_step", type=int, default=None,
                    help="also evaluate every Nth baseline epoch (trajectory, not just endpoints)")
    ap.add_argument("--include_onset", action="store_true",
                    help="also evaluate each perturbed run at its perturbation-onset epoch")
    ap.add_argument("--no-seed43", action="store_true",
                    help="skip the seed-43 replicate runs and their baselines")
    ap.add_argument("--seed43_only", action="store_true",
                    help="only the seed-43 replicate runs and their baselines")
    ap.add_argument("--baselines_only", action="store_true",
                    help="schedule only baseline epochs (with --baseline_step: a baseline trajectory)")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated substrings; any run whose name contains one "
                         "is skipped, e.g. --exclude seed43,late.")
    ap.add_argument("--recalibrate_bn", action="store_true",
                    help="Recalibrate BatchNorm running stats on clean ImageNet train "
                         "images before extracting features. Only affects ResNet50 (the "
                         "other architectures use LayerNorm). Writes to a '_bnrecal' "
                         "output filename so results coexist with uncalibrated ones.")
    ap.add_argument("--imagenet_train", type=Path, default=None,
                    help="ImageNet train split root (class subdirectories); default "
                         "<imagenet_root>/train. Used only when --recalibrate_bn is set.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.runs_root = args.runs_root or paths.path("runs_root")
    args.nod_dir = args.nod_dir or paths.path("nod_images")
    args.things_dir = args.things_dir or paths.path("things_fmri_images")
    if args.recalibrate_bn:
        args.imagenet_train = args.imagenet_train or paths.path("imagenet_root") / "train"

    suites = {
        "nod":    (PIPELINE / "configs" / "brain_frrsa_nod.yaml",    args.nod_dir),
        "things": (PIPELINE / "configs" / "brain_frrsa_things.yaml", args.things_dir),
    }

    jobs = build_jobs(args.runs_root, args.arch, include_seed43=not args.no_seed43,
                      baseline_step=args.baseline_step, include_onset=args.include_onset,
                      exclude=_excl(args.exclude), seed43_only=args.seed43_only,
                      baselines_only=args.baselines_only)
    todo = []
    have = 0
    for run_name, arch, tmode, epoch, ckpt in jobs:
        out_dir = args.output_dir / run_name
        # Resolved here from the same (architecture, training_mode) inference_core will
        # use, so both sides name the file identically -- see frrsa_result_filename.
        pp = resolve_preprocessing(args.preprocessing, arch, tmode)
        for ds in args.datasets:
            cfg, img_dir = suites[ds]
            out_json = out_dir / frrsa_result_filename(
                ds, epoch, pp, bn_recalibrated=args.recalibrate_bn)
            if out_json.exists() and not args.force:
                have += 1
                continue
            todo.append({
                "label": f"{run_name} {ds} ep{epoch}", "out_json": out_json,
                "run_name": run_name, "arch": arch, "tmode": tmode, "epoch": epoch,
                "ckpt": ckpt, "dataset": ds, "cfg_path": cfg, "img_dir": img_dir,
                "out_dir": out_dir, "pp": pp,
            })

    if args.limit:
        todo = todo[:args.limit]

    print(f"=== FR-RSA: {len(jobs)} run-epochs, "
          f"{have} already present, {len(todo)} to run ===")
    for j in todo:
        print(f"  [todo] {j['label']} -> {j['out_json'].name}")
    if args.dry_run:
        return

    failures = run_in_process(todo, args)
    print(f"\n=== {len(todo)-len(failures)}/{len(todo)} succeeded ===")
    for f in failures:
        print(f"  [FAIL] {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
