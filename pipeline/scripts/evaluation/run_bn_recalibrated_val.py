"""Re-evaluate the onset-epoch checkpoints with BatchNorm buffers recalibrated on clean data.

Why this exists
---------------
ResNet50's logged validation loss at the perturbation epoch reaches 141 (vs a
chance CE of 6.91). That is not weight damage: one epoch of training on blank or
noise images overwrites BatchNorm's running_mean/running_var with the statistics
of a degenerate input distribution, so validation then normalizes real images by
the wrong statistics and the logits come out ~20-40x too large. The model is
confidently wrong, and cross-entropy punishes that without bound.

ViT-B/16 and ConvNeXt-B use LayerNorm, which has no running buffers, so they
cannot show this artifact -- which is exactly why the raw comparison across
architectures is confounded.

This script separates the two effects. For each checkpoint it reports val CE and
top-1 twice:
  * `as_is`         -- BN running buffers exactly as the perturbed epoch left them
                       (reproduces the logged numbers).
  * `bn_recalibrated` -- BN running buffers reset and re-estimated from clean
                       ImageNet *train* images, with all learned weights frozen.
                       The difference between the two is the artifact; what
                       remains in `bn_recalibrated` is real damage to the weights.

For LayerNorm architectures the two are identical by construction (the script
records `n_bn_layers: 0`), which serves as a built-in control.

Evaluation is on a fixed per-class subset of ImageNet val, cached as a uint8
tensor so all checkpoints see byte-identical inputs and the JPEG decode happens
once rather than once per checkpoint.

Sanity check: the baseline `as_is` top-1 should reproduce the logged value in
<results_root>/imagenet_val_curves.json to within subset sampling error. The
script prints both so a preprocessing mismatch is visible immediately.

The same script produces three result directories the figures read:
  * `bn_recalibrated_val/`     -- ResNet-50 onsets on the 10k val subset (default).
  * `bn_recalibrated_val_50k/` -- full val split (`--val_per_class 50`) at chosen
                                  epochs (`--run_epochs`), for recovery trajectories.
  * `onset_val_10k/`           -- ViT-B/16 and ConvNeXt-B at their onsets
                                  (`--phases`).

Usage:
    python pipeline/scripts/evaluation/run_bn_recalibrated_val.py --arch resnet50_imagenet \
        --phases early mid late
    python pipeline/scripts/evaluation/run_bn_recalibrated_val.py --arch resnet50_imagenet \
        --phases early mid late --run_epochs 89 --val_per_class 50 \
        --output_dir <results_root>/bn_recalibrated_val_50k
    python pipeline/scripts/evaluation/run_bn_recalibrated_val.py --arch vitb16_imagenet \
        --phases early mid late --output_dir <results_root>/onset_val_10k
"""

import argparse
import json
import sys
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE = EVAL_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths  # noqa: E402
from temporal_perturbations.models.lightning_ckpt import load_lightning_state_dict  # noqa: E402

DEFAULT_OUT = paths.results_root() / 'bn_recalibrated_val'
DEFAULT_CACHE = paths.path('feature_cache') / 'bn_recal_cache'

# (run prefix, timm architecture)
ARCHES = {
    'resnet50_imagenet': 'resnet50',
    'vitb16_imagenet': 'vit_base_patch16_224',
    'convnext_b_imagenet': 'convnext_base',
}
PERTS = ['label_shuffle', 'image_noise', 'uniform_images']
TIMINGS = ['early', 'mid', 'late']

IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Resize/crop only -- normalization happens on GPU so the cache stays uint8.
_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.PILToTensor(),
])


def _class_dirs(root, split):
    """Sorted WNIDs = the standard ImageNet-1k class index order."""
    return sorted(p for p in (root / split).iterdir() if p.is_dir())


def _build_cache(root, split, per_class, cache_path, limit=None):
    if cache_path.is_file():
        d = torch.load(cache_path)
        return d['x'], d['y']
    print(f'  building {split} cache ({per_class}/class) from {root} -> {cache_path}')
    xs, ys = [], []
    for idx, cdir in enumerate(_class_dirs(root, split)):
        for f in sorted(cdir.iterdir())[:per_class]:
            try:
                xs.append(_TF(Image.open(f).convert('RGB')))
            except OSError as e:
                print(f'    skipping unreadable {f.name}: {e}')
                continue
            ys.append(idx)
        if limit and len(xs) >= limit:
            break
        if (idx + 1) % 200 == 0:
            print(f'    {idx + 1}/1000 classes, {len(xs)} images')
    x = torch.stack(xs)
    y = torch.tensor(ys)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'x': x, 'y': y}, cache_path)
    print(f'  cached {tuple(x.shape)} ({x.numel() / 1e9:.2f} GB uint8)')
    return x, y


def _norm(batch_u8, device):
    x = batch_u8.to(device, non_blocking=True).float().div_(255)
    return (x - IN_MEAN.to(device)) / IN_STD.to(device)


@torch.no_grad()
def _evaluate(model, x, y, device, batch_size):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for i in range(0, len(x), batch_size):
        xb = _norm(x[i:i + batch_size], device)
        yb = y[i:i + batch_size].to(device)
        logits = model(xb)
        loss_sum += torch.nn.functional.cross_entropy(
            logits, yb, reduction='sum').item()
        correct += (logits.argmax(-1) == yb).sum().item()
        n += len(yb)
    return loss_sum / n, correct / n


@torch.no_grad()
def _recalibrate_bn(model, x, device, batch_size):
    """Reset BN running buffers and re-estimate them from clean train images.

    momentum=None makes each BN use a cumulative moving average, so the result is
    the exact mean/var over the calibration set rather than an EMA that depends on
    batch order. Learned weights are untouched.

    Only the BN modules go into train mode -- the rest of the model stays in eval
    so that dropout / stochastic depth stay disabled. (Inert for ResNet50, which
    has neither, but ConvNeXt and ViT do.)
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for i in range(0, len(x), batch_size):
        model(_norm(x[i:i + batch_size], device))
    model.eval()
    return len(bns)


def _ckpt_path(runs_root, run, epoch):
    base = runs_root / run / 'checkpoints' / 'all_epochs'
    for name in (f'epoch_{epoch:03d}.ckpt', f'epoch_epoch={epoch:03d}.ckpt'):
        if (base / name).is_file():
            return base / name
    return None


def _onset_epochs(runs_root, arch, phases=TIMINGS, seed_suffix=''):
    """{phase: onset epoch}, read as the first checkpoint present in each run."""
    out = {}
    for t in phases:
        for p in PERTS:
            d = runs_root / f'{arch}_{p}_{t}{seed_suffix}' / 'checkpoints' / 'all_epochs'
            if not d.is_dir():
                continue
            eps = sorted(
                int(f.stem.split('=')[-1].split('_')[-1])
                for f in d.glob('epoch_*.ckpt')
            )
            if eps:
                out[t] = eps[0]
                break
    return out


def _logged_top1(run, epoch):
    p = paths.results_root() / 'imagenet_val_curves.json'
    if not p.is_file():
        return None
    with p.open() as f:
        r = json.load(f)['runs'].get(run, {}).get('val_acc1', {})
    return r.get(str(epoch))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arch', choices=list(ARCHES), action='append',
                    help='restrict to one architecture (repeatable)')
    ap.add_argument('--val_per_class', type=int, default=10,
                    help='ImageNet val images per class (50 = the full split)')
    ap.add_argument('--calib_images', type=int, default=8192,
                    help='clean train images used to re-estimate BN statistics')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--output_dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--cache_dir', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true',
                    help='recompute even if the output JSON exists')
    ap.add_argument('--phases', nargs='+', default=TIMINGS, choices=TIMINGS,
                    help='onset timings to evaluate (e.g. convnext_b_imagenet_image_noise_early)')
    ap.add_argument('--baseline_epochs', nargs='*', type=int, default=[],
                    help='extra baseline-only epochs to evaluate (e.g. 71, the end of '
                         'the first epoch after a late onset)')
    ap.add_argument('--run_epochs', nargs='*', type=int, default=[],
                    help='evaluate every perturbed run and the baseline at these epochs '
                         'instead of at each phase\'s onset (e.g. 89, the final epoch)')
    ap.add_argument('--imagenet_root', type=Path, default=None,
                    help='ImageNet root with train/ and val/ class folders '
                         '(default: paths key imagenet_root)')
    ap.add_argument('--runs_root', type=Path, default=None,
                    help='default: paths key runs_root')
    ap.add_argument('--seed_suffix', default='',
                    help='appended to every run name, baseline included (e.g. "_seed43")')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    args.imagenet_root = args.imagenet_root or paths.path('imagenet_root')
    runs_root = args.runs_root or paths.path('runs_root')
    sfx = args.seed_suffix

    arches = args.arch or list(ARCHES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for arch in arches:
        onsets = _onset_epochs(runs_root, arch, args.phases, sfx)
        if not onsets:
            print(f'WARNING: no perturbed runs found for {arch}')
            continue
        for t, onset in onsets.items():
            for ep in args.run_epochs or [onset]:
                jobs.append((arch, f'{arch}_baseline{sfx}', 'none', t, ep))
                for p in PERTS:
                    jobs.append((arch, f'{arch}_{p}_{t}{sfx}', p, t, ep))
        for ep in args.baseline_epochs:
            jobs.append((arch, f'{arch}_baseline{sfx}', 'none', 'extra', ep))

    # With --run_epochs the baseline is queued once per phase at the same epoch.
    unique = {}
    for job in jobs:
        unique.setdefault((job[1], job[4]), job)
    jobs = list(unique.values())
    print(f'{len(jobs)} (run, epoch) evaluations across {len(arches)} architectures')
    for arch, run, p, t, ep in jobs:
        print(f'  {run:45s} epoch {ep:3d}  ({p}/{t})')
    if args.dry_run:
        return

    device = torch.device(args.device)
    calib_per_class = max(1, args.calib_images // 1000)
    val_x, val_y = _build_cache(args.imagenet_root, 'val', args.val_per_class,
                                args.cache_dir / f'val_{args.val_per_class}pc.pt')
    cal_x, _ = _build_cache(args.imagenet_root, 'train', calib_per_class,
                            args.cache_dir / f'train_{calib_per_class}pc.pt',
                            limit=args.calib_images)
    print(f'val set {tuple(val_x.shape)}, calibration set {tuple(cal_x.shape)}')

    for arch, run, pert, timing, epoch in jobs:
        out_path = args.output_dir / f'{run}_epoch{epoch}.json'
        if out_path.is_file() and not args.force:
            print(f'[skip] {out_path.name}')
            continue
        ckpt = _ckpt_path(runs_root, run, epoch)
        if ckpt is None:
            print(f'[MISSING] no checkpoint for {run} epoch {epoch}')
            continue

        model = timm.create_model(ARCHES[arch], pretrained=False, num_classes=1000)
        model.load_state_dict(load_lightning_state_dict(str(ckpt)), strict=True)
        model.to(device)

        ce_a, top1_a = _evaluate(model, val_x, val_y, device, args.batch_size)
        n_bn = _recalibrate_bn(model, cal_x, device, args.batch_size)
        if n_bn:
            ce_b, top1_b = _evaluate(model, val_x, val_y, device, args.batch_size)
        else:
            ce_b, top1_b = ce_a, top1_a

        rec = {
            'run_name': run, 'architecture': arch, 'timm_model': ARCHES[arch],
            'perturb_type': pert, 'timing': timing, 'epoch': epoch,
            'n_bn_layers': n_bn,
            'val_images': int(len(val_x)), 'val_per_class': args.val_per_class,
            'calib_images': int(len(cal_x)),
            'as_is': {'val_ce_loss': ce_a, 'val_acc1': top1_a},
            'bn_recalibrated': {'val_ce_loss': ce_b, 'val_acc1': top1_b},
            'logged_val_acc1': _logged_top1(run, epoch),
            'checkpoint': str(ckpt.relative_to(runs_root)),
        }
        with out_path.open('w') as f:
            json.dump(rec, f, indent=2)

        logged = rec['logged_val_acc1']
        tag = f' | logged {logged * 100:.2f}%' if logged is not None else ''
        print(f'[{run} ep{epoch}] bn={n_bn:2d}  as-is CE={ce_a:8.3f} top1={top1_a * 100:5.2f}%'
              f'  ->  recal CE={ce_b:7.3f} top1={top1_b * 100:5.2f}%{tag}')
        del model
        torch.cuda.empty_cache()

    print(f'\nWrote {len(list(args.output_dir.glob("*.json")))} JSON files to {args.output_dir}')


if __name__ == '__main__':
    main()
