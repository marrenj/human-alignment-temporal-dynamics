"""Evaluate recovery-dynamics sub-epoch checkpoints on ImageNet validation.

Each ``_recovery_dynamics`` run in runs_root has logarithmically spaced
checkpoints within the first post-perturbation epoch. This script evaluates
each on ImageNet-1k val (50k images), with BN recalibration for ResNet-50.

Output: one JSON per run under ``<results_root>/subepoch_imagenet_val/``,
containing per-step accuracy and loss.

Usage:
    python pipeline/scripts/evaluation/run_subepoch_imagenet_val.py
    python pipeline/scripts/evaluation/run_subepoch_imagenet_val.py --arch resnet50
    python pipeline/scripts/evaluation/run_subepoch_imagenet_val.py --dry-run
"""

import argparse
import json
import re
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

DEFAULT_OUT = paths.results_root() / 'subepoch_imagenet_val'
DEFAULT_CACHE = paths.path('feature_cache') / 'bn_recal_cache'

ARCH_CONFIG = {
    'resnet50': {
        'timm_model': 'resnet50',
        'prefix': 'resnet50_imagenet',
        'has_bn': True,
        'phases': {
            'early': {'onset': 5},
            'mid':   {'onset': 30},
            'late':  {'onset': 70},
        },
    },
    'vitb16': {
        'timm_model': 'vit_base_patch16_224',
        'prefix': 'vitb16_imagenet',
        'has_bn': False,
        'phases': {
            'early': {'onset': 5},
            'mid':   {'onset': 30},
            'late':  {'onset': 70},
        },
    },
    'convnext_b': {
        'timm_model': 'convnext_base',
        'prefix': 'convnext_b_imagenet',
        'has_bn': False,
        'phases': {
            'early': {'onset': 5},
            'mid':   {'onset': 30},
            'late':  {'onset': 70},
        },
    },
}

PERTS = ['label_shuffle', 'image_noise', 'uniform_images']

IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.PILToTensor(),
])

_STEP_RE = re.compile(r'epoch_(\d+)_step_(\d+)\.ckpt$')


def _class_dirs(split):
    return sorted(p for p in (paths.path('imagenet_root') / split).iterdir() if p.is_dir())


def _build_cache(split, per_class, cache_path, limit=None):
    if cache_path.is_file():
        d = torch.load(cache_path)
        return d['x'], d['y']
    print(f'  building {split} cache ({per_class}/class) -> {cache_path}')
    xs, ys = [], []
    for idx, cdir in enumerate(_class_dirs(split)):
        for f in sorted(cdir.iterdir())[:per_class]:
            try:
                xs.append(_TF(Image.open(f).convert('RGB')))
            except OSError:
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
    bns = [m for m in model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
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


def _discover_subepoch_ckpts(run_dir):
    """Return [(epoch, step, path), ...] sorted by (epoch, step)."""
    sub_dir = run_dir / 'checkpoints' / 'sub_epoch'
    if not sub_dir.is_dir():
        return []
    out = []
    for f in sub_dir.glob('epoch_*_step_*.ckpt'):
        m = _STEP_RE.search(f.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), f))
    return sorted(out)


def _discover_runs(arches, runs_root):
    """Yield (arch_key, run_name, run_dir, onset_epoch, pert, phase_key)."""
    for arch_key in arches:
        cfg = ARCH_CONFIG[arch_key]
        prefix = cfg['prefix']
        for phase_key, phase in cfg['phases'].items():
            onset = phase['onset']
            for pert in PERTS:
                run_name = f'{prefix}_{pert}_{phase_key}_recovery_dynamics'
                run_dir = runs_root / run_name
                if run_dir.is_dir():
                    yield arch_key, run_name, run_dir, onset, pert, phase_key


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arch', choices=list(ARCH_CONFIG), action='append')
    ap.add_argument('--val_per_class', type=int, default=50,
                    help='ImageNet val images per class (50 = full split)')
    ap.add_argument('--calib_images', type=int, default=8192)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--runs_root', type=Path, default=None, help='Default: paths key runs_root.')
    ap.add_argument('--output_dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--cache_dir', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', dest='dry_run', action='store_true')
    args = ap.parse_args()

    arches = args.arch or list(ARCH_CONFIG)
    runs_root = args.runs_root or paths.path('runs_root')
    runs = list(_discover_runs(arches, runs_root))

    total_ckpts = 0
    for arch_key, run_name, run_dir, onset, pert, phase_key in runs:
        ckpts = _discover_subepoch_ckpts(run_dir)
        total_ckpts += len(ckpts)
        print(f'  {run_name:60s}  {len(ckpts)} sub-epoch ckpts  (onset={onset})')

    print(f'\n{len(runs)} runs, {total_ckpts} total checkpoints')
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    calib_per_class = max(1, args.calib_images // 1000)
    val_x, val_y = _build_cache('val', args.val_per_class,
                                args.cache_dir / f'val_{args.val_per_class}pc.pt')
    cal_x = None  # loaded lazily for BN arches

    for arch_key, run_name, run_dir, onset, pert, phase_key in runs:
        out_path = args.output_dir / f'{run_name}.json'
        if out_path.is_file() and not args.force:
            print(f'[skip] {run_name}')
            continue

        ckpts = _discover_subepoch_ckpts(run_dir)
        if not ckpts:
            print(f'[no ckpts] {run_name}')
            continue

        cfg = ARCH_CONFIG[arch_key]
        needs_bn = cfg['has_bn']
        if needs_bn and cal_x is None:
            cal_x, _ = _build_cache(
                'train', calib_per_class,
                args.cache_dir / f'train_{calib_per_class}pc.pt',
                limit=args.calib_images)

        results = []
        for epoch, step, ckpt_path in ckpts:
            model = timm.create_model(cfg['timm_model'], pretrained=False,
                                      num_classes=1000)
            try:
                model.load_state_dict(load_lightning_state_dict(str(ckpt_path)),
                                      strict=True)
            except RuntimeError as exc:
                print(f'  [corrupt ckpt, skipping] {ckpt_path.name}: {exc}')
                continue
            model.to(device)

            ce_raw, top1_raw = _evaluate(model, val_x, val_y, device,
                                         args.batch_size)

            if needs_bn:
                _recalibrate_bn(model, cal_x, device, args.batch_size)
                ce_bn, top1_bn = _evaluate(model, val_x, val_y, device,
                                           args.batch_size)
            else:
                ce_bn, top1_bn = ce_raw, top1_raw

            results.append({
                'epoch': epoch,
                'step': step,
                'val_ce_loss_raw': ce_raw,
                'val_acc1_raw': top1_raw,
                'val_ce_loss': ce_bn,
                'val_acc1': top1_bn,
                'checkpoint': str(ckpt_path.relative_to(runs_root)),
            })
            print(f'  [{run_name}] ep{epoch} step{step:5d}: '
                  f'top1={top1_bn * 100:.2f}%  (raw={top1_raw * 100:.2f}%)')

            del model
            torch.cuda.empty_cache()

        rec = {
            'run_name': run_name,
            'architecture': arch_key,
            'timm_model': cfg['timm_model'],
            'perturb_type': pert,
            'phase': phase_key,
            'onset_epoch': onset,
            'has_bn': needs_bn,
            'val_per_class': args.val_per_class,
            'steps': results,
        }
        with out_path.open('w') as f:
            json.dump(rec, f, indent=2)
        print(f'[wrote] {out_path.name}  ({len(results)} steps)')

    print(f'\nDone. Results in {args.output_dir}')


if __name__ == '__main__':
    main()
