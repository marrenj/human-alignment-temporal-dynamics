"""Extract ResNet-50 penultimate features on ImageNet val with BatchNorm re-estimated on clean images.

The cached cross-architecture embeddings in `<feature_cache>/cross_arch_embeddings/`
(from `extract_imagenet_embeddings.py`) were extracted with each checkpoint's stored BatchNorm running statistics. A
ResNet-50 checkpoint saved at the end of a perturbed epoch stores statistics of
noise or blank images, so its cached features are corrupted: 10-NN class purity
sits at chance and some feature norms are infinite, even for runs whose
recalibrated model keeps most of its top-1.

This script repeats that extraction exactly
(`trimap_imagenet.extract_embeddings`: same images, preprocessing,
batching and fp16 autocast) after re-estimating BatchNorm with
`run_bn_recalibrated_val._recalibrate_bn` on that script's 8,000-image
calibration cache -- the set every other recalibrated ResNet-50 result uses.
Both sides are recalibrated: the baseline at 5 / 30 / 70 and each perturbed run
at its onset. Outputs carry a `_bnrecal` suffix: the feature array next to the
originals in the feature cache, the `_synsets.json` / `_meta.json` sidecars under
`<results_root>/imagenet_embeddings/`. The originals are never touched.

Usage:
    python pipeline/scripts/evaluation/extract_imagenet_embeddings_bn_recalibrated.py
    python pipeline/scripts/evaluation/extract_imagenet_embeddings_bn_recalibrated.py --device cuda:0 --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE = EVAL_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(PIPELINE / 'scripts' / 'visualization'))

from temporal_perturbations import paths  # noqa: E402
from trimap_imagenet import (CACHE_DIR, SYNSETS_DIR,  # noqa: E402
                             build_and_load_model, extract_embeddings,
                             resolve_ckpt_path)
from run_bn_recalibrated_val import DEFAULT_CACHE, _recalibrate_bn  # noqa: E402

ARCHITECTURE = 'RN50'
PREFIX = 'resnet50_imagenet'
PERTS = ['label_shuffle', 'image_noise', 'uniform_images']
# onset timing -> onset epoch (ResNet-50)
ONSETS = {'early': 5, 'mid': 30, 'late': 70}
SUFFIX = '_bnrecal'


def jobs(seed_suffix=''):
    out = [(f'{PREFIX}_baseline{seed_suffix}', ep) for ep in ONSETS.values()]
    out += [(f'{PREFIX}_{p}_{phase}{seed_suffix}', ep)
            for phase, ep in ONSETS.items() for p in PERTS]
    return out


def output_paths(cache_dir, synsets_dir, run, epoch):
    name = f'{run}_epoch{epoch}{SUFFIX}'
    return (cache_dir / f'{name}.npy',
            synsets_dir / f'{name}_synsets.json',
            synsets_dir / f'{name}_meta.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--img_dir', type=Path, default=None, help='Default: <imagenet_root>/val.')
    ap.add_argument('--cache_dir', type=Path, default=CACHE_DIR)
    ap.add_argument('--synsets_dir', type=Path, default=SYNSETS_DIR)
    ap.add_argument('--calib_cache', type=Path, default=DEFAULT_CACHE / 'train_8pc.pt',
                    help='uint8 calibration images written by run_bn_recalibrated_val.py')
    ap.add_argument('--seed_suffix', default='',
                    help='replicate family, appended to every run name (e.g. "_seed43")')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    args.img_dir = args.img_dir or paths.path('imagenet_root') / 'val'

    all_jobs = jobs(args.seed_suffix)
    todo = []
    for run, epoch in all_jobs:
        npy, syn, meta = output_paths(args.cache_dir, args.synsets_dir, run, epoch)
        done = npy.is_file() and syn.is_file() and meta.is_file()
        print(f'  {run:45s} epoch {epoch:3d}  {"[done]" if done and not args.force else ""}')
        if args.force or not done:
            todo.append((run, epoch))
    print(f'{len(todo)} of {len(all_jobs)} extractions to run')
    if args.dry_run or not todo:
        return

    device = torch.device(args.device)
    cal_x = torch.load(args.calib_cache)['x']
    print(f'calibration set {tuple(cal_x.shape)} from {args.calib_cache}')

    for run, epoch in todo:
        ckpt = resolve_ckpt_path(run, epoch)
        model = build_and_load_model(ARCHITECTURE, ckpt, device)
        n_bn = _recalibrate_bn(model.net, cal_x, device, args.batch_size)
        if n_bn == 0:
            raise RuntimeError(f'{run} epoch {epoch}: no BatchNorm layers to recalibrate')
        embeddings, synsets = extract_embeddings(
            model, str(args.img_dir), args.batch_size, args.num_workers, device,
            ARCHITECTURE)
        norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
        if not np.isfinite(norms).all():
            raise RuntimeError(f'{run} epoch {epoch}: non-finite features after recalibration')

        npy, syn, meta = output_paths(args.cache_dir, args.synsets_dir, run, epoch)
        npy.parent.mkdir(parents=True, exist_ok=True)
        syn.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, embeddings)
        syn.write_text(json.dumps(synsets))
        meta.write_text(json.dumps({
            'run_name': run, 'epoch': epoch, 'architecture': ARCHITECTURE,
            'checkpoint': str(Path(ckpt).relative_to(paths.path('runs_root'))), 'n_bn_layers': n_bn,
            'calib_images': int(len(cal_x)), 'calib_cache': args.calib_cache.name,
            'n_images': int(len(embeddings)), 'dim': int(embeddings.shape[1]),
            'mean_norm': float(norms.mean()),
        }, indent=2))
        print(f'[{run} ep{epoch}] {embeddings.shape} bn={n_bn} mean norm {norms.mean():.2f} -> {npy.name}')
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
