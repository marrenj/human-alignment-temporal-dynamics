"""Extract penultimate ImageNet-val features for every baseline and perturbed run at its onset epoch.

Writes `{run}_epoch{E}.npy` to `<feature_cache>/cross_arch_embeddings/` and the matching
`{run}_epoch{E}_synsets.json` to `<results_root>/imagenet_embeddings/`, for the baseline and all nine perturbed runs of each architecture, each at its onset
epoch: ResNet-50 / ViT-B/16 / ConvNeXt-B at 5 / 30 / 70, CLIP ViT-B/32 at
5 / 13 / 28. These are the features read by the TriMaps (`visualization/trimap_imagenet.py`,
Fig 3B) and by `analysis/analyze_global_local_structure.py`.

The timm families go through `trimap_imagenet.extract_and_cache` (pooled
penultimate feature, per-recipe preprocessing, fp16 autocast); CLIP uses the same
extraction loop on the 768-d pre-projection CLS token. ResNet-50 files here carry the
checkpoints' stored BatchNorm statistics, which are corrupted at a perturbed epoch;
the recalibrated versions come from `extract_imagenet_embeddings_bn_recalibrated.py`.

Existing files are skipped unless `--force`.

Usage:
    python pipeline/scripts/evaluation/extract_imagenet_embeddings.py --dry-run
    python pipeline/scripts/evaluation/extract_imagenet_embeddings.py --device cuda:0
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE = EVAL_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PIPELINE / 'scripts' / 'visualization'))

from temporal_perturbations import paths  # noqa: E402
from trimap_imagenet import (CACHE_DIR, SYNSETS_DIR, extract_and_cache,  # noqa: E402
                             extract_embeddings, resolve_ckpt_path)

CLIP_PREFIX = 'clip_vitb32'
CLIP_ARCHITECTURE = 'CLIP ViT-B/32'
ONSET_EPOCH = {
    'resnet50_imagenet':   {'early': 5, 'mid': 30, 'late': 70},
    'vitb16_imagenet':     {'early': 5, 'mid': 30, 'late': 70},
    'convnext_b_imagenet': {'early': 5, 'mid': 30, 'late': 70},
    CLIP_PREFIX:           {'early': 5, 'mid': 13, 'late': 28},
}
ARCHITECTURE = {
    'resnet50_imagenet':   'RN50',
    'vitb16_imagenet':     'ViT-B/16',
    'convnext_b_imagenet': 'ConvNeXt-B',
    CLIP_PREFIX:           CLIP_ARCHITECTURE,
}
PERTS = ['label_shuffle', 'image_noise', 'uniform_images']

logger = logging.getLogger(__name__)


def run_name(prefix, pert, timing, seed_suffix=''):
    return f'{prefix}_{pert}_{timing}{seed_suffix}'


def jobs(seed_suffix=''):
    out = []
    for prefix, onsets in ONSET_EPOCH.items():
        for timing, epoch in onsets.items():
            out.append((prefix, f'{prefix}_baseline' + seed_suffix, epoch))
            out += [(prefix, run_name(prefix, p, timing, seed_suffix), epoch) for p in PERTS]
    return list(dict.fromkeys(out))


def extract_clip_and_cache(run, epoch, cache_dir, synsets_dir, img_dir, device, batch_size,
                           num_workers):
    from temporal_perturbations.models.clip_yfcc.feature_space import load_clip_feature_model

    model = load_clip_feature_model(str(resolve_ckpt_path(run, epoch)), clip_feature='cls',
                                    device=device).eval()
    embeddings, synset_ids = extract_embeddings(model, str(img_dir), batch_size, num_workers,
                                                device, CLIP_ARCHITECTURE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f'{run}_epoch{epoch}.npy', embeddings)
    synsets_dir.mkdir(parents=True, exist_ok=True)
    with (synsets_dir / f'{run}_epoch{epoch}_synsets.json').open('w') as f:
        json.dump(synset_ids, f)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--img_dir', type=Path, default=None, help='Default: <imagenet_root>/val.')
    ap.add_argument('--cache_dir', type=Path, default=CACHE_DIR)
    ap.add_argument('--synsets_dir', type=Path, default=SYNSETS_DIR)
    ap.add_argument('--seed_suffix', default='',
                    help='replicate family, appended to every run name (e.g. "_seed43")')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    args.img_dir = args.img_dir or paths.path('imagenet_root') / 'val'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    all_jobs = jobs(args.seed_suffix)
    todo = [j for j in all_jobs if args.force
            or not (args.cache_dir / f'{j[1]}_epoch{j[2]}.npy').exists()
            or not (args.synsets_dir / f'{j[1]}_epoch{j[2]}_synsets.json').exists()]
    print(f'{len(all_jobs)} run-epochs, {len(todo)} to extract')
    for prefix, run, epoch in todo:
        print(f'  {run} epoch {epoch}')
    if args.dry_run:
        return

    device = torch.device(args.device)
    for prefix, run, epoch in todo:
        t0 = time.time()
        if prefix == CLIP_PREFIX:
            extract_clip_and_cache(run, epoch, args.cache_dir, args.synsets_dir, args.img_dir,
                                   device, args.batch_size, args.num_workers)
        else:
            extract_and_cache(run, epoch, ARCHITECTURE[prefix], cache_dir=args.cache_dir,
                              img_dir=args.img_dir, device=device, batch_size=args.batch_size,
                              num_workers=args.num_workers, force=args.force,
                              synsets_dir=args.synsets_dir)
        print(f'{run} epoch {epoch}: {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
