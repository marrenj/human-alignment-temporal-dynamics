"""Characterize the collapse of CLIP ViT-B/32's image features after one epoch of uniform images.

After an early (epoch 5) uniform-images epoch, the 768-d CLS features of the
50,000 ImageNet val images lie on a one-dimensional arc. This script asks what
that arc encodes and why the collapsed model still classifies CIFAR-10 well
above chance:

1. Dimensionality (PCA variance fractions, participation ratio) of the cached
   ImageNet-val CLS features for the collapsed run, the other early CLIP runs,
   and the baseline, from `<feature_cache>/cross_arch_embeddings/`
   (written by evaluation/extract_imagenet_embeddings.py).
2. Twelve low-level statistics of every ImageNet val image as CLIP sees it
   (bicubic Resize(224) -> CenterCrop(224)), correlated (Spearman) with each
   model state's leading principal components.
3. CIFAR-10 (standard 50k/10k split, python pickle batches; `--cifar_dir`)
   linear probes, all fit with the paper's protocol
   (`linear_probe_eval._select_C` + `_fit_and_score_torch`), on: baseline and
   collapsed CLIP features, the collapsed features' arc axis alone, the
   collapsed features minus their top three ImageNet principal directions,
   the collapsed features minus everything the twelve statistics explain
   linearly, the twelve statistics alone, and raw pixels.

Every feature set is z-scored on train before its probe
(`linear_probe_eval.fit_standardizer`), as in the paper's probes.

Writes `<results_root>/clip_uniform_collapse{seed suffix}/summary.json` and
`pca_points.npz` (PC scores and luminance for every 10th ImageNet val image).
Intermediates are cached under `<feature_cache>/clip_uniform_collapse{seed suffix}/`.

Usage:
    python pipeline/scripts/analysis/analyze_clip_uniform_collapse.py
    python pipeline/scripts/analysis/analyze_clip_uniform_collapse.py --device cuda:1 --force
"""

import argparse
import json
import logging
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage, stats
from torchvision import transforms

ANALYSIS_DIR = Path(__file__).resolve().parent
PIPELINE = ANALYSIS_DIR.parent.parent
REPO_ROOT = PIPELINE.parent
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths
from temporal_perturbations.data.eval_transforms import ARCH_EVAL_PREPROCESSING
from temporal_perturbations.inference.linear_probe_eval import (
    PROBE_C_SELECTION, PROBE_FEATURE_SCALING, _fit_and_score_torch, _select_C, fit_standardizer,
    is_current_recipe)
from temporal_perturbations.models.clip_yfcc.feature_space import load_clip_feature_model

EMBED_DIR = paths.path('feature_cache') / 'cross_arch_embeddings'
PROBE_DIR = paths.results_root() / 'linear_probe_eval_scratch_std'
# Set from the CLI in main().
IMAGENET_VAL = CIFAR_DIR = None

BASELINE = 'clip_vitb32_baseline'
COLLAPSED = 'clip_vitb32_uniform_images_early'
ONSET = 5
# (label, run, epoch): every cached ImageNet-val embedding set of interest.
STATES = [
    ('baseline, epoch 5', BASELINE, 5),
    ('uniform images, epoch 5', COLLAPSED, 5),
    ('image noise, epoch 5', 'clip_vitb32_image_noise_early', 5),
    ('label shuffle, epoch 5', 'clip_vitb32_label_shuffle_early', 5),
    ('uniform images, epoch 10', COLLAPSED, 10),
    ('uniform images, epoch 31', COLLAPSED, 31),
    ('baseline, epoch 31', BASELINE, 31),
]
PLOT_STRIDE = 10
N_REMOVED_PCS = 3

STAT_NAMES = ['luminance', 'RMS contrast', 'saturation', 'colorfulness',
              'mean red', 'mean green', 'mean blue', 'red–green balance',
              'yellow–blue balance', 'edge density', 'high-frequency power', 'entropy']


def image_stats(rgb):
    """Twelve low-level statistics of one RGB image in [0, 1] (H x W x 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-12), 0.0)
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    edges = np.hypot(ndimage.sobel(y, 0), ndimage.sobel(y, 1)).mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(y - y.mean()))) ** 2
    h, w = y.shape
    fy, fx = np.meshgrid(np.arange(h) - h // 2, np.arange(w) - w // 2, indexing='ij')
    radius = np.hypot(fy / (h / 2), fx / (w / 2))
    high = power[radius > 0.5].sum() / max(power.sum(), 1e-12)
    hist = np.histogram(y, bins=64, range=(0, 1))[0] / y.size
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return [y.mean(), y.std(), sat.mean(), colorfulness, r.mean(), g.mean(), b.mean(),
            rg.mean(), yb.mean(), edges, high, entropy]


# The geometric half of CLIP's eval transform: the pixels CLIP is given.
_CLIP_VIEW = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])


def _imagenet_stats_one(path):
    im = _CLIP_VIEW(Image.open(path).convert('RGB'))
    return image_stats(np.asarray(im, np.float64) / 255)


def _uint8_stats_one(img):
    return image_stats(img.astype(np.float64) / 255)


def imagenet_paths():
    """ImageFolder order: sorted class folders, sorted files -- the row order of
    the cached embeddings."""
    return [str(f) for d in sorted(p for p in IMAGENET_VAL.iterdir() if p.is_dir())
            for f in sorted(d.iterdir())]


def imagenet_stats(cache, workers, force):
    path = cache / 'imagenet_val_stats.npy'
    if path.is_file() and not force:
        return np.load(path)
    paths = imagenet_paths()
    print(f'computing low-level stats for {len(paths)} ImageNet val images')
    with Pool(workers) as pool:
        out = np.array(pool.map(_imagenet_stats_one, paths, chunksize=64))
    np.save(path, out)
    return out


def participation_ratio(eigvals):
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


def pca(X, fit_stride=5):
    """Mean, principal axes and eigenvalues, fit on every `fit_stride`-th row."""
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X[::fit_stride] - mu, full_matrices=False)
    return mu, vt, s ** 2


def load_embeddings(run, epoch):
    X = np.load(EMBED_DIR / f'{run}_epoch{epoch}.npy').astype(np.float64)
    if X.shape != (50000, 768):
        raise ValueError(f'{run} epoch {epoch}: embeddings {X.shape}, expected (50000, 768)')
    return X


def imagenet_analysis(stats_in):
    states, points = [], {}
    lum = stats_in[:, 0]
    for label, run, epoch in STATES:
        X = load_embeddings(run, epoch)
        mu, vt, ev = pca(X)
        frac = ev / ev.sum()
        scores = (X - mu) @ vt[:3].T
        # PCs have no sign; orient PC1 so brighter images score higher.
        if stats.spearmanr(scores[:, 0], lum)[0] < 0:
            scores[:, 0] *= -1
        corr = {pc: {name: float(stats.spearmanr(scores[:, i], stats_in[:, j])[0])
                     for j, name in enumerate(STAT_NAMES)}
                for i, pc in enumerate(['PC1', 'PC2', 'PC3'])}
        design = np.column_stack([np.ones(len(X)), stats_in])
        coef, *_ = np.linalg.lstsq(design, scores[:, 0], rcond=None)
        r2 = 1 - np.var(scores[:, 0] - design @ coef) / np.var(scores[:, 0])
        # Position along the arc: the angle about the circle fitted to (PC1, PC2).
        a = np.column_stack([2 * scores[:, 0], 2 * scores[:, 1], np.ones(len(X))])
        c1, c2, _ = np.linalg.lstsq(a, scores[:, 0] ** 2 + scores[:, 1] ** 2, rcond=None)[0]
        angle = np.arctan2(scores[:, 1] - c2, scores[:, 0] - c1)
        states.append({
            'label': label, 'run': run, 'epoch': epoch,
            'pc_variance_fraction': [float(v) for v in frac[:10]],
            'effective_dim': participation_ratio(ev),
            'mean_norm': float(np.linalg.norm(X, axis=1).mean()),
            'spearman': corr,
            'pc1_r2_from_all_stats': float(r2),
            'spearman_pc1_vs_arc_angle': float(abs(stats.spearmanr(scores[:, 0], angle)[0])),
        })
        points[label] = scores[::PLOT_STRIDE, :2]
        print(f'{label:26s} PC1 {frac[0]:.3f}  eff.dim {participation_ratio(ev):6.1f}  '
              f'rho(PC1, luminance) {corr["PC1"]["luminance"]:+.3f}  R2(PC1 | stats) {r2:.3f}')
    return states, points


def load_cifar():
    def batch(name):
        with open(CIFAR_DIR / name, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        x = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return x, np.array(d[b'labels'])
    tr = [batch(f'data_batch_{i}') for i in range(1, 6)]
    x_tr, y_tr = np.concatenate([t[0] for t in tr]), np.concatenate([t[1] for t in tr])
    x_te, y_te = batch('test_batch')
    return x_tr, y_tr, x_te, y_te


@torch.no_grad()
def clip_features(run, epoch, split, images, device, cache, force):
    path = cache / f'cifar10_{split}_{run}_epoch{epoch}.npy'
    if path.is_file() and not force:
        return np.load(path)
    ckpt = paths.path('runs_root') / run / 'checkpoints' / 'all_epochs' / f'epoch_{epoch:03d}.ckpt'
    model = load_clip_feature_model(str(ckpt), clip_feature='cls', device=device).eval()
    tf = ARCH_EVAL_PREPROCESSING['CLIP ViT-B/32'].build(224)
    out = []
    for i in range(0, len(images), 500):
        x = torch.stack([tf(Image.fromarray(im)) for im in images[i:i + 500]]).to(device)
        out.append(model(x).float().cpu().numpy())
    feats = np.concatenate(out)
    np.save(path, feats)
    del model
    torch.cuda.empty_cache()
    return feats


def probe(name, x_tr, y_tr, x_te, y_te, device, logger):
    standardize = fit_standardizer(x_tr)
    x_tr, x_te = standardize(x_tr), standardize(x_te)
    C, sweep = _select_C(_fit_and_score_torch, x_tr, y_tr, logger, device=device)
    acc, info = _fit_and_score_torch(x_tr, y_tr, x_te, y_te, logger, device=device, C=C)
    print(f'  {name:48s} top-1 {acc:.4f}  (C={C:.3g}, {x_tr.shape[1]} features, '
          f'{"converged" if info["converged"] else "NOT converged"})')
    return {'feature_set': name, 'top1': float(acc), 'C': float(C),
            'n_features': int(x_tr.shape[1]), 'converged': bool(info['converged']),
            'c_selection': PROBE_C_SELECTION, 'feature_scaling': PROBE_FEATURE_SCALING,
            'c_sweep': sweep}


def cifar_analysis(imagenet_pca, device, cache, workers, force):
    x_tr, y_tr, x_te, y_te = load_cifar()
    print(f'CIFAR-10: {len(x_tr)} train, {len(x_te)} test')
    with Pool(workers) as pool:
        s_tr = np.array(pool.map(_uint8_stats_one, list(x_tr), chunksize=256))
        s_te = np.array(pool.map(_uint8_stats_one, list(x_te), chunksize=256))
    f_base = {s: clip_features(BASELINE, ONSET, s, x, device, cache, force)
              for s, x in (('train', x_tr), ('test', x_te))}
    f_coll = {s: clip_features(COLLAPSED, ONSET, s, x, device, cache, force)
              for s, x in (('train', x_tr), ('test', x_te))}
    for name, f in (('baseline', f_base), ('collapsed', f_coll)):
        if len(f['train']) != len(x_tr) or len(f['test']) != len(x_te):
            raise ValueError(f'{name} CIFAR-10 features do not match the splits')

    mu, vt = imagenet_pca
    top = vt[:N_REMOVED_PCS]

    def remove_top(X):
        Xc = X - mu
        return X - Xc @ top.T @ top

    def arc_axis(X):
        return (X - mu) @ vt[:1].T

    # Remove what the twelve statistics predict linearly (fit on train).
    design_tr = np.column_stack([np.ones(len(s_tr)), s_tr])
    design_te = np.column_stack([np.ones(len(s_te)), s_te])
    beta, *_ = np.linalg.lstsq(design_tr, f_coll['train'], rcond=None)
    resid_tr, resid_te = f_coll['train'] - design_tr @ beta, f_coll['test'] - design_te @ beta
    r2_stats = 1 - resid_te.var(0).sum() / f_coll['test'].var(0).sum()

    logger = logging.getLogger('probe')
    logging.basicConfig(level=logging.WARNING)
    sets = [
        ('baseline CLIP features', f_base['train'], f_base['test']),
        ('collapsed CLIP features', f_coll['train'], f_coll['test']),
        ('collapsed, arc axis (PC1) only', arc_axis(f_coll['train']), arc_axis(f_coll['test'])),
        (f'collapsed, top {N_REMOVED_PCS} PCs removed', remove_top(f_coll['train']),
         remove_top(f_coll['test'])),
        ('collapsed, low-level stats regressed out', resid_tr, resid_te),
        ('12 low-level image statistics', s_tr, s_te),
        ('raw pixels (32 x 32 x 3)', x_tr.reshape(len(x_tr), -1), x_te.reshape(len(x_te), -1)),
    ]
    probes = [probe(name, a.astype(np.float64), y_tr, b.astype(np.float64), y_te, device, logger)
              for name, a, b in sets]
    arc_te = arc_axis(f_coll['test'])[:, 0]
    if stats.spearmanr(arc_te, s_te[:, 0])[0] < 0:
        arc_te = -arc_te
    return {
        'n_train': int(len(x_tr)), 'n_test': int(len(x_te)),
        'probes': probes,
        'share_of_collapsed_variance_explained_by_stats': float(r2_stats),
        'spearman_arc_axis_vs_stats_cifar_test': {
            name: float(stats.spearmanr(arc_te, s_te[:, j])[0])
            for j, name in enumerate(STAT_NAMES)},
    }


def official_probe(run, epoch):
    path = PROBE_DIR / run / 'probe' / f'vtab_cifar10_epoch{epoch}_clip.json'
    with path.open() as f:
        rec = json.load(f)
    if not is_current_recipe(rec):
        raise ValueError(f'{path} was fit under an older probe recipe')
    return {'top1': rec['score'], 'n_train': rec['n_train'], 'n_test': rec['n_test']}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--output_dir', type=Path, default=None,
                    help='default <results_root>/clip_uniform_collapse{seed_suffix}')
    ap.add_argument('--imagenet_val', type=Path, default=None,
                    help='ImageNet val ImageFolder (default <imagenet_root>/val)')
    ap.add_argument('--cifar_dir', type=Path, default=None,
                    help='CIFAR-10 python batches (default <probe_datasets>/cifar-10-batches-py)')
    ap.add_argument('--force', action='store_true', help='recompute cached intermediates')
    ap.add_argument('--seed_suffix', default=os.environ.get('TP_SEED_SUFFIX', ''),
                    help='replicate family, appended to every run name (e.g. "_seed43"; '
                         'default $TP_SEED_SUFFIX)')
    args = ap.parse_args()

    global BASELINE, COLLAPSED, STATES, IMAGENET_VAL, CIFAR_DIR
    IMAGENET_VAL = args.imagenet_val or paths.path('imagenet_root') / 'val'
    CIFAR_DIR = args.cifar_dir or paths.path('probe_datasets') / 'cifar-10-batches-py'
    args.output_dir = args.output_dir or paths.results_root() / f'clip_uniform_collapse{args.seed_suffix}'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    BASELINE += args.seed_suffix
    COLLAPSED += args.seed_suffix
    STATES = [(label, run + args.seed_suffix, epoch) for label, run, epoch in STATES]

    cache = paths.path('feature_cache') / f'clip_uniform_collapse{args.seed_suffix}'
    cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    stats_in = imagenet_stats(cache, args.workers, args.force)
    states, points = imagenet_analysis(stats_in)
    coll = load_embeddings(COLLAPSED, ONSET)
    mu, vt, _ = pca(coll)
    cifar = cifar_analysis((mu, vt), device, cache, args.workers, args.force)

    summary = {
        'stat_names': STAT_NAMES,
        'imagenet_states': states,
        'cifar10': cifar,
        'official_cifar10_probe': {'baseline': official_probe(BASELINE, ONSET),
                                   'collapsed': official_probe(COLLAPSED, ONSET)},
        'plot_stride': PLOT_STRIDE,
    }
    with (args.output_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    np.savez(args.output_dir / 'pca_points.npz',
             luminance=stats_in[::PLOT_STRIDE, 0],
             **{label: pts for label, pts in points.items()})
    print(f'wrote {args.output_dir / "summary.json"}')


if __name__ == '__main__':
    main()
