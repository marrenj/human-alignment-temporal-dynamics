"""Mean image-pair distance at three levels of the ImageNet hierarchy, per run at the perturbed epoch.

Levels, on the penultimate features of the 50,000 ImageNet validation images
(the cached embeddings `analyze_global_local_structure.py` reads; ResNet-50 from
its BatchNorm-recalibrated cache):
  within_class      two images of the same ImageNet class
  between_class     two images of different classes in the same WordNet category
  between_category  two images of different WordNet categories
Pairs touching an 'Uncategorized' class are excluded from the two category-aware
levels, since that label groups classes with nothing in common.

Distances are cosine distances after subtracting the mean feature vector over all
50,000 images (`centered`), and without that step (`raw`). A perturbed epoch can
map every image close to one shared vector; raw distances then shrink toward 0 at
every level while the geometry of the small residual around that vector survives.
Centering separates the two.

Also records, over all 50,000 images, the effective dimensionality (participation
ratio of the feature covariance's eigenvalues) and the first principal component's
share of the variance.

Writes <results_root>/distance_levels_analysis{seed suffix}.json (one record per run and
per baseline at each onset).

Usage: python pipeline/scripts/analysis/analyze_distance_levels.py [--force]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_global_local_structure import (ARCH_SHORT, BASELINE_RUN, ONSET_EPOCH, PERTS,  # noqa: E402
                                            RESULTS_DIR, TIMINGS, _run_name,
                                            build_superclass_map, load_embeddings)

OUTPUT = RESULTS_DIR / 'distance_levels_analysis.json'
N_PAIRS = 20000
SEED = 0


def _sample_pairs(class_labels, cat_labels, rng):
    """(i, j) index arrays for each level."""
    by_class = {c: np.flatnonzero(class_labels == c) for c in np.unique(class_labels)}
    classes = np.array(list(by_class))
    categorized = np.flatnonzero(cat_labels >= 0)
    by_cat = {c: np.flatnonzero(cat_labels == c) for c in np.unique(cat_labels[categorized])}

    ci = rng.choice(classes, N_PAIRS)
    wi = np.array([rng.choice(by_class[c], 2, replace=False) for c in ci])

    bc_i, bc_j = [], []
    while len(bc_i) < N_PAIRS:
        i = rng.choice(categorized)
        j = rng.choice(by_cat[cat_labels[i]])
        if class_labels[j] != class_labels[i]:
            bc_i.append(i)
            bc_j.append(j)

    bk_i, bk_j = [], []
    while len(bk_i) < N_PAIRS:
        i, j = rng.choice(categorized, 2)
        if cat_labels[i] != cat_labels[j]:
            bk_i.append(i)
            bk_j.append(j)

    return {'within_class': (wi[:, 0], wi[:, 1]),
            'between_class': (np.array(bc_i), np.array(bc_j)),
            'between_category': (np.array(bk_i), np.array(bk_j))}


def _cosine_distance(x, i, j):
    a, b = x[i], x[j]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return 1.0 - num / np.maximum(den, 1e-12)


def level_distances(emb, pairs):
    x = emb.astype(np.float64)
    mean = x.mean(0)
    centered = x - mean
    out = {}
    for level, (i, j) in pairs.items():
        out[f'raw_{level}'] = float(_cosine_distance(x, i, j).mean())
        out[f'centered_{level}'] = float(_cosine_distance(centered, i, j).mean())
    out['mean_vector_norm'] = float(np.linalg.norm(mean))
    out['residual_norm'] = float(np.linalg.norm(centered, axis=1).mean())
    eig = np.clip(np.linalg.eigvalsh(centered.T @ centered / (len(x) - 1)), 0, None)
    out['effective_dim'] = float(eig.sum() ** 2 / (eig ** 2).sum())
    out['pc1_variance_share'] = float(eig.max() / eig.sum())
    return out


def _labels(synsets, synset_to_cat):
    classes = sorted(set(synsets))
    class_idx = {s: k for k, s in enumerate(classes)}
    cats = sorted({c for c in synset_to_cat.values() if c != 'Uncategorized'})
    cat_idx = {c: k for k, c in enumerate(cats)}
    class_labels = np.array([class_idx[s] for s in synsets])
    cat_labels = np.array([cat_idx.get(synset_to_cat.get(s, 'Uncategorized'), -1) for s in synsets])
    return class_labels, cat_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--output', type=Path, default=None,
                    help=f'default: {OUTPUT.name}, tagged with --seed_suffix')
    ap.add_argument('--seed_suffix', default=os.environ.get('TP_SEED_SUFFIX', ''),
                    help='replicate family, appended to every run name (e.g. "_seed43"; '
                         'default $TP_SEED_SUFFIX)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    sfx = args.seed_suffix
    if args.output is None:
        args.output = OUTPUT.with_name(f'{OUTPUT.stem}{sfx}{OUTPUT.suffix}')
    if args.output.exists() and not args.force:
        print(f'{args.output} exists; pass --force to recompute')
        return

    synset_to_cat = build_superclass_map()
    records = []
    for arch_key in ONSET_EPOCH:
        for timing in TIMINGS:
            onset = ONSET_EPOCH[arch_key][timing]
            base_run = BASELINE_RUN[arch_key] + sfx
            base_emb, base_syn = load_embeddings(arch_key, base_run, onset)
            class_labels, cat_labels = _labels(base_syn, synset_to_cat)
            pairs = _sample_pairs(class_labels, cat_labels, np.random.default_rng(SEED))
            runs = [('baseline', base_run, base_emb, base_syn)]
            for pert in PERTS:
                run = _run_name(arch_key, pert, timing, sfx)
                emb, syn = load_embeddings(arch_key, run, onset)
                runs.append((pert, run, emb, syn))
            for pert, run, emb, syn in runs:
                if syn != base_syn:
                    raise ValueError(f'{run} epoch {onset}: image order differs from its baseline')
                rec = {'arch': ARCH_SHORT[arch_key], 'arch_key': arch_key, 'timing': timing,
                       'onset_epoch': onset, 'pert': pert, 'run': run,
                       **level_distances(emb, pairs)}
                records.append(rec)
                print(f"{rec['arch']:10s} {timing:5s} {pert:15s} "
                      f"centered w/bc/bk {rec['centered_within_class']:.3f} "
                      f"{rec['centered_between_class']:.3f} {rec['centered_between_category']:.3f}  "
                      f"raw {rec['raw_within_class']:.4f} {rec['raw_between_class']:.4f} "
                      f"{rec['raw_between_category']:.4f}", flush=True)
    args.output.write_text(json.dumps(records, indent=1))
    print(f'Wrote {len(records)} records to {args.output}')


if __name__ == '__main__':
    main()
