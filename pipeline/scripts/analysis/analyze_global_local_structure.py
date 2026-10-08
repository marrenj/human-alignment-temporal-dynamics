"""Global vs. local category structure of penultimate features at the perturbed epoch, per run (Fig 3 C/D).

Reads the cached ImageNet-val embeddings in `<feature_cache>/cross_arch_embeddings/`
(image order in `<results_root>/imagenet_embeddings/`) and writes
`<results_root>/global_local_structure_analysis{seed suffix}.json`. ResNet-50 uses
the `_bnrecal` embeddings (BatchNorm re-estimated on clean images, from
`evaluation/extract_imagenet_embeddings_bn_recalibrated.py`) for both the run and its baseline: its plain
cached features were extracted with the corrupted BatchNorm statistics a
perturbed epoch leaves behind. ResNet-50 accuracy must likewise come from the
BatchNorm-recalibrated evaluation; there is no fallback to the logged value.

Hypothesis: models that keep SPoSE alignment + ImageNet accuracy retain
both global and local structure; those that keep SPoSE but lose ImageNet
retain global but lose local; those that lose both lose everything.

Metrics computed per condition (relative to baseline at matched onset epoch):

Global structure:
  - category_rdm_rho: Spearman corr of 16×16 superordinate category centroid RDM
  - category_silhouette: silhouette score at superordinate category level
  - between_category_rdm_std: std of between-category pairwise distances (structure richness)

Local structure:
  - class_rdm_rho: Spearman corr of 1000×1000 class centroid RDM
  - class_silhouette: silhouette score at class level (sampled)
  - knn_purity_k10: fraction of 10-NN that are same class
  - within_class_cosine_sim: mean within-class cosine similarity

Mixed / overall:
  - effective_dim: participation ratio (effective dimensionality) over all 50,000 images
  - rdm_std: std of full RDM upper triangle (overall structure richness)
  - mean_norm: mean L2 norm of embeddings (representation magnitude)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ANALYSIS_DIR = Path(__file__).resolve().parent
PIPELINE = ANALYSIS_DIR.parent.parent
REPO_ROOT = PIPELINE.parent

sys.path.insert(0, str(PIPELINE / 'scripts' / 'visualization'))
from arch_runs import (  # noqa: E402
    CLIP_PREFIX, RESULTS as RESULTS_DIR, rdm1854_corrected_suffix, require_current_probe,
)
from temporal_perturbations import paths  # noqa: E402

CACHE_DIR = paths.path('feature_cache') / 'cross_arch_embeddings'
SYNSETS_DIR = RESULTS_DIR / 'imagenet_embeddings'

ONSET_EPOCH = {
    'resnet50_imagenet':   {'early': 5,  'mid': 30, 'late': 70},
    'vitb16_imagenet':     {'early': 5,  'mid': 30, 'late': 70},
    'convnext_b_imagenet': {'early': 5,  'mid': 30, 'late': 70},
    CLIP_PREFIX:           {'early': 5,  'mid': 13, 'late': 28},
}

ARCH_SHORT = {
    'resnet50_imagenet': 'RN50',
    'vitb16_imagenet': 'ViT-B/16',
    'convnext_b_imagenet': 'ConvNeXt-B',
    CLIP_PREFIX: 'CLIP',
}

BASELINE_RUN = {
    'resnet50_imagenet':   'resnet50_imagenet_baseline',
    'vitb16_imagenet':     'vitb16_imagenet_baseline',
    'convnext_b_imagenet': 'convnext_b_imagenet_baseline',
    CLIP_PREFIX:           'clip_vitb32_baseline',
}

PERTS = ['label_shuffle', 'image_noise', 'uniform_images']
TIMINGS = ['early', 'mid', 'late']


def _run_name(key, pert, timing, seed_suffix=''):
    return f'{key}_{pert}_{timing}{seed_suffix}'


def clip_probe_top1(clip_ood_dir, run, epoch):
    """CLIP's ImageNet linear-probe top-1 at `epoch`; raises unless the record exists
    and was fit under the current probe recipe, so a re-run cannot mix recipes."""
    path = clip_ood_dir / run / f'ood_probe_eval_epoch{epoch}.json'
    if not path.is_file():
        raise FileNotFoundError(f'no CLIP ImageNet probe for {run} at epoch {epoch}: {path}')
    with path.open() as f:
        return require_current_probe(json.load(f), path)['imagenet_val']['top1_accuracy']


# Architectures whose embeddings must come from the BatchNorm-recalibrated cache.
EMBEDDING_SUFFIX = {'resnet50_imagenet': '_bnrecal'}


def _embedding_path(arch_key, run_name, epoch):
    return CACHE_DIR / f'{run_name}_epoch{epoch}{EMBEDDING_SUFFIX.get(arch_key, "")}.npy'


def _synsets_path(arch_key, run_name, epoch):
    return SYNSETS_DIR / f'{run_name}_epoch{epoch}{EMBEDDING_SUFFIX.get(arch_key, "")}_synsets.json'


def load_embeddings(arch_key, run_name, epoch):
    emb_path = _embedding_path(arch_key, run_name, epoch)
    syn_path = _synsets_path(arch_key, run_name, epoch)
    if not emb_path.is_file():
        return None, None
    emb = np.load(emb_path)
    with syn_path.open() as f:
        synsets = json.load(f)
    return emb, synsets


def build_superclass_map():
    cache_path = (REPO_ROOT / 'temporal_perturbations' / 'data'
                  / '_imagenet_superclass_cache.json')
    if cache_path.is_file():
        with cache_path.open() as f:
            return json.load(f)
    sys.path.insert(0, str(REPO_ROOT))
    from temporal_perturbations.data.imagenet_superclasses import load_synset_to_category
    return load_synset_to_category()


def compute_class_centroids(embeddings, synsets):
    unique_synsets = sorted(set(synsets))
    synset_to_idx = {s: i for i, s in enumerate(unique_synsets)}
    labels = np.array([synset_to_idx[s] for s in synsets])
    n_classes = len(unique_synsets)
    d = embeddings.shape[1]
    centroids = np.zeros((n_classes, d), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=int)
    for i in range(len(synsets)):
        c = labels[i]
        centroids[c] += embeddings[i].astype(np.float64)
        counts[c] += 1
    centroids /= counts[:, None]
    return centroids, unique_synsets, labels


def compute_category_centroids(class_centroids, unique_synsets, synset_to_category):
    cat_names = sorted(set(synset_to_category.values()))
    cat_to_idx = {c: i for i, c in enumerate(cat_names)}
    n_cats = len(cat_names)
    d = class_centroids.shape[1]
    centroids = np.zeros((n_cats, d), dtype=np.float64)
    counts = np.zeros(n_cats, dtype=int)
    class_cat_labels = []
    for i, syn in enumerate(unique_synsets):
        cat = synset_to_category.get(syn, 'Uncategorized')
        ci = cat_to_idx[cat]
        centroids[ci] += class_centroids[i]
        counts[ci] += 1
        class_cat_labels.append(ci)
    mask = counts > 0
    centroids[mask] /= counts[mask, None]
    return centroids, cat_names, np.array(class_cat_labels)


def cosine_rdm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_normed = X / norms
    sim = X_normed @ X_normed.T
    return 1.0 - sim


def rdm_upper_tri(rdm):
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def participation_ratio(X):
    X_centered = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total < 1e-12:
        return 0.0
    return total ** 2 / (eigenvalues ** 2).sum()


def sampled_silhouette(embeddings, labels, n_sample=5000, seed=42):
    from sklearn.metrics import silhouette_score
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n > n_sample:
        idx = rng.choice(n, n_sample, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or (counts < 2).sum() > len(unique) * 0.5:
        return np.nan
    mask = np.isin(labels, unique[counts >= 2])
    if mask.sum() < 100:
        return np.nan
    return silhouette_score(embeddings[mask], labels[mask], metric='cosine',
                            sample_size=min(2000, mask.sum()),
                            random_state=seed)


def knn_purity(embeddings, labels, k=10, n_sample=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n > n_sample:
        idx = rng.choice(n, n_sample, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_normed = embeddings / norms
    sim = X_normed @ X_normed.T
    np.fill_diagonal(sim, -np.inf)
    topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    purity = 0.0
    for i in range(len(labels)):
        neighbors = topk_idx[i]
        purity += (labels[neighbors] == labels[i]).mean()
    return purity / len(labels)


def mean_within_class_cosine_sim(embeddings, labels, n_sample_classes=200, seed=42):
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    if len(unique_labels) > n_sample_classes:
        sampled = rng.choice(unique_labels, n_sample_classes, replace=False)
    else:
        sampled = unique_labels

    sims = []
    for c in sampled:
        mask = labels == c
        class_emb = embeddings[mask]
        if len(class_emb) < 2:
            continue
        norms = np.linalg.norm(class_emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = class_emb / norms
        sim_mat = normed @ normed.T
        n = len(class_emb)
        triu = sim_mat[np.triu_indices(n, k=1)]
        sims.append(triu.mean())
    return np.mean(sims) if sims else np.nan


def compute_all_metrics(embeddings, synsets, synset_to_category):
    class_centroids, unique_synsets, class_labels = compute_class_centroids(embeddings, synsets)
    cat_centroids, cat_names, class_cat_labels = compute_category_centroids(
        class_centroids, unique_synsets, synset_to_category)

    image_cat_labels = np.array([class_cat_labels[class_labels[i]] for i in range(len(class_labels))])

    class_rdm = cosine_rdm(class_centroids)
    class_rdm_vec = rdm_upper_tri(class_rdm)

    cat_rdm = cosine_rdm(cat_centroids)
    cat_rdm_vec = rdm_upper_tri(cat_rdm)

    mean_norm = np.linalg.norm(embeddings, axis=1).mean()

    pr = participation_ratio(embeddings)

    class_sil = sampled_silhouette(embeddings, class_labels, n_sample=5000)
    cat_sil = sampled_silhouette(embeddings, image_cat_labels, n_sample=5000)

    knn_pur = knn_purity(embeddings, class_labels, k=10, n_sample=5000)

    wc_sim = mean_within_class_cosine_sim(embeddings, class_labels)

    rdm_s = class_rdm_vec.std() if len(class_rdm_vec) > 0 else 0.0

    return {
        'class_rdm_vec': class_rdm_vec,
        'cat_rdm_vec': cat_rdm_vec,
        'class_silhouette': class_sil,
        'category_silhouette': cat_sil,
        'knn_purity_k10': knn_pur,
        'within_class_cosine_sim': wc_sim,
        'effective_dim': pr,
        'rdm_std': rdm_s,
        'mean_norm': mean_norm,
        'cat_rdm_std': cat_rdm_vec.std() if len(cat_rdm_vec) > 0 else 0.0,
    }


def load_accuracy_data(seed_suffix=''):
    curves_path = RESULTS_DIR / f'imagenet_val_curves{seed_suffix}.json'
    if not curves_path.is_file():
        return {}
    with curves_path.open() as f:
        data = json.load(f)
    return data.get('runs', {})


def load_alignment_data(seed_suffix=''):
    """Load SPoSE alignment from posthoc_rdm1854 results."""
    import make_fig_cross_architecture as CA

    BN_RECALIBRATED_ARCHES = {'resnet50_imagenet'}

    out = {}
    for key in ONSET_EPOCH:
        suffix = rdm1854_corrected_suffix(key)
        if key in BN_RECALIBRATED_ARCHES:
            suffix = '_bn_recal' + suffix

        base_run = BASELINE_RUN[key] + seed_suffix
        base_traj = CA.load_traj(base_run, suffix)
        if base_traj is None:
            print(f'  WARNING: no baseline alignment for {key}')
            continue
        b_ep, b_rsa, _ = base_traj
        b_lookup = dict(zip(b_ep.astype(int).tolist(), b_rsa.tolist()))

        for timing in TIMINGS:
            onset = ONSET_EPOCH[key][timing]
            if onset in b_lookup:
                out[(key, 'baseline', timing)] = b_lookup[onset]
            for pert in PERTS:
                run = _run_name(key, pert, timing, seed_suffix)
                tr = CA.load_traj(run, suffix)
                if tr is None:
                    continue
                p_ep, p_rsa, _ = tr
                p_lookup = dict(zip(p_ep.astype(int).tolist(), p_rsa.tolist()))
                if onset in p_lookup:
                    out[(key, pert, timing)] = p_lookup[onset]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seed_suffix', default=os.environ.get('TP_SEED_SUFFIX', ''),
                    help='replicate family, appended to every run name (e.g. "_seed43"; '
                         'default $TP_SEED_SUFFIX); also tags the output file')
    sfx = ap.parse_args().seed_suffix

    print('Building superclass map...')
    synset_to_cat = build_superclass_map()
    print(f'  {len(synset_to_cat)} synsets mapped')

    print('\nLoading accuracy data...')
    acc_curves = load_accuracy_data(sfx)

    print('Loading alignment data...')
    alignment = load_alignment_data(sfx)
    print(f'  {len(alignment)} alignment values')

    clip_ood_dir = RESULTS_DIR / 'imagenet_ood_eval_scratch_clip_cls'
    bn_recal_dir = RESULTS_DIR / 'bn_recalibrated_val'

    results = []

    for arch_key in ONSET_EPOCH:
        arch_short = ARCH_SHORT[arch_key]
        base_run = BASELINE_RUN[arch_key] + sfx

        for timing in TIMINGS:
            onset = ONSET_EPOCH[arch_key][timing]

            # Load baseline embeddings
            base_emb, base_syn = load_embeddings(arch_key, base_run, onset)
            if base_emb is None:
                print(f'  SKIP: no baseline embeddings for {arch_key} epoch {onset}')
                continue

            print(f'\n=== {arch_short} / {timing} (epoch {onset}) ===')
            print('  Computing baseline metrics...')
            base_metrics = compute_all_metrics(base_emb, base_syn, synset_to_cat)

            # Get baseline accuracy
            if arch_key == CLIP_PREFIX:
                base_acc = clip_probe_top1(clip_ood_dir, base_run, onset)
            elif arch_key == 'resnet50_imagenet':
                # RN50 uses BN-recalibrated
                bn_path = bn_recal_dir / f'{base_run}_epoch{onset}.json'
                with bn_path.open() as f:
                    base_acc = json.load(f)['bn_recalibrated']['val_acc1']
            else:
                base_acc = acc_curves.get(base_run, {}).get('val_acc1', {}).get(str(onset))

            base_align = alignment.get((arch_key, 'baseline', timing))

            print(f'  Baseline: acc={base_acc}, align={base_align}')

            for pert in PERTS:
                run = _run_name(arch_key, pert, timing, sfx)
                pert_emb, pert_syn = load_embeddings(arch_key, run, onset)
                if pert_emb is None:
                    print(f'  SKIP: no embeddings for {run}')
                    continue

                print(f'  Computing {pert} metrics...')
                pert_metrics = compute_all_metrics(pert_emb, pert_syn, synset_to_cat)

                # Compute relative metrics (perturbed vs baseline)
                cat_rdm_rho = stats.spearmanr(base_metrics['cat_rdm_vec'],
                                               pert_metrics['cat_rdm_vec'])[0]
                class_rdm_rho = stats.spearmanr(base_metrics['class_rdm_vec'],
                                                 pert_metrics['class_rdm_vec'])[0]

                # Get perturbed accuracy
                if arch_key == CLIP_PREFIX:
                    pert_acc = clip_probe_top1(clip_ood_dir, run, onset)
                elif arch_key == 'resnet50_imagenet':
                    bn_path = bn_recal_dir / f'{run}_epoch{onset}.json'
                    with bn_path.open() as f:
                        pert_acc = json.load(f)['bn_recalibrated']['val_acc1']
                else:
                    pert_acc = acc_curves.get(run, {}).get('val_acc1', {}).get(str(onset))

                pert_align = alignment.get((arch_key, pert, timing))

                # Retained fractions
                acc_retained = pert_acc / base_acc if (base_acc and pert_acc) else None
                align_retained = pert_align / base_align if (base_align and pert_align) else None

                row = {
                    'arch': arch_short,
                    'arch_key': arch_key,
                    'pert': pert,
                    'timing': timing,
                    'onset_epoch': onset,
                    'bn_recalibrated_features': arch_key in EMBEDDING_SUFFIX,
                    # Raw readouts
                    'base_acc': base_acc,
                    'pert_acc': pert_acc,
                    'base_align': base_align,
                    'pert_align': pert_align,
                    'acc_retained': acc_retained,
                    'align_retained': align_retained,
                    # Global structure
                    'category_rdm_rho': cat_rdm_rho,
                    'category_silhouette_base': base_metrics['category_silhouette'],
                    'category_silhouette_pert': pert_metrics['category_silhouette'],
                    'cat_rdm_std_base': base_metrics['cat_rdm_std'],
                    'cat_rdm_std_pert': pert_metrics['cat_rdm_std'],
                    # Local structure
                    'class_rdm_rho': class_rdm_rho,
                    'class_silhouette_base': base_metrics['class_silhouette'],
                    'class_silhouette_pert': pert_metrics['class_silhouette'],
                    'knn_purity_base': base_metrics['knn_purity_k10'],
                    'knn_purity_pert': pert_metrics['knn_purity_k10'],
                    'within_class_sim_base': base_metrics['within_class_cosine_sim'],
                    'within_class_sim_pert': pert_metrics['within_class_cosine_sim'],
                    # Overall
                    'effective_dim_base': base_metrics['effective_dim'],
                    'effective_dim_pert': pert_metrics['effective_dim'],
                    'rdm_std_base': base_metrics['rdm_std'],
                    'rdm_std_pert': pert_metrics['rdm_std'],
                    'mean_norm_base': base_metrics['mean_norm'],
                    'mean_norm_pert': pert_metrics['mean_norm'],
                }
                results.append(row)
                acc_s = f'{acc_retained:.4f}' if acc_retained else 'N/A'
                aln_s = f'{align_retained:.4f}' if align_retained else 'N/A'
                print(f'    cat_rdm_rho={cat_rdm_rho:.4f}  class_rdm_rho={class_rdm_rho:.4f}  '
                      f'knn={pert_metrics["knn_purity_k10"]:.4f}  '
                      f'acc_ret={acc_s}  align_ret={aln_s}')

    # Save results
    out_path = RESULTS_DIR / f'global_local_structure_analysis{sfx}.json'
    # Convert numpy types
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = [{k: to_serializable(v) for k, v in row.items()} for row in results]
    with out_path.open('w') as f:
        json.dump(serializable, f, indent=2)
    print(f'\nSaved {len(results)} rows to {out_path}')

    # Print summary table
    print('\n' + '=' * 120)
    print(f'{"Arch":<10} {"Pert":<16} {"Tim":<6} '
          f'{"AccRet":>8} {"AlnRet":>8} | '
          f'{"CatRDM":>8} {"ClsRDM":>8} {"kNN":>8} {"CatSil":>8} {"ClsSil":>8} {"WCSim":>8} {"EffDim":>8}')
    print('-' * 120)
    for r in sorted(results, key=lambda x: (x['arch'], x['pert'], x['timing'])):
        acc_r = f"{r['acc_retained']:.3f}" if r['acc_retained'] else 'N/A'
        aln_r = f"{r['align_retained']:.3f}" if r['align_retained'] else 'N/A'
        cat_sil_r = r['category_silhouette_pert']
        cls_sil_r = r['class_silhouette_pert']
        print(f'{r["arch"]:<10} {r["pert"]:<16} {r["timing"]:<6} '
              f'{acc_r:>8} {aln_r:>8} | '
              f'{r["category_rdm_rho"]:>8.4f} {r["class_rdm_rho"]:>8.4f} '
              f'{r["knn_purity_pert"]:>8.4f} '
              f'{cat_sil_r:>8.4f} {cls_sil_r:>8.4f} '
              f'{r["within_class_sim_pert"]:>8.4f} '
              f'{r["effective_dim_pert"]:>8.1f}')

    # Correlation analysis
    print('\n\n=== CORRELATION ANALYSIS ===')
    print('Pearson r between retained fractions and structural metrics:\n')

    valid = [r for r in results if r['acc_retained'] is not None and r['align_retained'] is not None]

    acc_ret = np.array([r['acc_retained'] for r in valid])
    aln_ret = np.array([r['align_retained'] for r in valid])

    metrics = {
        'category_rdm_rho': [r['category_rdm_rho'] for r in valid],
        'class_rdm_rho': [r['class_rdm_rho'] for r in valid],
        'knn_purity': [r['knn_purity_pert'] for r in valid],
        'cat_silhouette': [r['category_silhouette_pert'] for r in valid],
        'cls_silhouette': [r['class_silhouette_pert'] for r in valid],
        'wc_cosine_sim': [r['within_class_sim_pert'] for r in valid],
        'effective_dim': [r['effective_dim_pert'] for r in valid],
        'rdm_std': [r['rdm_std_pert'] for r in valid],
        'mean_norm': [r['mean_norm_pert'] for r in valid],
        'cat_sil_retained': [r['category_silhouette_pert'] / r['category_silhouette_base']
                             if r['category_silhouette_base'] and not np.isnan(r['category_silhouette_base'])
                             else np.nan for r in valid],
        'cls_sil_retained': [r['class_silhouette_pert'] / r['class_silhouette_base']
                             if r['class_silhouette_base'] and not np.isnan(r['class_silhouette_base'])
                             else np.nan for r in valid],
    }

    print(f'{"Metric":<22} {"r(Acc)":<12} {"p(Acc)":<12} {"r(Align)":<12} {"p(Align)":<12}')
    print('-' * 70)
    for name, vals in metrics.items():
        vals = np.array(vals, dtype=float)
        mask = np.isfinite(vals) & np.isfinite(acc_ret)
        if mask.sum() < 3:
            continue
        r_acc, p_acc = stats.pearsonr(vals[mask], acc_ret[mask])
        mask2 = np.isfinite(vals) & np.isfinite(aln_ret)
        r_aln, p_aln = stats.pearsonr(vals[mask2], aln_ret[mask2])
        star_acc = '***' if p_acc < 0.001 else '**' if p_acc < 0.01 else '*' if p_acc < 0.05 else ''
        star_aln = '***' if p_aln < 0.001 else '**' if p_aln < 0.01 else '*' if p_aln < 0.05 else ''
        print(f'{name:<22} {r_acc:>7.3f}{star_acc:<4} {p_acc:>10.4f}  {r_aln:>7.3f}{star_aln:<4} {p_aln:>10.4f}')

    # KEY TEST: Can we separate global from local?
    # Group conditions by their behavioral profile
    print('\n\n=== BEHAVIORAL GROUPING ===')
    print('Grouping by SPoSE alignment retained fraction and ImageNet accuracy retained fraction\n')

    keep_both = []
    keep_align_lose_acc = []
    lose_both = []

    for r in valid:
        acc_r = r['acc_retained']
        aln_r = r['align_retained']
        # Thresholds: "keep" = >90%, "lose" = <80%
        if acc_r >= 0.90 and aln_r >= 0.90:
            keep_both.append(r)
        elif aln_r >= 0.90 and acc_r < 0.90:
            keep_align_lose_acc.append(r)
        elif aln_r < 0.90:
            lose_both.append(r)

    # Also try continuous: no threshold
    print(f'Keep both (acc>=90%, align>=90%): {len(keep_both)} conditions')
    print(f'Keep align, lose acc (align>=90%, acc<90%): {len(keep_align_lose_acc)} conditions')
    print(f'Lose both (align<90%): {len(lose_both)} conditions')

    for group_name, group in [('Keep both', keep_both),
                               ('Keep align, lose acc', keep_align_lose_acc),
                               ('Lose both', lose_both)]:
        if not group:
            continue
        print(f'\n--- {group_name} ---')
        for r in group:
            print(f'  {r["arch"]}/{r["pert"]}/{r["timing"]}: '
                  f'acc_ret={r["acc_retained"]:.3f}, align_ret={r["align_retained"]:.3f}, '
                  f'cat_rdm_rho={r["category_rdm_rho"]:.4f}, class_rdm_rho={r["class_rdm_rho"]:.4f}, '
                  f'knn={r["knn_purity_pert"]:.4f}')
        g_cat = np.mean([r['category_rdm_rho'] for r in group])
        g_cls = np.mean([r['class_rdm_rho'] for r in group])
        g_knn = np.mean([r['knn_purity_pert'] for r in group])
        g_cat_sil = np.nanmean([r['category_silhouette_pert'] for r in group])
        g_cls_sil = np.nanmean([r['class_silhouette_pert'] for r in group])
        print(f'  MEAN: cat_rdm_rho={g_cat:.4f}, class_rdm_rho={g_cls:.4f}, '
              f'knn={g_knn:.4f}, cat_sil={g_cat_sil:.4f}, cls_sil={g_cls_sil:.4f}')

    # Try different thresholds
    print('\n\n=== SENSITIVITY TO THRESHOLDS ===')
    for acc_thresh in [0.95, 0.90, 0.85, 0.80]:
        for aln_thresh in [0.95, 0.90, 0.85, 0.80]:
            kb = [r for r in valid if r['acc_retained'] >= acc_thresh and r['align_retained'] >= aln_thresh]
            kl = [r for r in valid if r['align_retained'] >= aln_thresh and r['acc_retained'] < acc_thresh]
            lb = [r for r in valid if r['align_retained'] < aln_thresh]
            if len(kb) > 0 and len(kl) > 0 and len(lb) > 0:
                print(f'\nacc_thresh={acc_thresh:.2f}, aln_thresh={aln_thresh:.2f}: '
                      f'KB={len(kb)}, KL={len(kl)}, LB={len(lb)}')
                for gname, g in [('KB', kb), ('KL', kl), ('LB', lb)]:
                    print(f'  {gname}: cat_rdm_rho={np.mean([r["category_rdm_rho"] for r in g]):.4f}, '
                          f'class_rdm_rho={np.mean([r["class_rdm_rho"] for r in g]):.4f}, '
                          f'knn={np.mean([r["knn_purity_pert"] for r in g]):.4f}')


if __name__ == '__main__':
    main()
