"""Print the derived statistics the paper's Results section quotes and no figure script prints.

Everything here is recomputed from existing outputs, no model is run:
  <results_root>/global_local_structure_analysis.json   (Fig 3 C/D structure metrics)
  <results_root>/distance_levels_analysis.json          (analyze_distance_levels.py)
  corrected-preprocessing THINGS-fMRI FR-RSA files      (via figS_neural_alignment_by_roi's loader)
  linear-probe summaries                                (via the Fig 5 loaders)
  CLIP-HBA-Behavior embeddings                          (via figS_spose_dimensions' loaders)

Correlations across runs are over the 36 perturbed runs at their perturbed epoch.
Usage: python pipeline/scripts/analysis/results_section_numbers.py [--skip_cliphba]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'visualization'))

from arch_runs import RESULTS, SEED_SUFFIX, structure_analysis_path  # noqa: E402

STRUCTURE_JSON = structure_analysis_path()
DISTANCE_JSON = RESULTS / f'distance_levels_analysis{SEED_SUFFIX}.json'

FINE_GRAINED = ['vtab/pets', 'vtab/food101', 'vtab/cars', 'fgvc_aircraft', 'vtab/flowers']
COARSE = ['vtab/cifar10', 'stl10', 'mnist']
N_CLASSES = {'vtab/pets': 37, 'vtab/food101': 101, 'vtab/cars': 196, 'fgvc_aircraft': 100,
             'vtab/flowers': 102, 'vtab/cifar10': 10, 'stl10': 10, 'mnist': 10}
OBJECT_SELECTIVE = ['LOC', 'lLOC', 'rLOC', 'lFFA', 'rFFA', 'lEBA', 'rEBA', 'lOFA', 'rOFA']
V1 = 'va-1'
COLLAPSE_WITHIN_CLASS_SIM = 0.95


def _partial(x, y, z):
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)[0]


def _corr(label, x, y):
    r, p = stats.pearsonr(x, y)
    rs, ps = stats.spearmanr(x, y)
    print(f'  {label}: r = {r:.2f} (p = {p:.3g}), r_s = {rs:.2f} (p = {ps:.3g})')


def _key(arch, timing, pert):
    return arch, timing, pert


def structure_rows():
    return json.loads(STRUCTURE_JSON.read_text())


def _williams(x, y1, y2, rank):
    """Williams's T2 for r(x, y1) vs r(x, y2), as given in Steiger (1980), eq. 7; df = n - 3."""
    if rank:
        x, y1, y2 = (stats.rankdata(v) for v in (x, y1, y2))
    n = len(x)
    r12, r13, r23 = np.corrcoef(x, y1)[0, 1], np.corrcoef(x, y2)[0, 1], np.corrcoef(y1, y2)[0, 1]
    det = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    t = (r12 - r13) * np.sqrt((n - 1) * (1 + r23) /
                              (2 * (n - 1) / (n - 3) * det + ((r12 + r13) / 2) ** 2 * (1 - r23) ** 3))
    return t, 2 * stats.t.sf(abs(t), n - 3)


def _compare(label, x, acc, aln):
    for rank in (True, False):
        t, p = _williams(x, acc, aln, rank)
        print(f"  {label}, top-1 vs alignment correlation ({'ranks' if rank else 'Pearson'}): "
              f"Williams t({len(x) - 3}) = {t:.2f}, p = {p:.2g}")


def accuracy_vs_alignment(rows):
    print('Top-1 retention vs behavioral-alignment retention at the onset')
    acc = np.array([r['acc_retained'] for r in rows])
    aln = np.array([r['align_retained'] for r in rows])
    _corr('all 36 runs', acc, aln)
    groups = {
        'without the runs keeping >= 0.5 of both': ~((acc >= 0.5) & (aln >= 0.5)),
        'without the runs whose alignment rose': aln <= 1.0,
        'without the runs keeping >= 0.5 of top-1': acc < 0.5,
        'only the runs keeping < 0.5 of both': (acc < 0.5) & (aln < 0.5),
    }
    for label, keep in groups.items():
        _corr(f'{label} (n = {keep.sum()})', acc[keep], aln[keep])
    loo = [stats.pearsonr(np.delete(acc, i), np.delete(aln, i))[0] for i in range(len(acc))]
    print(f'  leave-one-out Pearson r: {min(loo):.2f} to {max(loo):.2f}')


def local_structure(rows):
    print('10-NN class purity retained (perturbed / baseline)')
    acc = np.array([r['acc_retained'] for r in rows])
    aln = np.array([r['align_retained'] for r in rows])
    knn = np.array([r['knn_purity_pert'] / r['knn_purity_base'] for r in rows])
    _corr('vs top-1 retention', knn, acc)
    _corr('vs alignment retention', knn, aln)
    _compare('10-NN retained', knn, acc, aln)


def collapsed_runs(rows):
    col = [r for r in rows if np.isfinite(r['within_class_sim_pert'])
           and r['within_class_sim_pert'] >= COLLAPSE_WITHIN_CLASS_SIM]
    print(f'Collapsed runs (within-class cosine similarity >= {COLLAPSE_WITHIN_CLASS_SIM}): {len(col)}')
    base = [r['within_class_sim_base'] for r in rows]
    print(f'  baseline within-class similarity range {min(base):.2f}-{max(base):.2f}')
    knn = np.array([r['knn_purity_pert'] / r['knn_purity_base'] for r in col])
    aln = np.array([r['align_retained'] for r in col])
    sep = np.array([r['category_silhouette_pert'] for r in col])
    sup = [r for r in col if r['arch'] != 'CLIP']
    print(f'  max 10-NN purity retained {knn.max():.2f}; alignment retained {aln.min():.2f}-{aln.max():.2f}')
    print(f'  supervised collapsed runs: {len(sup)}, max top-1 retained '
          f'{max(r["acc_retained"] for r in sup):.3f}')
    _corr('S vs alignment retention', sep, aln)
    _corr('10-NN purity retained vs alignment retention', knn, aln)
    for r in col:
        if r['arch'] == 'ConvNeXt-B':
            print(f"  ConvNeXt-B {r['timing']:5s} {r['pert']:15s} dS = "
                  f"{r['category_silhouette_pert'] - r['category_silhouette_base']:+.3f} "
                  f"S = {r['category_silhouette_pert']:+.3f} align {r['align_retained']:.2f}")


def rdm_spread(rows):
    print('Class-centroid RDM standard deviation, perturbed / baseline > 1.02')
    for r in rows:
        ratio = r['rdm_std_pert'] / r['rdm_std_base']
        if ratio > 1.02:
            print(f"  {r['arch']:10s} {r['timing']:5s} {r['pert']:15s} x{ratio:.2f} "
                  f"(category-centroid RDM std {r['cat_rdm_std_base']:.3f} -> {r['cat_rdm_std_pert']:.3f}; "
                  f"dS {r['category_silhouette_pert'] - r['category_silhouette_base']:+.3f}; "
                  f"align {r['align_retained']:.2f})")
    print('  baseline class-centroid RDM std by onset:')
    seen = {}
    for r in rows:
        seen[(r['arch'], r['onset_epoch'])] = r['rdm_std_base']
    for (arch, ep), v in sorted(seen.items()):
        print(f'    {arch:10s} epoch {ep:2d}: {v:.3f}')


def _gaps(r):
    bk = r['centered_between_category']
    return ((r['centered_between_class'] - r['centered_within_class']) / bk,
            (bk - r['centered_between_class']) / bk)


def distance_levels(rows):
    dl = json.loads(DISTANCE_JSON.read_text())
    base = {(r['arch'], r['timing']): r for r in dl if r['pert'] == 'baseline'}
    struct = {_key(r['arch'], r['timing'], r['pert']): r for r in rows}
    recs = []
    for r in dl:
        if r['pert'] == 'baseline':
            continue
        b = base[(r['arch'], r['timing'])]
        s = struct[_key(r['arch'], r['timing'], r['pert'])]
        cg, kg = _gaps(r)
        cgb, kgb = _gaps(b)
        recs.append(dict(name=f"{r['arch']} {r['timing']} {r['pert']}", cg=cg, kg=kg, cgb=cgb, kgb=kgb,
                         cr=cg / cgb, kr=kg / kgb, acc=s['acc_retained'], aln=s['align_retained'],
                         mean_over_resid=r['mean_vector_norm'] / r['residual_norm'],
                         base_mean_over_resid=b['mean_vector_norm'] / b['residual_norm']))
    get = lambda k: np.array([x[k] for x in recs])
    print('Distance levels: class gap (d_c - d_w)/d_k, category gap (d_k - d_c)/d_k, ratio to baseline')
    _corr('category-gap ratio vs top-1 retention', get('kr'), get('acc'))
    _corr('category-gap ratio vs alignment retention', get('kr'), get('aln'))
    _corr('class-gap ratio vs top-1 retention', get('cr'), get('acc'))
    _corr('class-gap ratio vs alignment retention', get('cr'), get('aln'))
    _compare('category-gap ratio', get('kr'), get('acc'), get('aln'))
    _compare('class-gap ratio', get('cr'), get('acc'), get('aln'))
    print(f"  partial, category gap | class gap: top-1 {_partial(get('kr'), get('acc'), get('cr')):.2f}, "
          f"alignment {_partial(get('kr'), get('aln'), get('cr')):.2f}")
    print(f"  partial, class gap | category gap: top-1 {_partial(get('cr'), get('acc'), get('kr')):.2f}, "
          f"alignment {_partial(get('cr'), get('aln'), get('kr')):.2f}")
    print('  runs sorted by alignment retention:')
    for x in sorted(recs, key=lambda x: x['aln']):
        print(f"    {x['name']:36s} class gap {x['cg']:.3f} (base {x['cgb']:.3f}, x{x['cr']:.2f})  "
              f"category gap {x['kg']:.3f} (base {x['kgb']:.3f}, x{x['kr']:.2f})  "
              f"mean/residual {x['mean_over_resid']:.1f} (base {x['base_mean_over_resid']:.2f})  "
              f"top-1 {x['acc']:.3f} align {x['aln']:.2f}")
    print('  baseline gaps by onset:')
    for (arch, timing), b in base.items():
        cg, kg = _gaps(b)
        print(f'    {arch:10s} {timing:5s} (epoch {b["onset_epoch"]:2d}): class gap {cg:.3f}, category gap {kg:.3f}')


def onset_representation_change(rows):
    """Rank all 36 runs on each immediate representational measure."""
    import fig5_linear_probe_transfer as F6
    prefix = {'RN50': 'resnet50_imagenet', 'ViT-B/16': 'vitb16_imagenet',
              'ConvNeXt-B': 'convnext_b_imagenet', 'CLIP': 'clip_vitb32'}
    dist = {}
    for run_name, arch_prefix, pt, timing, onset in F6.discover_perturbed_runs():
        dist[(arch_prefix, pt, timing)] = 1 - F6.load_m2m_rsa_at_epoch(run_name, arch_prefix, onset)
    dl = {_key(r['arch'], r['timing'], r['pert']): r
          for r in json.loads(DISTANCE_JSON.read_text())}
    recs = []
    for r in rows:
        d = dl[_key(r['arch'], r['timing'], r['pert'])]
        b = dl[_key(r['arch'], r['timing'], 'baseline')]
        recs.append({'name': f"{r['arch']} {r['timing']} {r['pert']}",
                     'distance': dist[(prefix[r['arch']], r['pert'], r['timing'])],
                     'eff_dim': d['effective_dim'], 'eff_dim_base': b['effective_dim'],
                     'eff_dim_frac': d['effective_dim'] / b['effective_dim'],
                     'pc1': d['pc1_variance_share'],
                     'F': r['category_rdm_rho'], 'S': r['category_silhouette_pert'],
                     'knn_frac': r['knn_purity_pert'] / r['knn_purity_base']})
    print('Immediate representational change, all runs ranked (most extreme first)')
    for key, reverse in [('distance', True), ('eff_dim_frac', False), ('eff_dim', False),
                         ('pc1', True), ('F', False), ('S', False), ('knn_frac', False)]:
        ranked = sorted(recs, key=lambda x: x[key], reverse=reverse)
        print(f'  {key}: ' + '; '.join(
            f"{x['name']} {x[key]:.3f}" + (f" (ED {x['eff_dim_base']:.1f} -> {x['eff_dim']:.2f})"
                                          if key.startswith('eff') else '') for x in ranked[:4]))
        clip = next(i for i, x in enumerate(ranked) if x['name'] == 'CLIP early uniform_images')
        print(f'    CLIP early uniform_images: rank {clip + 1} of {len(ranked)}, {ranked[clip][key]:.3f}')


def transfer_vs_distance():
    """Fig 5C's Spearman correlation, pooled and split by checkpoint epoch."""
    import fig5_linear_probe_transfer as F6
    pts, _ = F6._distance_gap_points()
    print('Transfer-probe gap vs representational distance (Spearman)')
    splits = {
        'pooled, all checkpoints': pts,
        'pooled, without CLIP': [q for q in pts if not q['arch'].startswith('clip')],
        'onset': [q for q in pts if q['epoch'] == q['onset']],
        'onset + 1': [q for q in pts if q['epoch'] == q['onset'] + 1],
        'final': [q for q in pts if q['epoch'] > q['onset'] + 1],
    }
    for label, sel in splits.items():
        rs, p = stats.spearmanr([q['dist'] for q in sel], [q['gap'] for q in sel])
        print(f'  {label}: n = {len(sel)}, r_s = {rs:.2f}, p = {p:.2g}')


def roi_changes():
    import figS_neural_alignment_by_roi as C
    deltas = C.load_deltas('things')
    print('THINGS-fMRI per-ROI change at the onset (perturbed - baseline, in-sample LR-RSA)')
    n_v1, n_midlate = 0, 0
    for (arch, pert), series in deltas.items():
        for i, timing in enumerate(['early', 'mid', 'late']):
            obj = np.mean([series[roi][i] for roi in OBJECT_SELECTIVE])
            v1 = series[V1][i]
            if i > 0:
                n_midlate += 1
                n_v1 += v1 > 0.03
            top = sorted(((series[roi][i], roi) for roi in series), reverse=True)[:3]
            print(f'  {arch:22s} {pert:15s} {timing:5s} object-selective mean {obj:+.3f}  V1 {v1:+.3f}  '
                  f'LOC {series["LOC"][i]:+.3f}  top: ' + ', '.join(f'{r} {v:+.3f}' for v, r in top))
    print(f'  V1 > +0.03 in {n_v1} of {n_midlate} mid- and late-onset runs')


def probes():
    import make_fig2_test_performance_defs_in_caption as F2
    import fig5_linear_probe_transfer as F6

    def retained(p, b, ds):
        chance = 1 / N_CLASSES[ds]
        return (p[ds]['score'] - chance) / (b[ds]['score'] - chance)

    print('Linear probes, chance-corrected retained fraction: fine-grained mean vs coarse mean')
    rows = []
    for arch_key, _, _ in F2.ARCHES:
        base = F2._baseline_name(arch_key)
        final = 31 if arch_key.startswith('clip') else 89
        for pert in ['image_noise', 'uniform_images', 'label_shuffle']:
            for timing in F2.TIMINGS:
                run = F2._run_name(arch_key, pert, timing)
                for tag, ep in [('onset', F2.ONSET_EPOCH[arch_key][timing]), ('final', final)]:
                    p = F6.load_probe_scores_all_datasets(run, ep)
                    b = F6.load_probe_scores_all_datasets(base, ep)
                    fine = np.mean([retained(p, b, ds) for ds in FINE_GRAINED])
                    coarse = np.mean([retained(p, b, ds) for ds in COARSE])
                    per = {ds: retained(p, b, ds) for ds in FINE_GRAINED + COARSE}
                    rows.append((run, tag, ep, fine, coarse, per))
    onset = [r for r in rows if r[1] == 'onset']
    lossy = [r for r in onset if min(r[3], r[4]) < 0.9]
    print(f'  onset: {len(lossy)} runs with either group < 0.9; fine-grained lower in '
          f'{sum(r[3] < r[4] for r in lossy)}')
    for run, tag, ep, fine, coarse, per in rows:
        if tag == 'final' and not run.startswith('clip'):
            continue
        print(f'  {run:42s} {tag:5s} epoch {ep:2d}: fine {fine:.2f} coarse {coarse:.2f}  ' +
              ' '.join(f'{ds.split("/")[-1]} {v:.2f}' for ds, v in per.items()))
    # Seed 43 was not probed on ImageNet-1k / its 16 categories.
    for case in () if SEED_SUFFIX else (F6.CASE_T, F6.CASE_K):
        onset_ep = case['onset']
        for ds in ('imagenet1k', 'imagenet1k_16superclass'):
            print(f"  {case['label']} {ds}: perturbed {F6.load_probe_score(case['run'], ds, onset_ep)}, "
                  f"baseline {F6.load_probe_score(case['baseline'], ds, onset_ep)}")
    return onset


def probes_vs_structure(onset, rows):
    names = {}
    for r in rows:
        names[_key(r['arch'], r['timing'], r['pert'])] = r
    import make_fig2_test_performance_defs_in_caption as F2
    arch_label = {a: lab for a, lab, _ in F2.ARCHES}
    short = {'RN50': 'RN50', 'ViT-B/16': 'ViT-B/16', 'ConvNeXt-B': 'ConvNeXt-B', 'CLIP ViT-B/32': 'CLIP'}
    fine, acc, knn = [], [], []
    for arch_key, _, _ in F2.ARCHES:
        for pert in ['image_noise', 'uniform_images', 'label_shuffle']:
            for timing in F2.TIMINGS:
                run = F2._run_name(arch_key, pert, timing)
                rec = next(o for o in onset if o[0] == run)
                s = names[_key(short[arch_label[arch_key]], timing, pert)]
                fine.append(rec[3])
                acc.append(s['acc_retained'])
                knn.append(s['knn_purity_pert'] / s['knn_purity_base'])
    _corr('fine-grained retained vs top-1 retention', np.array(fine), np.array(acc))
    _corr('fine-grained retained vs 10-NN purity retained', np.array(fine), np.array(knn))


def cliphba():
    import figS_spose_dimensions as F4
    print('CLIP-HBA-Behavior: behavioral-alignment change and mean fractional variance change (66 dims)')
    beh = F4._compute_full_behavioral_delta_rho()
    baselines, perts = F4._load_embeddings()
    for pt, df in beh.items():
        g = df.groupby('epoch')['delta_rho'].mean()
        dv = F4._compute_delta_variance(baselines, perts, pt).groupby('epoch')['delta_var'].mean()
        print(f'  {pt:15s} delta rho: max {g.max():+.3f} at onset {g.idxmax()}, min {g.min():+.3f} at '
              f'onset {g.idxmin()}, mean over onsets >= 14 {g[g.index >= 14].mean():+.3f}; '
              f'dVar mean over onsets >= 14 {dv[dv.index >= 14].mean():+.2f}; onsets 2-5 dVar '
              + ' '.join(f'{v:+.2f}' for e, v in dv.items() if 2 <= e <= 5)
              + '; onsets 2-5 delta rho ' + ' '.join(f'{v:+.3f}' for e, v in g.items() if 2 <= e <= 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--skip_cliphba', action='store_true',
                    help='skip the CLIP-HBA-Behavior numbers (they reload every embedding)')
    args = ap.parse_args()
    rows = structure_rows()
    accuracy_vs_alignment(rows)
    local_structure(rows)
    collapsed_runs(rows)
    rdm_spread(rows)
    distance_levels(rows)
    onset_representation_change(rows)
    roi_changes()
    onset = probes()
    probes_vs_structure(onset, rows)
    transfer_vs_distance()
    if not args.skip_cliphba:
        cliphba()


if __name__ == '__main__':
    main()
