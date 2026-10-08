"""Robustness of the Figure 3C-D structure-readout correlations to architecture and onset.

Recomputes the four correlations Figure 3D reports -- category fidelity F and category
separability S, each against top-1 retained and alignment retained -- from the same
<results_root>/global_local_structure_analysis{SEED_SUFFIX}.json, under:

  pooled          all 36 perturbed runs (reproduces Figure 3D)
  fixed effects   residuals after removing architecture, onset, architecture + onset, or
                  architecture x onset means (n = 36, k dummy covariates)
  leave one out   each architecture (n = 27) and each onset (n = 24) dropped in turn
  within group    each architecture (n = 9) and each onset (n = 12) alone; the within-group
                  Fisher z's are combined by an inverse-variance mean with Cochran's Q

Pearson r CIs are Fisher-z with se = 1/sqrt(n - 3 - k), as in Figure 3D; Spearman CIs use
the Bonett-Wright se. Partial r controls for the other of F and S or for 10-NN class purity.
p_perm is Freedman-Lane: the outcome's residuals under the covariates are permuted freely and
re-residualized, which is valid for the additive two-factor model where within-cell shuffling
is not. Williams's test compares r(F, top-1) with r(F, alignment) and r(S, alignment) with
r(S, top-1), zero-order and after partialling out the other of F and S.

Writes <results_root>/structure_robustness{SEED_SUFFIX}.json, the appendix table
<figures_root>/table_structure_robustness{SEED_SUFFIX}.tex, and prints markdown tables.

Usage:
    python pipeline/scripts/analysis/structure_robustness.py
    TP_SEED_SUFFIX=_seed43 python pipeline/scripts/analysis/structure_robustness.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'visualization'))

from arch_runs import FIGURES, RESULTS, SEED_SUFFIX, structure_analysis_path  # noqa: E402

OUT_JSON = RESULTS / f'structure_robustness{SEED_SUFFIX}.json'
OUT_TEX = FIGURES / f'table_structure_robustness{SEED_SUFFIX}.tex'
ARCHS = ['RN50', 'ViT-B/16', 'ConvNeXt-B', 'CLIP']
ONSETS = ['early', 'mid', 'late']
PERTS = ['image_noise', 'uniform_images', 'label_shuffle']
RELATIONS = [('F', 'acc'), ('F', 'aln'), ('S', 'acc'), ('S', 'aln')]
RELATION_LABEL = {('F', 'acc'): 'F, top-1', ('F', 'aln'): 'F, align',
                  ('S', 'acc'): 'S, top-1', ('S', 'aln'): 'S, align'}
OTHER = {'F': 'S', 'S': 'F'}
FIXED_EFFECTS = {
    'architecture': ('arch',),
    'onset': ('onset',),
    'architecture + onset': ('arch', 'onset'),
    'architecture x onset': ('cell',),
}


def load(path):
    rows = [r for r in json.loads(path.read_text()) if r['pert'] != 'baseline']
    have = sorted((r['arch'], r['timing'], r['pert']) for r in rows)
    want = sorted((a, t, p) for a in ARCHS for t in ONSETS for p in PERTS)
    if have != want:
        raise SystemExit(f'{path}: expected the 36-run grid; missing {sorted(set(want) - set(have))}, '
                         f'unexpected {sorted(set(have) - set(want))}')
    return {
        'F': np.array([r['category_rdm_rho'] for r in rows]),
        'S': np.array([r['category_silhouette_pert'] for r in rows]),
        'acc': np.array([r['acc_retained'] for r in rows]),
        'aln': np.array([r['align_retained'] for r in rows]),
        'knn': np.array([r['knn_purity_pert'] for r in rows]),
        'arch': np.array([r['arch'] for r in rows]),
        'onset': np.array([r['timing'] for r in rows]),
        'cell': np.array([f"{r['arch']}|{r['timing']}" for r in rows]),
    }


def _design(n, factors, covariates):
    cols = [np.ones(n)]
    for f in factors:
        cols += [(f == level).astype(float) for level in np.unique(f)[1:]]
    return np.column_stack(cols + list(covariates))


def _resid(v, X):
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def _perm_p(rx, ry, X, n_perm, rng):
    """Freedman-Lane: rx is orthogonal to X, so only the re-residualized norm of ry changes."""
    perm = ry[np.argsort(rng.random((n_perm, len(ry))), axis=1)]
    perm_resid = perm - (perm @ X) @ np.linalg.pinv(X)
    r_perm = (perm @ rx) / np.linalg.norm(perm_resid, axis=1)
    return float((np.sum(np.abs(r_perm) >= abs(rx @ ry) - 1e-12) + 1) / (n_perm + 1))


def corr(d, xk, yk, sel=None, factors=(), covars=(), rank=False, n_perm=0, rng=None):
    sel = np.ones(len(d['F']), bool) if sel is None else sel
    x, y, cov = d[xk][sel], d[yk][sel], [d[c][sel] for c in covars]
    if rank:
        x, y, cov = stats.rankdata(x), stats.rankdata(y), [stats.rankdata(c) for c in cov]
    X = _design(len(x), [d[f][sel] for f in factors], cov)
    n, k = len(x), X.shape[1] - 1
    rx, ry = _resid(x, X), _resid(y, X)
    rx, ry = rx / np.linalg.norm(rx), ry / np.linalg.norm(ry)
    r = float(rx @ ry)
    df = n - 2 - k
    se = np.sqrt((1 + r ** 2 / 2 if rank else 1) / (n - 3 - k))
    z = np.arctanh(r)
    out = {'r': r, 'lo': float(np.tanh(z - 1.96 * se)), 'hi': float(np.tanh(z + 1.96 * se)),
           'p': float(2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r ** 2)), df)), 'n': n, 'k': k}
    if n_perm:
        out['p_perm'] = _perm_p(rx, ry, X, n_perm, rng)
    return out


def williams(d, xk, sel=None, factors=(), covars=()):
    """Williams's T2 (Steiger 1980, eq. 7) for r(x, acc) - r(x, aln), on residuals after
    `factors` and `covars`; n is reduced by the number of covariates."""
    stat = corr(d, xk, 'acc', sel, factors, covars)
    r12 = stat['r']
    r13 = corr(d, xk, 'aln', sel, factors, covars)['r']
    r23 = corr(d, 'acc', 'aln', sel, factors, covars)['r']
    n = stat['n'] - stat['k']
    det = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    t = (r12 - r13) * np.sqrt((n - 1) * (1 + r23) /
                              (2 * (n - 1) / (n - 3) * det + ((r12 + r13) / 2) ** 2 * (1 - r23) ** 3))
    return {'diff': r12 - r13, 't': float(t), 'df': n - 3, 'p': float(2 * stats.t.sf(abs(t), n - 3))}


def meta(results):
    z = np.array([np.arctanh(r['r']) for r in results])
    w = np.array([r['n'] - 3 - r['k'] for r in results], float)
    zbar, se = w @ z / w.sum(), 1 / np.sqrt(w.sum())
    q = float(w @ (z - zbar) ** 2)
    dfq = len(z) - 1
    return {'r': float(np.tanh(zbar)), 'lo': float(np.tanh(zbar - 1.96 * se)),
            'hi': float(np.tanh(zbar + 1.96 * se)), 'p': float(2 * stats.norm.sf(abs(zbar) / se)),
            'Q': q, 'p_Q': float(stats.chi2.sf(q, dfq)), 'I2': max(0.0, (q - dfq) / q) if q else 0.0}


def relation_block(d, sel=None, factors=(), n_perm=0, rng=None):
    out = {'F-S': corr(d, 'F', 'S', sel, factors)}
    for xk, yk in RELATIONS:
        out[f'{xk}-{yk}'] = {
            'pearson': corr(d, xk, yk, sel, factors, n_perm=n_perm, rng=rng),
            'spearman': corr(d, xk, yk, sel, factors, rank=True, n_perm=n_perm, rng=rng),
            'partial_other': corr(d, xk, yk, sel, factors, covars=(OTHER[xk],), n_perm=n_perm, rng=rng),
            'partial_knn': corr(d, xk, yk, sel, factors, covars=('knn',), n_perm=n_perm, rng=rng),
        }
    for xk in ('F', 'S'):
        out[f'williams_{xk}'] = williams(d, xk, sel, factors)
        out[f'williams_{xk}_partial'] = williams(d, xk, sel, factors, covars=(OTHER[xk],))
    return out


def _metas(blocks):
    return {f'{xk}-{yk}': {m: meta([b[f'{xk}-{yk}'][m] for b in blocks])
                           for m in ('pearson', 'spearman', 'partial_other')}
            for xk, yk in RELATIONS}


def analyze(d, n_perm, seed):
    rng = np.random.default_rng(seed)
    res = {'pooled': relation_block(d, n_perm=n_perm, rng=rng)}
    res['fixed_effects'] = {name: relation_block(d, factors=f, n_perm=n_perm, rng=rng)
                            for name, f in FIXED_EFFECTS.items()}
    res['leave_one_architecture_out'] = {a: relation_block(d, sel=d['arch'] != a, n_perm=n_perm, rng=rng)
                                         for a in ARCHS}
    res['leave_one_onset_out'] = {t: relation_block(d, sel=d['onset'] != t, n_perm=n_perm, rng=rng)
                                  for t in ONSETS}
    res['within_architecture'] = {a: relation_block(d, sel=d['arch'] == a, n_perm=n_perm, rng=rng)
                                  for a in ARCHS}
    res['within_onset'] = {t: relation_block(d, sel=d['onset'] == t, n_perm=n_perm, rng=rng)
                           for t in ONSETS}
    res['within_architecture_meta'] = _metas(res['within_architecture'].values())
    res['within_onset_meta'] = _metas(res['within_onset'].values())
    res['within_architecture_leave_one_run_out'] = {
        a: {f'{xk}-{yk}': _leave_one_run_out(d[xk][d['arch'] == a], d[yk][d['arch'] == a])
            for xk, yk in RELATIONS}
        for a in ARCHS}
    return res


def _leave_one_run_out(x, y):
    """Range of Pearson r with any one run dropped, since n = 9 lets one run carry r."""
    loo = [stats.pearsonr(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))]
    return {'min': float(min(loo)), 'max': float(max(loo))}


MEASURE_LABEL = {'pearson': 'Pearson r', 'spearman': 'Spearman r',
                 'partial_other': 'partial r, other of F and S', 'partial_knn': 'partial r, 10-NN purity'}


def _cell(s, p_key):
    star = '*' if s[p_key] < 0.05 else ' '
    return f"{s['r']:+.2f} [{s['lo']:+.2f},{s['hi']:+.2f}]{star}"


def _table(title, blocks, measure, p_key='p_perm'):
    ptxt = 'permutation p' if p_key == 'p_perm' else 'p'
    print(f'\n### {title}: {MEASURE_LABEL[measure]} [95% CI] (* {ptxt} < 0.05)\n')
    print('| subset | n | r(F,S) | ' + ' | '.join(RELATION_LABEL[r] for r in RELATIONS) + ' |')
    print('|---|---|---|' + '---|' * len(RELATIONS))
    for label, block in blocks.items():
        cells = [block[f'{x}-{y}'][measure] for x, y in RELATIONS]
        print(f"| {label} | {cells[0]['n']} | {block['F-S']['r']:+.2f} | "
              + ' | '.join(_cell(c, p_key) for c in cells) + ' |')


def _meta_table(title, metas):
    print(f'\n### {title}\n')
    print('| measure | ' + ' | '.join(RELATION_LABEL[r] for r in RELATIONS) + ' |')
    print('|---|' + '---|' * len(RELATIONS))
    for m in ('pearson', 'spearman', 'partial_other'):
        cells = [metas[f'{x}-{y}'][m] for x, y in RELATIONS]
        print(f'| {MEASURE_LABEL[m]} | ' + ' | '.join(
            f"{c['r']:+.2f} [{c['lo']:+.2f},{c['hi']:+.2f}] p={c['p']:.2g}; Q={c['Q']:.1f} (p={c['p_Q']:.2f})"
            for c in cells) + ' |')


def _williams_table(title, blocks):
    print(f'\n### {title}\n')
    print('| subset | F: top-1 minus align | F given S | S: align minus top-1 | S given F |')
    print('|---|---|---|---|---|')
    for label, block in blocks.items():
        cells = []
        for key, sign in (('williams_F', 1), ('williams_F_partial', 1),
                          ('williams_S', -1), ('williams_S_partial', -1)):
            w = block[key]
            cells.append(f"{sign * w['diff']:+.2f}, t({w['df']}) = {sign * w['t']:+.2f}, p = {w['p']:.2g}")
        print(f'| {label} | ' + ' | '.join(cells) + ' |')


def report(res):
    headline = {'pooled (Figure 3D)': res['pooled'],
                **{f'{k} fixed effects': v for k, v in res['fixed_effects'].items()}}
    loo = {**{f'without {a}': v for a, v in res['leave_one_architecture_out'].items()},
           **{f'without {t} onset': v for t, v in res['leave_one_onset_out'].items()}}
    within = {**res['within_architecture'],
              **{f'{t} onset': v for t, v in res['within_onset'].items()}}
    for measure in MEASURE_LABEL:
        _table('Pooled and fixed effects (n = 36)', headline, measure)
    for measure in MEASURE_LABEL:
        _table('Leave one architecture or onset out', loo, measure)
    for measure in MEASURE_LABEL:
        _table('Within one architecture or onset', within, measure)
    print('\n### Within one architecture, Pearson r range with any one run left out (n = 8)\n')
    print('| architecture | ' + ' | '.join(RELATION_LABEL[r] for r in RELATIONS) + ' |')
    print('|---|' + '---|' * len(RELATIONS))
    for a, block in res['within_architecture_leave_one_run_out'].items():
        print(f'| {a} | ' + ' | '.join(f"{block[f'{x}-{y}']['min']:+.2f} to {block[f'{x}-{y}']['max']:+.2f}"
                                        for x, y in RELATIONS) + ' |')
    _meta_table('Within-architecture correlations combined (inverse-variance Fisher z over 4 x 9 runs)',
                res['within_architecture_meta'])
    _meta_table('Within-onset correlations combined (inverse-variance Fisher z over 3 x 12 runs)',
                res['within_onset_meta'])
    _williams_table("Williams's test: F tracks top-1 more than alignment; S tracks alignment more than top-1",
                    {**headline, **loo, **within})


TEX_ARCH = {'RN50': 'RN50', 'ViT-B/16': 'ViT-B/16', 'ConvNeXt-B': 'ConvNeXt-B', 'CLIP': 'CLIP ViT-B/32'}
TEX_COLUMNS = [('F-acc', 'pearson'), ('F-aln', 'pearson'), ('S-acc', 'pearson'), ('S-aln', 'pearson'),
               ('F-acc', 'partial_other'), ('F-aln', 'partial_other'),
               ('S-acc', 'partial_other'), ('S-aln', 'partial_other')]


def _tex_r(s, p_key):
    v = f"{abs(s['r']):.2f}"
    return ('$-$' if s['r'] < 0 else '') + v + ('$^{*}$' if s[p_key] < 0.05 else '')


def latex_table(res, n_perm):
    """Appendix table, plain ASCII LaTeX (booktabs) like the paper's other tables."""
    def row(label, block, n=None, p_key='p_perm'):
        rfs = f"{block['F-S']['r']:.2f}" if 'F-S' in block else '--'
        n = n or block['F-acc']['pearson']['n']
        return (f'{label} & {n} & {rfs} & '
                + ' & '.join(_tex_r(block[rel][m], p_key) for rel, m in TEX_COLUMNS) + r' \\')

    n_perm_tex = f'{n_perm:,}'.replace(',', '{,}')
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{\textbf{Category-structure correlations of Figure~\ref{fig:storyboard}C,~D by '
        r'architecture and onset.} Pearson $r$ across perturbed runs at the perturbed epoch; $n$, '
        r'runs. FE, fixed effects: $r$ after removing architecture, onset, or architecture $\times$ onset '
        r'means. Combined, inverse-variance mean of the within-group Fisher $z$. $r_{FS}$, $r$ between '
        r'$F$ and $S$. $F \mid S$ and $S \mid F$, partial $r$ controlling for the other measure. '
        rf'$^{{*}}$, $p < 0.05$: permutation test ({n_perm_tex} permutations; Freedman--Lane for '
        r'fixed effects), and $z$-test for combined rows.}',
        r'\label{tab:structure_robustness}',
        r'\small',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{lrrrrrrrrrr}',
        r'\toprule',
        r' & & & \multicolumn{2}{c}{$F$} & \multicolumn{2}{c}{$S$} & \multicolumn{2}{c}{$F \mid S$} '
        r'& \multicolumn{2}{c}{$S \mid F$} \\',
        r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}',
        r'Runs & $n$ & $r_{FS}$ & top-1 & align. & top-1 & align. & top-1 & align. & top-1 & align. \\',
        r'\midrule',
        row(r'All (Figure~\ref{fig:storyboard}D)', res['pooled']),
        row('Architecture FE', res['fixed_effects']['architecture']),
        row('Onset FE', res['fixed_effects']['onset']),
        row(r'Architecture $\times$ onset FE', res['fixed_effects']['architecture x onset']),
        r'\midrule',
        *[row(f'Without {TEX_ARCH[a]}', res['leave_one_architecture_out'][a]) for a in ARCHS],
        *[row(f'{TEX_ARCH[a]} only', res['within_architecture'][a]) for a in ARCHS],
        row('Architectures combined', res['within_architecture_meta'],
            n=r'$4 \times 9$', p_key='p'),
        r'\midrule',
        *[row(f'Without {t} onset', res['leave_one_onset_out'][t]) for t in ONSETS],
        *[row(f'{t.capitalize()} onset only', res['within_onset'][t]) for t in ONSETS],
        row('Onsets combined', res['within_onset_meta'], n=r'$3 \times 12$', p_key='p'),
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--structure_json', type=Path, default=structure_analysis_path())
    ap.add_argument('--output', type=Path, default=OUT_JSON)
    ap.add_argument('--table', type=Path, default=OUT_TEX)
    ap.add_argument('--n_perm', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    d = load(args.structure_json)
    res = analyze(d, args.n_perm, args.seed)
    report(res)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'source': args.structure_json.name, 'n_perm': args.n_perm,
                                       'seed': args.seed, **res}, indent=1))
    args.table.write_text(latex_table(res, args.n_perm))
    print(f'\nwrote {args.output}\nwrote {args.table}')


if __name__ == '__main__':
    main()
