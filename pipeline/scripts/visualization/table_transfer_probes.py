"""Appendix tables and supplementary CSV for the linear-probe transfer figures.

Reads the same probe records as `fig5_linear_probe_transfer.py`
(the figure's constants are imported from it), for the seed-42 runs the paper reports, and
writes:

  <figures_root>/table_transfer_probes.tex   five booktabs tables, ASCII-only, pasted into the appendix
  <figures_root>/transfer_probe_scores.csv   one row per probe (dataset x checkpoint), with the
                                   selected C, the selection settings and convergence status
  stdout                           every number the appendix prose quotes

Uncertainty is test-set sampling: a binomial standard error per dataset, propagated to the
21-dataset mean; for a perturbed - baseline gap the two variances are added, which bounds
the paired variance whenever the two models' per-image correctness is non-negatively
correlated (per-image predictions are not stored, so the exact paired value cannot be
computed).

    python pipeline/scripts/visualization/table_transfer_probes.py
"""

import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if os.environ.get('TP_SEED_SUFFIX'):
    raise SystemExit('unset TP_SEED_SUFFIX: the paper reports seed 42 only')

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig5_linear_probe_transfer as F  # noqa: E402
import paper_style as S  # noqa: E402
from arch_runs import (CLIP_PREFIX, FIGURES, baseline_run, clip_imagenet_probe,  # noqa: E402
                       require_current_probe)

OUT_TEX = FIGURES / 'table_transfer_probes.tex'
OUT_CSV = FIGURES / 'transfer_probe_scores.csv'
IMAGENET_TRAIN_IMAGES = 1_281_167

ARCHS = ['resnet50_imagenet', 'vitb16_imagenet', 'convnext_b_imagenet', CLIP_PREFIX]
ARCH_LABEL = {'resnet50_imagenet': 'RN50', 'vitb16_imagenet': 'ViT-B/16',
              'convnext_b_imagenet': 'ConvNeXt-B', CLIP_PREFIX: 'CLIP ViT-B/32'}
PERT_LABEL = {'image_noise': 'image noise', 'uniform_images': 'uniform images',
              'label_shuffle': 'label shuffle'}
TIMINGS = ['early', 'mid', 'late']

# (display name, task, split source) in table order; fine-grained first, as in the per-dataset panel.
DATASETS = {
    'vtab/pets': ('Oxford-IIIT Pets', 'pet breeds', 'VTAB'),
    'vtab/food101': ('Food-101', 'dishes', 'VTAB'),
    'vtab/cars': ('Stanford Cars', 'car models', 'VTAB'),
    'fgvc_aircraft': ('FGVC Aircraft', 'aircraft variants', 'official'),
    'vtab/flowers': ('Flowers-102', 'flower species', 'VTAB'),
    'vtab/cifar10': ('CIFAR-10', 'objects', 'VTAB'),
    'vtab/cifar100': ('CIFAR-100', 'objects', 'VTAB'),
    'stl10': ('STL-10', 'objects', 'official'),
    'vtab/caltech101': ('Caltech-101', 'objects', 'VTAB'),
    'voc2007': ('PASCAL VOC 2007', 'objects (box crops)', 'official'),
    'vtab/dtd': ('DTD', 'textures', 'VTAB'),
    'vtab/svhn': ('SVHN', 'house numbers', 'VTAB'),
    'mnist': ('MNIST', 'digits', 'official'),
    'gtsrb': ('GTSRB', 'traffic signs', 'official'),
    'country211': ('Country211', 'country of photo', 'official'),
    'renderedsst2': ('Rendered SST-2', 'sentiment of rendered text', 'official'),
    'vtab/eurosat': ('EuroSAT', 'satellite land use', 'VTAB'),
    'vtab/resisc45': ('RESISC45', 'aerial scenes', 'VTAB'),
    'vtab/clevr_count_all': ('CLEVR count', 'number of objects', 'VTAB'),
    'vtab/clevr_closest_object_distance': ('CLEVR distance', 'distance to closest object', 'VTAB'),
    'vtab/kitti_closest_vehicle_distance': ('KITTI distance', 'distance to closest vehicle', 'VTAB'),
}
assert sorted(DATASETS) == sorted(F.TRANSFER_DATASETS)
assert list(DATASETS)[:len(F.FINE_GRAINED)] == F.FINE_GRAINED
N_DS = len(DATASETS)

IMAGENET_PROBES = {'imagenet1k': '1000-class', 'imagenet1k_16superclass': '16-category'}


def probe_results(run, epoch):
    run_dir = F.probe_root(run) / run
    for name in (f'linear_probe_eval_epoch{epoch:03d}.json', f'linear_probe_eval_epoch{epoch}.json'):
        if (run_dir / name).exists():
            return F._read_json(str(run_dir / name))['results']
    raise FileNotFoundError(f'{run_dir}: no linear_probe_eval_epoch{epoch}.json')


def suite(run, epoch):
    res = probe_results(run, epoch)
    missing = [ds for ds in DATASETS if ds not in res]
    if missing:
        raise ValueError(f'{run} epoch {epoch}: transfer suite missing {missing}')
    return {ds: require_current_probe(res[ds], f'{run} epoch {epoch} {ds}') for ds in DATASETS}


def binom_se(block):
    p = block['score']
    return math.sqrt(p * (1 - p) / block['n_test'])


def mean_and_se(blocks):
    m = np.mean([b['score'] for b in blocks.values()])
    se = math.sqrt(sum(binom_se(b) ** 2 for b in blocks.values())) / len(blocks)
    return m, se


def perturbed_runs():
    """(arch, perturbation, timing, onset, run name, baseline name) for the 36 perturbed runs."""
    out = []
    for arch in ARCHS:
        for timing in TIMINGS:
            onset = S.ONSET_EPOCH[arch][timing]
            for pt in F.PERTURB_TYPES:
                out.append((arch, pt, timing, onset, f'{arch}_{pt}_{timing}', baseline_run(arch)))
    return out


def collect():
    """{(run, epoch): suite} for every checkpoint the figure compares, and the 108 pairs."""
    ckpts, pairs = {}, []
    for arch, pt, timing, onset, run, bl in perturbed_runs():
        for role, epoch in (('onset', onset), ('onset+1', onset + 1),
                            ('final', F.FINAL_EPOCHS[arch])):
            for r in (run, bl):
                if (r, epoch) not in ckpts:
                    ckpts[(r, epoch)] = suite(r, epoch)
            pairs.append(dict(arch=arch, pert=pt, timing=timing, onset=onset, run=run,
                              baseline=bl, epoch=epoch, role=role))
    for p in pairs:
        p['P'], p['se_P'] = mean_and_se(ckpts[(p['run'], p['epoch'])])
        p['B'], p['se_B'] = mean_and_se(ckpts[(p['baseline'], p['epoch'])])
        p['gap'] = p['P'] - p['B']
        p['se_gap'] = math.hypot(p['se_P'], p['se_B'])
    return ckpts, pairs


def pct(x, nd=1):
    return f'{100 * x:.{nd}f}'


def signed_num(x_pp, nd=1):
    s = f'{x_pp:.{nd}f}'
    if float(s) == 0:
        return f'{abs(float(s)):.{nd}f}'
    return f'{"+" if x_pp > 0 else ""}{s}'


def signed(x_pp, nd=1):
    return f'${signed_num(x_pp, nd)}$'


def gap_pm(x_pp, hw_pp):
    return f'${signed_num(x_pp)} \\pm {hw_pp:.1f}$'


def thousands(n):
    return f'{n:,}'


def table_datasets(ckpts):
    meta = {}
    for blocks in ckpts.values():
        for ds, b in blocks.items():
            meta.setdefault(ds, set()).add((b['n_classes'], b['n_train'], b['n_test']))
    for ds, m in meta.items():
        if len(m) != 1:
            raise ValueError(f'{ds}: split sizes differ across checkpoints: {m}')
    rows = [r'\begin{table}[H]', r'\centering', r'\small',
            r'\caption{Datasets of the transfer suite (Figure~\ref{fig:linear_probe_generalization}). '
            r"Train and test are the probe's fitting and scoring sets; the C-selection "
            r'split takes 20\% of Train (Appendix~\ref{app:transfer_probes}). VTAB: the train and test '
            r'splits of the VTAB task definitions \citep{zhai2020vtab}, whose validation split is not '
            r"used; official: the dataset's own train and test splits. FGVC Aircraft is "
            r'labeled at the variant level and probed on its train split, not train+val. PASCAL VOC 2007 '
            r'has one image per annotated object box, cropped to the box. STL-10 uses its labeled '
            r'images only. The ImageNet rows are the other probes: the 1000-class and 16-category probes of '
            r'Figure~\ref{fig:linear_probe_generalization}A share one class-stratified 100,000-image '
            r'subset of the ImageNet training set (100 per class); the CLIP ViT-B/32 ImageNet probe of '
            r'Figure~\ref{fig:transfer_retention_distance}A uses the full training set.}',
            r'\label{tab:transfer_datasets}',
            r'\begin{tabular}{lllrrr}', r'\toprule',
            r'Dataset & Task & Splits & Classes & Train & Test \\', r'\midrule',
            r'\multicolumn{6}{l}{\emph{Fine-grained}} \\']
    for i, (ds, (name, task, src)) in enumerate(DATASETS.items()):
        if i == len(F.FINE_GRAINED):
            rows.append(r'\multicolumn{6}{l}{\emph{Other}} \\')
        (k, ntr, nte), = meta[ds]
        rows.append(f'{name} & {task} & {src} & {k} & {thousands(ntr)} & {thousands(nte)} \\\\')
    rows += [r'\midrule', r'\multicolumn{6}{l}{\emph{ImageNet probes}} \\',
             r'ImageNet, 1000-class & objects & subset & 1000 & 100,000 & 50,000 \\',
             r'ImageNet, 16-category & superordinate category & subset & 16 & 100,000 & 50,000 \\',
             r'ImageNet, CLIP only & objects & full & 1000 & 1,281,167 & 50,000 \\',
             r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(rows), meta


def table_baselines(pairs):
    rows = [r'\begin{table}[H]', r'\centering', r'\small',
            r'\caption{Transfer-probe mean (\%, mean top-1 over the 21 datasets of '
            r'Table~\ref{tab:transfer_datasets}) of the unperturbed baselines at every epoch that '
            r'Figure~\ref{fig:transfer_retention_distance} compares with a perturbed run. $t$ is the '
            r'onset epoch: 5, 30, and 70 for the ImageNet classifiers and 5, 13, and 28 for CLIP '
            r'ViT-B/32; final is epoch 89 (CLIP ViT-B/32: 31).}',
            r'\label{tab:transfer_baselines}',
            r'\begin{tabular}{lrrrrrrr}', r'\toprule',
            r' & \multicolumn{2}{c}{Early} & \multicolumn{2}{c}{Mid} & \multicolumn{2}{c}{Late} & \\',
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
            r'Model & $t$ & $t+1$ & $t$ & $t+1$ & $t$ & $t+1$ & Final \\', r'\midrule']
    for arch in ARCHS:
        vals = {}
        for p in pairs:
            if p['arch'] == arch:
                key = 'final' if p['role'] == 'final' else (p['timing'], p['role'])
                vals[key] = p['B']
        cells = [pct(vals[(t, r)]) for t in TIMINGS for r in ('onset', 'onset+1')]
        cells.append(pct(vals['final']))
        rows.append(f'{ARCH_LABEL[arch]} & ' + ' & '.join(cells) + r' \\')
    rows += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(rows)


def table_runs(pairs, max_hw_gap):
    by_key = {(p['run'], p['role']): p for p in pairs}
    rows = [r'\begin{table}[H]', r'\centering', r'\small',
            r'\caption{Transfer-probe mean of every perturbed run (\%) and its gap to the baseline at '
            r'the same epoch (perturbed $-$ baseline, percentage points); the baselines are in '
            r'Table~\ref{tab:transfer_baselines}. $t$, onset epoch; final, epoch 89 (CLIP ViT-B/32: 31). '
            r'Test-set sampling alone puts at most $\pm' + f'{max_hw_gap:.2f}'
            + r'$~pp (95\%) on any gap (Appendix~\ref{app:transfer_probes}). '
            r'Figure~\ref{fig:transfer_retention_distance}A plots the ratio of the two final-epoch '
            r'means and Figure~\ref{fig:transfer_retention_distance}B the gaps.}',
            r'\label{tab:transfer_runs}',
            r'\begin{tabular}{llrrrrrrr}', r'\toprule',
            r' & & & \multicolumn{3}{c}{Perturbed} & \multicolumn{3}{c}{Gap} \\',
            r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
            r'Onset & Perturbation & $t$ & $t$ & $t+1$ & Final & $t$ & $t+1$ & Final \\']
    roles = ('onset', 'onset+1', 'final')
    for arch in ARCHS:
        rows += [r'\midrule', rf'\multicolumn{{9}}{{l}}{{\emph{{{ARCH_LABEL[arch]}}}}} \\']
        for timing in TIMINGS:
            onset = S.ONSET_EPOCH[arch][timing]
            for pt in F.PERTURB_TYPES:
                ps = [by_key[(f'{arch}_{pt}_{timing}', r)] for r in roles]
                rows.append(f'{timing} & {PERT_LABEL[pt]} & {onset} & '
                            + ' & '.join(pct(p['P']) for p in ps) + ' & '
                            + ' & '.join(signed(100 * p['gap']) for p in ps) + r' \\')
    rows += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(rows)


def table_final_per_dataset(ckpts):
    fin = F.FINAL_EPOCHS
    base = {arch: ckpts[(baseline_run(arch), fin[arch])] for arch in ARCHS}
    case_l = ckpts[(F.CASE_L['run'], fin[CLIP_PREFIX])]
    rows = [r'\begin{table}[H]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
            r'\caption{Per-dataset linear-probe top-1 (\%) at the final epoch (89; CLIP ViT-B/32: 31) '
            r'for the four baselines and for case L (CLIP ViT-B/32, image noise at epoch 5), and '
            r"L's gap to its baseline (percentage points). "
            r'$\pm$, 95\% test-set sampling interval of the gap (Appendix~\ref{app:transfer_probes}). '
            r'The gaps are Figure~\ref{fig:linear_probe_generalization}B.}',
            r'\label{tab:transfer_final_datasets}',
            r'\begin{tabular}{lrrrrrr}', r'\toprule',
            r' & \multicolumn{4}{c}{Baseline} & & \\',
            r'\cmidrule(lr){2-5}',
            r'Dataset & RN50 & ViT-B/16 & ConvNeXt-B & CLIP & L & Gap of L \\',
            r'\midrule', r'\multicolumn{7}{l}{\emph{Fine-grained}} \\']
    for i, (ds, (name, _, _)) in enumerate(DATASETS.items()):
        if i == len(F.FINE_GRAINED):
            rows.append(r'\multicolumn{7}{l}{\emph{Other}} \\')
        gap = 100 * (case_l[ds]['score'] - base[CLIP_PREFIX][ds]['score'])
        hw = 196 * math.hypot(binom_se(case_l[ds]), binom_se(base[CLIP_PREFIX][ds]))
        rows.append(f'{name} & ' + ' & '.join(pct(base[a][ds]['score']) for a in ARCHS)
                    + f' & {pct(case_l[ds]["score"])} & {gap_pm(gap, hw)} \\\\')
    means = {a: mean_and_se(base[a])[0] for a in ARCHS}
    lm, lse = mean_and_se(case_l)
    bm, bse = mean_and_se(base[CLIP_PREFIX])
    rows += [r'\midrule',
             'Mean & ' + ' & '.join(pct(means[a]) for a in ARCHS)
             + f' & {pct(lm)} & {gap_pm(100 * (lm - bm), 196 * math.hypot(lse, bse))} \\\\',
             r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(rows)


def imagenet_probe_series(case):
    """Panel A ImageNet-probe points for one case: [(label, epoch or step, readout, P, B)]."""
    run, rec, bl, onset = case['run'], case['recovery_run'], case['baseline'], case['onset']
    first = onset + 1
    out = []
    for label in sorted(F._probe_epochs(run)) + sorted(F._probe_epochs(rec)):
        is_sub = label >= 1000
        src = rec if is_sub else run
        res = probe_results(src, label)
        base_ep = first if is_sub else label
        for ds, readout in IMAGENET_PROBES.items():
            if ds not in res:
                continue
            b = probe_results(bl, base_ep)[ds]
            if b.get('max_train_samples') != res[ds].get('max_train_samples'):
                raise ValueError(f'{src} {label} {ds}: baseline fit on a different train set')
            where = (f'epoch {first}, step {label - first * 1000 + 1}' if is_sub
                     else f'epoch {label}')
            out.append((where, label, readout, res[ds], b))
    return out


def table_imagenet_probes():
    series = {c['label']: imagenet_probe_series(c) for c in (F.CASE_T, F.CASE_K)}
    order = []
    for c in ('T', 'K'):
        for where, label, *_ in series[c]:
            if where not in order:
                order.append(where)
    order.sort(key=lambda w: (int(w.split()[1].rstrip(',')), 'step' not in w,
                              int(w.split()[-1]) if 'step' in w else 0))
    cell = {(c, where, readout): (p, b) for c in series for where, _, readout, p, b in series[c]}
    rows = [r'\begin{table}[H]', r'\centering', r'\small',
            r'\caption{ImageNet linear-probe top-1 (\%) of cases T (ViT-B/16, image noise at epoch 70) '
            r'and K (ConvNeXt-B, label shuffle at epoch 70), perturbed / baseline, behind the 1000-class '
            r'and 16-category readouts of Figure~\ref{fig:linear_probe_generalization}A. Checkpoints '
            r'inside epoch 71 come from a re-run of that epoch saved after the listed number of '
            r'optimizer steps (834 per epoch) and are compared with the baseline at the end of '
            r'epoch 71; the epoch-71 row is the original run.}',
            r'\label{tab:transfer_imagenet_probes}',
            r'\begin{tabular}{lcccc}', r'\toprule',
            r' & \multicolumn{2}{c}{T} & \multicolumn{2}{c}{K} \\',
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
            r'Checkpoint & 1000-class & 16-category & 1000-class & 16-category \\', r'\midrule']
    for where in order:
        cells = []
        for c in ('T', 'K'):
            for readout in IMAGENET_PROBES.values():
                pb = cell.get((c, where, readout))
                cells.append('--' if pb is None else f'{pct(pb[0]["score"])} / {pct(pb[1]["score"])}')
        label = where.replace('epoch 70', 'end of epoch 70')
        rows.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    rows += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(rows), series


CSV_FIELDS = ['architecture', 'run', 'role', 'perturbation', 'onset_timing', 'onset_epoch',
              'epoch', 'step_in_epoch', 'dataset', 'dataset_name', 'group', 'n_classes', 'n_train',
              'n_test', 'top1', 'top5', 'top1_se', 'C', 'n_C_evaluated', 'selection_max_iter',
              'selection_val_fraction', 'selection_val_seed', 'n_val', 'converged', 'n_iter',
              'n_features', 'n_constant_dims_dropped', 'preprocessing', 'bn_recalibrated', 'solver']


def csv_row(arch, run, meta, epoch, ds, name, group, b, step=''):
    cs = b.get('c_sweep', {})
    return dict(architecture=ARCH_LABEL[arch], run=run, epoch=epoch, step_in_epoch=step,
                dataset=ds, dataset_name=name, group=group, n_classes=b.get('n_classes'),
                n_train=b['n_train'], n_test=b['n_test'], top1=b['score'],
                top5=b.get('top5_accuracy'), top1_se=binom_se(b), C=b['C'],
                n_C_evaluated=len(cs.get('C_grid', [])), selection_max_iter=cs.get('sweep_max_iter'),
                selection_val_fraction=cs.get('val_fraction'), selection_val_seed=cs.get('val_seed'),
                n_val=cs.get('n_val'), converged=b['converged'], n_iter=b['n_iter'],
                n_features=b.get('n_features'), n_constant_dims_dropped=b.get('n_constant_dims_dropped'),
                preprocessing=b['preprocessing'], bn_recalibrated=arch == 'resnet50_imagenet',
                solver=b['solver'], **meta)


def write_csv(ckpts, imagenet_series):
    rows = []
    meta_of = {}
    for arch, pt, timing, onset, run, bl in perturbed_runs():
        meta_of[run] = (arch, dict(role='perturbed', perturbation=pt, onset_timing=timing,
                                   onset_epoch=onset))
        meta_of[bl] = (arch, dict(role='baseline', perturbation='', onset_timing='', onset_epoch=''))
    for (run, epoch), blocks in sorted(ckpts.items()):
        arch, meta = meta_of[run]
        for ds, b in blocks.items():
            group = 'fine-grained' if ds in F.FINE_GRAINED else 'other'
            rows.append(csv_row(arch, run, meta, epoch, ds, DATASETS[ds][0], group, b))
    for case in (F.CASE_T, F.CASE_K):
        arch = case['arch_prefix']
        meta = dict(role='perturbed', perturbation=case['pert'], onset_timing='late',
                    onset_epoch=case['onset'])
        bmeta = dict(role='baseline', perturbation='', onset_timing='', onset_epoch='')
        seen_b = set()
        for where, label, readout, p, b in imagenet_series[case['label']]:
            is_sub = label >= 1000
            epoch = label // 1000 if is_sub else label
            step = label - epoch * 1000 + 1 if is_sub else ''
            run = case['recovery_run'] if is_sub else case['run']
            name = f'ImageNet {readout} (100k-image train subset)'
            rows.append(csv_row(arch, run, meta, epoch, p['dataset'], name, 'imagenet', p, step))
            if (epoch, p['dataset']) not in seen_b:
                seen_b.add((epoch, p['dataset']))
                rows.append(csv_row(arch, case['baseline'], bmeta, epoch, b['dataset'], name,
                                    'imagenet', b))
    for arch, pt, timing, onset, run, bl in perturbed_runs():
        if arch != CLIP_PREFIX:
            continue
        for r, meta in ((run, dict(role='perturbed', perturbation=pt, onset_timing=timing,
                                   onset_epoch=onset)),
                        (bl, dict(role='baseline', perturbation='', onset_timing='', onset_epoch=''))):
            if r == bl and any(x['run'] == bl and x['dataset'] == 'imagenet1k_full' for x in rows):
                continue
            rec = clip_imagenet_record(r, F.FINAL_EPOCHS[arch])
            rows.append(csv_row(arch, r, meta, F.FINAL_EPOCHS[arch], 'imagenet1k_full',
                                'ImageNet 1000-class (full train set)', 'imagenet', rec))
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def clip_imagenet_record(run, epoch):
    """CLIP's ImageNet probe (x-axis of the retention panel) in the transfer records' block shape."""
    d = clip_imagenet_probe(run, epoch)
    iv = d['imagenet_val']
    if abs(iv['top1_accuracy'] - F.load_clip_imagenet_probe(run, epoch)) > 0:
        raise ValueError(f'{run} epoch {epoch}: not the record the figure reads')
    # The CLIP probe records come in two layouts (fit details at top level or under
    # `probe_fit`, `n_test` or `n_samples`), and neither stores the training-set size or the
    # preprocessing tag; the probe is fit on the full ImageNet training set.
    fit = d.get('probe_fit', {})
    return dict(dataset='imagenet1k_full', n_classes=1000,
                n_train=d.get('n_train', IMAGENET_TRAIN_IMAGES),
                n_test=iv.get('n_test', iv.get('n_samples')),
                score=iv['top1_accuracy'], top5_accuracy=iv.get('top5_accuracy'), C=d['C'],
                c_sweep=d['c_sweep'], converged=d.get('converged', fit.get('converged')),
                n_iter=d.get('n_iter', fit.get('n_iter')), n_features=d['n_features'],
                n_constant_dims_dropped=d.get('n_constant_dims_dropped', d.get('n_constant_features')),
                preprocessing=d.get('preprocessing'), solver=d.get('solver', d.get('probe_solver')))


def report(ckpts, pairs):
    print('\n== Records')
    blocks = [b for s in ckpts.values() for b in s.values()]
    cap = Counter(b['c_sweep']['sweep_max_iter'] for b in blocks)
    usual_cap = cap.most_common(1)[0][0]
    cap_ckpts = sum(any(b['c_sweep']['sweep_max_iter'] != usual_cap for b in s.values())
                    for s in ckpts.values())
    print(f'{len(ckpts)} checkpoints, {len(blocks)} probes; '
          f'solver {Counter(b["solver"] for b in blocks)}; recipe {Counter(b["c_selection"] for b in blocks)}; '
          f'val seed {Counter(b["c_sweep"]["val_seed"] for b in blocks)}; '
          f'val fraction {Counter(b["c_sweep"]["val_fraction"] for b in blocks)}; '
          f'max_train_samples {Counter(b.get("max_train_samples") for b in blocks)}')
    print(f'  selection cap {dict(cap)} ({cap_ckpts} checkpoints hold a record with another cap); '
          f'C candidates {dict(Counter(len(b["c_sweep"]["C_grid"]) for b in blocks))}')
    nc = [(r, e, ds, b['n_iter'], b['C']) for (r, e), s in ckpts.items() for ds, b in s.items()
          if not b['converged']]
    print(f'  not converged: {len(nc)} {nc}')
    iters = [b['n_iter'] for b in blocks]
    print(f'  final-fit iterations: median {np.median(iters):.0f}, max {max(iters)}')
    const = [b.get('n_constant_dims_dropped', 0) or 0 for b in blocks]
    print(f'  constant dims dropped in {sum(c > 0 for c in const)} probes (max {max(const)})')
    print(f'  selected C: {Counter(b["C"] for b in blocks).most_common(4)}; '
          f'at 1e-6: {sum(b["C"] == 1e-6 for b in blocks)}, at 1e6: {sum(b["C"] == 1e6 for b in blocks)}')
    diffC = mismatch = total = 0
    ratios = []
    for p in pairs:
        sp, sb = ckpts[(p['run'], p['epoch'])], ckpts[(p['baseline'], p['epoch'])]
        for ds in DATASETS:
            total += 1
            diffC += sp[ds]['C'] != sb[ds]['C']
            ratios.append(abs(math.log10(sp[ds]['C'] / sb[ds]['C'])))
            mismatch += sp[ds]['c_sweep']['sweep_max_iter'] != sb[ds]['c_sweep']['sweep_max_iter']
    print(f'  pairs: {total}; C differs in {diffC} ({100 * diffC / total:.0f}%), '
          f'max |log10 ratio| {max(ratios):.1f}; selection caps differ in {mismatch}')
    for r in ('onset', 'onset+1', 'final'):
        ps = [p for p in pairs if p['role'] == r]
        d = sum(ckpts[(p['run'], p['epoch'])][ds]['C'] != ckpts[(p['baseline'], p['epoch'])][ds]['C']
                for p in ps for ds in DATASETS)
        print(f'    {r}: C differs in {d} of {len(ps) * N_DS}')

    print('\n== Test-set sampling')
    ses = [binom_se(b) for b in blocks]
    print(f'per-dataset 95% half-width: {196 * min(ses):.2f}-{196 * max(ses):.2f} pp')
    per_ds = {ds: max(196 * binom_se(s[ds]) for s in ckpts.values()) for ds in DATASETS}
    print('  largest per dataset:', {DATASETS[d][0]: round(v, 2) for d, v in
                                     sorted(per_ds.items(), key=lambda kv: -kv[1])[:5]})
    hw_mean = 196 * max(max(p['se_P'], p['se_B']) for p in pairs)
    hw_gap = 196 * max(p['se_gap'] for p in pairs)
    print(f'transfer mean 95% half-width <= {hw_mean:.2f} pp; gap 95% half-width <= {hw_gap:.2f} pp')
    small = [p for p in pairs if abs(p['gap']) * 100 <= 1.96 * 100 * p['se_gap']]
    print(f'gaps within their own 95% sampling interval: {len(small)} of {len(pairs)} '
          f'({Counter(p["role"] for p in small)})')

    print('\n== Claims')
    sup_final = [p for p in pairs if p['role'] == 'final' and p['arch'] != CLIP_PREFIX]
    lo = min(sup_final, key=lambda p: p['P'] / p['B'])
    print(f'supervised final retained min {lo["P"] / lo["B"]:.4f} ({lo["run"]}); '
          f'max |gap| {max(abs(p["gap"]) for p in sup_final) * 100:.2f} pp')
    clip_final = sorted((p['P'] / p['B'], p['run']) for p in pairs
                        if p['role'] == 'final' and p['arch'] == CLIP_PREFIX)
    print(f'  CLIP final retained: {[(round(r, 3), n) for r, n in clip_final]}')
    for c in (F.CASE_T, F.CASE_K):
        vals = {p['role']: round(p['P'] / p['B'], 3) for p in pairs if p['run'] == c['run']}
        print(f'  case {c["label"]} retained: {vals}')
    case_l = ckpts[(F.CASE_L['run'], F.FINAL_EPOCHS[CLIP_PREFIX])]
    bl = ckpts[(baseline_run(CLIP_PREFIX), F.FINAL_EPOCHS[CLIP_PREFIX])]
    gaps = {ds: 100 * (case_l[ds]['score'] - bl[ds]['score']) for ds in DATASETS}
    fg = np.mean([gaps[ds] for ds in F.FINE_GRAINED])
    ot = np.mean([gaps[ds] for ds in DATASETS if ds not in F.FINE_GRAINED])
    print(f'  case L final: fine-grained {fg:.2f} pp vs other {ot:.2f} pp; '
          f'fine-grained all below every other? {max(gaps[d] for d in F.FINE_GRAINED) < min(gaps[d] for d in DATASETS if d not in F.FINE_GRAINED)}')


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    ckpts, pairs = collect()
    t_ds, _ = table_datasets(ckpts)
    max_hw_gap = 196 * max(p['se_gap'] for p in pairs)
    t_in, series = table_imagenet_probes()
    tables = [t_ds, table_baselines(pairs), table_runs(pairs, max_hw_gap),
              table_final_per_dataset(ckpts), t_in]
    OUT_TEX.write_text('% Generated by pipeline/scripts/visualization/table_transfer_probes.py -- edit the script, not this file.\n\n'
                       + '\n\n'.join(tables) + '\n')
    n = write_csv(ckpts, series)
    print(f'wrote {OUT_TEX.name} and {OUT_CSV.name} ({n} rows)')
    report(ckpts, pairs)


if __name__ == '__main__':
    main()
