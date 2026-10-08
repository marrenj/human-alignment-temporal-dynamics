"""Loader for the per-epoch 1854-image behavioral alignment of the from-scratch runs.

Data: <results_root>/posthoc_rdm1854/<run>/test_things_behavioral_full1854_rsa_summary{suffix}.json
(per-epoch RSA + CKA vs the 66-d human SPoSE embeddings over all 1854 THINGS images),
written by pipeline/scripts/evaluation/run_behavioral_rsa.py.
"""

import json

import numpy as np

from arch_runs import RESULTS

RDM1854 = RESULTS / 'posthoc_rdm1854'


def load_traj(run, suffix=''):
    """Return (epochs, rsa, cka) arrays for a run, or None.

    `suffix` selects a tagged variant of the aggregate JSON (e.g. the
    corrected-preprocessing `_in1k_bicubic`; see arch_runs.rdm1854_corrected_suffix).
    Empty reads the untagged THINGS-preprocessing file.
    """
    p = RDM1854 / run / f'test_things_behavioral_full1854_rsa_summary{suffix}.json'
    if not p.is_file():
        return None
    with p.open() as f:
        d = json.load(f)
    rows = []
    for r in d.get('results', []):
        try:
            ep = int(r['epoch'])
        except (KeyError, TypeError, ValueError):
            continue
        rsa = r.get('rsa_results', {}).get('behavioral_rdm', {}).get('score')
        cka = r.get('cka_results', {}).get('behavioral_cka', {}).get('score')
        if rsa is None or cka is None:
            continue
        rows.append((ep, float(rsa), float(cka)))
    if not rows:
        return None
    a = np.array(sorted(rows))
    return a[:, 0], a[:, 1], a[:, 2]
