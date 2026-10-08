"""Layer-reweighted RSA (a.k.a. FR-RSA) engine for the non-CLIP-HBA architectures.

The idea: instead of comparing a single model RDM against a brain-ROI RDM,
extract a feature vector from *every* layer, build one RDM per layer, and fit a
non-negative ridge that reweights the layers so their weighted combination best
predicts the ROI RDM. Early layers can dominate the fit for early visual ROIs
(V1) while late layers dominate for high-level ROIs (IT/FFA) — the weights are
the object of interest.

Two things differ from the CLIP-HBA neural path (``inference_core.run_inference``):

1. **Features come from intermediate layers**, not just the final embedding, via
   forward hooks (:class:`FeatureExtractor` + :func:`select_layers`).

2. **Scoring has an opt-in depth-fair mode.** The headline by default
   (``scoring="insample"``) is an *in-sample* correlation: one inner
   leave-conditions-out CV picks the ridge alpha, then weights are refit and
   scored on all condition-pairs. Its upward bias grows with the number of
   layers, so raw comparison across architectures of different depth is
   confounded. ``scoring="both"`` additionally runs an OUTER
   leave-conditions-out loop for an unbiased, depth-fair ``cv_score`` — ~5-6x
   the compute, hence opt-in, and ``cv_score`` is ``None`` without it.

Feature-vector conventions :
  - conv / 4D outputs ``[N, C, H, W]``   -> flatten to ``[N, C*H*W]`` (spatial preserved)
  - transformer / 3D  ``[N, tokens, D]`` -> CLS token ``[N, D]`` (index 0)
  - pooled / 2D       ``[N, D]``         -> unchanged

The fitting functions (:func:`fit_frrsa`, :func:`leave_conditions_out_splits`,
:func:`nnls_ridge`) take plain numpy arrays and are model-free, so they unit-test
without a GPU or a checkpoint.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import nnls
from scipy.stats import rankdata, spearmanr

# scipy emits RuntimeWarnings for constant/degenerate correlation inputs; we
# handle those explicitly in safe_spearman, so silence the noise.
warnings.filterwarnings("ignore", message=".*constant.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*invalid value.*", category=RuntimeWarning)


# Feature extraction

def _pool_layer_output(out: torch.Tensor) -> torch.Tensor:
    """Collapse a raw layer output to a per-sample feature vector ``[N, D]``.

    Follows the standard ``extract_features`` conventions:
      - 4D ``[N, C, H, W]`` conv map  -> flatten C*H*W (keeps spatial layout)
      - 3D ``[N, tokens, D]`` tokens  -> CLS token (index 0 along the token axis)
      - 2D ``[N, D]``                 -> unchanged
      - anything else                 -> flatten trailing dims
    """
    if isinstance(out, (tuple, list)):
        # Some transformer blocks return (hidden_state, ...) — take the first tensor.
        out = out[0]
    if out.dim() == 3:
        out = out[:, 0, :]  # CLS token — assumes batch-first, class token at index 0
    return out.reshape(out.size(0), -1)


class FeatureExtractor:
    """Register forward hooks on named modules and capture pooled per-layer features.

    ``layers`` is a list of ``(label, module)`` pairs (see :func:`select_layers`).
    On each forward pass the hook stores the *pooled* output (:func:`_pool_layer_output`)
    on CPU as float32, so the caller never holds full 4D activations longer than a
    single batch.
    """

    def __init__(self, layers: Sequence[Tuple[str, nn.Module]]):
        self.labels = [label for label, _ in layers]
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        for label, module in layers:
            self._hooks.append(module.register_forward_hook(self._make_hook(label)))

    def _make_hook(self, label: str):
        def hook(_module, _inp, output):
            self.features[label] = _pool_layer_output(output).detach().to("cpu", torch.float32)
        return hook

    def pop(self) -> Dict[str, torch.Tensor]:
        """Return the features captured on the last forward pass and clear the buffer."""
        feats = self.features
        self.features = {}
        return feats

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


def select_layers(model: nn.Module, architecture: Optional[str] = None) -> List[Tuple[str, nn.Module]]:
    """Pick the ``(label, module)`` extraction points for a model, per architecture family.

    Extraction is at **residual-block granularity** (matching the standard
    auto_detect and this module's per-transformer-block handling), by *structural*
    inspection of stable attribute names, with ``architecture`` as a hint. Covers:
      - timm ResNet   (stem + each block ``layerI.J`` + ``global_pool``)   ~16 blocks (RN50)
      - timm ConvNeXt (stem + each block ``stagesI.blocks.J`` + head pool) ~36 blocks (ConvNeXt-B)
      - timm ViT      (``blocks.0``..``blocks.N`` + ``norm``)              12 blocks (ViT-B/16)
      - CLIP visual   (``visual.transformer.0``.. + ``visual.ln_post``) for
                       clip_contrastive

    Raises ValueError if the structure is unrecognised, listing the top-level
    modules so the caller can add a branch.
    """
    m = model.module if hasattr(model, "module") else model  # unwrap DataParallel
    layers: List[Tuple[str, nn.Module]] = []

    # --- timm ResNet: stem + each residual (bottleneck) block ----------------------
    if hasattr(m, "layer1") and hasattr(m, "layer4"):
        if hasattr(m, "act1"):
            layers.append(("stem", m.act1))              # [N, 64, 112, 112]
        for i in range(1, 5):
            stage = getattr(m, f"layer{i}", None)
            if stage is None:
                continue
            blocks = list(stage.children())               # residual (bottleneck) blocks
            if blocks:
                for j, blk in enumerate(blocks):
                    layers.append((f"layer{i}.{j}", blk))  # [N, C, H, W] post-residual
            else:
                layers.append((f"layer{i}", stage))
        if hasattr(m, "global_pool"):
            layers.append(("global_pool", m.global_pool))  # [N, C] pre-logit
        return layers

    # --- timm ConvNeXt: stem + each block within each stage ------------------------
    if hasattr(m, "stages") and hasattr(m, "stem") and not hasattr(m, "blocks"):
        layers.append(("stem", m.stem))
        for i, stage in enumerate(m.stages):
            blocks = getattr(stage, "blocks", None)       # ConvNeXtStage.blocks (Sequential)
            if blocks is not None and len(list(blocks)) > 0:
                for j, blk in enumerate(blocks):
                    layers.append((f"stages.{i}.blocks.{j}", blk))  # [N, C, H, W]
            else:
                layers.append((f"stages.{i}", stage))
        if hasattr(m, "head") and hasattr(m.head, "global_pool"):
            layers.append(("head.global_pool", m.head.global_pool))
        return layers

    # --- timm ViT -----------------------------------------------------------------
    if hasattr(m, "blocks"):
        for i, blk in enumerate(m.blocks):
            layers.append((f"blocks.{i}", blk))      # [N, tokens, D] -> CLS
        if hasattr(m, "norm") and isinstance(m.norm, nn.Module):
            layers.append(("norm", m.norm))
        return layers

    # --- CLIP-style visual tower (clip_contrastive) -------------
    visual = getattr(m, "visual", None)
    if visual is not None and hasattr(visual, "transformer"):
        for i, blk in enumerate(visual.transformer):
            layers.append((f"visual.transformer.{i}", blk))  # [N, tokens, D] -> CLS
        if hasattr(visual, "ln_post"):
            layers.append(("visual.ln_post", visual.ln_post))
        return layers

    top = [name for name, _ in m.named_children()]
    raise ValueError(
        f"select_layers: unrecognised model structure for architecture={architecture!r}. "
        f"Top-level modules: {top}. Add a dispatch branch for this family."
    )


# RDM construction

def compute_layer_rdm_vector(
    features: np.ndarray,
    metric: str = "cosine",
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Upper-triangular dissimilarity vector for one layer's condition-mean features.

    ``features`` is ``[n_conditions, D]`` (already averaged to the reference-RDM
    condition set). Uses torch matmul (optionally on ``device``) so the high-D
    flattened conv RDMs stay fast. Constant rows yield 0 similarity rather than NaN.
    """
    X = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
    if device is not None:
        X = X.to(device)

    if metric == "cosine":
        Xn = torch.nn.functional.normalize(X, dim=1, eps=1e-8)
        sim = Xn @ Xn.t()
    elif metric == "correlation":
        Xc = X - X.mean(dim=1, keepdim=True)
        Xn = torch.nn.functional.normalize(Xc, dim=1, eps=1e-8)
        sim = Xn @ Xn.t()
    else:
        raise ValueError(f"Unknown layer_rdm metric {metric!r}; expected 'cosine' or 'correlation'.")

    rdm = 1.0 - sim
    n = rdm.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    vec = rdm[iu[0], iu[1]]
    return torch.nan_to_num(vec, nan=0.0).cpu().numpy()


# Cross-validation splits (leave-conditions-out)

def _triu_endpoints(n_conditions: int) -> Tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(n_conditions, k=1)
    return iu[0], iu[1]


def leave_conditions_out_splits(
    n_conditions: int,
    n_folds: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split *conditions* (not pairs) into folds; return (train_conds, test_conds) index arrays.

    Splitting at the condition level is what prevents train/test leakage in RSA:
    a held-out condition participates in ~n_conditions pairs, so a naive split of
    the pair vector would leak that condition's representational fingerprint into
    training. Here a held-out condition appears in *no* training pair.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_conditions)
    folds = np.array_split(perm, n_folds)
    splits = []
    for k in range(n_folds):
        test_conds = np.sort(folds[k])
        train_conds = np.sort(np.concatenate([folds[j] for j in range(n_folds) if j != k]))
        splits.append((train_conds, test_conds))
    return splits


def _both_in_mask(conds: np.ndarray, rows: np.ndarray, cols: np.ndarray, n_conditions: int) -> np.ndarray:
    """Boolean mask over pairs whose *both* endpoints lie in ``conds``."""
    keep = np.zeros(n_conditions, dtype=bool)
    keep[conds] = True
    return keep[rows] & keep[cols]


# Non-negative ridge fit

def nnls_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Non-negative ridge via augmented NNLS: min_{w>=0} ||Xw - y||^2 + alpha||w||^2.

    Adds ``sqrt(alpha) * I`` rows (target 0) so the NNLS solver pays a cost for
    weight magnitude — exactly ridge, with the w>=0 constraint kept.
    """
    n_features = X.shape[1]
    if alpha and alpha > 0:
        X_aug = np.vstack([X, np.sqrt(alpha) * np.eye(n_features)])
        y_aug = np.concatenate([y, np.zeros(n_features)])
    else:
        X_aug, y_aug = X, y
    w, _ = nnls(X_aug, y_aug)
    return w


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman rho, returning (0.0, 1.0) for constant/degenerate inputs."""
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(p)


# Design-matrix preparation

def _normalize_columns(X: np.ndarray, method: str) -> np.ndarray:
    """Per-layer (column) normalization of the stacked RDM design matrix.

    Without this, late layers (higher-variance RDMs) dominate the regression and
    early-layer weights collapse to zero. 'l2' (the recommended default)
    gives every layer equal magnitude so weights reflect relative contribution.
    """
    if method == "none":
        return X
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        if method == "l2":
            nrm = np.linalg.norm(col)
            X[:, j] = col / nrm if nrm > 1e-12 else 0.0
        elif method == "rank":
            X[:, j] = rankdata(col) / col.size
        elif method == "zscore":
            std = col.std()
            X[:, j] = (col - col.mean()) / std if std > 1e-12 else 0.0
        else:
            raise ValueError(f"Unknown normalize_layers method {method!r}.")
    return X


def _dedupe_columns(X: np.ndarray, labels: List[str], thresh: float = 0.999) -> Tuple[np.ndarray, List[str]]:
    """Drop near-identical layer columns (corr > thresh) that would split weight arbitrarily."""
    keep = np.ones(X.shape[1], dtype=bool)
    for i in range(X.shape[1]):
        if not keep[i]:
            continue
        for j in range(i + 1, X.shape[1]):
            if not keep[j]:
                continue
            if np.std(X[:, i]) < 1e-12 or np.std(X[:, j]) < 1e-12:
                continue
            if np.corrcoef(X[:, i], X[:, j])[0, 1] > thresh:
                keep[j] = False
    return X[:, keep], [l for l, k in zip(labels, keep) if k]


def build_design_matrix(
    layer_rdm_vectors: Dict[str, np.ndarray],
    normalize_layers: str = "l2",
    dedupe: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Stack per-layer RDM upper-tri vectors into a design matrix ``X`` [n_pairs, n_layers]."""
    labels = list(layer_rdm_vectors.keys())
    X = np.column_stack([layer_rdm_vectors[l] for l in labels]).astype(np.float64)
    X = _normalize_columns(X, normalize_layers)
    if dedupe:
        X, labels = _dedupe_columns(X, labels)
    return X, labels


# The fit: nested-CV score + in-sample score

_DEFAULT_ALPHAS = np.logspace(-6, 3, 20)


def _select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    train_conds: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n_conditions: int,
    alphas: np.ndarray,
    n_folds: int,
    seed: int,
) -> float:
    """Inner leave-conditions-out CV over ``train_conds`` to pick the ridge alpha.

    Splits the *training* conditions into folds; trains on both-in-inner-train
    pairs, scores on both-in-inner-val pairs. Centering uses inner-train stats.
    """
    if len(train_conds) < 2 * n_folds:
        return float(alphas[len(alphas) // 2])  # too few conds to split; midpoint alpha

    # Sub-split the training conditions (indices are into [0, n_conditions)).
    rng = np.random.RandomState(seed)
    perm = train_conds[rng.permutation(len(train_conds))]
    inner_folds = np.array_split(perm, n_folds)

    best_alpha, best_score = float(alphas[0]), -np.inf
    for alpha in alphas:
        scores = []
        for k in range(n_folds):
            val_conds = inner_folds[k]
            tr_conds = np.concatenate([inner_folds[j] for j in range(n_folds) if j != k])
            tr = _both_in_mask(tr_conds, rows, cols, n_conditions)
            va = _both_in_mask(val_conds, rows, cols, n_conditions)
            if tr.sum() < X.shape[1] + 1 or va.sum() < 3:
                continue
            xm, ym = X[tr].mean(axis=0), y[tr].mean()
            w = nnls_ridge(X[tr] - xm, y[tr] - ym, alpha)
            pred = (X[va] - xm) @ w
            rho, _ = safe_spearman(y[va], pred)
            scores.append(rho)
        if scores and np.mean(scores) > best_score:
            best_score, best_alpha = float(np.mean(scores)), float(alpha)
    return best_alpha


def fit_frrsa(
    layer_rdm_vectors: Dict[str, np.ndarray],
    target_rdm_vector: np.ndarray,
    n_conditions: int,
    alphas: Optional[np.ndarray] = None,
    normalize_layers: str = "l2",
    scoring: str = "insample",
    n_outer_folds: int = 5,
    n_inner_folds: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """Fit layer-reweighted RSA against one target ROI RDM.

    The standard (default, ``scoring="insample"``) runs a single inner
    leave-conditions-out CV to pick the ridge alpha, refits on all condition
    pairs, and reports the in-sample correlation. ``scoring="both"`` additionally
    runs an OUTER leave-conditions-out loop for a depth-fair cross-validated score
    (unbiased across architectures of different depth) — ~5-6x more compute, so
    it's opt-in.

    Args:
        layer_rdm_vectors: {layer_label -> upper-tri RDM vector}, all length
            ``n_conditions*(n_conditions-1)/2`` and row-aligned to the target.
        target_rdm_vector: the ROI's upper-tri RDM vector (same length).
        n_conditions: number of stimulus conditions (720 THINGS / 1000 NOD).
        alphas: ridge grid; defaults to logspace(-6, 3, 20).
        normalize_layers: per-layer column normalization ('l2' recommended).
        scoring: 'insample' (standard, default) or 'both' (also compute cv_score).
        n_outer_folds / n_inner_folds: outer / inner CV fold counts.

    Returns dict with:
        cv_score, cv_score_std      -- nested leave-conditions-out score (mean/std over outer
                                       folds); None unless scoring="both"
        insample_score, insample_p  -- standard score (fit + score on all pairs); the headline
        alpha                        -- alpha chosen for the in-sample full-data fit
        layer_weights                -- {label -> raw weight}  (from full-data fit)
        layer_weights_normalized     -- {label -> |w| / sum|w|}
        n_layers                     -- layers surviving dedupe
        dropped_layers               -- layers collapsed by the near-duplicate (corr>0.999) dedupe
    """
    if alphas is None:
        alphas = _DEFAULT_ALPHAS

    y = np.asarray(target_rdm_vector, dtype=np.float64)
    valid = ~np.isnan(y)
    X, labels = build_design_matrix(layer_rdm_vectors, normalize_layers=normalize_layers)
    n_layers = len(labels)
    # Layers collapsed by the near-duplicate dedupe (corr > 0.999); recorded for
    # transparency so it's always visible which blocks were merged for this fit.
    dropped_layers = [l for l in layer_rdm_vectors if l not in set(labels)]
    rows, cols = _triu_endpoints(n_conditions)

    # ---- nested-CV score (leave-conditions-out, depth-fair) — OPT-IN, expensive ---
    # The outer loop (~5-6x the cost of the fit) is skipped by default; enable it
    # with scoring="both" when you want the depth-fair cross-architecture number.
    if scoring in ("cv", "both"):
        outer = leave_conditions_out_splits(n_conditions, n_outer_folds, seed=seed)
        fold_scores = []
        for f, (train_conds, test_conds) in enumerate(outer):
            alpha = _select_alpha(
                X, y, train_conds, rows, cols, n_conditions, alphas, n_inner_folds, seed=seed + f + 1
            )
            tr = _both_in_mask(train_conds, rows, cols, n_conditions) & valid
            te = _both_in_mask(test_conds, rows, cols, n_conditions) & valid
            if tr.sum() < n_layers + 1 or te.sum() < 3:
                continue
            xm, ym = X[tr].mean(axis=0), y[tr].mean()
            w = nnls_ridge(X[tr] - xm, y[tr] - ym, alpha)
            pred = (X[te] - xm) @ w
            rho, _ = safe_spearman(y[te], pred)
            fold_scores.append(rho)
        cv_score = float(np.mean(fold_scores)) if fold_scores else 0.0
        cv_std = float(np.std(fold_scores)) if fold_scores else 0.0
        n_cv = len(fold_scores)
    else:
        cv_score, cv_std, n_cv = None, None, 0

    # ---- in-sample score (standard: inner CV picks alpha, refit + score on ALL) ----
    all_conds = np.arange(n_conditions)
    insample_alpha = _select_alpha(
        X, y, all_conds, rows, cols, n_conditions, alphas, n_inner_folds, seed=seed
    )
    xm_all, ym_all = X[valid].mean(axis=0), y[valid].mean()
    w_full = nnls_ridge(X[valid] - xm_all, y[valid] - ym_all, insample_alpha)
    pred_full = (X[valid] - xm_all) @ w_full
    insample_score, insample_p = safe_spearman(y[valid], pred_full)

    wsum = float(np.abs(w_full).sum())
    weights = {l: float(w) for l, w in zip(labels, w_full)}
    weights_norm = {l: (float(abs(w)) / wsum if wsum > 0 else 0.0) for l, w in zip(labels, w_full)}

    return {
        "cv_score": cv_score,
        "cv_score_std": cv_std,
        "n_cv_folds": n_cv,
        "insample_score": float(insample_score),
        "insample_p": float(insample_p),
        "alpha": float(insample_alpha),
        "layer_weights": weights,
        "layer_weights_normalized": weights_norm,
        "n_layers": n_layers,
        "dropped_layers": dropped_layers,
    }
