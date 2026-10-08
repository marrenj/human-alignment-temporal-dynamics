from temporal_perturbations.models.clip_hba.clip_hba_utils import (
    initialize_cliphba_model,
    load_dora_checkpoint,
    load_dora_state_dict,
)
from temporal_perturbations.models.factory import build_model
from temporal_perturbations.training.trainer import load_model_checkpoint
from temporal_perturbations.data.spose_dimensions import classnames66
import torch
import torch.nn.functional as F
import os
import numpy as np
import scipy.io
import pandas as pd
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy.stats import spearmanr, pearsonr, rankdata
from tqdm import tqdm
from torch.utils.data import DataLoader

from temporal_perturbations.utils.logging import setup_logger

from temporal_perturbations.inference.extract_embeddings import extract_embeddings


def center_gram(gram, unbiased=False):
    """Double-center a symmetric Gram matrix for CKA computation.
    Equivalent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.
    Args:
        gram (np.ndarray): A num_examples x num_examples symmetric matrix.
        unbiased (bool): Whether to apply the unbiased HSIC correction
            (recommended when N is small, e.g. N=48).
    Returns:
        np.ndarray: Centered Gram matrix of the same shape.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.copy()
    if unbiased:
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]
    return gram


def compute_cka(X, Y, unbiased=False):
    """Compute linear Centered Kernel Alignment (CKA) between two activation matrices.
    Uses the linear kernel (K = X X^T) and the HSIC-based formulation from
    Kornblith et al., "Similarity of Neural Network Representations Revisited"
    (ICML 2019, https://arxiv.org/abs/1905.00414).
    CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    where HSIC is estimated via the Frobenius inner product of the centered
    Gram matrices:  HSIC(K, L) = <K_c, L_c>_F
    Args:
        X (np.ndarray | torch.Tensor): [N, D1] activation matrix (e.g. model embeddings).
        Y (np.ndarray | torch.Tensor): [N, D2] activation matrix (e.g. SPoSE targets).
        unbiased (bool): Use the unbiased HSIC estimator. Recommended for
            small N (e.g. N=48). CKA may still carry a small bias.
    Returns:
        float: CKA score in [0, 1], where 1 means the two representations
            are identical up to orthogonal transformation and isotropic scaling.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of examples, "
            f"got {X.shape[0]} and {Y.shape[0]}."
        )

    gram_x = X @ X.T  # [N, N] linear kernel
    gram_y = Y @ Y.T  # [N, N] linear kernel

    gram_x = center_gram(gram_x, unbiased=unbiased)
    gram_y = center_gram(gram_y, unbiased=unbiased)

    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return float(scaled_hsic / (normalization_x * normalization_y))


def compute_model_rdm(embeddings, dataset_name, annotations_file, distance_metric, categories=None):
    """
    Compute a representational dissimilarity matrix from embeddings.

    Args:
        embeddings (np.ndarray | torch.Tensor): Row-wise embedding matrix.
        distance_metric (str): Dissimilarity to apply. Supports ``'pearson'`` and
            ``'cosine'``.

    Returns:
        np.ndarray: Square RDM whose entries represent pairwise dissimilarities.

    Raises:
        ValueError: If ``distance_metric`` is not a supported option.
    """

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings)

    if (
        dataset_name == "things"
        and categories is not None
        and annotations_file is not None
        and "concept" in pd.read_csv(annotations_file, nrows=1).columns
    ):
        # average the embeddings by concept using provided category labels
        df = pd.DataFrame(embeddings)
        df["concept"] = categories
        embeddings = df.groupby("concept").mean().to_numpy()

    elif dataset_name == "nod" and categories is not None:
        # average per ImageNet synset; groupby sorts alphabetically, which
        # matches the order in the NOD fMRI RDMs (built from filenames.txt).
        df = pd.DataFrame(embeddings)
        df["category"] = categories
        embeddings = df.groupby("category").mean().to_numpy()

    if distance_metric is None:
        raise ValueError("model_rdm_distance_metric must be provided in the config.")

    if distance_metric == 'pearson':
        rdm = 1 - np.corrcoef(embeddings)
        np.fill_diagonal(rdm, 0)
        return rdm
    elif distance_metric == 'cosine':
        rdm = 1 - cosine_similarity(embeddings)
        np.fill_diagonal(rdm, 0)
        return rdm
    else:
        raise ValueError(f"Invalid distance metric: {distance_metric}")


def build_spose_target_rdm(annotations_file, distance_metric, image_order=None):
    """Build the human SPoSE target RDM from a behavioral annotations CSV.

    The CSV is `index_col=0`; its first column is the image name and the
    remaining columns are the 66-d SPoSE embedding. The target RDM is the
    pairwise dissimilarity of those embeddings under `distance_metric`, i.e. the
    same construction `trainer.compute_behavioral_rsa` uses for its in-loop
    target (`compute_model_rdm(targets, dataset_name="targets", ...)`).

    If `image_order` is given (e.g. the image_names returned by
    `extract_embeddings`), the rows are reordered to match it so the target RDM
    is row-aligned with the model RDM. Returns the upper-triangular vector.
    """
    df = pd.read_csv(annotations_file, index_col=0)
    name_col = df.columns[0]
    if image_order is not None:
        df = df.set_index(name_col).loc[list(image_order)].reset_index()
        targets = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    else:
        targets = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    rdm = compute_model_rdm(
        targets, dataset_name="targets", annotations_file=None,
        distance_metric=distance_metric,
    )
    return rdm[np.triu_indices_from(rdm, k=1)]


def prepare_reference_rdms(config: dict) -> dict:
    """
    Load one or more reference RDMs based on the provided config.

    Supports:
        - config['reference_rdm_paths']: mapping of ROI -> path (one or more paths to reference RDM(s))
    """
    reference_rdms = {}
    if "reference_rdm_paths" not in config:
        raise ValueError("reference_rdm_paths must be provided for RSA-based evaluation.")
    mapping = config["reference_rdm_paths"]

    if mapping:
        for roi_name, reference_rdm_path_str in mapping.items():

            reference_rdm_path = Path(reference_rdm_path_str)
            
            suffix = reference_rdm_path.suffix.lower()

            if suffix == ".npy":
                rdm = np.load(reference_rdm_path)
            elif suffix == ".npz":
                with np.load(reference_rdm_path) as data:
                    if len(data.files) == 1:
                        rdm = data[data.files[0]]
                    else:
                        raise ValueError(
                            f"Multiple arrays found in {reference_rdm_path}. "
                            "Please provide a single array in the .npz file."
                        )
            elif suffix == ".mat":
                mat = scipy.io.loadmat(str(reference_rdm_path))
                candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
                if len(candidate_keys) == 1:
                    rdm = mat[candidate_keys[0]]
                else:
                    raise ValueError(
                         f"Multiple variables found in {reference_rdm_path}. "
                         "Please provide a single array in the .mat file."
                        )
            elif suffix in (".csv", ".tsv", ".txt"):
                delimiter = "\t" if suffix == ".tsv" else ","
                rdm = np.loadtxt(reference_rdm_path, delimiter=delimiter)
            else:
                raise ValueError(f"Unsupported reference RDM format: {reference_rdm_path.suffix}")

            rdm = np.asarray(rdm)
            # If square, flatten upper triangle (k=1 to drop diagonal)
            if rdm.ndim == 2 and rdm.shape[0] == rdm.shape[1]:
                tri_idx = np.triu_indices_from(rdm, k=1)
                rdm = rdm[tri_idx]
            reference_rdms[roi_name] = rdm

    elif mapping is None:
        raise ValueError(
            "reference_rdm_paths must be provided for RSA-based evaluation."
        )
    else:
        raise ValueError("reference_rdm_paths is empty; provide at least one reference RDM path.")

    return reference_rdms


def _permutation_p_gpu(A_sym, B_upper, observed_rho, n, n_permutations, rng,
                       device, batch_size=200):
    """GPU-batched permutation test on n×n matrices."""
    A_gpu = torch.tensor(A_sym, device=device, dtype=torch.float64)
    B_gpu = torch.tensor(B_upper, device=device, dtype=torch.float64)

    count_ge = 0
    for start in range(0, n_permutations, batch_size):
        k = min(batch_size, n_permutations - start)
        perms = np.stack([rng.permutation(n) for _ in range(k)])
        perms_gpu = torch.tensor(perms, device=device, dtype=torch.long)
        A_rows = A_gpu[perms_gpu]
        col_idx = perms_gpu.unsqueeze(1).expand(-1, n, -1)
        A_perm = torch.gather(A_rows, 2, col_idx)
        null_rhos = (A_perm * B_gpu.unsqueeze(0)).sum(dim=(1, 2))
        count_ge += (null_rhos >= observed_rho).sum().item()

    return count_ge


def _permutation_p_cpu(a_proc, b_proc, observed_rho, n, n_permutations, rng):
    """Original CPU loop over upper-tri vectors."""
    pairs_i, pairs_j = np.triu_indices(n, k=1)
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        pi = perm[pairs_i]
        pj = perm[pairs_j]
        a_idx = np.minimum(pi, pj)
        b_idx = np.maximum(pi, pj)
        new_idx = a_idx * n - a_idx * (a_idx + 1) // 2 + b_idx - a_idx - 1
        null_rho = np.dot(a_proc[new_idx], b_proc)
        if null_rho >= observed_rho:
            count_ge += 1
    return count_ge


def permutation_p_rdm(vec_a, vec_b, observed_rho, similarity_metric,
                      n_permutations=10000, seed=42):
    """Condition-label permutation p-value for an RDM comparison (Kriegeskorte et al., 2008)."""
    m = len(vec_a)
    n = int((1 + math.sqrt(1 + 8 * m)) / 2)
    if n * (n - 1) // 2 != m:
        raise ValueError(f"Vector length {m} is not a valid upper-triangular size")

    pairs_i, pairs_j = np.triu_indices(n, k=1)

    if similarity_metric == "spearman":
        a_proc = rankdata(vec_a).astype(np.float64)
        b_proc = rankdata(vec_b).astype(np.float64)
    elif similarity_metric == "pearson":
        a_proc = np.asarray(vec_a, dtype=np.float64)
        b_proc = np.asarray(vec_b, dtype=np.float64)
    else:
        raise ValueError(f"Invalid similarity metric: {similarity_metric}")

    a_proc = a_proc - a_proc.mean()
    norm_a = np.linalg.norm(a_proc)
    if norm_a > 0:
        a_proc /= norm_a

    b_proc = b_proc - b_proc.mean()
    norm_b = np.linalg.norm(b_proc)
    if norm_b > 0:
        b_proc /= norm_b

    rng = np.random.default_rng(seed)

    use_gpu = torch.cuda.is_available() and n >= 100
    if use_gpu:
        A_sym = np.zeros((n, n), dtype=np.float64)
        A_sym[pairs_i, pairs_j] = a_proc
        A_sym[pairs_j, pairs_i] = a_proc
        B_upper = np.zeros((n, n), dtype=np.float64)
        B_upper[pairs_i, pairs_j] = b_proc
        device = torch.device("cuda")
        count_ge = _permutation_p_gpu(
            A_sym, B_upper, observed_rho, n, n_permutations, rng, device)
    else:
        count_ge = _permutation_p_cpu(
            a_proc, b_proc, observed_rho, n, n_permutations, rng)

    return (count_ge + 1) / (n_permutations + 1)


def compute_rdm_similarity(model_rdm_upper_tri_vector, reference_rdm_upper_tri_vector,
                           similarity_metric, n_permutations=10000, permutation_seed=42):
    """Compare two RDMs using a correlation on their upper-triangular entries.

    P-values are computed via condition-label permutation testing
    (Kriegeskorte et al., 2008), not parametric tests.
    """
    if similarity_metric is None:
        raise ValueError("rsa_similarity_metric must be provided in the config.")

    if similarity_metric == 'spearman':
        rho, _ = spearmanr(model_rdm_upper_tri_vector, reference_rdm_upper_tri_vector)
    elif similarity_metric == 'pearson':
        rho, _ = pearsonr(model_rdm_upper_tri_vector, reference_rdm_upper_tri_vector)
    else:
        raise ValueError(f"Invalid similarity metric: {similarity_metric}")

    p_value = permutation_p_rdm(
        model_rdm_upper_tri_vector, reference_rdm_upper_tri_vector,
        rho, similarity_metric, n_permutations, permutation_seed,
    )
    return rho, p_value


def run_inference(config):

    os.makedirs(config['inference_save_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(
        config['inference_save_dir'],
        f'inference_log_{config.get("dataset", "unknown")}_{config["evaluation_type"]}.txt',
    )
    logger = setup_logger(log_file)
    logger.info("Run timestamp: %s", timestamp)

    # Persist the config snapshot alongside outputs
    snapshot_path = Path(config['inference_save_dir']) / "inference_config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    logger.info("Saved config snapshot to %s", snapshot_path)
    
    training_mode = config["training_mode"]
    cuda_cfg = config.get("cuda", 0)

    if config['cuda'] == -1:
        device = torch.device("cuda")
    elif config['cuda'] == 0:
        device = torch.device("cuda:0")
    elif config['cuda'] == 1:
        device = torch.device("cuda:1")
    else:
        raise ValueError("Unsupported cuda setting. Expected -1, 0, or 1.")

    if training_mode == 'scratch':
        # Build the same timm ViT/ResNet used during training
        model = build_model(
            architecture=config.get('architecture'),
            pretrained=False,
            clip_hba_backbone=None,
            vision_layers=None,
            transformer_layers=None,
            rank=None,
            cuda=cuda_cfg,
            device=device,
            wandb_watch_model=False,
            wandb_log_freq=None,
            num_classes=config.get('num_classes', 1000),
        )

        # Resolve checkpoint path and derive epoch label
        weights_path = Path(config['model_weights_path'])
        if weights_path.is_file():
            epoch = weights_path.stem.replace('epoch', '').replace('_model', '')
            load_model_checkpoint(model, str(weights_path), logger)


        else:
            # Directory mode: look inside model_checkpoints/ for epoch*_model.pth files
            ckpt_dir = (
                weights_path
                if weights_path.name == 'model_checkpoints'
                else weights_path / 'model_checkpoints'
            )
            ckpt_files = sorted(ckpt_dir.glob('epoch*_model.pth'))
            if not ckpt_files:
                raise FileNotFoundError(f"No model checkpoints found under {ckpt_dir}")
            requested_epoch = config.get('epoch')
            if requested_epoch is not None:
                epoch = str(requested_epoch)
                candidate = ckpt_dir / f'epoch{epoch}_model.pth'
                if not candidate.exists():
                    raise FileNotFoundError(f"Requested epoch checkpoint not found: {candidate}")
                load_model_checkpoint(model, str(candidate), logger)

            else:
                latest = ckpt_files[-1]
                epoch = latest.stem.replace('epoch', '').replace('_model', '')
                load_model_checkpoint(model, str(latest), logger)

        model.to(device)

    elif training_mode == 'scratch_lightning':
        # timm ResNet/ViT/ConvNeXt trained by an external PyTorch-Lightning
        # pipeline: weights are nested under `state_dict` with a `model.` prefix in
        # a Lightning .ckpt (see temporal_perturbations/models/lightning_ckpt.py).
        # Build the timm model, strip `model.`, and load. Checkpoints live at
        # <run>/checkpoints/all_epochs/epoch_{E:03d}.ckpt (some runs also use the
        # `epoch_epoch={E:03d}.ckpt` variant) — both are handled.
        from temporal_perturbations.models.lightning_ckpt import load_lightning_state_dict
        import re as _re

        model = build_model(
            architecture=config.get('architecture'),
            pretrained=False,
            clip_hba_backbone=None,
            vision_layers=None,
            transformer_layers=None,
            rank=None,
            cuda=cuda_cfg,
            device=device,
            wandb_watch_model=False,
            wandb_log_freq=None,
            num_classes=config.get('num_classes', 1000),
        )

        def _parse_lightning_epoch(name):
            m = _re.search(r"epoch(?:_epoch=|_)(\d+)\.ckpt$", name)
            return str(int(m.group(1))) if m else None

        weights_path = Path(config['model_weights_path'])
        if weights_path.is_file():
            ckpt_path = weights_path
            epoch = (str(int(config['epoch'])) if config.get('epoch') is not None
                     else _parse_lightning_epoch(weights_path.name))
        else:
            all_epochs = (
                weights_path if weights_path.name == 'all_epochs'
                else weights_path / 'checkpoints' / 'all_epochs'
            )
            requested_epoch = config.get('epoch')
            if requested_epoch is None:
                raise ValueError("scratch_lightning directory mode requires --set epoch=<E>.")
            epoch = str(int(requested_epoch))
            candidates = [
                all_epochs / f"epoch_{int(requested_epoch):03d}.ckpt",
                all_epochs / f"epoch_epoch={int(requested_epoch):03d}.ckpt",
            ]
            hits = [c for c in candidates if c.exists()]
            if not hits:
                raise FileNotFoundError(
                    f"No Lightning checkpoint for epoch {requested_epoch} under {all_epochs} "
                    f"(looked for epoch_{int(requested_epoch):03d}.ckpt / "
                    f"epoch_epoch={int(requested_epoch):03d}.ckpt)."
                )
            ckpt_path = hits[0]

        state_dict = load_lightning_state_dict(ckpt_path, map_location='cpu')
        target = model.module if hasattr(model, 'module') else model
        target.load_state_dict(state_dict, strict=True)
        logger.info(
            "Loaded scratch_lightning %s epoch %s from %s",
            config.get('architecture'), epoch, ckpt_path.name,
        )
        model.to(device)

    elif training_mode == 'clip_contrastive':
        # CLIP-from-scratch trained on YFCC15M (Lightning checkpoint).
        # The permissive Lightning unpickler skips the checkpoint's pickled config classes.
        from temporal_perturbations.models.clip_yfcc import load_clip_lightning

        weights_path = Path(config['model_weights_path'])
        epoch = None
        if weights_path.is_file():
            ckpt_path = weights_path
        elif weights_path.is_dir():
            # Find the requested epoch's Lightning ckpt under <weights_path>/checkpoints/
            ckpt_dir = (
                weights_path
                if weights_path.name == "checkpoints"
                else weights_path / "checkpoints"
            )
            requested_epoch = config.get("epoch")
            if requested_epoch is None:
                # Default to last.ckpt if present, otherwise the highest-numbered.
                last = ckpt_dir / "last.ckpt"
                if last.is_file():
                    ckpt_path = last
                    epoch = "last"
                else:
                    candidates = sorted(ckpt_dir.glob("*epoch_epoch=*.ckpt"))
                    if not candidates:
                        raise FileNotFoundError(f"No Lightning ckpts under {ckpt_dir}")
                    ckpt_path = candidates[-1]
            else:
                epoch_str = f"{int(requested_epoch):03d}"
                candidates = list(ckpt_dir.glob(f"*epoch_epoch={epoch_str}.ckpt"))
                if not candidates:
                    candidates = list(ckpt_dir.glob(f"epoch_{epoch_str}.ckpt"))
                if not candidates:
                    raise FileNotFoundError(
                        f"No Lightning ckpt for epoch {requested_epoch} under {ckpt_dir}"
                    )
                ckpt_path = sorted(candidates)[0]
        else:
            raise FileNotFoundError(f"model_weights_path not found: {weights_path}")

        if epoch is None:
            # Prefer an explicitly-passed epoch (so output filenames match the caller's
            # `epoch{E}` convention); else parse it from the checkpoint filename like
            # `monitor_val_cifar100_epoch_acc1-epoch_epoch=017.ckpt` / `epoch_017.ckpt`.
            if config.get("epoch") is not None:
                epoch = str(int(config["epoch"]))
            else:
                import re as _re
                m = _re.search(r"epoch[_=](\d+)\.ckpt$", ckpt_path.name)
                epoch = m.group(1) if m else "last"

        config_preset = config.get("clip_yfcc_config_preset", "vit_b_32")
        model = load_clip_lightning(
            ckpt_path=str(ckpt_path),
            config_preset=config_preset,
            map_location=device,
        )
        model.to(device)
        logger.info("Loaded CLIP-YFCC Lightning ckpt: %s (epoch %s)", ckpt_path, epoch)

    else:
        # Fine-tuning: CLIP-HBA with DoRA adapters
        model = initialize_cliphba_model(
            backbone_name=config['backbone'],
            classnames=classnames66,
            vision_layers=config['vision_layers'],
            transformer_layers=config['transformer_layers'],
            rank=config['rank'],
            dora_dropout=0.1,
            logger=logger,
        )

        weights_path = Path(config['model_weights_path'])
        loaded_direct = False
        if weights_path.is_file():
            load_dora_state_dict(
                model,
                torch.load(weights_path, map_location=device, weights_only=True),
                source=str(weights_path),
            )
            epoch = weights_path.stem.replace('epoch', '').replace('_dora_params', '')
            loaded_direct = True

        else:
            dora_dir = (
                weights_path
                if weights_path.name == "dora_params"
                else weights_path / "dora_params"
            )
            checkpoint_root = dora_dir.parent
            checkpoint_files = sorted(dora_dir.glob("epoch*_dora_params.pth"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No DoRA checkpoints found under {dora_dir}")
            requested_epoch = config.get("epoch")
            if requested_epoch is not None:
                epoch = str(requested_epoch)
                candidate = dora_dir / f"epoch{epoch}_dora_params.pth"
                if not candidate.exists():
                    raise FileNotFoundError(f"Requested epoch checkpoint not found: {candidate}")
            else:
                latest_ckpt = checkpoint_files[-1]
                epoch = latest_ckpt.stem.replace('epoch', '').replace('_dora_params', '')

        model = model.to(device)
        if not loaded_direct:
            load_dora_checkpoint(model, checkpoint_root=checkpoint_root, epoch=epoch, map_location=device)

    model.eval()

    if "evaluation_type" not in config:
        logger.error("evaluation_type is not set in the config file.")
        raise ValueError("evaluation_type must be set in the config file.")
    evaluation_type = config["evaluation_type"]

    logger.info("\n=== Processing epoch %s ===", epoch)

    results = None
    score = None

    if evaluation_type in ("neural", "behavioral"):
        logger.info("--- FILES ---")
        logger.info("  img_dir:          %s", config['img_dir'])
        logger.info("  annotations_file: %s", config['annotations_file'])
        logger.info("  model_weights:    %s", config['model_weights_path'])
        for rdm_name, rdm_path in config.get("reference_rdm_paths", {}).items():
            logger.info("  reference_rdm[%s]: %s", rdm_name, rdm_path)

        # extract embeddings from the model
        embedding_outputs = extract_embeddings(
            model=model,
            dataset_name=config['dataset'],
            img_dir=config['img_dir'],
            annotations_file=config['annotations_file'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            max_images_per_category=config['max_images_per_category'],
            device=device,
            logger=logger,
        )

        embeddings = embedding_outputs["embeddings"]
        categories = embedding_outputs["categories"]
        logger.info("--- EMBEDDINGS ---")
        logger.info("  Raw embedding matrix shape: %s", tuple(np.array(embeddings).shape if not hasattr(embeddings, 'shape') else embeddings.shape))
        logger.info("  Unique categories extracted: %d", len(set(categories)) if categories is not None else 0)

        # create an RDM from the model's embeddings
        model_rdm = compute_model_rdm(
            embeddings,
            dataset_name=config['dataset'],
            annotations_file=config['annotations_file'],
            categories=categories,
            distance_metric=config["model_rdm_distance_metric"],
        )

        # Cache the per-image concept×dim matrix for offline dimension-level
        # analyses. THINGS behavioral RSA uses max_images_per_category=1, so the
        # raw embeddings are already 48×66 and aligned to annotations_file row order.
        if evaluation_type == "behavioral" and config.get("dataset") == "things":
            raw = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
            if categories:
                grouped = pd.DataFrame(raw).assign(concept=categories).groupby("concept").mean()
                matrix, labels = grouped.to_numpy(), list(grouped.index)
            else:
                ann = pd.read_csv(config["annotations_file"], index_col=0)
                matrix, labels = raw, ann["image_name"].tolist()
            emb_dir = Path(config.get('embeddings_save_dir') or config['inference_save_dir'])
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.save(emb_dir / f"things_embeddings_epoch{epoch}.npy", matrix)
            with open(emb_dir / f"things_concepts_epoch{epoch}.json", "w") as f:
                json.dump(labels, f)
            logger.info("Cached things embeddings: %s (%dx%d) + label list",
                        emb_dir / f"things_embeddings_epoch{epoch}.npy", *matrix.shape)

        logger.info("--- MODEL RDM ---")
        logger.info("  Shape: %s  distance_metric: %s", model_rdm.shape, config["model_rdm_distance_metric"])
        logger.info("  Min: %.6f  Max: %.6f  Mean: %.6f", float(model_rdm.min()), float(model_rdm.max()), float(model_rdm.mean()))

        # get the upper triangular vector of the model RDM
        model_rdm_upper_tri_indices = np.triu_indices_from(model_rdm, k=1)
        model_rdm_upper_tri_vector = model_rdm[model_rdm_upper_tri_indices]
        logger.info("  Upper-tri vector length: %d", len(model_rdm_upper_tri_vector))

        # load reference RDM(s)
        reference_rdms_upper_tri_vectors = prepare_reference_rdms(config)
        reference_rdm_distance_metric = config["reference_rdm_distance_metric"]

        logger.info("--- REFERENCE RDMs ---")
        for rdm_name, rdm_vec in reference_rdms_upper_tri_vectors.items():
            logger.info(
                "  [%s] upper-tri length: %d  Min: %.6f  Max: %.6f  Mean: %.6f",
                rdm_name, len(rdm_vec), float(rdm_vec.min()), float(rdm_vec.max()), float(rdm_vec.mean()),
            )

        # compute RSA scores for each reference RDM
        rsa_similarity_metric = config["rsa_similarity_metric"]

        logger.info("--- RSA ---")
        logger.info("  similarity_metric: %s", rsa_similarity_metric)

        rsa_results = {}
        for reference_rdm_name, reference_rdm_upper_tri_vector in reference_rdms_upper_tri_vectors.items():
            if reference_rdm_upper_tri_vector.shape != model_rdm_upper_tri_vector.shape:
                raise ValueError(
                    f"Reference RDM for ROI '{reference_rdm_name}' has shape "
                    f"{reference_rdm_upper_tri_vector.shape} but model RDM has shape "
                    f"{model_rdm_upper_tri_vector.shape}"
                )

            rho, p_value = compute_rdm_similarity(
                model_rdm_upper_tri_vector,
                reference_rdm_upper_tri_vector,
                similarity_metric=rsa_similarity_metric,
            )

            rsa_results[reference_rdm_name] = {
                "epoch": epoch,
                "evaluation_type": evaluation_type,
                "dataset": config['dataset'],
                "reference_rdm_name": reference_rdm_name,
                "score": float(rho),
                "p_value": float(p_value),
                "p_method": "permutation",
                "n_permutations": 10000,
                "rsa_similarity_metric": rsa_similarity_metric,
                "model_rdm_distance_metric": config["model_rdm_distance_metric"],
                "reference_rdm_distance_metric": reference_rdm_distance_metric,
            }
            logger.info(
                "  [%s] rho=%.4f  p=%.4g",
                reference_rdm_name, rho, p_value,
            )

        results = rsa_results
        # Use the first reference RDM's rho as the headline scalar score.
        score = float(next(iter(rsa_results.values()))["score"]) if rsa_results else None

        results_json_path = (
            Path(config['inference_save_dir'])
            / f"inference_results_{config['dataset']}_{evaluation_type}_epoch{epoch}.json"
        )
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved inference results (JSON) to %s", results_json_path)

    elif evaluation_type == "frrsa":
        # Layer-reweighted RSA (FR-RSA) for non-CLIP-HBA architectures: per-layer
        # features -> per-layer RDMs -> a non-negative ridge that reweights the layers
        # to predict each ROI's RDM. See temporal_perturbations/inference/frrsa_core.py.
        #
        # The body lives in frrsa_eval.run_frrsa_eval so run_frrsa.py can
        # call it in a loop with the stimulus set decoded once, instead of paying a
        # fresh subprocess and a full re-decode per (run, epoch, dataset).
        from temporal_perturbations.inference.frrsa_eval import run_frrsa_eval

        results, _ = run_frrsa_eval(
            model=model, config=config, device=device, training_mode=training_mode,
            epoch=epoch, logger=logger,
        )
        score = float(next(iter(results.values()))["score"]) if results else None
    elif evaluation_type == "imagenet_val":
        # ImageNet top-1 / top-5 classification accuracy (scratch models only)
        if training_mode != 'scratch':
            raise ValueError(
                "evaluation_type='imagenet_val' requires training_mode='scratch'. "
                "Use 'behavioral' or 'neural' for finetune models."
            )
        from temporal_perturbations.data.imagenet_dataset import ImagenetDataset

        val_dataset = ImagenetDataset(img_dir=config['img_dir'], split='val')
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=(device.type == 'cuda'),
        )
        logger.info("Running ImageNet val evaluation on %d images ...", len(val_dataset))

        top1_correct = 0
        top5_correct = 0
        total = 0
        with torch.no_grad():
            for _image_names, images, labels in tqdm(val_loader, desc="ImageNet val"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                top1_correct += (logits.argmax(dim=1) == labels).sum().item()
                top5_preds = logits.topk(5, dim=1).indices
                top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

        top1_acc = top1_correct / total
        top5_acc = top5_correct / total
        score = top1_acc
        results = {
            "epoch": epoch,
            "evaluation_type": "imagenet_val",
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "total_samples": total,
        }
        logger.info(
            "ImageNet val — Epoch %s: Top-1 %.4f | Top-5 %.4f (%d samples)",
            epoch, top1_acc, top5_acc, total,
        )
        results_json_path = (
            Path(config['inference_save_dir'])
            / f"inference_results_imagenet_val_epoch{epoch}.json"
        )
    
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved inference results (JSON) to %s", results_json_path)

    elif evaluation_type == "cka":
        # Extract CLIP-HBA last-layer activations for the 48 inference images
        embedding_outputs = extract_embeddings(
            model=model,
            dataset_name=config['dataset'],
            img_dir=config['img_dir'],
            annotations_file=config['annotations_file'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            max_images_per_category=config['max_images_per_category'],
            device=device,
            logger=logger,
        )
        model_activations = embedding_outputs["embeddings"]  # [N, 66] tensor

        # Load the SPoSE target embeddings from the same annotation CSV.
        # The CSV has index_col=0, column 0 = image_name, columns 1-66 = embedding dims,
        # in the same row order as the dataset, so alignment with model_activations is exact.
        annotations_df = pd.read_csv(config['inference_annotations_file'], index_col=0)
        spose_embeddings = annotations_df.iloc[:, 1:].values.astype(np.float64)  # [N, 66]

        n_images = model_activations.shape[0]
        if n_images != spose_embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between number of model activations ({n_images}) "
                f"and SPoSE embeddings ({spose_embeddings.shape[0]})."
            )

        unbiased = config.get('cka_unbiased', False)
        cka_score = compute_cka(model_activations, spose_embeddings, unbiased=unbiased)
        score = cka_score

        results = {
            "epoch": epoch,
            "evaluation_type": "cka",
            "dataset": config['dataset'],
            "cka_score": cka_score,
            "unbiased": unbiased,
            "n_images": n_images,
        }
        logger.info(
            "CKA score (epoch %s, n=%d, unbiased=%s): %.4f",
            epoch, n_images, unbiased, cka_score,
        )
        results_json_path = (
            Path(config['inference_save_dir'])
            / f"inference_results_{config['dataset']}_cka_epoch{epoch}.json"
        )
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved CKA results (JSON) to %s", results_json_path)

    elif evaluation_type == "linear_probe":
        # Linear-probe eval for vision-only models on the 27 paper classification datasets.
        if training_mode in ('finetune', 'clip_contrastive'):
            raise ValueError(
                "evaluation_type='linear_probe' is for vision-only models "
                "(scratch / scratch_lightning); use run_linear_probe_eval.py for CLIP."
            )
        from temporal_perturbations.data.eval_transforms import IMAGENET_SUPERVISED_BILINEAR
        from temporal_perturbations.inference.linear_probe_eval import (
            run_linear_probe,
            PAPER_PROBE_DATASETS,
        )

        # Keyed on training_mode, not architecture: `scratch` models are trained by
        # data/imagenet_dataset.py -- ImageNet stats, Resize(256) -> CenterCrop(224),
        # torchvision's default bilinear.
        preprocessing = IMAGENET_SUPERVISED_BILINEAR

        # _extract_features raises on raw timm models (forward_features guard).
        if training_mode in ('scratch', 'scratch_lightning'):
            from temporal_perturbations.models.scratch.scratch_utils import TimmFeatureWrapper
            model = TimmFeatureWrapper(model).to(device).eval()

        probe_dir = Path(config['inference_save_dir']) / f"linear_probe_epoch{epoch}"
        datasets_arg = config.get('linear_probe_datasets')
        if datasets_arg in (None, 'paper'):
            datasets = list(PAPER_PROBE_DATASETS)
        elif isinstance(datasets_arg, list):
            datasets = datasets_arg
        else:
            datasets = [d.strip() for d in str(datasets_arg).split(',') if d.strip()]

        results = run_linear_probe(
            model,
            output_dir=probe_dir,
            epoch=epoch,
            preprocessing=preprocessing,
            datasets=datasets,
            image_size=config.get('linear_probe_image_size', 224),
            batch_size=config.get('linear_probe_batch_size', config.get('batch_size', 256)),
            num_workers=config.get('num_workers', 8),
            device=device,
            force=bool(config.get('force', False)),
            logger=logger,
        )

        headline = results.get('imagenet1k', {}).get('score') if results else None
        if headline is None and results:
            for v in results.values():
                if isinstance(v, dict) and 'score' in v:
                    headline = v['score']
                    break
        score = float(headline) if headline is not None else None

        results_json_path = (
            Path(config['inference_save_dir'])
            / f"inference_results_linear_probe_epoch{epoch}.json"
        )
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, default=str)
        logger.info("Saved linear-probe results (JSON) to %s", results_json_path)

    else:
        raise ValueError(
            f"Unsupported combination of dataset '{config.get('dataset', 'unknown')}' "
            f"and evaluation_type '{evaluation_type}'."
        )

    if score is None:
        logger.info("Score for epoch %s (%s): None", epoch, evaluation_type)
    else:
        logger.info("Score for epoch %s (%s): %.4f", epoch, evaluation_type, score)

    return {
        "epoch": epoch,
        "score": score,
        "results": results,
    }
