"""Reusable core for comparing CLIP-HBA representations between two checkpoints.

Shared so a sweep driver comparing hundreds of perturbed-run checkpoints
against their baselines can build the model and the frozen
visual-prefix cache ONCE and reuse them across every comparison, instead of
paying that cost per checkpoint pair. See `vit_prefix_cache.py` for why the
prefix is shared across every checkpoint that has the same
(backbone, vision_layers, transformer_layers).
"""

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from temporal_perturbations.paths import resolve_config_paths
from temporal_perturbations.data.spose_dimensions import classnames66
from temporal_perturbations.data.things_dataset import ThingsBehavioralDataset, ThingsFMRIDataset
from temporal_perturbations.inference.inference_core import (
    compute_cka,
    compute_model_rdm,
    compute_rdm_similarity,
)
from temporal_perturbations.inference.vit_prefix_cache import text_features, visual_prefix, visual_suffix
from temporal_perturbations.models.clip_hba.clip_hba_utils import (
    initialize_cliphba_model,
    load_dora_checkpoint,
    load_dora_state_dict,
)


def load_stimulus_configs(config_paths, repo_root, img_dir_override=None):
    """Load one or more stimulus-set YAML configs, resolving paths against repo_root."""
    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (Path(repo_root) / p)

    configs = []
    for config_path in config_paths:
        with open(_resolve(config_path)) as f:
            cfg = yaml.safe_load(f)
        if img_dir_override and len(config_paths) == 1:
            cfg["img_dir"] = img_dir_override
        resolve_config_paths(cfg)
        cfg["annotations_file"] = str(_resolve(cfg["annotations_file"]))
        cfg["img_dir"] = str(_resolve(cfg["img_dir"]))
        cfg["stimulus_set"] = cfg.get("stimulus_set", Path(config_path).stem)
        configs.append(cfg)
    return configs


def build_model_and_caches(stimulus_configs, device, logger):
    """Initialize the CLIP-HBA architecture and cache each stimulus set's frozen
    visual prefix ONCE. Returns (model, tokenized_prompts, prefix_caches)."""
    # All stimulus-set configs compare the SAME two checkpoints, so the
    # architecture (backbone/vision_layers/transformer_layers/rank) must match
    # across configs -- initialize once from the first.
    arch_cfg = stimulus_configs[0]
    model = initialize_cliphba_model(
        backbone_name=arch_cfg["backbone"],
        classnames=classnames66,
        vision_layers=arch_cfg["vision_layers"],
        transformer_layers=arch_cfg["transformer_layers"],
        rank=arch_cfg["rank"],
        dora_dropout=0.1,
        logger=logger,
    ).to(device)
    model.eval()
    tokenized_prompts = model.tokenized_prompts.to(device)

    logger.info("[cache] building frozen visual-prefix cache per stimulus set "
                "(shared across every checkpoint/epoch compared)...")
    prefix_caches = {cfg["stimulus_set"]: _cache_prefix(model, cfg, device, logger) for cfg in stimulus_configs}
    return model, tokenized_prompts, prefix_caches


def load_checkpoint(model, checkpoint_dir, epoch, device):
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_file():
        load_dora_state_dict(
            model,
            torch.load(checkpoint_dir, map_location=device, weights_only=True),
            source=str(checkpoint_dir),
        )
        return
    load_dora_checkpoint(model, checkpoint_root=checkpoint_dir, epoch=epoch, map_location=device)


def available_epochs(checkpoint_dir):
    """Epoch numbers with an existing dora_params checkpoint under checkpoint_dir."""
    checkpoint_dir = Path(checkpoint_dir)
    dora_dir = checkpoint_dir / "dora_params" if (checkpoint_dir / "dora_params").is_dir() else checkpoint_dir
    return {int(p.stem.replace("epoch", "").replace("_dora_params", "")) for p in dora_dir.glob("epoch*_dora_params.pth")}


def compare_checkpoints(model, tokenized_prompts, prefix_caches, checkpoint_dir_a, checkpoint_dir_b,
                         epochs, label_a, label_b, out_dir, stimulus_configs, device, force, logger):
    """Compare checkpoint_dir_a vs checkpoint_dir_b at each epoch, for each stimulus
    set, writing one JSON per (epoch, stimulus_set). Returns the list of results written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for epoch in epochs:
        pending = []
        for cfg in stimulus_configs:
            out_json = out_dir / (f"model_to_model_rsa_{cfg['stimulus_set']}_"
                                   f"{label_a}_vs_{label_b}_epoch{epoch}.json")
            if out_json.exists() and not force:
                logger.info("[skip] %s exists", out_json)
            else:
                pending.append((cfg, out_json))
        if not pending:
            continue

        load_checkpoint(model, checkpoint_dir_a, epoch, device)
        model.eval()
        emb_a = {
            cfg["stimulus_set"]: embed_from_cache(
                model, prefix_caches[cfg["stimulus_set"]], tokenized_prompts, device, cfg["batch_size"])
            for cfg, _ in pending
        }

        load_checkpoint(model, checkpoint_dir_b, epoch, device)
        model.eval()
        emb_b = {
            cfg["stimulus_set"]: embed_from_cache(
                model, prefix_caches[cfg["stimulus_set"]], tokenized_prompts, device, cfg["batch_size"])
            for cfg, _ in pending
        }

        for cfg, out_json in pending:
            name = cfg["stimulus_set"]
            cache = prefix_caches[name]
            r = compare_embeddings(cfg, emb_a[name], cache["categories"], cache["names"],
                         emb_b[name], cache["categories"], cache["names"], epoch)

            result = {
                "epoch": epoch,
                "stimulus_set": name,
                "run_a": str(checkpoint_dir_a), "label_a": label_a,
                "run_b": str(checkpoint_dir_b), "label_b": label_b,
                **r,
            }
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(
                "[done] %s epoch%s: RSA(pearson) rho=%.4f (p=%.3g)  RSA(spearman) rho=%.4f (p=%.3g)  CKA=%.4f -> %s",
                name, epoch, r["rsa"]["pearson"]["rho"], r["rsa"]["pearson"]["p_value"],
                r["rsa"]["spearman"]["rho"], r["rsa"]["spearman"]["p_value"], r["cka"]["score"], out_json.name,
            )
            written.append(result)
    return written


# Internals

def grouped_embeddings(embeddings, dataset_name, categories):
    """Concept/category-average embeddings the same way `compute_model_rdm`
    does internally, so CKA and RSA operate on identically aligned rows."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings)
    if not categories:
        return embeddings
    key = "concept" if dataset_name == "things" else "category"
    df = pd.DataFrame(embeddings)
    df[key] = categories
    return df.groupby(key).mean().to_numpy()


def _build_stimulus_dataset(cfg):
    """Build the underlying THINGS image dataset for the frozen-prefix cache.

    Only `dataset: things` is wired up here (both default configs use it);
    NOD would need its own cache-building branch if ever added.
    """
    if cfg["dataset"] != "things":
        raise NotImplementedError(
            f"Cached-prefix path only supports dataset='things' (got {cfg['dataset']!r})."
        )
    has_categories = "concept" in pd.read_csv(cfg["annotations_file"], nrows=1).columns
    if has_categories:
        dataset = ThingsFMRIDataset(img_annotations_file=cfg["annotations_file"], img_dir=cfg["img_dir"])
    else:
        dataset = ThingsBehavioralDataset(img_annotations_file=cfg["annotations_file"], img_dir=cfg["img_dir"])
    return dataset, has_categories


def _cache_prefix(model, cfg, device, logger):
    """Run every stimulus through the frozen prefix ONCE; reused for every
    checkpoint/epoch compared against this stimulus set for the rest of the run."""
    dataset, has_categories = _build_stimulus_dataset(cfg)
    use_autocast = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg["num_workers"], pin_memory=use_autocast)

    visual = model.clip_model.visual
    n_blocks = len(visual.transformer.resblocks)
    split_at = n_blocks - cfg["vision_layers"]
    pos_embedding = model.pos_embedding

    names, categories, prefix_chunks = [], [], []
    with torch.no_grad():
        for batch in loader:
            if has_categories:
                image_names, images, cats = batch
                categories.extend(list(cats))
            else:
                image_names, images, _ = batch
            names.extend(list(image_names))
            images = images.to(device, non_blocking=use_autocast)
            with torch.amp.autocast("cuda") if use_autocast else nullcontext():
                pfx = visual_prefix(visual, images, pos_embedding, split_at)
            prefix_chunks.append(pfx.permute(1, 0, 2).contiguous())  # -> [B, L, W]

    prefix_cache = torch.cat(prefix_chunks, dim=0)
    logger.info(
        "[cache] %s: prefix %s (%.2f GB), %d stimuli, split at block %d",
        cfg["stimulus_set"], tuple(prefix_cache.shape),
        prefix_cache.element_size() * prefix_cache.nelement() / 1e9,
        len(names), split_at,
    )
    return {"prefix": prefix_cache, "names": names, "categories": categories, "split_at": split_at}


def embed_from_cache(model, cache, tokenized_prompts, device, batch_size):
    """Recompute only the last `vision_layers` blocks + text tower for the
    currently-loaded checkpoint, reusing the cached frozen prefix."""
    visual = model.clip_model.visual
    split_at = cache["split_at"]
    prefix_cache = cache["prefix"]
    use_autocast = device.type == "cuda"
    with torch.no_grad():
        with torch.amp.autocast("cuda") if use_autocast else nullcontext():
            txt = text_features(model.clip_model, tokenized_prompts)
            logit_scale = model.clip_model.logit_scale.exp()
            logits = []
            for i in range(0, prefix_cache.shape[0], batch_size):
                chunk = prefix_cache[i:i + batch_size].permute(1, 0, 2)  # [L,B,W]
                img = visual_suffix(visual, chunk, split_at)
                img = img / img.norm(dim=-1, keepdim=True)
                logits.append((logit_scale * (img @ txt.t())).float().cpu())
    return torch.cat(logits, dim=0).numpy()


def compare_embeddings(cfg, emb_a, cats_a, names_a, emb_b, cats_b, names_b, epoch):
    if names_a != names_b:
        raise ValueError(
            f"Stimulus order mismatch between run A and run B at epoch {epoch} "
            f"for stimulus_set={cfg['stimulus_set']!r}; both runs must be evaluated "
            "on the identical stimulus set/order."
        )

    model_metric = cfg["model_rdm_distance_metric"]
    rdm_a = compute_model_rdm(emb_a, dataset_name=cfg["dataset"], annotations_file=cfg["annotations_file"],
                               categories=cats_a, distance_metric=model_metric)
    rdm_b = compute_model_rdm(emb_b, dataset_name=cfg["dataset"], annotations_file=cfg["annotations_file"],
                               categories=cats_b, distance_metric=model_metric)
    vec_a = rdm_a[np.triu_indices_from(rdm_a, k=1)]
    vec_b = rdm_b[np.triu_indices_from(rdm_b, k=1)]

    rsa_scores = {}
    for sim_metric in ("pearson", "spearman"):
        rho, p = compute_rdm_similarity(vec_a, vec_b, sim_metric)
        rsa_scores[sim_metric] = {
            "rho": float(rho), "p_value": float(p),
            "p_method": "permutation", "n_permutations": 10000,
        }

    grouped_a = grouped_embeddings(emb_a, cfg["dataset"], cats_a)
    grouped_b = grouped_embeddings(emb_b, cfg["dataset"], cats_b)
    cka_unbiased = cfg.get("cka_unbiased", True)
    cka_score = compute_cka(grouped_a, grouped_b, unbiased=cka_unbiased)

    return {
        "dataset": cfg["dataset"],
        "n_stimuli": int(grouped_a.shape[0]),
        "rsa": {"model_rdm_distance_metric": model_metric, **rsa_scores},
        "cka": {"unbiased": cka_unbiased, "score": float(cka_score)},
    }
