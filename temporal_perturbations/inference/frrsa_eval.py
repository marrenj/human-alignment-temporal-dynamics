"""One FR-RSA evaluation, callable directly.

`inference_core.run_inference`'s `frrsa` branch calls this, and so does the sweep
driver `pipeline/scripts/evaluation/run_frrsa.py`, which loops in-process with a
`stimulus_cache` so the stimulus set is decoded once per (dataset, preprocessing
recipe) rather than once per (run, epoch, dataset) -- 8,640 THINGS-fMRI JPEGs take
~10 minutes to decode.

The result JSON is written to `config["frrsa_save_dir"]` (the paper layout is
`<results_root>/frrsa/<run>/`), falling back to `config["inference_save_dir"]`.
"""

from __future__ import annotations

import json
from pathlib import Path

from temporal_perturbations.data.eval_transforms import (
    frrsa_result_filename,
    resolve_preprocessing,
)


def run_frrsa_eval(*, model, config, device, training_mode, epoch, logger,
                   stimulus_cache=None, reference_rdms=None):
    """Layer RDMs -> per-ROI reweighted ridge -> results dict, written to JSON.

    `stimulus_cache` is an optional pre-decoded batch list from
    `extract_layer_embeddings.build_neural_stimulus_cache`; when None the images are
    decoded here, which is the original per-call behaviour. `reference_rdms` is
    likewise optional and cached by the driver -- it is only a handful of .npy reads,
    but they are the same reads on every checkpoint.

    Returns (results, results_json_path).
    """
    from temporal_perturbations.inference import frrsa_core
    from temporal_perturbations.inference.extract_layer_embeddings import (
        extract_layer_rdm_vectors,
    )
    # Lazy: inference_core imports this module, so a top-level import would cycle.
    from temporal_perturbations.inference.inference_core import prepare_reference_rdms

    frrsa_cfg = config.get("frrsa", {}) or {}
    normalize_layers = frrsa_cfg.get("normalize_layers", "l2")
    n_outer = int(frrsa_cfg.get("n_outer_folds", frrsa_cfg.get("n_folds", 5)))
    n_inner = int(frrsa_cfg.get("n_inner_folds", frrsa_cfg.get("n_folds", 5)))
    layer_metric = frrsa_cfg.get(
        "layer_rdm_distance_metric", config.get("model_rdm_distance_metric", "cosine")
    )
    rdm_device = device if frrsa_cfg.get("rdm_on_gpu", False) else None
    frrsa_seed = int(frrsa_cfg.get("seed", 42))
    scoring = frrsa_cfg.get("scoring", "insample")  # 'insample' (standard) | 'both'
    frrsa_n_jobs = int(frrsa_cfg.get("n_jobs", 8))   # parallel per-ROI fits (1 = serial)

    # FR-RSA covers ImageNet-supervised and CLIP-contrastive checkpoints, none of which
    # were trained on the THINGS-statistics squash-resize the dataset classes default
    # to. `preprocessing: things_legacy` reproduces the dataset-class default; "auto"
    # (default) resolves from the architecture / training_mode.
    frrsa_preprocessing = resolve_preprocessing(
        config.get("preprocessing", "auto"), config.get("architecture"), training_mode
    )

    logger.info("--- FR-RSA (layer-reweighted neural RSA) ---")
    logger.info(
        "  dataset=%s architecture=%s normalize=%s scoring=%s folds=%d/%d metric=%s",
        config.get("dataset"), config.get("architecture"), normalize_layers,
        scoring, n_outer, n_inner, layer_metric,
    )
    logger.info("  preprocessing=%s (%s resize, mean=%s, %s)",
                frrsa_preprocessing.tag, frrsa_preprocessing.resize_mode,
                frrsa_preprocessing.mean, frrsa_preprocessing.interpolation)

    extraction = extract_layer_rdm_vectors(
        model=model,
        dataset_name=config["dataset"],
        img_dir=config["img_dir"],
        annotations_file=config["annotations_file"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_images_per_category=config["max_images_per_category"],
        device=device,
        architecture=config.get("architecture"),
        preprocessing=frrsa_preprocessing,
        layer_rdm_metric=layer_metric,
        rdm_device=rdm_device,
        stimulus_cache=stimulus_cache,
        logger=logger,
    )
    layer_rdm_vectors = extraction["layer_rdm_vectors"]
    n_conditions = extraction["n_conditions"]
    logger.info("  layers=%d conditions=%d", len(layer_rdm_vectors), n_conditions)

    if reference_rdms is None:
        reference_rdms = prepare_reference_rdms(config)
    model_pairs = next(iter(layer_rdm_vectors.values())).shape[0]

    # Validate every ROI reference RDM matches the model condition count up front.
    roi_items = list(reference_rdms.items())
    for roi_name, ref_vec in roi_items:
        if ref_vec.shape[0] != model_pairs:
            raise ValueError(
                f"ROI '{roi_name}' reference RDM has {ref_vec.shape[0]} pairs but the "
                f"model layer RDMs have {model_pairs} ({n_conditions} conditions). "
                f"Check that the reference RDM matches the dataset's condition count."
            )

    # Fit all ROIs. Each is an independent, deterministic, CPU-bound ridge fit that
    # reuses the same model layer-RDMs, so they parallelize cleanly across cores.
    # Results are identical to a serial loop (same seed, same computation) -- joblib
    # just spreads the work; n_jobs=1 falls back to the plain loop.
    fit_kwargs = dict(
        layer_rdm_vectors=layer_rdm_vectors,
        n_conditions=n_conditions,
        normalize_layers=normalize_layers,
        scoring=scoring,
        n_outer_folds=n_outer,
        n_inner_folds=n_inner,
        seed=frrsa_seed,
    )
    if frrsa_n_jobs == 1 or len(roi_items) <= 1:
        fits = [frrsa_core.fit_frrsa(target_rdm_vector=ref_vec, **fit_kwargs)
                for _, ref_vec in roi_items]
    else:
        from joblib import Parallel, delayed
        logger.info("  fitting %d ROIs in parallel (n_jobs=%d) ...",
                    len(roi_items), frrsa_n_jobs)
        fits = Parallel(n_jobs=frrsa_n_jobs)(
            delayed(frrsa_core.fit_frrsa)(target_rdm_vector=ref_vec, **fit_kwargs)
            for _, ref_vec in roi_items
        )

    bn_recalibrated = config.get("bn_recalibrated", False)

    results = {}
    for (roi_name, _ref_vec), fit in zip(roi_items, fits):
        # Headline = the cross-validated score when computed, else the in-sample score.
        headline = fit["cv_score"] if fit["cv_score"] is not None else fit["insample_score"]
        results[roi_name] = {
            "epoch": epoch,
            "evaluation_type": "frrsa",
            "dataset": config["dataset"],
            "reference_rdm_name": roi_name,
            "score": headline,
            "cv_score": fit["cv_score"],
            "cv_score_std": fit["cv_score_std"],
            "n_cv_folds": fit["n_cv_folds"],
            "insample_score": fit["insample_score"],
            "insample_p": fit["insample_p"],
            "alpha": fit["alpha"],
            "n_layers": fit["n_layers"],
            "dropped_layers": fit["dropped_layers"],
            "n_conditions": n_conditions,
            "layer_weights": fit["layer_weights_normalized"],
            "normalize_layers": normalize_layers,
            "layer_rdm_distance_metric": layer_metric,
            "rsa_similarity_metric": "spearman",
            "bn_recalibrated": bn_recalibrated,
        }
        cv_str = (
            f"cv={fit['cv_score']:.4f} (±{fit['cv_score_std']:.4f}) "
            if fit["cv_score"] is not None else ""
        )
        logger.info(
            "  [%s] %sinsample=%.4f alpha=%.2e n_layers=%d",
            roi_name, cv_str, fit["insample_score"], fit["alpha"], fit["n_layers"],
        )

    # Shared with the FR-RSA drivers, which test for this exact path to skip completed
    # work and to confirm output was produced.
    results_json_path = (
        Path(config.get("frrsa_save_dir") or config["inference_save_dir"])
        / frrsa_result_filename(config["dataset"], epoch, frrsa_preprocessing,
                                bn_recalibrated=bn_recalibrated)
    )
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, default=str)
    logger.info("Saved FR-RSA results (JSON) to %s", results_json_path)

    return results, results_json_path
