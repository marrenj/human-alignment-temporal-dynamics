# Temporal Perturbations

Code for *Examining the temporal dynamics of human representational alignment by probing model learning windows*
(arXiv, 2026), which studies how a single perturbed training epoch, applied at a controlled time, changes task
performance, transfer, and human behavioral and neural alignment over the rest of training.

Four from-scratch model families are trained once as baselines, then forked at an onset epoch and perturbed for
one epoch with **image noise**, **label shuffle**, or **uniform images**:

| Family | Data / objective | Epochs | Onsets (early / mid / late) |
|---|---|---|---|
| ResNet-50 (RN50) | ImageNet-1k, supervised | 90 | 5 / 30 / 70 |
| ViT-B/16 | ImageNet-1k, supervised | 90 | 5 / 30 / 70 |
| ConvNeXt-B | ImageNet-1k, supervised | 90 | 5 / 30 / 70 |
| CLIP ViT-B/32 | YFCC15M, contrastive | 32 | 5 / 13 / 28 |

A second setting fine-tunes CLIP-HBA (a CLIP ViT-L/14 with DoRA adapters regressed onto 66-d SPoSE embeddings;
Zhao et al., 2025, arXiv:2502.04658) with a perturbation at every onset epoch (appendix).

## Setup

Requirements: Linux, Python 3.11, and conda. Regenerating the figures needs only a CPU. Training and the
evaluations need an NVIDIA GPU with CUDA 12.1 (the from-scratch families were trained on 4-8 GPUs).

```bash
conda env create -f environment.yml          # PyTorch 2.2.1 + CUDA 12.1 wheels and requirements.txt
conda activate temporal_perturbations
python -m nltk.downloader wordnet            # WordNet categories used by Figure 4's embedding maps
python -m pytest -q                          # sanity check (one test skips unless the datasets are configured)
```

Without conda: `pip install -r requirements.txt` into a Python 3.11 environment with PyTorch already installed.

Only for the evaluations and training: copy `pipeline/paths.example.yaml` to `pipeline/paths.yaml` and fill in
the dataset and checkpoint locations (see [Full reproduction](#full-reproduction)). The VTAB transfer datasets are
exported once by `pipeline/scripts/downloads/export_vtab_datasets.py`, which needs TensorFlow and
`task_adaptation`; install those in a separate environment, since TensorFlow conflicts with this one.

## Quick start: regenerate the paper figures

The aggregated results every figure reads are shipped in `pipeline/results/` (46 MB), so the figures, tables, and
the numbers quoted in the text regenerate on a CPU with no datasets, checkpoints, or `paths.yaml`:

```bash
python pipeline/scripts/visualization/fig2_performance_alignment.py   # writes pipeline/figures/Fig2_*.{pdf,png}
python pipeline/scripts/analysis/results_section_numbers.py           # numbers quoted in the Results text
```

or open `pipeline/notebook/pipeline.ipynb` and run all cells: sections 1-6 find their results already present
and skip, and section 7 draws every figure.

## What data is included

| Included | Location |
|---|---|
| SPoSE embeddings (1,854 THINGS objects x 66 dimensions) and the 48 / 1,806 splits | `pipeline/data/spose_embedding66d_rescaled*.csv` |
| SPoSE dimension labels | `pipeline/data/labels66.txt` |
| Human 48-object triplet RDM | `pipeline/data/RDM48_triplet.mat` |
| THINGS-fMRI ROI RDMs (27 ROIs, 3 subjects averaged) | `pipeline/data/things_fmri/reference_rdms/` |
| NOD ROI RDMs (14 ROIs) | `pipeline/data/nod/reference_rdms/` |
| Aggregated evaluation results read by the figures | `pipeline/results/` |
| CLIP-HBA 48 x 66 SPoSE-space embeddings | `pipeline/results/cliphba_embeddings/` |
| 2-D TriMap coordinates and 2-D PCA points of the embedding-map panels | `pipeline/results/trimap/`, `pipeline/results/clip_uniform_collapse*/pca_points.npz` |

Not included: image datasets (THINGS, THINGS-fMRI stimuli, NOD, ImageNet, YFCC15M, transfer datasets; see
`pipeline/datasets.yaml` for sources), model checkpoints, and the high-dimensional ImageNet-validation features
(768-2,048 dimensions per image, 24 GB). The embedding-map panels in Figures 3 and 4 are therefore drawn from the
saved 2-D coordinates; recomputing the coordinates themselves requires the checkpoints and ImageNet validation set
(`visualization/trimap_imagenet.py`, `visualization/trimap_recovery.py`, with `FORCE_ANALYSIS = True` in the notebook).
The reference RDMs can be rebuilt from the raw fMRI releases with `pipeline/scripts/preprocessing/`.

## Figures and the scripts that make them

| Paper element | Script (`pipeline/scripts/`) | Output (`pipeline/figures/`) |
|---|---|---|
| Figure 1 | schematic, not data-generated | |
| Figure 2 | `visualization/fig2_performance_alignment.py` | `Fig2_performance_alignment_defs_in_caption` |
| Figure 3 | `visualization/fig3_storyboard.py` | `Fig3_storyboard_defs_in_caption` |
| Figure 4 | `visualization/fig5_linear_probe_transfer.py` | `Fig6_linear_probe_generalization_defs_in_caption` |
| CLIP-HBA fine-tuning (appendix) | `visualization/figS_spose_dimensions.py` | `Fig4_spose_dimensions_defs_in_caption` |
| Recovery after the perturbed epoch (appendix) | `visualization/fig2_performance_alignment.py` | `FigS_recovery_after_onset` |
| Category fidelity and separability (appendix) | `visualization/fig3_storyboard.py` | `FigS_fidelity_separability` |
| Recovery trajectories (appendix) | `visualization/fig4_recovery_trajectories.py` | `Fig5_recovery_trajectories_defs_in_caption` |
| Transfer retention and distance to baseline (appendix) | `visualization/fig5_linear_probe_transfer.py` | `FigS_transfer_retention_distance` |
| Transfer-probe tables (appendix) | `visualization/table_transfer_probes.py` | `table_transfer_probes.tex`, `transfer_probe_scores.csv` |
| CKA robustness (appendix) | `visualization/figS_cka_behavioral_alignment.py` | `FigS_cka_behavioral_alignment` |
| Baseline behavioral alignment (appendix) | `visualization/figS_baseline_behavioral_alignment.py` | `Fig1b_baseline_alignment` |
| Baseline neural alignment, THINGS-fMRI / NOD (appendix) | `visualization/figS_baseline_neural_alignment.py` | `Fig_neural_trajectories_by_roi_{things,nod}` |
| Neural alignment by ROI, THINGS-fMRI / NOD (appendix) | `visualization/figS_neural_alignment_by_roi.py` | `Fig_neural_critical_window_by_roi_{things,nod}_canonical` |
| CLIP uniform-image collapse (appendix) | `visualization/figS_clip_uniform_collapse.py` | `FigS_clip_uniform_collapse` |
| Numbers quoted in the Results text | `analysis/results_section_numbers.py` (stdout) | |
| Structure-correlation robustness | `analysis/structure_robustness.py` | `table_structure_robustness.tex` |

Output file names keep the names the manuscript's `\includegraphics` uses.

## Full reproduction

### 1. Configure paths

All external locations (datasets, checkpoints) are read from one file; no path is hard-coded.

```bash
cp pipeline/paths.example.yaml pipeline/paths.yaml   # then edit
```

Any key can be overridden with an environment variable `TP_<KEY>`. `pipeline/datasets.yaml` lists where to obtain
each dataset. Outputs default to `pipeline/results/`, `pipeline/figures/`, and `pipeline/cache/`.

### 2. Train

**From-scratch families.** `pipeline/configs/scratch_runs.yaml` holds the exact arguments of every run (baselines,
perturbed forks, and the sub-epoch `*_recovery_dynamics` runs). Train each baseline first; each perturbed run
resumes from its baseline's checkpoint at epoch *E*-1, is perturbed for epoch *E* only, and trains unperturbed to
the end of the schedule.

```bash
torchrun --nproc_per_node <gpus> pipeline/scripts/training/train_from_scratch.py <args from scratch_runs.yaml>
```

The GPU count is part of the recipe (it fixes the effective batch size through gradient accumulation).

**Fork exactness check.** `pipeline/scripts/training/fork_check.py` runs an unperturbed fork of a baseline (same
checkpoint and arguments as the perturbed runs) for two whole epochs from the onset and checks that its logged
training loss is bit-identical to the baseline's at every step both runs logged (every 50 steps). On a Slurm cluster,
`ACCOUNT=<account> PARTITION=<partition> GPU_TYPE=<baseline GPU type> pipeline/slurm/submit_fork_checks.sh` submits one job per family;
each writes `<runs_root>/fork_checks/fork_check_<family>_<onset>/fork_check_report.json`.

**CLIP-HBA.** `pipeline/scripts/training/train_cliphba.py --config pipeline/configs/training_config.yaml` trains one
run; `run_cliphba_perturbation.py` runs one perturbation task (onset, length, type, seed).

### 3. Evaluate

Each producer writes to `pipeline/results/<subdir>` (commands are `python pipeline/scripts/evaluation/<script>`):

| Results | Command |
|---|---|
| `imagenet_val_curves.json`, `lr_schedule.json` | `extract_imagenet_val_curves.py`, `extract_lr_schedule.py` (from the training logs) |
| `spose_chance_floor.json` | `compute_spose_chance_floor.py` |
| `posthoc_rdm1854/` (behavioral alignment, 1,854 THINGS images) | `run_behavioral_rsa.py`; sub-epoch recovery runs add `--run_name <run>_recovery_dynamics --sub_epoch`; RN50 uses `run_behavioral_rsa_bn_recalibrated.py` |
| `bn_recalibrated_val/`, `bn_recalibrated_val_50k/`, `onset_val_10k/` (ImageNet top-1) | `run_bn_recalibrated_val.py` (see its docstring for the three protocols) |
| `subepoch_imagenet_val/` | `run_subepoch_imagenet_val.py` |
| `linear_probe_eval_scratch_std/`, `linear_probe_eval_scratch_bn_recal_std/` (21-dataset transfer probes; 1000-class and 16-category ImageNet probes) | `run_linear_probe_eval.py`, `run_linear_probe_eval_bn_recalibrated.py` (after `downloads/export_vtab_datasets.py`) |
| `imagenet_ood_eval_scratch_clip_cls/` (CLIP ViT-B/32 ImageNet probes) | `run_clip_imagenet_ood_probe.py`; the dense epoch grid: `run_clip_imagenet_probe.py` |
| `model_to_model_rsa_scratch_*/` (distance from baseline) | `run_model_to_model_rsa.py --architecture {RN50 --recalibrate_bn, ViT-B/16, ConvNeXt-B, "CLIP ViT-B/32"}` |
| `frrsa/` (layer-reweighted neural RSA, THINGS-fMRI and NOD) | `run_frrsa.py --baseline_step 5 --include_onset`; `run_frrsa.py --arch resnet50 --recalibrate_bn` |
| `cliphba_embeddings/` | `extract_cliphba_things_embeddings.py --offset 0`, then `--offset 10` |
| `imagenet_embeddings/` | `extract_imagenet_embeddings.py`, `extract_imagenet_embeddings_bn_recalibrated.py` |

Then the analyses in `pipeline/scripts/analysis/` (`analyze_global_local_structure.py`,
`analyze_distance_levels.py`, `analyze_clip_uniform_collapse.py`) and the embedding maps
(`visualization/trimap_imagenet.py`, `visualization/trimap_recovery.py`).

## Methodological notes

- **Evaluation preprocessing follows each checkpoint's training recipe** (`temporal_perturbations/data/eval_transforms.py`):
  ImageNet normalization with Resize(256)/CenterCrop(224) for the supervised families (bilinear for RN50, bicubic
  otherwise), CLIP normalization with Resize(224) for CLIP ViT-B/32. Result files carry the recipe as a tag.
- **ResNet-50 BatchNorm recalibration.** A checkpoint saved at the end of a perturbed epoch stores BatchNorm
  statistics of noise or blank images; RN50 evaluations re-estimate them on 8 clean ImageNet training images per class.
- **CLIP ViT-B/32 is scored in the 768-d pre-projection CLS space**, the analog of the supervised models' pooled
  penultimate features (`temporal_perturbations/models/clip_yfcc/feature_space.py`).
- **Linear probes** z-score the features with training-split statistics, select the L2 strength per dataset and
  checkpoint on a held-out 20% split with a cold-started sweep (Kornblith et al., 2019; Radford et al., 2021), then
  refit on the full training split (`temporal_perturbations/inference/linear_probe_eval.py`). Every probe record carries
  the recipe markers `c_selection: val_sweep_v2` and `feature_scaling: standardized`, and the figure scripts refuse
  records without them (`arch_runs.require_current_probe`).

## Layout

```
temporal_perturbations/     library: models, datasets, perturbations, evaluation cores, trainers
  paths.py                  the single source of filesystem locations
  scratch_training/         Lightning trainer for the four from-scratch families
  training/                 CLIP-HBA fine-tuning loop
pipeline/
  paths.example.yaml        template for pipeline/paths.yaml
  datasets.yaml             dataset manifest (sources, paths keys)
  configs/                  training and evaluation configs; scratch_runs.yaml
  data/                     small reference data (SPoSE embeddings, THINGS-fMRI / NOD reference RDMs)
  notebook/pipeline.ipynb   end-to-end wrapper
  scripts/{downloads,preprocessing,training,evaluation,analysis,visualization}/
  results/                  aggregated results read by the figures
tests/                      pytest suite (`python -m pytest -q`)
```
