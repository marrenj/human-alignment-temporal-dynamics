#!/bin/bash
# One fork check: an unperturbed fork of a baseline for EPOCHS whole epochs, then an exact
# loss comparison against the baseline (pipeline/scripts/training/fork_check.py).
#
# Submit through submit_fork_checks.sh, which sets the account, partition, GPU count and
# resources per family and passes FAMILY / ONSET / EPOCHS / TP_REPO via --export.
#SBATCH --job-name=fork_check
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

: "${FAMILY:?FAMILY must be set (resnet50, vitb16, convnext_b, clip_vitb32)}"
: "${TP_REPO:?TP_REPO must point at the repository checkout}"
ONSET=${ONSET:-early}
EPOCHS=${EPOCHS:-2}

# Environment activation; override TP_ENV_SETUP if your environment is set up differently.
: "${TP_ENV_SETUP:=module load Anaconda3 && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate temporal_perturbations}"
set +u
eval "$TP_ENV_SETUP"
set -u

cd "$TP_REPO"
echo "Host $(hostname), job ${SLURM_JOB_ID:-?}: fork check ${FAMILY} onset ${ONSET}, ${EPOCHS} epochs"
nvidia-smi --query-gpu=index,name,driver_version --format=csv
python -c "import torch, pytorch_lightning as pl; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'lightning', pl.__version__)"

python pipeline/scripts/training/fork_check.py run --family "$FAMILY" --onset "$ONSET" --epochs "$EPOCHS"
python pipeline/scripts/training/fork_check.py compare --family "$FAMILY" --onset "$ONSET" --epochs "$EPOCHS"
