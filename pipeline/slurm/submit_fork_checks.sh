#!/bin/bash
# Submit the fork checks on a Slurm cluster: one job per family, each on its baseline's GPU count
# (the count sets DDP sharding and grad accumulation, so it must match the baseline).
#
#   ACCOUNT=<account> GPU_TYPE=<gres type of the baseline GPUs> \
#       pipeline/slurm/submit_fork_checks.sh                  # all four families
#   ACCOUNT=<account> pipeline/slurm/submit_fork_checks.sh resnet50 vitb16
#
# Settings (env vars). Cluster-specific ones can live in pipeline/slurm/cluster.env
# (gitignored; see cluster.env.example), which is read first, so a variable given on the
# command line still wins:
#   ACCOUNT      Slurm account (required; `sacctmgr show assoc user=$USER` lists yours)
#   PARTITION    GPU partition (default batch_gpu)
#   QOS          Slurm QOS, passed to sbatch --qos (default: the account's default)
#   GPU_TYPE     GPU gres type; set it to the model the baselines trained on, since
#                bit-exact results are only expected on the same GPU model (default: any)
#   NODELIST     node(s) to run on, passed to sbatch --nodelist (default: any)
#   DRY_RUN=1    only check the requests with `sbatch --test-only` (reports whether and
#                when each job would start); nothing is queued
#   ONSET        early | mid | late (default early)
#   EPOCHS       whole epochs to run the fork for, from the onset epoch (default 2)
#   TIME         wall time per job (default: per family below; two epochs plus validation)
#   CPUS_PER_GPU / MEM_PER_GPU   per-GPU CPU cores and memory (default 10 / 48G)
#   MAIL_USER / MAIL_TYPE   Slurm email notifications (default: none / ALL when MAIL_USER is set)
#   TP_ENV_SETUP command that activates the Python environment (see fork_check.slurm)
# Dataset and run locations come from pipeline/paths.yaml (or TP_<KEY> env vars):
# runs_root, imagenet_train_dir, imagenet_val_dir, yfcc15m_parquet_dir, val_dataset_root.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Local cluster settings; entries are written as VAR=${VAR:-value} so the command line wins.
if [ -f "$REPO/pipeline/slurm/cluster.env" ]; then
    # shellcheck source=/dev/null
    source "$REPO/pipeline/slurm/cluster.env"
fi
: "${ACCOUNT:?set ACCOUNT (or put it in pipeline/slurm/cluster.env)}"
PARTITION=${PARTITION:-batch_gpu}
GPU_TYPE=${GPU_TYPE:-}
NODELIST=${NODELIST:-}
QOS=${QOS:-}
MAIL_USER=${MAIL_USER:-}
MAIL_TYPE=${MAIL_TYPE:-ALL}
ONSET=${ONSET:-early}
EPOCHS=${EPOCHS:-2}
CPUS_PER_GPU=${CPUS_PER_GPU:-10}
MEM_PER_GPU=${MEM_PER_GPU:-48G}
LOG_DIR="$REPO/pipeline/slurm/logs"
mkdir -p "$LOG_DIR"

# GPU counts of the baselines in pipeline/configs/scratch_runs.yaml (fork_check.py run
# re-checks this against the allocation and refuses to start on a mismatch).
declare -A GPUS=([resnet50]=4 [vitb16]=6 [convnext_b]=6 [clip_vitb32]=8)
# Default wall time for EPOCHS=2; scale with EPOCHS or set TIME from your baselines' epoch times.
declare -A DEFAULT_TIME=([resnet50]=12:00:00 [vitb16]=16:00:00 [convnext_b]=24:00:00 [clip_vitb32]=48:00:00)

families=("$@")
[ ${#families[@]} -eq 0 ] && families=(resnet50 vitb16 convnext_b clip_vitb32)
# Check every name before submitting anything, so a typo or a stray flag queues nothing.
for family in "${families[@]}"; do
    if [ -z "${GPUS[$family]+x}" ]; then
        echo "unknown family '$family' (resnet50, vitb16, convnext_b, clip_vitb32); settings go in env vars, e.g. DRY_RUN=1" >&2
        exit 1
    fi
done

test_only=()
[ "${DRY_RUN:-0}" = 1 ] && test_only=(--test-only)

for family in "${families[@]}"; do
    n=${GPUS[$family]}
    mem_num=${MEM_PER_GPU%G}
    sbatch "${test_only[@]}" \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --gres="gpu:${GPU_TYPE:+$GPU_TYPE:}$n" \
        ${NODELIST:+--nodelist="$NODELIST"} \
        ${QOS:+--qos="$QOS"} \
        ${MAIL_USER:+--mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE"} \
        --cpus-per-task=$((CPUS_PER_GPU * n)) \
        --mem="$((mem_num * n))G" \
        --time="${TIME:-${DEFAULT_TIME[$family]}}" \
        --job-name="fork_check_${family}_${ONSET}" \
        --output="$LOG_DIR/%x_%j.out" \
        --export="ALL,FAMILY=$family,ONSET=$ONSET,EPOCHS=$EPOCHS,TP_REPO=$REPO" \
        "$REPO/pipeline/slurm/fork_check.slurm"
done
