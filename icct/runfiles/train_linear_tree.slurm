#!/bin/bash
# Train Linear Tree (behavioral cloning from the oracle, linear-model leaves) for a
# given environment (seeds 0-4).
# Uses the seed-matched MLP-Max oracle trained by train_oracle.slurm: distillation
# seed X is taught by the oracle trained with seed X. train_oracle.slurm MUST be run
# for this ENV_NAME first, using the same LOG_ROOT.
#
# Usage: ENV_NAME=cart sbatch train_linear_tree.slurm
#
#SBATCH --job-name=linear_tree
#SBATCH --output=/scratch/gilbreth/%u/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --account=rpaleja
#SBATCH --qos=training
#SBATCH --partition=training
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --array=0-4

module load conda
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
eval "$(mamba shell hook --shell bash)"
mamba activate icct_jmlr

curruser=$(whoami)
LOG_ROOT=${LOG_ROOT:-/scratch/gilbreth/$curruser/log}
cd "$SLURM_SUBMIT_DIR"

case $ENV_NAME in
  cart)               ORACLE_DIR=cart_oracle ;;
  lunar)              ORACLE_DIR=lunar_oracle ;;
  lane_keeping)        ORACLE_DIR=lk_oracle ;;
  ring_accel)          ORACLE_DIR=ring_accel_oracle ;;
  ring_lane_changing)  ORACLE_DIR=ring_lc_oracle ;;
  figure8)             ORACLE_DIR=figure8_oracle ;;
  *)
    echo "Unknown ENV_NAME: $ENV_NAME"; exit 1 ;;
esac

SAVE_BASE=$LOG_ROOT/${ENV_NAME}_linear_tree

SEED=$SLURM_ARRAY_TASK_ID
ORACLE_PATH=$LOG_ROOT/${ORACLE_DIR}/seed${SEED}
echo "=== Linear Tree: env=$ENV_NAME seed=$SEED oracle=$ORACLE_PATH/best_model ==="
python -u icct/dagger/train_linear_tree.py \
  --env_name $ENV_NAME \
  --oracle_load_path $ORACLE_PATH \
  --oracle_load_file best_model \
  --seed $SEED \
  --n_rollouts 50 \
  --eval_episodes 20 \
  --save ${SAVE_BASE}/seed${SEED} \
  --gpu
