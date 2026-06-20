#!/bin/bash
#SBATCH --job-name=viper_big
#SBATCH --output=/scratch/gilbreth/%u/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --account=rpaleja
#SBATCH --qos=training
#SBATCH --partition=training
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --array=0-29

module load conda
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
eval "$(mamba shell hook --shell bash)"
mamba activate icct_jmlr

curruser=$(whoami)
LOG_ROOT=${LOG_ROOT:-/scratch/gilbreth/$curruser/log}
cd "$SLURM_SUBMIT_DIR"

MAX_DEPTH=6  # up to 64 leaves

declare -A ORACLE_MAP
ORACLE_MAP[cart]=cart_oracle
ORACLE_MAP[lunar]=lunar_oracle
ORACLE_MAP[lane_keeping]=lk_oracle
ORACLE_MAP[figure8]=figure8_oracle
ORACLE_MAP[ring_accel]=ring_accel_oracle
ORACLE_MAP[ring_lane_changing]=ring_lc_oracle

ENVS=(cart lunar lane_keeping figure8 ring_accel ring_lane_changing)
ENV=${ENVS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))
ORACLE_DIR=${ORACLE_MAP[$ENV]}
ORACLE_PATH=$LOG_ROOT/${ORACLE_DIR}/seed${SEED}
SAVE_BASE=$LOG_ROOT/${ENV}_viper_big

echo "=== VIPER-Big: env=$ENV seed=$SEED max_depth=$MAX_DEPTH oracle=$ORACLE_PATH/best_model ==="
python -u icct/dagger/train_dagger.py \
  --env_name $ENV \
  --oracle_load_path $ORACLE_PATH \
  --oracle_load_file best_model \
  --seed $SEED \
  --max_depth $MAX_DEPTH \
  --n_rollouts 10 \
  --iterations 80 \
  --max_samples 1000000 \
  --q_dagger \
  --n_q_samples 10 \
  --eval_episodes 20 \
  --save ${SAVE_BASE}/seed${SEED}
