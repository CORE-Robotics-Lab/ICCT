#!/bin/bash
#SBATCH --job-name=viper_small
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

# VIPER-Small: max_depth matches best ICCT leaf count per env
# cart=depth1(2 leaves), lunar=depth3(8 leaves), others=depth4(16 leaves)

declare -A ORACLE_MAP
declare -A DEPTH_MAP
ORACLE_MAP[cart]=cart_oracle;        DEPTH_MAP[cart]=1
ORACLE_MAP[lunar]=lunar_oracle;      DEPTH_MAP[lunar]=3
ORACLE_MAP[lane_keeping]=lk_oracle;          DEPTH_MAP[lane_keeping]=4
ORACLE_MAP[figure8]=figure8_oracle;          DEPTH_MAP[figure8]=4
ORACLE_MAP[ring_accel]=ring_accel_oracle;    DEPTH_MAP[ring_accel]=4
ORACLE_MAP[ring_lane_changing]=ring_lc_oracle; DEPTH_MAP[ring_lane_changing]=4

ENVS=(cart lunar lane_keeping figure8 ring_accel ring_lane_changing)
ENV=${ENVS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))
DEPTH=${DEPTH_MAP[$ENV]}
ORACLE_DIR=${ORACLE_MAP[$ENV]}
ORACLE_PATH=$LOG_ROOT/${ORACLE_DIR}/seed${SEED}
SAVE_BASE=$LOG_ROOT/${ENV}_viper_small

echo "=== VIPER-Small: env=$ENV seed=$SEED max_depth=$DEPTH oracle=$ORACLE_PATH/best_model ==="
python -u icct/dagger/train_dagger.py \
  --env_name $ENV \
  --oracle_load_path $ORACLE_PATH \
  --oracle_load_file best_model \
  --seed $SEED \
  --max_depth $DEPTH \
  --n_rollouts 10 \
  --iterations 80 \
  --max_samples 1000000 \
  --q_dagger \
  --n_q_samples 10 \
  --eval_episodes 20 \
  --save ${SAVE_BASE}/seed${SEED}
