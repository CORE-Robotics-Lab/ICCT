#!/bin/bash
#SBATCH --job-name=hinton_distil
#SBATCH --output=/scratch/gilbreth/%u/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --account=rpaleja
#SBATCH --qos=training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=training
#SBATCH --mem=16G
#SBATCH --array=0-4

module load conda
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
eval "$(mamba shell hook --shell bash)"
mamba activate icct_jmlr

curruser=$(whoami)
LOG_ROOT=${LOG_ROOT:-/scratch/gilbreth/$curruser/log}
cd "$SLURM_SUBMIT_DIR"

# Per-env config: oracle dir, num_leaves (paper values), save path
case $ENV_NAME in
  cart)
    ORACLE_DIR=cart_oracle
    NUM_LEAVES=4
    N_ROLLOUTS=200
    SAVE_PATH=$LOG_ROOT/ip_hinton_l4
    ;;
  lunar)
    ORACLE_DIR=lunar_oracle
    NUM_LEAVES=8
    N_ROLLOUTS=200
    SAVE_PATH=$LOG_ROOT/ll_hinton_l8
    ;;
  lane_keeping)
    ORACLE_DIR=lk_oracle
    NUM_LEAVES=16
    N_ROLLOUTS=200
    SAVE_PATH=$LOG_ROOT/lk_hinton_l16_fix
    ;;
  figure8)
    ORACLE_DIR=figure8_oracle
    NUM_LEAVES=16
    N_ROLLOUTS=100
    SAVE_PATH=$LOG_ROOT/fig8_hinton_l16
    ;;
  ring_accel)
    ORACLE_DIR=ring_accel_oracle
    NUM_LEAVES=16
    N_ROLLOUTS=100
    SAVE_PATH=$LOG_ROOT/ring_accel_hinton_l16
    ;;
  ring_lane_changing)
    ORACLE_DIR=ring_lc_oracle
    NUM_LEAVES=16
    N_ROLLOUTS=100
    SAVE_PATH=$LOG_ROOT/ring_lc_hinton_l16
    ;;
  *)
    echo "Unknown ENV_NAME: $ENV_NAME"
    exit 1
    ;;
esac

SEED=$SLURM_ARRAY_TASK_ID
ORACLE_PATH=$LOG_ROOT/${ORACLE_DIR}/seed${SEED}
echo "=== Hinton distillation: env=$ENV_NAME seed=$SEED leaves=$NUM_LEAVES oracle=$ORACLE_PATH/best_model ==="
python -u icct/dagger/train_hinton.py \
  --env_name $ENV_NAME \
  --oracle_load_path $ORACLE_PATH \
  --oracle_load_file best_model \
  --seed $SEED \
  --num_leaves $NUM_LEAVES \
  --n_rollouts $N_ROLLOUTS \
  --epochs 10000 \
  --batch_size 256 \
  --lr 3e-4 \
  --lam 0.1 \
  --log_every 500 \
  --eval_freq 500 \
  --n_eval_episodes 5 \
  --save_path ${SAVE_PATH}/seed${SEED} \
  --gpu
