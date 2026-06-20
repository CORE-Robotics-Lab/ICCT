#!/bin/bash
#SBATCH --job-name=mlp_lower_sp
#SBATCH --output=/scratch/gilbreth/%u/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --time=23:30:00
#SBATCH --account=rpaleja
#SBATCH --qos=training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=training
#SBATCH --mem=32G
#SBATCH --array=0-4

module load conda
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
eval "$(mamba shell hook --shell bash)"
mamba activate icct_jmlr

curruser=$(whoami)
LOG_ROOT=${LOG_ROOT:-/scratch/gilbreth/$curruser/log}
cd "$SLURM_SUBMIT_DIR"

# MLP-Lower: small-sized MLP (~same active params as ICCT at deployment)
# with L1 sparsity regularization
case $ENV_NAME in
  cart)
    LR=5e-4
    BATCH_SIZE=1024
    MIN_REWARD=900
    SAVE_PATH=$LOG_ROOT/ip_mlp_lower_l1
    SPARSITY_COEFF=1e-4
    ;;
  lunar)
    LR=5e-4
    BATCH_SIZE=256
    MIN_REWARD=225
    SAVE_PATH=$LOG_ROOT/ll_mlp_lower_l1
    SPARSITY_COEFF=1e-4
    ;;
  lane_keeping)
    LR=3e-4
    BATCH_SIZE=1024
    MIN_REWARD=420
    SAVE_PATH=$LOG_ROOT/lk_mlp_lower_l1
    SPARSITY_COEFF=5e-4
    ;;
  figure8)
    LR=6e-4
    BATCH_SIZE=1024
    MIN_REWARD=900
    SAVE_PATH=$LOG_ROOT/fig8_mlp_lower_l1
    SPARSITY_COEFF=1e-4
    ;;
  ring_accel)
    LR=5e-4
    BATCH_SIZE=1024
    MIN_REWARD=120
    SAVE_PATH=$LOG_ROOT/ring_accel_mlp_lower_l1
    SPARSITY_COEFF=1e-4
    ;;
  ring_lane_changing)
    LR=5e-4
    BATCH_SIZE=1024
    MIN_REWARD=1200
    SAVE_PATH=$LOG_ROOT/ring_lc_mlp_lower_l1
    SPARSITY_COEFF=1e-4
    ;;
  *)
    echo "Unknown ENV_NAME: $ENV_NAME"
    exit 1
    ;;
esac

SEED=$SLURM_ARRAY_TASK_ID
echo "=== Training MLP-Lower sparse: env=$ENV_NAME seed=$SEED ==="
python -u icct/runfiles/train.py \
  --env_name $ENV_NAME \
  --policy_type mlp \
  --mlp_size small \
  --seed $SEED \
  --lr $LR \
  --buffer_size 1000000 \
  --batch_size $BATCH_SIZE \
  --gamma 0.99 \
  --tau 0.01 \
  --learning_starts 10000 \
  --eval_freq 1500 \
  --min_reward $MIN_REWARD \
  --training_steps 500000 \
  --log_interval 4 \
  --save_path ${SAVE_PATH}/seed${SEED} \
  --mlp_sparsity_reg \
  --mlp_sparsity_type l1 \
  --mlp_sparsity_coeff $SPARSITY_COEFF \
  --gpu
