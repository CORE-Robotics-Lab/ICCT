#!/bin/bash
#SBATCH --job-name=cddt
#SBATCH --output=/scratch/gilbreth/%u/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=2
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

case $ENV_NAME in
  cart)
    LR=5e-4
    BATCH_SIZE=1024
    MIN_REWARD=900
    NUM_LEAVES=4
    LEARNING_STARTS=10000
    TRAINING_STEPS=500000
    SAVE_PATH=$LOG_ROOT/ip_cddt_ctrl_sparse
    ;;
  lunar)
    LR=5e-4
    BATCH_SIZE=256
    MIN_REWARD=225
    NUM_LEAVES=8
    LEARNING_STARTS=10000
    TRAINING_STEPS=500000
    SAVE_PATH=$LOG_ROOT/ll_cddt_ctrl_sparse
    ;;
  lane_keeping)
    LR=3e-4
    BATCH_SIZE=1024
    MIN_REWARD=420
    NUM_LEAVES=16
    LEARNING_STARTS=10000
    TRAINING_STEPS=500000
    SAVE_PATH=$LOG_ROOT/lk_cddt_ctrl_sparse
    ;;
  figure8)
    LR=6e-4
    BATCH_SIZE=1024
    MIN_REWARD=900
    NUM_LEAVES=16
    LEARNING_STARTS=10000
    TRAINING_STEPS=500000
    SAVE_PATH=$LOG_ROOT/fig8_cddt_ctrl_sparse
    ;;
  ring_accel)
    LR=5e-4
    BATCH_SIZE=1024
    MIN_REWARD=120
    NUM_LEAVES=16
    LEARNING_STARTS=5000
    TRAINING_STEPS=100000
    SAVE_PATH=$LOG_ROOT/ring_accel_cddt_ctrl_sparse
    ;;
  ring_lane_changing)
    LR=5e-4
    BATCH_SIZE=1024
    MIN_REWARD=1200
    NUM_LEAVES=16
    LEARNING_STARTS=10000
    TRAINING_STEPS=500000
    SAVE_PATH=$LOG_ROOT/ring_lc_cddt_ctrl_sparse
    ;;
  *)
    echo "Unknown ENV_NAME: $ENV_NAME"
    exit 1
    ;;
esac

SEED=$SLURM_ARRAY_TASK_ID
echo "=== Training CDDT-controllers + L1 sparse: env=$ENV_NAME seed=$SEED ==="
python -u icct/runfiles/train.py \
  --env_name $ENV_NAME \
  --policy_type ddt \
  --seed $SEED \
  --lr $LR \
  --buffer_size 1000000 \
  --batch_size $BATCH_SIZE \
  --gamma 0.99 \
  --tau 0.01 \
  --learning_starts $LEARNING_STARTS \
  --eval_freq 1500 \
  --min_reward $MIN_REWARD \
  --training_steps $TRAINING_STEPS \
  --log_interval 4 \
  --save_path ${SAVE_PATH}/seed${SEED} \
  --submodels \
  --num_leaves $NUM_LEAVES \
  --sparse_submodel_type 1 \
  --l1_reg_coeff 1e-4 \
  --l1_hard_attn \
  --gpu
