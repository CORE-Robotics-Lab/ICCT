#!/bin/bash
# Train ICCT variants (M1–M5) for a given environment and method (seeds 0-4).
#
# METHOD controls which variant to train:
#   cddt         — M1: CDDT (soft routing, no submodels)
#   cddt_ctrl    — M2: CDDT-controllers (soft routing + submodels)
#   icct_static  — M3: ICCT-static (hard routing, no submodels)
#   icct         — M4: ICCT-complete (hard routing + submodels)
#   icct_l1      — M5-a: ICCT-L1-sparse (hard routing + L1-regularized submodels)
#   icct_1feat   — M5-b: ICCT-1-feature (hard routing + 1-feature-per-leaf submodels)
#   icct_2feat   — M5-b: ICCT-2-feature
#   icct_3feat   — M5-b: ICCT-3-feature
#
# Usage: METHOD=icct ENV_NAME=cart sbatch train_icct.slurm
#
#SBATCH --job-name=icct
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

# Per-environment hyperparameters (shared across all ICCT methods)
case $ENV_NAME in
  cart)
    LR=5e-4; BATCH_SIZE=1024; MIN_REWARD=900; NUM_LEAVES=4
    ;;
  lunar)
    LR=5e-4; BATCH_SIZE=256;  MIN_REWARD=225; NUM_LEAVES=8
    ;;
  lane_keeping)
    LR=3e-4; BATCH_SIZE=1024; MIN_REWARD=420; NUM_LEAVES=16
    ;;
  ring_accel)
    LR=5e-4; BATCH_SIZE=1024; MIN_REWARD=120; NUM_LEAVES=16
    TRAINING_STEPS=100000; LEARNING_STARTS=5000
    ;;
  ring_lane_changing)
    LR=5e-4; BATCH_SIZE=1024; MIN_REWARD=1200; NUM_LEAVES=16
    ;;
  figure8)
    LR=6e-4; BATCH_SIZE=1024; MIN_REWARD=900;  NUM_LEAVES=16
    ;;
  *)
    echo "Unknown ENV_NAME: $ENV_NAME"; exit 1 ;;
esac

# Default training steps
TRAINING_STEPS=${TRAINING_STEPS:-500000}
LEARNING_STARTS=${LEARNING_STARTS:-10000}

# Per-method flags
case ${METHOD:-icct} in
  cddt)
    EXTRA_FLAGS=""
    SAVE_PATH=$LOG_ROOT/${ENV_NAME}_cddt
    ;;
  cddt_ctrl)
    EXTRA_FLAGS="--submodels --sparse_submodel_type 0"
    SAVE_PATH=$LOG_ROOT/${ENV_NAME}_cddt_ctrl
    ;;
  icct_static)
    EXTRA_FLAGS="--hard_node"
    SAVE_PATH=$LOG_ROOT/${ENV_NAME}_icct_static
    ;;
  icct)
    EXTRA_FLAGS="--hard_node --submodels --sparse_submodel_type 0"
    SAVE_PATH=$LOG_ROOT/${ENV_NAME}_icct
    ;;
  icct_l1)
    EXTRA_FLAGS="--hard_node --submodels --sparse_submodel_type 1 --l1_reg_coeff 1e-4 --l1_hard_attn"
    SAVE_PATH=$LOG_ROOT/${ENV_NAME}_icct_l1
    ;;
  icct_1feat)
    EXTRA_FLAGS="--hard_node --submodels --sparse_submodel_type 2 --num_sub_features 1 --argmax_tau 1.0"
    SAVE_PATH=$LOG_ROOT/${ENV_NAME}_icct_1feat
    ;;
  icct_2feat)
    EXTRA_FLAGS="--hard_node --submodels --sparse_submodel_type 2 --num_sub_features 2 --argmax_tau 1.0"
    SAVE_PATH=$LOG_ROOT/${ENV_NAME}_icct_2feat
    ;;
  icct_3feat)
    EXTRA_FLAGS="--hard_node --submodels --sparse_submodel_type 2 --num_sub_features 3 --argmax_tau 1.0"
    SAVE_PATH=$LOG_ROOT/${ENV_NAME}_icct_3feat
    ;;
  *)
    echo "Unknown METHOD: $METHOD"; exit 1 ;;
esac

# Leaf count varies by METHOD for cart/lunar (per paper Table 1/Table 2).
# lane_keeping/ring_accel/ring_lane_changing/figure8 are uniformly 16 leaves
# for every method, so no override is needed there.
case $ENV_NAME in
  cart)
    case ${METHOD:-icct} in
      cddt|cddt_ctrl|icct|icct_3feat) NUM_LEAVES=2 ;;
      icct_static)                    NUM_LEAVES=32 ;;
      icct_l1|icct_2feat)             NUM_LEAVES=4 ;;
      icct_1feat)                     NUM_LEAVES=8 ;;
    esac
    ;;
  lunar)
    case ${METHOD:-icct} in
      icct_static) NUM_LEAVES=32 ;;
      *)           NUM_LEAVES=8 ;;
    esac
    ;;
esac

SEED=$SLURM_ARRAY_TASK_ID
echo "=== ICCT (method=${METHOD:-icct}): env=$ENV_NAME seed=$SEED leaves=$NUM_LEAVES ==="
python -u icct/runfiles/train.py \
  --env_name $ENV_NAME \
  --policy_type ddt \
  --seed $SEED \
  --num_leaves $NUM_LEAVES \
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
  $EXTRA_FLAGS \
  --gpu
