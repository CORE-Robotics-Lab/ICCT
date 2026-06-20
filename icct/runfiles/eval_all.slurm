#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --account=rpaleja
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/gilbreth/%u/logs/%x_%j.out

module load conda
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
eval "$(mamba shell hook --shell bash)"
mamba activate icct_jmlr

# Set PROJECT to the repo root and LOG_ROOT to where models are saved
PROJECT=${PROJECT:-$SLURM_SUBMIT_DIR}
LOG_ROOT=${LOG_ROOT:-/scratch/gilbreth/$(whoami)/log}
RESULTS_FILE=$PROJECT/eval_results.txt
cd $PROJECT

echo "=====================================================" > $RESULTS_FILE
echo "Evaluation Results — $(date)" >> $RESULTS_FILE
echo "=====================================================" >> $RESULTS_FILE

run_eval() {
    local ENV=$1
    local METHOD=$2
    local LOG_DIR=$3       # absolute path to method dir (no seed suffix)
    local NUM_EPS=$4
    local MAX_SEEDS=$5

    echo "" >> $RESULTS_FILE
    echo "-----------------------------------------------------" >> $RESULTS_FILE
    echo "ENV: $ENV  |  METHOD: $METHOD" >> $RESULTS_FILE
    echo "-----------------------------------------------------" >> $RESULTS_FILE

    for seed in $(seq 0 $((MAX_SEEDS - 1))); do
        MODEL=${LOG_DIR}/seed${seed}/best_model.zip
        if [ ! -f "$MODEL" ]; then
            echo "  seed$seed: MISSING" >> $RESULTS_FILE
            continue
        fi

        echo -n "  seed$seed: " >> $RESULTS_FILE
        python -u icct/runfiles/test.py \
            --env_name $ENV \
            --seed 42 \
            --load_path ${LOG_DIR}/seed${seed} \
            --num_episodes $NUM_EPS \
            --load_file best_model \
            --nn \
            --gpu 2>&1 | \
        awk '
            /fuzzy results:/     { getline; getline; mean=$0; getline; std=$0 }
            END { printf "mean=%.2f std=%.2f\n", mean, std }
        ' >> $RESULTS_FILE
    done
}

NUM_EPS=20
S=5  # seeds

# Only SAC-trained (test.py-loadable) methods are covered here: the 8 ICCT/CDDT
# methods (train_icct.slurm) plus the MLP family (oracle/sparse/plain).
# DAgger/VIPER/Hinton save decision-tree/distilled models that test.py cannot
# load (test.py only does SAC.load(...)) -- those report their own eval metrics
# directly in their training logs instead.

declare -A ORACLE_PREFIX
declare -A MLP_PREFIX
ORACLE_PREFIX[cart]=cart;       MLP_PREFIX[cart]=ip
ORACLE_PREFIX[lunar]=lunar;     MLP_PREFIX[lunar]=ll
ORACLE_PREFIX[lane_keeping]=lk; MLP_PREFIX[lane_keeping]=lk
ORACLE_PREFIX[ring_accel]=ring_accel;        MLP_PREFIX[ring_accel]=ring_accel
ORACLE_PREFIX[ring_lane_changing]=ring_lc;    MLP_PREFIX[ring_lane_changing]=ring_lc
ORACLE_PREFIX[figure8]=figure8; MLP_PREFIX[figure8]=fig8

for ENV in cart lunar lane_keeping ring_accel ring_lane_changing figure8; do
  OP=${ORACLE_PREFIX[$ENV]}
  MP=${MLP_PREFIX[$ENV]}

  # ---- ICCT/CDDT family (train_icct.slurm: $LOG_ROOT/${ENV}_<method>) ----
  for METHOD in cddt cddt_ctrl icct_static icct icct_l1 icct_1feat icct_2feat icct_3feat; do
    run_eval $ENV $METHOD $LOG_ROOT/${ENV}_${METHOD} $NUM_EPS $S
  done

  # ---- CDDT-ctrl-L1 baseline ----
  run_eval $ENV cddt_ctrl_l1 $LOG_ROOT/${MP}_cddt_ctrl_sparse $NUM_EPS $S

  # ---- MLP family ----
  run_eval $ENV mlp_max       $LOG_ROOT/${OP}_oracle        $NUM_EPS $S
  run_eval $ENV mlp_max_l1    $LOG_ROOT/${MP}_mlp_sparse_l1 $NUM_EPS $S
  run_eval $ENV mlp_upper     $LOG_ROOT/${MP}_mlp_upper     $NUM_EPS $S
  run_eval $ENV mlp_upper_l1  $LOG_ROOT/${MP}_mlp_upper_l1  $NUM_EPS $S
  run_eval $ENV mlp_lower     $LOG_ROOT/${MP}_mlp_lower     $NUM_EPS $S
  run_eval $ENV mlp_lower_l1  $LOG_ROOT/${MP}_mlp_lower_l1  $NUM_EPS $S
done

echo "" >> $RESULTS_FILE
echo "=====================================================" >> $RESULTS_FILE
echo "Done — $(date)" >> $RESULTS_FILE
echo "=====================================================" >> $RESULTS_FILE

echo "Results written to $RESULTS_FILE"
