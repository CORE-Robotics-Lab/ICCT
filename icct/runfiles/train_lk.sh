export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -u train.py \
  --env_name lane_keeping \
  --policy_type ddt \
  --seed 0 \
  --num_leaves 16 \
  --lr 3e-4 \
  --ddt_lr 3e-4 \
  --buffer_size 1000000 \
  --batch_size 1024 \
  --gamma 0.99 \
  --tau 0.01 \
  --learning_starts 10000 \
  --eval_freq 1500 \
  --min_reward 420 \
  --training_steps 500000 \
  --log_interval 4 \
  --save_path log/lk \
  --use_individual_alpha \
  --submodels \
  --hard_node \
  --argmax_tau 1.0 \
  --sparse_submodel_type 2 \
  --num_sub_features 2 \
  | tee train_lk.log
