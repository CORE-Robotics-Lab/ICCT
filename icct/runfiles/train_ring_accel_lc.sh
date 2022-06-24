export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -u train.py \
  --env_name ring_lane_changing \
  --policy_type ddt \
  --seed 0 \
  --num_leaves 16 \
  --lr 5e-4 \
  --ddt_lr 5e-4 \
  --buffer_size 1000000 \
  --batch_size 1024 \
  --gamma 0.99 \
  --tau 0.01 \
  --learning_starts 5000 \
  --eval_freq 2500 \
  --min_reward 1200 \
  --training_steps 500000 \
  --log_interval 4 \
  --save_path log/ring_lc \
  --use_individual_alpha \
  --submodels \
  --hard_node \
  --argmax_tau 1.0 \
  --sparse_submodel_type 2 \
  --num_sub_features 3 \
  | tee train_ring_accel_lc.log