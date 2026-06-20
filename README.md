# ICCT Implementation

This is the codebase for "[Learning Interpretable, High-Performing Policies for Autonomous Driving](http://www.roboticsproceedings.org/rss18/p068.pdf)", published at [Robotics: Science and Systems (RSS), 2022](http://www.roboticsproceedings.org/rss18/index.html).

Authors: [Rohan Paleja*](https://rohanpaleja.com/), [Yaru Niu*](https://www.yaruniu.com/), [Andrew Silva](https://www.andrew-silva.com/), Chace Ritchie, Sugju Choi, [Matthew Gombolay](https://core-robotics.gatech.edu/people/matthew-gombolay/)

\* indicates co-first authors.

<p align="center">
    <img src="assets/trained_icct.gif" width=800><br/>
    <em>Trained High-Performance ICCT Policies in Six Tested Domains.</em>
</p>

---

## Environment Setup

Tested on Python 3.8.20, PyTorch 2.4.1, CUDA 12.1, Rocky Linux 9.7 / RHEL 9, kernel `5.14.0-611.16.1.el9_7.x86_64`, no-sudo cluster.

If you hit dependency conflicts on a different OS or kernel, use the provided `Dockerfile` in `flow/` as a starting point, or build a container from `python:3.8-slim` installing these exact versions.

All `#SBATCH` directives (`--account=rpaleja`, `--partition=training`, `--qos=training`) and the default `LOG_ROOT` (`/scratch/gilbreth/$USER/log`) are tuned for our cluster. On a different cluster, edit these in every script under `icct/runfiles/` to match your account, partition, QOS, and scratch path.

### 1. Create the conda/mamba environment

```bash
mamba create -n icct_jmlr python=3.8 -y
mamba activate icct_jmlr
python -m pip install --upgrade pip setuptools
```

The environment is named `icct_jmlr` throughout; all SLURM scripts use `mamba activate icct_jmlr`. If you create the environment with a different name, update the `mamba activate` line in every script under `icct/runfiles/`.

This assumes `mamba` is already initialized for your shell (i.e. you've run `mamba shell init --shell bash` at some point, or your cluster's `mamba`/`conda` module does this for you, so `mamba activate` works without a "Shell not initialized" error). If `mamba create -n ...` fails with `Permission denied`, your shell isn't initialized yet and mamba is defaulting to a read-only system path; run `mamba shell init --shell bash`, restart your shell, and retry. You can also use conda if that module is preloaded.

### 2. Install build dependencies and OpenGL libraries

```bash
mamba install -c conda-forge glew=2.3.0 libglvnd=1.7.0 libgl libglu libglx patchelf=0.17.2 swig -y
```

- OpenGL packages (`glew`, `libglvnd`, `libgl`, `libglu`, `libglx`) are needed for MuJoCo and rendering
- `swig` is needed to build `box2d-py` (used by `LunarLanderContinuous-v2`)

Set these in the same shell session before building any C extensions:

```bash
export CONDA_PREFIX=$(mamba info --base)/envs/icct_jmlr
export CPATH="$CONDA_PREFIX/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

These exports are session-only. Re-run them (or add to `~/.bashrc` / your job script) before any build step that compiles C extensions, i.e. before `pip install mujoco-py`.

### 3. Install MuJoCo

MuJoCo 2.1 has been free since 2022, no license key required.

Download and extract:
```bash
mkdir -p $HOME/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
tar -xzf /tmp/mujoco210.tar.gz -C $HOME/.mujoco
# Result: $HOME/.mujoco/mujoco210/
```

Then build `mujoco-py`:
```bash
python -m pip install "Cython==0.29.37"
python -m pip install mujoco-py==2.1.2.14
```

Also add MuJoCo's own lib directory to `LD_LIBRARY_PATH` (needed at runtime, not just build time):
```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH"
```

If the build fails with missing GL headers, set:
```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
```
then re-run the `python -m pip install mujoco-py` line.

MuJoCo is only needed for the `cart` (InvertedPendulum) environment. If you are not running `cart`, skip this step.

### 4. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

If `box2d-py` fails to build with a `swig` not found error, this usually happens when a broken `swig` stub (e.g. at `~/.local/bin/swig`) shadows the conda env's real swig binary. Fix by ensuring the conda env's bin precedes `~/.local/bin`:
```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
python -m pip install -r requirements.txt
```

`requirements.txt` is a full pip freeze of the verified environment. A few versions worth knowing about:
- `torch==2.4.1` (CUDA 12)
- `numpy==1.23.5`, pinned deliberately. `highway-env` (used by `lane_keeping`) calls the deprecated `np.float` alias, which still works on 1.23.x but was removed in numpy 1.24+, causing a hard crash on environment creation.
- `stable-baselines3==1.1.0a11`
- `gym==0.17.0`
- `highway-env==1.4`
- `eclipse-sumo==1.23.1` (includes SUMO, sumolib, traci, no manual SUMO install needed)
- `ray==2.10.0`, not used directly by `icct`, but required transitively since `flow/flow/utils/rllib.py` imports `ray.cloudpickle`, and that module is imported by every traffic-env file.

### 5. Install Flow (bundled)

A pinned fork of Flow is included in `flow/`. Install it in editable mode so the SUMO environments are importable:

```bash
python -m pip install -e flow/ --no-deps
```

`--no-deps` is required because `flow/setup.py` pins very old versions of `pandas`, `scipy`, `gym`, etc. that conflict with what `requirements.txt` already installed. All real runtime dependencies are already satisfied from Step 4, so `--no-deps` is safe.

Flow is used only for the traffic environments (`ring_accel`, `ring_lane_changing`, `figure8`). The `lane_keeping` environment is from `highway-env` and does not require Flow.

### 6. Install this package

```bash
python -m pip install -e .
```

### 7. Smoke test

Make sure MuJoCo is on `LD_LIBRARY_PATH` (if not already set from Step 3):
```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

```bash
# Non-traffic environment (no SUMO needed)
python icct/runfiles/train.py --env_name cart --seed 0 --training_steps 1000

# Traffic environment (requires SUMO via eclipse-sumo)
python icct/runfiles/train.py --env_name ring_accel --seed 0 --training_steps 500
```

---

## SUMO Notes

SUMO is installed automatically via `pip install eclipse-sumo` (included in `requirements.txt`), no manual download or `SUMO_HOME` configuration required. The `eclipse-sumo` package ships `sumo`, `sumolib`, and `traci` and registers the `sumo` binary on PATH within the conda environment.

If you see `TraCIException: Connection refused`, verify SUMO is on your PATH:
```bash
which sumo   # should point to your conda env's bin/
sumo --version
```

On a SLURM cluster, use `module load conda` and `conda activate` in your job script before running, no `module load sumo` needed since it comes from pip.

The warning `Environment variable SUMO_HOME is not set properly` is harmless. It only disables XML schema validation for SUMO config files; simulation behavior is unaffected. Since SUMO comes from `eclipse-sumo` via pip rather than a manual install, there's no separate `SUMO_HOME` directory to point at.

---

## Training

In this codebase, we provide all the methods presented in the paper including CDDT (M1), CDDT-controllers (M2), ICCT-static (M3), ICCT-complete (M4), ICCT-L1-sparse (M5-a), ICCT-n-feature (M5-b), MLP-Max (large), MLP-Upper (medium), and MLP-Lower (small). Run `python icct/runfiles/train.py --help` to check all the options for training. Examples of training ICCT-2-feature can be found in `icct/runfiles/`. All the methods are trained using [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) in our paper. We also provide the implementation for [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477). Here we provide instructions on using method-specific arguments.

This extension additionally provides CDDT-controllers-L1, and an L1-sparse version of each MLP size: MLP-Max-L1, MLP-Upper-L1, MLP-Lower-L1.

* Arguments for all methods
  * `--env_name`: environment to run on
  * `--alg_type`: use SAC or TD3
  * `--policy_type`: use DDT or MLP as the policy network
  * `--seed`: set the seed number
  * `--gpu`: add to use GPU
  * `--lr`: the learning rate
  * `--buffer_size`: the buffer size
  * `--batch_size`: the batch size
  * `--gamma`: the discount factor
  * `--tau`: the soft update coefficient (between 0 and 1) in SAC
  * `--learning_starts`: how many steps of the model to collect transitions for before learning starts
  * `--training_steps`: total steps for training the model
  * `--min_reward`: the minimum reward to save the model
  * `--save_path`: the path to save the models and logged files
  * `--n_eval_episodes`: the number of episodes for each evaluation during training
  * `--eval_freq`: evaluation frequence (evaluating the model for every fixed number of steps) of the model during training
  * `--log_interval`: the number of episodes before logging
* Arguments for MLP:
  * `--mlp_size`: choose the size of MLP to use (large: MLP-Max; medium: MLP-Upper; small: MLP-Lower)
  * `--mlp_sparsity_reg`: apply sparsity regularization to the MLP actor, used for the L1 baselines added in this extension (MLP-Max-L1, MLP-Upper-L1, MLP-Lower-L1)
  * `--mlp_sparsity_type`: l1 or l2
  * `--mlp_sparsity_coeff`: the coefficient of the sparsity regularization
  * `--mlp_sparsity_include_bias`: include bias terms in the sparsity regularization
* Arguments for DDT (including ICCT):
  * `--num_leaves`: the number of leaves used in ddt (2^n)
  * `--ddt_lr`: a specific learning rate used for DDT (the policy network), the learning rate for the critic network will be specified by `--lr`
  * `--use_individual_alpha`: if use different alphas for different nodes (sometimes it helps boost the performance)
  * To activate CDDT (M1), only set `--policy_type` to `ddt`, and do not use `--submodels` or `--hard_node`
  * To activate CDDT-controllers (M2), use `--submodels` and set `--sparse_submodel_type` to 0
  * To activate CDDT-controllers-L1, use `--submodels`, set `--sparse_submodel_type` to 1, and do not use `--hard_node` (M2 plus the L1 arguments below)
  * To activate ICCT-static (M3), use `--hard_node`
  * To activate ICCT-complete (M4), use `--hard_node`, `--submodels`, and set `--sparse_submodel_type` to 0
  * To activate ICCT-L1-sparse (M5-a), use `--hard_node`, `--submodels`, set `--sparse_submodel_type` to 1, and use the following arguments:
    * `--l1_reg_coeff`: the coefficient of the L1 regularization
    * `--l1_reg_bias`: if consider biases in the L1 loss (not recommended)
    * `--l1_hard_attn`: if only sample one leaf node's linear controller to perform L1 regularization for each update, and this can be helpful in enforcing sparsity on each linear controller
    * We choose L1 regularization over L2 because L1 is more likely to push coefficients to zeros
  * To activate ICCT-n-feature (M5-b, "n" is the number of features selected by each leaf's linear sub-controller), use `--hard_node`, `--submodels`, set `--sparse_submodel_type` to 2, and use the following arguments:
    * `--num_sub_features`: the number of chosen features for submodels
    * `--argmax_tau`: the temperature of the diff_argmax function
    * `--use_gumbel_softmax`: include to replace the Argmax operation in the paper with Gumbel-Softmax

### Running on a SLURM cluster

This extension adds SLURM scripts in `icct/runfiles/` so you do not have to call `train.py` directly. Each script trains 5 seeds as a job array in one `sbatch` call:

```bash
METHOD=icct ENV_NAME=cart sbatch icct/runfiles/train_icct.slurm
```

Set `LOG_ROOT` to control where models are saved (defaults to `/scratch/gilbreth/$USER/log`), and use the same `LOG_ROOT` for every script in a run.

| Script | Methods |
|---|---|
| `train_icct.slurm` | CDDT, CDDT-controllers, ICCT-static, ICCT-complete, ICCT-L1-sparse, ICCT-1/2/3-feature (set `METHOD`) |
| `train_cddt_ctrl_l1.slurm` | CDDT-controllers-L1 |
| `train_oracle.slurm` | MLP-Max (also the oracle used for distillation, see Imitation Learning below) |
| `train_mlp_sparse.slurm` | MLP-Max-L1 |
| `train_mlp_upper.slurm` / `train_mlp_upper_sparse.slurm` | MLP-Upper / MLP-Upper-L1 |
| `train_mlp_lower.slurm` / `train_mlp_lower_sparse.slurm` | MLP-Lower / MLP-Lower-L1 |
| `train_dagger.slurm` | DT w/ DAgger (see Imitation Learning below, needs the oracle) |
| `train_hinton.slurm` | Hinton soft decision tree distillation (needs the oracle) |
| `train_viper_big.slurm` / `train_viper_small.slurm` | VIPER-Big / VIPER-Small (need the oracle) |
| `train_linear_tree.slurm` | Linear Tree (see Imitation Learning below, needs the oracle) |

---

## Loading and Testing

All the MLP and DDT-based methods are evaluated in real time throughout the training process. Here we provide modules to load and test trained models:

```bash
python icct/runfiles/test.py \
  --env_name cart \
  --load_path /path/to/model/seed0 \
  --load_file best_model \
  --num_episodes 20 \
  --nn --gpu
```

For each DDT-based method, two types of performance can be output:
  * Fuzzy performance: the performance is evaluated by directly loading the trained model
  * Crisp performance: the performance is evaluated by a processed discretized (crisp) model. The discretization process is proposed in https://arxiv.org/pdf/1903.09338.pdf

For any ICCT methods, fuzzy and crisp performance will be the same, while the crisp performance of CDDT (CDDT-Crisp) or CDDT-controllers (CDDT-controllers Crisp) will change and usually drop drastically.

This extension adds `eval_all.slurm` to batch-evaluate every SAC-trained method (the ICCT/CDDT family, CDDT-controllers-L1, and the MLP family) across all environments and seeds in one job, writing one `eval_results.txt`:

```bash
sbatch icct/runfiles/eval_all.slurm
```

It does not cover DAgger, VIPER, or Hinton, since those save decision-tree or distilled models in a format `test.py` cannot load; those report their own eval metrics directly in their training logs instead.

---

## Visualization of Learning Curves

During training, the training process can be monitored by tensorboard. Please run `tensorboard --logdir TARGET_PATH`, where `TARGET_PATH` is the path to your saved log files. We also provide visualization of mean rollout rewards and mean evaluation rewards throughout the training process of multiple runs (seeds). The csv files of these two kinds of rewards are saved in the same folder of the trained models. Please copy the csv files from different runs (seeds) and different methods in the same tested domain to one folder. Run `learning_curve_plot.py` in `icct/plot/` and include the following arguments:
* `--log_dir`: the path to the data
* `--eval_freq`: evaluation frequence used during training (has to be the same as the one in training)
* `--n_eval_episodes`: the number of episodes for each evaluation during training (has to be the same as the one in training)
* `--eval_smooth_window_size`: the sliding window size to smooth the evaluation rewards
* `--non_eval_sample_freq`: the sample frequence of the rollout rewards for plotting
* `--non_eval_smooth_window_size`: the sliding window size to smooth the sampled rollout rewards

---

## Imitation Learning - DAgger

We provide an implementation of imitation learning by decision trees using [Dataset Aggregation (DAgger)](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf). The oracle models are picked from the best of MLP-Max from five seeds trained by SAC.

This extension changes how the oracle is supplied (see below), and adds three more distillation methods that also use a trained MLP-Max oracle as the teacher: [VIPER](https://arxiv.org/abs/1805.08328) (Big and Small variants), Hinton-style soft decision tree distillation, and Linear Tree (behavioral cloning into a tree with a linear model at each leaf, via the `lineartree` package).

Train the oracle first, for every environment:
```bash
ENV_NAME=cart sbatch icct/runfiles/train_oracle.slurm
# repeat for: lunar, lane_keeping, ring_accel, ring_lane_changing, figure8
```

This saves 5 seeded oracles per environment to `$LOG_ROOT/<env>_oracle/seed{0-4}/best_model.zip`. Each distillation script below reads from the same `LOG_ROOT` and matches seeds: distillation seed X is taught by the oracle trained with seed X. The oracle must finish training before submitting these, or a distillation run can grab a GPU and fail before its matching oracle exists.

```bash
# DT w/ DAgger
ENV_NAME=lunar sbatch icct/runfiles/train_dagger.slurm

# Hinton soft decision tree distillation
ENV_NAME=lunar sbatch icct/runfiles/train_hinton.slurm

# VIPER-Big and VIPER-Small loop over all 6 environments internally in one job
sbatch icct/runfiles/train_viper_big.slurm
sbatch icct/runfiles/train_viper_small.slurm

# Linear Tree
ENV_NAME=lunar sbatch icct/runfiles/train_linear_tree.slurm
```

* Arguments for `train_dagger.py` (also used by VIPER, since VIPER is DAgger with Q-value weighting):
  * `--oracle_load_path`: directory containing the pre-trained oracle model
  * `--oracle_load_file`: filename of the oracle, without `.zip` (always `best_model` for the seed-matched oracles produced by `train_oracle.slurm`)
  * `--max_depth`: maximum depth of the fitted decision tree (leaves = 2^depth)
  * `--n_rollouts`: number of rollout episodes per DAgger iteration
  * `--iterations`: number of DAgger iterations (the dataset grows each round)
  * `--eval_episodes`: episodes used to evaluate the decision tree after each iteration
  * `--q_dagger`: VIPER only, weight training samples by Q-value
  * `--n_q_samples`: VIPER only, number of Q-value samples per state used for weighting
  * `--max_samples`: VIPER only, cap on total dataset size across iterations
  * `--load`: path to a saved `best_dt.pkl`. Skips training and evaluates that tree directly, useful to re-test a tree without retraining
* Arguments for `train_linear_tree.py` (same idea as `train_dagger.py`, but fits one linear-tree model by behavioral cloning from a single batch of oracle rollouts, no DAgger iterations):
  * `--oracle_load_path` / `--oracle_load_file`: same as above
  * `--n_rollouts`: number of oracle rollouts to fit the tree on
  * `--eval_episodes`: episodes used to evaluate the fitted tree
  * `--load`: path to a saved `best_dt.pkl`. Skips training and evaluates that tree directly
* Arguments for `train_hinton.py`:
  * `--num_leaves`: tree leaves (should match the ICCT num_leaves for the environment)
  * `--n_rollouts`: rollouts used to collect the distillation dataset
  * `--epochs`: gradient steps on the distillation objective
  * `--lam`: regularization weight balancing entropy loss against reward loss
  * `--lr`: learning rate for the soft decision tree
  * `--load`: path to a saved `best_model.pt`. Skips training and evaluates that tree directly, same purpose as `--load` in `train_dagger.py`
  * `--eval_episodes`: episodes to evaluate when using `--load` (default 20)

VIPER-Big always fits a depth-6 decision tree. VIPER-Small and DAgger use a depth matched to the environment's ICCT leaf count.

---

## Citation

```bibtex
@inproceedings{icct-rss-22,
  title={Learning Interpretable, High-Performing Policies for Autonomous Driving},
  author={Paleja, Rohan and Niu, Yaru and Silva, Andrew and Ritchie, Chace and Choi, Sugju and Gombolay, Matthew},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2022}
}
```

## Acknowledgments

Parts of this codebase are based on:
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [DDTs](https://github.com/CORE-Robotics-Lab/Interpretable_DDTS_AISTATS2020)
- [VIPER](https://github.com/obastani/viper/)
- [Flow](https://github.com/flow-project/flow)
