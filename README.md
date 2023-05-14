# ICCT Implementation
[**Talk (RSS 2022)**](https://www.youtube.com/watch?v=V17vQSZP5Zs&t=6798s) | [**Robot Demo**](https://sites.google.com/view/icctree) | [**Paper (RSS 2022)**](https://www.roboticsproceedings.org/rss18/p068.pdf) 

This is the codebase for "[Learning Interpretable, High-Performing Policies for Autonomous Driving](http://www.roboticsproceedings.org/rss18/p068.pdf)", which is published in [Robotics: Science and Systems (RSS), 2022](http://www.roboticsproceedings.org/rss18/index.html).

Authors: [Rohan Paleja*](https://rohanpaleja.com/), [Yaru Niu*](https://www.yaruniu.com/), [Andrew Silva](https://www.andrew-silva.com/), Chace Ritchie, Sugju Choi, [Matthew Gombolay](https://core-robotics.gatech.edu/people/matthew-gombolay/)

\* indicates co-first authors.

<p align="center">
    <img src="assets/trained_icct.gif" width=800><br/>
    <em >Trained High-Performance ICCT Polices in Six Tested Domains.</em>
</p>

## Dependencies
* [PyTorch](https://pytorch.org/) 1.5.0 (GPU)
* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 1.1.0a11 (verified but not required)
* [OpenAI Gym](https://github.com/openai/gym)
* [box2d-py](https://github.com/openai/box2d-py)
* [MuJoCo](https://mujoco.org/), [mujoco-py](https://github.com/openai/mujoco-py)
* [highway-env](https://github.com/eleurent/highway-env)
* [Flow](https://github.com/flow-project/flow) and [SUMO](https://github.com/eclipse/sumo) ([Installing Flow and SUMO](https://flow.readthedocs.io/en/latest/flow_setup.html#installing-flow-and-sumo))
* [scikit-learn](https://scikit-learn.org/stable/install.html)

## Installation of ICCT
  ```
  git clone https://github.com/CORE-Robotics-Lab/ICCT.git
  cd ICCT
  pip install -e .
  ```

## Training
In this codebase, we provide all the methods presented in the paper including CDDT (M1), CDDT-controllers (M2), ICCT-static (M3), ICCT-complete (M4), ICCT-L1-sparse (M5-a), ICCT-n-feature (M5-b), MLP-Max (large), MLP-Upper (medium), and MLP-Lower (small). Run `python icct/runfiles/train.py --help` to check all the options for training. Examples of training ICCT-2-feature can be found in `icct/runfiles/`. All the methods are trained using [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) in our paper. We also provide the implementation for [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477). Here we provide instructions on using method-specific arguments.
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
* Arguments for DDT (including ICCT):
  * `--num_leaves`: the number of leaves used in ddt (2^n)
  * `--ddt_lr`: a specific learning rate used for DDT (the policy network), the learning rate for the critic network will be specified by `--lr`
  * `--use_individual_alpha`: if use different alphas for different nodes (sometimes it helps boost the performance)
  * To activate CDDT (M1), only set `--policy_type` to `ddt`, and do not use `--submodels` or `--hard_node`
  * To activate CDDT-controllers (M2), use `--submodels` and set `--sparse_submodel_type` to 0
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

## Loading and Testing
All the MLP and DDT-based methods are evaluated in real time throughout the training process. Here we provide modules to load and test trained models. Please set up arguments and run `sh test.sh` in `icct/runfiles/`. For each DDT-based method, two types of performance can be output: 
  * Fuzzy performance: the performance is evaluated by directly loading the trained model
  * Crisp performance: the performance is evaluated by a processed discretized (crisp) model. The discretization process is proposed in https://arxiv.org/pdf/1903.09338.pdf

For any ICCT methods, fuzzy and crisp performance will be the same, while the crisp performance of CDDT (CDDT-Crisp) or CDDT-controllers (CDDT-controllers Crisp) will change and usually drop drastically.

## Visualization of Learning Curves
During training, the training process can be monitored by tensorboard. Please run `tensorboard --logdir TARGET_PATH`, where `TARGET_PATH` is the path to your saved log files. We also provide visualization of mean rollout rewards and mean evaluation rewards througout the training process of multiple runs (seeds). The csv files of these two kinds of rewards are saved in the same folder of the trained models. Please copy the csv files from different runs (seeds) and different methods in the same tested domain to one folder. Run `learning_curve_plot.py` in `icct/plot/` and include the following the arguments:
* `--log_dir`: the path to the data
* `--eval_freq`: evaluation frequence used during training (has to be the same as the one in training)
* `--n_eval_episodes`: the number of episodes for each evaluation during training (has to be the same as the one in training)
* `--eval_smooth_window_size`: the sliding window size to smooth the evaluation rewards
* `--non_eval_sample_freq`: the sample frequence of the rollout rewards for plotting
* `--non_eval_smooth_window_size`: the sliding window size to smooth the sampled rollout rewards

## Imitation Learning - DAgger
We provide an implementation of imitation learning by decision trees using [Dataset Aggregation (DAgger)](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf). Please set up arguments and run `sh train.sh` in `icct/dagger/`. The oracle models are picked from the best of MLP-Max from five seeds trained by SAC, which can be found in `icct/dagger/oracle_models/`. We have improved the implementation of DAgger since paper submission and update the results averaged over five seeds as follows.
| Environment  | Inverted Pendulum | Lunar Lander | Lane Keeping | Single-Lane Ring | Multi-Lane Ring | Figure-8 |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| Number of Leaves | 32 | 32 | 16 | 16 | 32 | 16 |
| Best Rollout Performance | $853.1\pm38.2$ | $245.7\pm8.4$ | $393.1\pm14.2$ | $121.9\pm0.03$ | $1260.4\pm4.6$ | $1116.4\pm8.3$ |
| Evaluation Performance | $776.6\pm54.2$ | $184.7\pm17.3$ | $395.2\pm13.8$ | $121.5\pm0.01$ | $1249.4\pm3.4$  | $1113.8\pm9.5$ |
| Oracle Performance | $1000.0$ | $301.2$ | $494.1$ | $122.29$ | $1194.5$ | $1126.3$ |

## Citation
If you find our paper or repo helpful to your research, please consider citing the paper:
```
@inproceedings{icct-rss-22,
  title={Learning Interpretable, High-Performing Policies for Autonomous Driving},
  author={Paleja, Rohan and Niu, Yaru and Silva, Andrew and Ritchie, Chace and Choi, Sugju and Gombolay, Matthew},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2022}
}
```

## Acknowledgments
Some parts of this codebase are inspired from or based on several public repos:
* [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [DDTs](https://github.com/CORE-Robotics-Lab/Interpretable_DDTS_AISTATS2020)
* [VIPER](https://github.com/obastani/viper/)
