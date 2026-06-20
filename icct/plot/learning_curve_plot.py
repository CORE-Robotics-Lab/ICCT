# Created by Yaru Niu

import argparse
from learning_curve_plotter import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Plotting Learning Curves')
    parser.add_argument('--log_dir', help='the path to the data', type=str, default='results')
    parser.add_argument('--eval_freq', help='evaluation frequence used during training', type=int, default=1500)
    parser.add_argument('--n_eval_episodes', help='the number of episodes for each evaluation during training', type=int, default=5)
    parser.add_argument('--eval_smooth_window_size', help='the sliding window size to smooth the evaluation rewards', type=int, default=1)
    parser.add_argument('--non_eval_sample_freq', help='the sample frequence of the rollout rewards for plotting ', type=int, default=2000)
    parser.add_argument('--non_eval_smooth_window_size', help='the sliding window size to smooth the sampled rollout rewards', type=int, default=1)
    parser.add_argument('--env_name', help='the environment name of the raw data', type=str)
    parser.add_argument('--show_legend', help='if show the legend in the figure', action='store_true', default=False)

    # Results table & Pareto plot args
    parser.add_argument('--results_table', help='generate results table (reward + param counts)',
                        action='store_true', default=False)
    parser.add_argument('--pareto', help='generate Pareto plot (reward vs. active params)',
                        action='store_true', default=False)
    parser.add_argument('--latex', help='print results as LaTeX table',
                        action='store_true', default=False)
    parser.add_argument('--model_dir', help='directory containing saved models (for param counting)',
                        type=str, default=None)
    parser.add_argument('--alg_type', help='RL algorithm type for loading models',
                        type=str, choices=['sac', 'td3'], default='sac')
    parser.add_argument('--threshold', help='active parameter threshold',
                        type=float, default=0.005)
    parser.add_argument('--save_csv', help='save results table to this CSV path',
                        type=str, default=None)

    args = parser.parse_args()
    plotter = Learning_Curve_Plotter(log_dir=args.log_dir,
                                     eval_freq=args.eval_freq,
                                     n_eval_episodes=args.n_eval_episodes,
                                     eval_smooth_window_size=args.eval_smooth_window_size,
                                     non_eval_sample_freq=args.non_eval_sample_freq,
                                     non_eval_smooth_window_size=args.non_eval_smooth_window_size,
                                     env_name=args.env_name,
                                     show_legend=args.show_legend)

    if args.results_table or args.pareto or args.latex:
        df = plotter.generate_results_table(
            model_dir=args.model_dir,
            alg_type=args.alg_type,
            threshold=args.threshold,
        )
        if args.save_csv:
            df.to_csv(args.save_csv, index=False)
            print(f"Results saved to {args.save_csv}")
        if args.results_table:
            print(df.to_string(index=False))
        if args.latex:
            plotter.print_latex_table(df)
        if args.pareto:
            plotter.plot_pareto(df)
    else:
        plotter.process_data()
        plotter.plot()