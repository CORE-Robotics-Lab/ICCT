"""
Count active (interpretable) parameters in trained ICCT / MLP policies.

"Active" = parameters whose absolute value exceeds `threshold`.
For ICCT policies this gives the number of features actively used in routing
decisions and leaf controllers — the quantity plotted on the Pareto figure.

Usage (via learning_curve_plot.py):
    python icct/plot/learning_curve_plot.py \
      --log_dir /path/to/logs --env_name cart \
      --results_table --model_dir /path/to/models \
      --threshold 0.005
"""

import torch


def count_active_params(model, policy_type='ddt', threshold=0.005, include_bias=True):
    """
    :param model: loaded SAC/TD3 model (from Alg.load())
    :param policy_type: 'ddt', 'mlp', or 'oblique_tree'
    :param threshold: absolute value threshold below which a param is inactive
    :param include_bias: whether to include bias terms in the count
    :return: dict with 'active_params', 'total_params', 'analytical_active_params'
    """
    actor = model.policy.actor

    if policy_type in ('ddt', 'oblique_tree'):
        return _count_ddt_params(actor, threshold, include_bias)
    elif policy_type == 'mlp':
        return _count_mlp_params(actor, threshold, include_bias)
    else:
        raise ValueError(f'Unknown policy_type: {policy_type}')


def _count_ddt_params(actor, threshold, include_bias):
    ddt = actor.ddt
    total = 0
    active = 0

    def _tally(t):
        nonlocal total, active
        if t is None or not isinstance(t, torch.Tensor):
            return
        flat = t.detach().cpu().float()
        total += flat.numel()
        active += int((flat.abs() > threshold).sum().item())

    # Routing: per-node feature weights and comparator thresholds
    _tally(ddt.layers)
    _tally(ddt.comparators)

    # Leaf outputs / submodel parameters
    if hasattr(ddt, 'lin_models') and ddt.lin_models is not None:
        # M2 / M4 / M5-a: per-leaf nn.Linear controllers
        for lin in ddt.lin_models:
            _tally(lin.weight)
            if include_bias and lin.bias is not None:
                _tally(lin.bias)
    elif hasattr(ddt, 'sub_weights') and ddt.sub_weights is not None:
        # M5-b: feature-selection submodels (sub_scalars selects, sub_weights scales)
        _tally(ddt.sub_weights)
        if include_bias:
            _tally(ddt.sub_biases)
    else:
        # M1 / M3: direct leaf output params
        if hasattr(ddt, 'action_mus'):
            _tally(ddt.action_mus)
        if hasattr(ddt, 'action_stds'):
            _tally(ddt.action_stds)

    # Analytical active-param count for M5-b (feature-selection) models.
    # By construction each leaf uses exactly num_sub_features input features,
    # so we know the count without needing a threshold.
    analytical = None
    if (hasattr(ddt, 'sparse_submodel_type') and ddt.sparse_submodel_type == 2
            and hasattr(ddt, 'num_sub_features')):
        n_leaves = ddt.num_leaves
        n_feats = ddt.num_sub_features
        out_dim = ddt.output_dim
        analytical = n_leaves * n_feats * out_dim
        if include_bias:
            analytical += n_leaves * out_dim

    return {
        'active_params': active,
        'total_params': total,
        'analytical_active_params': analytical,
    }


def _count_mlp_params(actor, threshold, include_bias):
    total = 0
    active = 0

    for name, param in actor.named_parameters():
        if not include_bias and 'bias' in name:
            continue
        flat = param.detach().cpu().float()
        total += flat.numel()
        active += int((flat.abs() > threshold).sum().item())

    return {
        'active_params': active,
        'total_params': total,
        'analytical_active_params': None,
    }
