# Created by Yaru Niu

import numpy as np
import sys
from icct.core.icct import ICCT
import torch


def convert_to_crisp(fuzzy_model, training_data):
    new_weights = []
    new_comps = []
    device = fuzzy_model.device

    weights = np.abs(fuzzy_model.layers.cpu().detach().numpy())
    most_used = np.argmax(weights, axis=1)
    for comp_ind, comparator in enumerate(fuzzy_model.comparators):
        comparator = comparator.item()
        divisor = abs(fuzzy_model.layers[comp_ind][most_used[comp_ind]].item())
        if divisor == 0:
            divisor = 1
        comparator /= divisor
        new_comps.append([comparator])
        max_ind = most_used[comp_ind]
        new_weight = np.zeros(len(fuzzy_model.layers[comp_ind].data))
        new_weight[max_ind] = fuzzy_model.layers[comp_ind][most_used[comp_ind]].item() / divisor
        new_weights.append(new_weight)

    new_input_dim = fuzzy_model.input_dim
    new_weights = np.array(new_weights)
    new_comps = np.array(new_comps)
    new_alpha = fuzzy_model.alpha
    new_alpha = 9999999. * new_alpha.cpu().detach().numpy() / np.abs(new_alpha.cpu().detach().numpy())
    crispy_model = ICCT(input_dim=new_input_dim,
                        output_dim=fuzzy_model.output_dim,
                        weights=new_weights,
                        comparators=new_comps,
                        leaves=fuzzy_model.leaf_init_information,
                        alpha=new_alpha,
                        use_individual_alpha=fuzzy_model.use_individual_alpha,
                        use_submodels=fuzzy_model.use_submodels,
                        hard_node=False,
                        sparse_submodel_type=fuzzy_model.sparse_submodel_type,
                        l1_hard_attn=False,
                        num_sub_features=fuzzy_model.num_sub_features,
                        use_gumbel_softmax=fuzzy_model.use_gumbel_softmax,
                        device=device).to(device)
    if hasattr(fuzzy_model, 'action_mus'):
        crispy_model.action_mus.data = fuzzy_model.action_mus.data
    crispy_model.action_stds.data = fuzzy_model.action_stds.data
    if fuzzy_model.use_submodels:
        if fuzzy_model.sparse_submodel_type != 2:
            crispy_model.lin_models = fuzzy_model.lin_models
        else:
            crispy_model.sub_scalars = fuzzy_model.sub_scalars
            crispy_model.sub_weights = fuzzy_model.sub_weights
            crispy_model.sub_biases = fuzzy_model.sub_biases

    return crispy_model
