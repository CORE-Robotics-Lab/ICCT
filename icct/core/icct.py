# Created by Yaru Niu
# Reference file by Andrew Silva: https://github.com/CORE-Robotics-Lab/Interpretable_DDTS_AISTATS2020/blob/master/interpretable_ddts/agents/ddt.py

import torch.nn as nn
import torch
import numpy as np
import typing as t
import torch.nn.functional as F
import time


class ICCT(nn.Module):
    def __init__(self,
                 input_dim: int,
                 weights: t.Union[t.List[np.array], np.array, None],
                 comparators: t.Union[t.List[np.array], np.array, None],
                 alpha: t.Union[t.List[np.array], np.array, None],
                 leaves: t.Union[None, int, t.List],  
                 output_dim: t.Optional[int] = None,
                 use_individual_alpha=False,
                 device: str = 'cpu',
                 use_submodels: bool = False,
                 hard_node: bool = False,
                 argmax_tau: float = 1.0,
                 sparse_submodel_type = 0,
                 fs_submodel_version = 0,
                 l1_hard_attn = False,
                 num_sub_features = 1,
                 use_gumbel_softmax = False,
                 alg_type = 'sac'):
        super(ICCT, self).__init__()
        """
        Initialize the Interpretable Continuous Control Tree (ICCT)

        :param input_dim: (observation/feature) input dimensionality
        :param weights: the weight vector for each node to initialize
        :param comparators: the comparator vector for each node to initialize
        :param alpha: the alpha to initialize
        :param leaves: the number of leaves of ICCT
        :param output_dim: (action) output dimensionality
        :param use_individual_alpha: whether use different alphas for different nodes 
                                    (sometimes it helps boost the performance)
        :param device: which device should ICCT run on [cpu|cuda]
        :param use_submodels: whether use linear sub-controllers (submodels)
        :param hard_node: whether use differentiable crispification (this arg does not
                          influence the differentiable crispification procedure in the 
                          sparse linear controllers)
        :param argmax_tau: the temperature of the diff_argmax function
        :param sparse_submodel_type: the type of the sparse sub-controller, 1 for L1 
                                    regularization, 2 for feature selection, other 
                                    values (default: 0) for not sparse
        :param fs_submodel_version: the version of feature-section submodel to use
        :param l1_hard_attn: whether only sample one linear controller to perform L1 
                             regularization for each update when using l1-reg submodels
        :param num_sub_features: the number of chosen features for sparse sub-controllers
        :param use_gumbel_softmax: whether use gumble softmax instead of the differentiable 
                                   argmax (diff_argmax) proposed in the paper
        :param alg_type: current supported RL methods [SAC|TD3] (the results in the paper 
                         were obtained by SAC)
        """
        self.device = device
        self.leaf_init_information = leaves
        self.hard_node = hard_node
        self.argmax_tau = argmax_tau

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = None
        self.comparators = None
        self.use_submodels = use_submodels
        self.sparse_submodel_type = sparse_submodel_type
        self.fs_submodel_version = fs_submodel_version
        self.l1_hard_attn = l1_hard_attn
        self.num_sub_features = num_sub_features
        self.use_gumbel_softmax = use_gumbel_softmax
        self.use_individual_alpha = use_individual_alpha
        self.alg_type = alg_type

        self.init_comparators(comparators)
        self.init_weights(weights)
        self.init_alpha(alpha)
        self.init_paths()
        self.init_leaves()
        self.sig = nn.Sigmoid()
        self.num_leaves = self.layers.size(0) + 1
        
        if self.use_submodels:
            self.init_submodels() 
            
        if self.alg_type == 'td3':
            self.tanh = nn.Tanh()

    def init_submodels(self):
        if self.sparse_submodel_type != 2:
            self.lin_models = nn.ModuleList([nn.Linear(self.input_dim, self.output_dim) for _ in range(self.num_leaves)])
            if self.sparse_submodel_type == 1:
                self.leaf_attn = None
        else:
            self.sub_scalars = nn.Parameter(torch.zeros(self.num_leaves, self.output_dim, self.input_dim).to(self.device), requires_grad=True)
            self.sub_weights = nn.Parameter(torch.zeros(self.num_leaves, self.output_dim, self.input_dim).to(self.device), requires_grad=True)
            self.sub_biases = nn.Parameter(torch.zeros(self.num_leaves, self.output_dim, self.input_dim).to(self.device), requires_grad=True)

            nn.init.xavier_normal_(self.sub_scalars.data)
            nn.init.xavier_normal_(self.sub_weights.data)
            nn.init.xavier_normal_(self.sub_biases.data)
            
    def init_comparators(self, comparators):
        if comparators is None:
            comparators = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    comparators.append(np.random.normal(0, 1.0, 1))
        new_comps = torch.tensor(comparators, dtype=torch.float).to(self.device)
        new_comps.requires_grad = True
        self.comparators = nn.Parameter(new_comps, requires_grad=True)

    def init_weights(self, weights):
        if weights is None:
            weights = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    weights.append(np.random.rand(self.input_dim))

        new_weights = torch.tensor(weights, dtype=torch.float).to(self.device)
        new_weights.requires_grad = True
        self.layers = nn.Parameter(new_weights, requires_grad=True)

    def init_alpha(self, alpha):
        if alpha is None:
            if self.use_individual_alpha:
                alphas = []
                if type(self.leaf_init_information) is int:
                    depth = int(np.floor(np.log2(self.leaf_init_information)))
                else:
                    depth = 4
                for level in range(depth):
                    for node in range(2**level):
                        alphas.append([1.0])
            else:
                alphas = [1.0]
        else:
            alphas = alpha
        self.alpha = torch.tensor(alphas, dtype=torch.float).to(self.device)
        self.alpha.requires_grad = True
        self.alpha = nn.Parameter(self.alpha, requires_grad=True)

    def init_paths(self):
        if type(self.leaf_init_information) is list:
            left_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)), dtype=torch.float)
            right_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)), dtype=torch.float)
            for n in range(0, len(self.leaf_init_information)):
                for i in self.leaf_init_information[n][0]:
                    left_branches[i][n] = 1.0
                for j in self.leaf_init_information[n][1]:
                    right_branches[j][n] = 1.0
        else:
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            left_branches = torch.zeros((2 ** depth - 1, 2 ** depth), dtype=torch.float)
            for n in range(0, depth):
                row = 2 ** n - 1
                for i in range(0, 2 ** depth):
                    col = 2 ** (depth - n) * i
                    end_col = col + 2 ** (depth - 1 - n)
                    if row + i >= len(left_branches) or end_col >= len(left_branches[row]):
                        break
                    left_branches[row + i, col:end_col] = 1.0
            right_branches = torch.zeros((2 ** depth - 1, 2 ** depth), dtype=torch.float)
            left_turns = np.where(left_branches == 1)
            for row in np.unique(left_turns[0]):
                cols = left_turns[1][left_turns[0] == row]
                start_pos = cols[-1] + 1
                end_pos = start_pos + len(cols)
                right_branches[row, start_pos:end_pos] = 1.0
        left_branches.requires_grad = False
        right_branches.requires_grad = False
        self.left_path_sigs = nn.Parameter(left_branches.to(self.device), requires_grad=False)
        self.right_path_sigs = nn.Parameter(right_branches.to(self.device), requires_grad=False)

    def init_leaves(self):
        if type(self.leaf_init_information) is list:
            new_leaves = [leaf[-1] for leaf in self.leaf_init_information]
        else:
            new_leaves = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4

            last_level = np.arange(2**(depth-1)-1, 2**depth-1)
            going_left = True
            leaf_index = 0
            self.leaf_init_information = []
            for level in range(2**depth):
                curr_node = last_level[leaf_index]
                turn_left = going_left
                left_path = []
                right_path = []
                while curr_node >= 0:
                    if turn_left:
                        left_path.append(int(curr_node))
                    else:
                        right_path.append(int(curr_node))
                    prev_node = np.ceil(curr_node / 2) - 1
                    if curr_node // 2 > prev_node:
                        turn_left = False
                    else:
                        turn_left = True
                    curr_node = prev_node
                if going_left:
                    going_left = False
                else:
                    going_left = True
                    leaf_index += 1
                new_probs = np.random.uniform(0, 1, self.output_dim)  # *(1.0/self.output_dim)
                self.leaf_init_information.append([sorted(left_path), sorted(right_path), new_probs])
                new_leaves.append(new_probs)

        labels = torch.tensor(new_leaves, dtype=torch.float).to(self.device)
        labels.requires_grad = True
        if not self.use_submodels:
            self.action_mus = nn.Parameter(labels, requires_grad=True)
            torch.nn.init.xavier_uniform_(self.action_mus)

        if self.alg_type == 'sac':
            self.action_stds = nn.Parameter(labels.detach().clone(), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.action_stds)

    def diff_argmax(self, logits, dim=-1):
        tau = self.argmax_tau
        sample = self.use_gumbel_softmax
        
        if sample:
            gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            logits = logits + gumbels
        
        y_soft = (logits/tau).softmax(-1)
        # straight through
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft

        return ret     
    

    def fs_submodels(self, input):
        ## feature-selection sparse linear sub-controller
        if self.fs_submodel_version == 1:
            # version 1 was used for inverted pendulum in our paper's experiments
            # we found that version 1 has better performance than version 0 for inverted pendulum
            # version 1 and version 0 have the same forward pass procedure with only precision difference

            # sub_scalars, sub_weights, sub_biases: [num_leaves, output_dim, input_dim]
            w = self.sub_weights
            batch_size = input.size(0)
            num_leaves, output_dim, input_dim = w.size(0), w.size(1), w.size(2)
            
            new_w = 0
            for i in range(self.num_sub_features):
                if not i == 0:
                    w = w - w * onehot_weights
                # onehot_weights: [num_leaves, output_dim, input_dim]
                onehot_weights = self.diff_argmax(torch.abs(w))
                new_w = onehot_weights + new_w
                
            # new_s: [num_leaves, output_dim, input_dim]
            new_s = new_w * self.sub_scalars
            # new_i: [batch_size, num_leaves, output_dim, input_dim]
            new_i = new_w.unsqueeze(0).expand(batch_size, num_leaves, output_dim, input_dim) * input.unsqueeze(1).unsqueeze(2).expand(batch_size, num_leaves, output_dim, input_dim)
            # new_b: [num_leaves, output_dim]
            new_b = (new_w * self.sub_biases).sum(-1)
            # ret: [batch_size, num_leaves, output_dim]
            ret =(new_s * new_i).sum(-1) + new_b
            return ret
        else:
            # version 0 was used for all domains except inverted pendulum in our paper's experiments
            
            ret = []
            w = self.sub_weights
            
            for i in range(self.num_sub_features):
                if not i == 0:
                    w = w - w * onehot_weights
            
                # onehot_weights: [num_leaves, output_dim, input_dim]
                onehot_weights = self.diff_argmax(torch.abs(w))

                # new_w: [num_leaves, output_dim, input_dim]
                # new_s: [num_leaves, output_dim, 1]
                # new_b: [num_leaves, output_dim, 1]
                new_w = onehot_weights
                new_s = (self.sub_scalars * onehot_weights).sum(-1).unsqueeze(-1)
                new_b = (self.sub_biases * onehot_weights).sum(-1).unsqueeze(-1)

                # input: [batch_size, input_dim]
                # output: [num_leaves, output_dim, batch_size]
                output = new_s * torch.matmul(new_w, input.transpose(0, 1)) + new_b
                ret.append(output.permute(2, 0, 1))
            return torch.sum(torch.stack(ret, dim=0), dim=0)  


    def forward(self, input_data, embedding_list=None):
        # self.comparators: [num_node, 1]

        if self.hard_node:
            ## node crispification
            weights = torch.abs(self.layers)
            # onehot_weights: [num_nodes, num_leaves]
            onehot_weights = self.diff_argmax(weights)
            # divisors: [num_node, 1]
            divisors = (weights * onehot_weights).sum(-1).unsqueeze(-1)
            # fill 0 with 1
            divisors_filler = torch.zeros(divisors.size()).to(divisors.device)
            divisors_filler[divisors==0] = 1
            divisors = divisors + divisors_filler
            new_comps = self.comparators / divisors
            new_weights = self.layers * onehot_weights / divisors
            new_alpha = self.alpha
        else:
            new_comps = self.comparators
            new_weights = self.layers
            new_alpha = self.alpha

        # original input_data dim: [batch_size, input_dim]
        input_copy = input_data.clone()
        input_data = input_data.t().expand(new_weights.size(0), *input_data.t().size())
        # layers: [num_node, input_dim]
        # input_data dim: [batch_size, num_node, input_dim]
        input_data = input_data.permute(2, 0, 1)
        # after discretization, some weights can be -1 depending on their origal values
        # comp dim: [batch_size, num_node, 1]
        comp = new_weights.mul(input_data)
        comp = comp.sum(dim=2).unsqueeze(-1)
        comp = comp.sub(new_comps.expand(input_data.size(0), *new_comps.size()))
        if self.use_individual_alpha:
            comp = comp.mul(new_alpha.expand(input_data.size(0), *new_alpha.size()))
        else:
            comp = comp.mul(new_alpha)
        if self.hard_node:
            ## outcome crispification
            # sig_vals: [batch_size, num_node, 2]
            sig_vals = self.diff_argmax(torch.cat((comp, torch.zeros((input_data.size(0), self.layers.size(0), 1)).to(comp.device)), dim=-1))
            
            sig_vals = torch.narrow(sig_vals, 2, 0, 1).squeeze(-1)
        else:
            sig_vals = self.sig(comp)
        # sig_vals: [batch_size, num_node]
        sig_vals = sig_vals.view(input_data.size(0), -1)
        # one_minus_sig: [batch_size, num_node]
        one_minus_sig = torch.ones(sig_vals.size()).to(sig_vals.device)
        one_minus_sig = torch.sub(one_minus_sig, sig_vals)
        
        # left_path_probs: [num_leaves, num_nodes]
        left_path_probs = self.left_path_sigs.t()
        right_path_probs = self.right_path_sigs.t()
        # left_path_probs: [batch_size, num_leaves, num_nodes]
        left_path_probs = left_path_probs.expand(input_data.size(0), *left_path_probs.size()) * sig_vals.unsqueeze(
            1)
        right_path_probs = right_path_probs.expand(input_data.size(0),
                                                   *right_path_probs.size()) * one_minus_sig.unsqueeze(1)
        # left_path_probs: [batch_size, num_nodes, num_leaves]
        left_path_probs = left_path_probs.permute(0, 2, 1)
        right_path_probs = right_path_probs.permute(0, 2, 1)

        # We don't want 0s to ruin leaf probabilities, so replace them with 1s so they don't affect the product
        left_filler = torch.zeros(self.left_path_sigs.size()).to(left_path_probs.device)
        left_filler[self.left_path_sigs == 0] = 1
        right_filler = torch.zeros(self.right_path_sigs.size()).to(left_path_probs.device)
        right_filler[self.right_path_sigs == 0] = 1

        # left_path_probs: [batch_size, num_nodes, num_leaves]
        left_path_probs = left_path_probs.add(left_filler)
        right_path_probs = right_path_probs.add(right_filler)

        # probs: [batch_size, 2*num_nodes, num_leaves]
        probs = torch.cat((left_path_probs, right_path_probs), dim=1)
        # probs: [batch_size, num_leaves]
        probs = probs.prod(dim=1)

        if self.use_submodels and self.sparse_submodel_type == 1:
            # here we choose L1 regularization over L2 because L1 is more likely to push coefficients to zeros
            self.leaf_attn = probs.clone().detach().sum(dim=0) / input_data.size(0)
            if self.l1_hard_attn:
                # if only sample one leaf node's linear controller to perform L1 regularization for each update
                # this can be helpful in enforcing sparsity on each linear controller
                distribution = torch.distributions.Categorical(self.leaf_attn)
                attn_idx = distribution.sample()
                self.leaf_attn = torch.zeros(probs.size(1))
                self.leaf_attn[attn_idx] = 1
        
        if self.use_submodels:
            if self.sparse_submodel_type != 2:
                output = torch.zeros((self.num_leaves, input_data.size(0), self.output_dim)).to(self.device)
                # input_copy [batch_size, input_dim]
                for e, i in enumerate(self.lin_models):
                    output[e] = i(input_copy)
            else:
                output = self.fs_submodels(input_copy).transpose(0, 1)
            actions = torch.bmm(probs.reshape(-1, 1, self.num_leaves), output.transpose(0, 1))
            mus = actions.squeeze(1)
        else:
            # self.action_mus: [num_leaves, output_dim]
            # mus: [batch_size, output_dim]
            mus = probs.mm(self.action_mus)

        if self.alg_type == 'sac':
            stds = probs.mm(self.action_stds)
            stds = torch.clamp(stds, -20, 2).view(input_data.size(0), -1) 
            return mus, stds
        else:
            # TD3 here outputs deterministic policies during training
            mus = self.tanh(mus)
            return mus
