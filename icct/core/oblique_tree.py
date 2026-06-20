# Hinton-style Oblique Soft Decision Tree for continuous control
#
# Based on "Distilling a Neural Network Into a Soft Decision Tree"
# (Frosst & Hinton, 2017), adapted for RL with continuous actions.
#
# Each internal node performs an oblique (multi-feature) split:
#   p_i(x) = sigmoid(alpha * (w_i^T x + b_i))
# Each leaf has a learned constant action output.
# The tree output is a probability-weighted sum over leaf outputs.

import torch
import torch.nn as nn
import numpy as np


class ObliqueTree(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_leaves: int = 16,
                 use_individual_alpha: bool = False,
                 device: str = 'cpu',
                 hard_node: bool = False,
                 argmax_tau: float = 1.0,
                 use_gumbel_softmax: bool = False,
                 alg_type: str = 'sac'):
        """
        :param input_dim: observation dimensionality
        :param output_dim: action dimensionality
        :param num_leaves: number of leaves (must be power of 2)
        :param use_individual_alpha: per-node sigmoid temperature vs global
        :param device: 'cpu' or 'cuda'
        :param hard_node: straight-through hard routing at inference
        :param argmax_tau: temperature for hard routing
        :param use_gumbel_softmax: add Gumbel noise during hard routing
        :param alg_type: 'sac' (outputs mus + log_stds) or 'td3' (tanh mus)
        """
        super(ObliqueTree, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hard_node = hard_node
        self.argmax_tau = argmax_tau
        self.use_gumbel_softmax = use_gumbel_softmax
        self.use_individual_alpha = use_individual_alpha
        self.alg_type = alg_type

        depth = int(np.floor(np.log2(num_leaves)))
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_nodes = self.num_leaves - 1

        # ---- Internal node parameters ----
        self.node_weights = nn.Parameter(
            torch.randn(self.num_nodes, input_dim, device=device) * 0.1,
            requires_grad=True
        )
        self.node_biases = nn.Parameter(
            torch.randn(self.num_nodes, 1, device=device) * 0.1,
            requires_grad=True
        )

        # Alpha (sigmoid temperature)
        if use_individual_alpha:
            self.alpha = nn.Parameter(
                torch.ones(self.num_nodes, 1, device=device),
                requires_grad=True
            )
        else:
            self.alpha = nn.Parameter(
                torch.ones(1, device=device),
                requires_grad=True
            )

        # ---- Leaf parameters ----
        self.action_mus = nn.Parameter(
            torch.empty(self.num_leaves, output_dim, device=device),
            requires_grad=True
        )
        nn.init.xavier_uniform_(self.action_mus)

        if alg_type == 'sac':
            self.action_stds = nn.Parameter(
                torch.empty(self.num_leaves, output_dim, device=device),
                requires_grad=True
            )
            nn.init.xavier_uniform_(self.action_stds)

        if alg_type == 'td3':
            self.tanh = nn.Tanh()

        # ---- Path matrices (fixed, not learned) ----
        self._init_paths()

        self.sig = nn.Sigmoid()
        self.last_probs = None  # for visualization

    def _init_paths(self):
        """
        Build left/right path indicator matrices using BFS node ordering.
        left_path_sigs[n, l]  = 1 iff path to leaf l goes LEFT  at node n.
        right_path_sigs[n, l] = 1 iff path to leaf l goes RIGHT at node n.
        Node i has left child 2i+1 and right child 2i+2.
        """
        left_branches  = torch.zeros(self.num_nodes, self.num_leaves)
        right_branches = torch.zeros(self.num_nodes, self.num_leaves)

        for leaf_idx in range(self.num_leaves):
            node = self.num_nodes + leaf_idx
            while node > 0:
                parent = (node - 1) // 2
                if node == 2 * parent + 1:
                    left_branches[parent, leaf_idx] = 1.0
                else:
                    right_branches[parent, leaf_idx] = 1.0
                node = parent

        self.left_path_sigs  = nn.Parameter(left_branches.to(self.device),  requires_grad=False)
        self.right_path_sigs = nn.Parameter(right_branches.to(self.device), requires_grad=False)

        # on_path[n, l] = 1 iff node n is an ancestor of leaf l (precomputed for penalty)
        on_path = left_branches + right_branches
        self.on_path = nn.Parameter(on_path.to(self.device), requires_grad=False)

        # Depth-scaled lambda coefficients: lam * 2^(-depth), depth = floor(log2(i+1))
        depths = torch.tensor(
            [int(np.floor(np.log2(i + 1))) for i in range(self.num_nodes)],
            dtype=torch.float32
        )
        self.node_lam_scale = nn.Parameter((0.5 ** depths).to(self.device), requires_grad=False)

    def inner_node_penalty(self, sig_vals: torch.Tensor, probs: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
        """
        Hinton penalty (Eq. 4): binary cross-entropy at each inner node.

        alpha_i = reach-weighted mean routing probability at node i:
            alpha_i = sum_n [reach_i(x_n) * p_i(x_n)] / sum_n [reach_i(x_n)]

        penalty_i = -0.5 * log(alpha_i) - 0.5 * log(1 - alpha_i)

        Lambda is depth-scaled as in the paper: lam_i = lam * 2^(-depth_i),
        so deeper nodes are regularized less.

        :param sig_vals: routing probabilities at each node [batch_size, num_nodes]
        :param probs: leaf reach probabilities [batch_size, num_leaves]
        :param lam: base penalty weight
        """
        # reach[b, n] = probability of sample b reaching inner node n
        reach = torch.mm(probs, self.on_path.t())  # [B, num_nodes]

        # Reach-weighted average routing probability at each node
        alpha = (reach * sig_vals).sum(0) / (reach.sum(0) + 1e-8)  # [num_nodes]
        alpha = alpha.clamp(1e-8, 1.0 - 1e-8)

        # Binary cross-entropy penalty (maximised when alpha = 0.5, i.e. balanced split)
        penalty = -0.5 * (torch.log(alpha) + torch.log(1.0 - alpha))  # [num_nodes]

        return (self.node_lam_scale * lam * penalty).sum()

    def diff_argmax(self, logits, dim=-1):
        """Straight-through differentiable argmax with optional Gumbel noise."""
        if self.use_gumbel_softmax:
            gumbels = -torch.empty_like(logits).exponential_().log()
            logits = logits + gumbels
        y_soft = (logits / self.argmax_tau).softmax(dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft

    def forward(self, input_data, embedding_list=None):
        """
        :param input_data: [batch_size, input_dim]
        :return: SAC: (mus, log_stds) | TD3: tanh(mus)
        """
        activations = torch.mm(input_data, self.node_weights.t()) + self.node_biases.t()

        if self.use_individual_alpha:
            activations = activations * self.alpha.t()
        else:
            activations = activations * self.alpha

        if self.hard_node:
            comp = activations.unsqueeze(-1)
            sig_vals = self.diff_argmax(
                torch.cat((comp, torch.zeros_like(comp)), dim=-1)
            )
            sig_vals = sig_vals[:, :, 0]
        else:
            sig_vals = self.sig(activations)

        self.sig_vals = sig_vals  # stored for inner_node_penalty in train_distil
        one_minus_sig = 1.0 - sig_vals

        left_path_probs  = self.left_path_sigs.unsqueeze(0)  * sig_vals.unsqueeze(2)
        right_path_probs = self.right_path_sigs.unsqueeze(0) * one_minus_sig.unsqueeze(2)

        left_filler  = (self.left_path_sigs  == 0).float().unsqueeze(0)
        right_filler = (self.right_path_sigs == 0).float().unsqueeze(0)
        left_path_probs  = left_path_probs  + left_filler
        right_path_probs = right_path_probs + right_filler

        probs = (left_path_probs * right_path_probs).prod(dim=1)
        self.last_probs = probs.cpu().detach().numpy()
        self.probs = probs

        mus = torch.mm(probs, self.action_mus)

        if self.alg_type == 'sac':
            stds = torch.mm(probs, self.action_stds)
            stds = torch.clamp(stds, -20, 2)
            return mus, stds
        else:
            return self.tanh(mus)

    def train_distil(self,
                     teacher: nn.Module,
                     observations: torch.Tensor,
                     epochs: int = 200,
                     batch_size: int = 256,
                     lr: float = 3e-4,
                     lam: float = 0.1,
                     log_every: int = 20,
                     optimizer: torch.optim.Optimizer = None):
        """
        Distil a frozen teacher MLP into this tree (Frosst & Hinton 2017).

        Hinton-style distillation (Frosst & Hinton 2017) adapted for continuous control.

        Uses MSE between the tree's expected action and the oracle's pre-tanh mean μ_t,
        plus the Hinton inner node penalty for balanced routing:

            L = ||Σ_l reach_l(x) · μ_l - μ_t(x)||² + lam * inner_node_penalty

        The Hinton architecture (soft sigmoid routing, leaf outputs, reach-weighted
        inner node penalty) is preserved. See GMM NLL note in the loss computation
        for a more principled but less stable alternative.

        :param teacher: frozen oracle MLP actor (nn.Module)
        :param observations: [N, input_dim] tensor of environment observations
        :param epochs: number of passes over the observation dataset
        :param batch_size: mini-batch size
        :param lr: Adam learning rate (used only if optimizer is None)
        :param lam: leaf usage penalty weight
        :param log_every: print loss interval
        :param optimizer: optional external Adam instance (pass to avoid reset across chunks)
        :return: list of (epoch, loss) tuples for plotting
        """
        teacher = teacher.to(self.device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        observations = observations.to(self.device)
        N = observations.size(0)
        loss_log = []

        for epoch in range(1, epochs + 1):
            self.train()
            perm = torch.randperm(N, device=self.device)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, N, batch_size):
                obs = observations[perm[start:start + batch_size]]

                with torch.no_grad():
                    # obs is already extracted features (from get_sa_pair / collect_observations),
                    # so bypass extract_features and go directly to latent_pi -> mu.
                    # Calling get_action_dist_params(obs) would re-run extract_features,
                    # which fails for Dict-obs envs (e.g. lane_keeping uses CombinedExtractor).
                    latent_pi = teacher.latent_pi(obs)
                    t_mus = teacher.mu(latent_pi)

                # Forward to populate self.probs and self.sig_vals [B, L]
                self(obs)

                # MSE between tree's expected action and oracle's pre-tanh mean.
                # Tree's expected action = weighted sum of leaf means (same as forward output).

                # NOTE: An alternative is GMM NLL (Gaussian Mixture Model Negative Log-Likelihood):
                #   loss = -logsumexp_l [ log(reach_l) + log N(μ_t; μ_l, σ_l²) ]
                # GMM NLL is the principled probabilistic Hinton analog for continuous outputs
                # and encourages leaf specialization, but suffers from std collapse (loss → -inf).
                # MSE is simpler and directly optimizes the inference-time output.
                pred_mus = torch.mm(self.probs, self.action_mus)  # [B, D]
                loss = ((pred_mus - t_mus) ** 2).mean() + self.inner_node_penalty(self.sig_vals, self.probs, lam=lam)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            loss_log.append((epoch, avg_loss))

            if epoch % log_every == 0 or epoch == 1:
                print(f'[distil] epoch {epoch:4d} | loss {avg_loss:.6f}')

        return loss_log