# Created by Yaru Niu

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy

from icct.rl_helpers.td3_policies import TD3Policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from icct.core.icct import ICCT
from stable_baselines3.common.type_aliases import Schedule


class DDTActor(BasePolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        ddt_kwargs: Dict[str, Any] = None,
    ):
        super(DDTActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.ddt_kwargs = ddt_kwargs

        action_dim = get_action_dim(self.action_space)
        last_layer_dim = features_dim
        self.ddt = ICCT(input_dim=features_dim,
                        output_dim=action_dim,
                        weights=None,
                        comparators=None,
                        leaves=ddt_kwargs['num_leaves'],
                        alpha=None,
                        use_individual_alpha=ddt_kwargs['use_individual_alpha'],
                        device=ddt_kwargs['device'],
                        use_submodels=ddt_kwargs['submodels'],
                        hard_node=ddt_kwargs['hard_node'],
                        argmax_tau=ddt_kwargs['argmax_tau'],
                        sparse_submodel_type=ddt_kwargs['sparse_submodel_type'],
                        fs_submodel_version = ddt_kwargs['fs_submodel_version'],
                        l1_hard_attn=ddt_kwargs['l1_hard_attn'],
                        num_sub_features=ddt_kwargs['num_sub_features'],
                        use_gumbel_softmax=ddt_kwargs['use_gumbel_softmax'],
                        alg_type=ddt_kwargs['alg_type']).to(ddt_kwargs['device'])
        print(self.ddt.state_dict())


    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        return self.ddt(features)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.forward(observation)


class DDT_TD3Policy(TD3Policy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            ddt_kwargs: Dict[str, Any] = None,
    ):
        self.ddt_kwargs = ddt_kwargs
        super(DDT_TD3Policy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DDTActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs['ddt_kwargs'] = self.ddt_kwargs
        return DDTActor(**actor_kwargs).to(self.device)

register_policy("DDT_TD3Policy", DDT_TD3Policy)