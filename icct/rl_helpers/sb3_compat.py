from typing import Dict

import gym
import torch as th
from torch import nn

try:
    # Preferred: re-export everything if available (newer stable-baselines3)
    from stable_baselines3.common.torch_layers import (
        BaseFeaturesExtractor,
        CombinedExtractor,
        FlattenExtractor,
        NatureCNN,
        create_mlp,
        get_actor_critic_arch,
    )
    # expose is_image_space if present
    try:
        from stable_baselines3.common.torch_layers import is_image_space  # type: ignore
    except Exception:
        is_image_space = None
except Exception:
    # Fallback: import available pieces and provide a minimal CombinedExtractor
    from stable_baselines3.common.torch_layers import (
        BaseFeaturesExtractor,
        FlattenExtractor,
        NatureCNN,
        create_mlp,
        get_actor_critic_arch,
    )
    try:
        # helper from SB3 preprocessing
        from stable_baselines3.common.preprocessing import is_image_space
    except Exception:
        def is_image_space(space: gym.Space) -> bool:  # type: ignore
            return getattr(space, 'shape', None) is not None and len(getattr(space, 'shape', ())) >= 2

    class CombinedExtractor(BaseFeaturesExtractor):
        """Minimal compatibility implementation of SB3's CombinedExtractor.

        It creates per-key extractors for Dict observation spaces and concatenates
        their outputs. For image subspaces, it uses NatureCNN; otherwise,
        it uses FlattenExtractor.
        """

        def __init__(self, observation_space: gym.spaces.Dict):
            assert isinstance(observation_space, gym.spaces.Dict)
            extractors = {}
            total_dim = 0
            for key, subspace in observation_space.spaces.items():
                if is_image_space and is_image_space(subspace):
                    ext = NatureCNN(subspace)
                else:
                    ext = FlattenExtractor(subspace)
                extractors[str(key)] = ext
                total_dim += int(ext.features_dim)

            super(CombinedExtractor, self).__init__(observation_space, int(total_dim))
            self.extractors = nn.ModuleDict(extractors)

        def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
            encoded = []
            # observation keys may be strings
            for key, extractor in self.extractors.items():
                obs = observations[key]
                encoded.append(extractor(obs))
            return th.cat(encoded, dim=1)
