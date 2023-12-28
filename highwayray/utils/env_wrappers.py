import copy
import logging
import random
from collections import defaultdict
from math import cos, sin
from typing import Optional, Tuple, Dict, Any

import numpy as np
from gymnasium.spaces import Box, Dict
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env



def get_rllib_compatible_env(env_class, return_class=False):
    env_name = env_class.__name__

    class MA(env_class):
        def __init__(self, config, *args, **kwargs):
            env_class.__init__(self, config, *args, **kwargs)
        @property
        def observation_space(self):
            ret = super(MA, self).observation_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            id = random.sample(list(ret.keys()), 1)
            return ret[id[0]]

        @property
        def action_space(self):
            ret = super(MA, self).action_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            id = random.sample(list(ret.keys()), 1)
            return ret[id[0]]


        def reset(self,
                  *,
                  seed: Optional[int] = None,
                  options: Optional[dict] = None, ):
            return env_class.reset(self, seed=0)

