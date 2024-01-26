import copy
import logging
import random
from collections import defaultdict
from math import cos, sin
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text, Any

import numpy as np
from gymnasium.spaces import Box, Dict
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from matplotlib import pyplot as plt
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env
from highway_env import utils
Observation = TypeVar("Observation")
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.common.observation import GrayscaleObservation, TimeToCollisionObservation, LidarObservation, OccupancyGridObservation
def normalize_observation(obs):
    # 将obs的值限制在0-255的范围内
    obs = np.clip(obs, 0, 255)
    # 将obs转换为uint8类型
    obs = np.divide(obs, 255)
    obs = obs.astype(np.float32)
    return obs
def get_rllib_compatible_env(env_class, return_class=False):
    env_name = env_class.__name__

    class MA(env_class):
        def __init__(self, config, *args, **kwargs):
            env_class.__init__(self, config, *args, **kwargs)
            self.render_mode = config["render_mode"]
        # @property
        # def observation_space(self):
        #     ret = super(MA, self).observation_space
        #     print(ret)
        #     return ret
        def define_spaces(self) -> None:
            """
            Set the types and spaces of observation and action from config.
            """
            self.observation_type = observation_factory(self, self.config["observation"])
            self.action_type = action_factory(self, self.config["action"])
            # if isinstance(self.observation_space, GrayscaleObservation):
            self.observation_space = self.observation_type.space()
            if self.config["observation"]["type"]=="GrayscaleObservation":
                new_shape=(self.observation_space.shape[1],self.observation_space.shape[2],
                                                  self.observation_space.shape[0])
                self.observation_space=Box(low=0*self.observation_space.low.reshape(new_shape),
                                           high=self.observation_space.high.reshape(new_shape)/255,
                                           shape=new_shape,dtype=np.float32)
            # print(self.observation_space)
            # elif isinstance(self.observation_space, TimeToCollisionObservation):
            #     self.observation_space = self.observation_type.space()
            #     pass
            # elif isinstance(self.observation_space, LidarObservation):
            #     self.observation_space = self.observation_type.space()
            #     pass
            # elif isinstance(self.observation_space, OccupancyGridObservation):
            #     self.observation_space = self.observation_type.space()
            #     pass
            # else:
            #     self.observation_space = self.observation_type.space()

            #Box(0, 255, (128, 64, 1), uint8)
            self.action_space = self.action_type.space()

        def step(self, action: Union[Action, int]) -> tuple[Any, Any, Any, Any, Any]:
            obs, reward, terminated, truncated, info = super(MA, self).step(action)
            if self.config["observation"]["type"]=="GrayscaleObservation":
                return normalize_observation(obs.transpose(1,2,0)), reward, terminated, truncated, info
            else:
                return obs, reward, terminated, truncated, info

        def _reward(self, action: Action) -> float:
            """
            The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
            :param action: the last action performed
            :return: the corresponding reward
            """
            rewards = self._rewards(action)
            # print(rewards)
            reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
            # print("new_r:",reward)
            if self.config["normalize_reward"]:
                reward = utils.lmap(reward,
                                    [self.config["collision_reward"],
                                     self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                    [0, 1])
            reward *= rewards['on_road_reward']
            return reward
        def _rewards(self, action: Action):
            neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
            lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
                else self.vehicle.lane_index[2]
            # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
            forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
            scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

            if np.cos(self.vehicle.heading)>0.95:
                forward_reward=forward_speed/5
            elif np.cos(self.vehicle.heading)<0.5 and np.cos(self.vehicle.heading)>0:
                forward_reward=-1
            else:
                forward_reward=-3

            return {
                "collision_reward": float(self.vehicle.crashed),
                "right_lane_reward": lane / max(len(neighbours) - 1, 1),
                "high_speed_reward": np.clip(scaled_speed, 0, 1),
                "on_road_reward": float(self.vehicle.on_road),
                "forward_reward":forward_reward
            }

        def _is_terminated(self) -> bool:
            """The episode is over if the ego vehicle crashed."""
            return (self.vehicle.crashed or
                    self.config["offroad_terminal"] and not self.vehicle.on_road or np.cos(self.vehicle.heading)<0)

        def _is_truncated(self) -> bool:
            """The episode is truncated if the time limit is reached."""
            return self.time >= self.config["duration"]

        def reset(self,
                  *,
                  seed: Optional[int] = None,
                  options: Optional[dict] = None,
                  ) -> Tuple[Observation, dict]:
            obs, info=super(MA, self).reset(seed=seed, options=options)
            if self.config["observation"]["type"]=="GrayscaleObservation":
                return normalize_observation(obs.transpose(1,2,0)), info
            else:
                return obs, info
    MA.__name__ = env_name
    MA.__qualname__ = env_name
    register_env(env_name, lambda config: MA(config))
    if return_class:
        return env_name, MA
    else:
        return env_name



config = {
    # "observation": {
    #     "type": "GrayscaleObservation",
    #     "observation_shape": (64, 64),
    #     "stack_size": 1,
    #     "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
    #     "scaling": 1.75
    # },
    # "observation": {
    #     "type": "OccupancyGrid",
    #     "vehicles_count": 15,
    #     "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    #     "features_range": {
    #         "x": [-100, 100],
    #         "y": [-100, 100],
    #         "vx": [-20, 20],
    #         "vy": [-20, 20]
    #     },
    #     "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    #     "grid_step": [5, 5],
    #     "absolute": False
    # },
    # "observation": {
    #     "type": "TimeToCollision",
    #     "horizon": 20
    # },
    # "manual_control": True,
    "action": {
        "type": "ContinuousAction",
        "steering_range": [-np.pi/3, np.pi/3]
    },
    "render_mode": "rgb_array",
    'offroad_terminal':True,
    "collision_reward": -3,
    "right_lane_reward": 0.5,
    "high_speed_reward": 1,
    "on_road_reward": 0.2,
    "forward_speed": 1,
    "simulation_frequency": 15,
    "policy_frequency": 15,
}
if __name__ == "__main__":
    name, ENV = get_rllib_compatible_env(HighwayEnv, return_class=True)
    print(name)
    env = ENV(config)
    print(env.action_space)

    obs, info = env.reset()
    plt.ion()
    fig1 = plt.figure('frame')
    print(obs.shape)
    for _ in range(100):
        # action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step([0,0.1])
        # ax1 = fig1.add_subplot(1, 1, 1)
        # ax1.axis('off')  # 关掉坐标轴
        # ax1.imshow(obs[:,:,0], cmap=plt.get_cmap('gray'))
        # for i in range(obs.shape[0]):
        #     print(obs[i,:,0])
        plt.pause(0.1)
        # fig1.clf()
        env.render()
        if done:
            env.reset()
    plt.ioff()
