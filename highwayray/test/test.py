import argparse

import numpy as np
from highway_env.envs import HighwayEnv
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
import gymnasium as gym


from matplotlib import pyplot as plt
from IPython import display # 可视化gym
from matplotlib import animation # 可视化gym
import os

from utils.env_wrappers import get_rllib_compatible_env


def myMkdir(path):
    pathSplit = path.split("/")
    # print(pathSplit)
    pathAccu = 0
    for split in pathSplit:
        if pathAccu == 0:
            pathAccu = split
        else:
            pathAccu = pathAccu + "/" + split
        # print(pathAccu)
        if not os.path.exists(pathAccu):
            os.mkdir(pathAccu)
# 显示gym渲染窗口的函数，在运行过程中将 env.render() 替换为 show_state(env, step, info).
def show_state(env, step=0, n_frame=0, info=""):
    plt.figure(3)
    plt.clf()
    frame = env.render()
    plt.imshow(frame)
    plt.title("Step frame info: %d %d %s" % (step, n_frame, info))
    plt.axis('off')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    return frame

def display_frames_as_gif(frames, directory, imageName, path):
    imagePath = path + directory
    myMkdir(imagePath)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1)
    anim.save(imagePath + imageName, writer='ffmpeg', fps=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=[
        'highway-v0',
        'highway-fast-v0',
        'merge-v0',
        'roundabout-v0',
        'parking-v0',
        'intersection-v0'
    ], default="highway-fast-v0", type=str)
    parser.add_argument("--render_mode", type=str, default='rgb_array')
    parser.add_argument("--checkpoint_path", type=str, default="/home/lzqw/PycharmProjects/Highway_ray/highwayray/test/checkpoint_000049")
    args = parser.parse_args()
    env_config={
        'render_mode': 'rgb_array',
        # "observation": {
        #     "type": "GrayscaleObservation",
        #     "observation_shape": (240, 320),
        #     "stack_size": 1,
        #     "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        #     "scaling": 1.75,
        # },
        "action": {
            "type": "ContinuousAction",
            "steering_range": [-np.pi / 3, np.pi / 3],
            "longitudinal": True,
            "lateral": True,
        },
        'offroad_terminal': True,
        "collision_reward": -3,
        "right_lane_reward": 0.5,
        "high_speed_reward": 1,
        "on_road_reward": 0.2,
        "forward_reward": 2,
        "simulation_frequency": 15,
        "policy_frequency": 15,
    }
    config = dict(
        env=get_rllib_compatible_env(HighwayEnv),
        env_config=env_config,
        num_gpus=0,
    )
    policy_function = PPO(config)
    policy_function.restore(args.checkpoint_path )
    _,ENV=get_rllib_compatible_env(HighwayEnv,True)
    env=ENV(env_config)
    #env = gym.make(args.env)
    o, info = env.reset()
    d = {"__all__": False}

    gif_frame = []
    frames = 0
    # gif_frame.append(show_state(env, frames, len(gif_frame), args.env))
    r_e=0
    for i in range(1, 100000):
        a=policy_function.compute_single_action(o)
        # print(a)
        o,r,d,t,info=env.step(a)
        # print(np.cos(env.vehicle.heading))
        r_e+=r
        # frames += 1
        # # gif_frame.append(show_state(env, frames, len(gif_frame), args.env))
        if d:  # This is important!
            o, info = env.reset()
            # print(r_e)
            r_e=0
            d = False
        env.render()
    env.close()
