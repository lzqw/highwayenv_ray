import argparse
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
import gymnasium as gym
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=[
        'highway-v0',
        'highway-fast-v0',
        'merge-v0',
        'roundabout-v0',
        'parking-v0',
        'intersection-v0'
    ], default="highway-v0", type=str)
    parser.add_argument("--render_mode", type=str, default='rgb_array')
    parser.add_argument("--checkpoint_path", type=str, default="PATH2CHECKPOINT")
    args = parser.parse_args()
    config = dict(
        env=args.env,
        env_config={
            'render_mode': args.render_mode
        },
        num_gpus=0,
    )
    policy_function = PPO(config)
    policy_function.restore(args.checkpoint_path )
    env = gym.make(args.env)
    o, info = env.reset()
    d = {"__all__": False}

    for i in range(1, 100000):
        o, r, d, truncate, info = env.step(policy_function.compute_actions(o, policy_id="default"))

        if d["__all__"]:  # This is important!
            o, info = env.reset()
            d = {"__all__": False}
        env.render()
    env.close()
