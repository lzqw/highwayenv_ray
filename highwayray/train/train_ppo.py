import os
import sys
from ray import tune
from highwayray.utils.train import train
from highwayray.utils.utils import get_train_parser
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import highway_env
import gymnasium as gym

if __name__ == "__main__":
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "TEST"

    stop = int(1000000)

    config = dict(

        env=tune.grid_search(
            [
                'highway-v0',
                'highway-fast-v0',
                'merge-v0',
                'roundabout-v0',
                'parking-v0',
                'intersection-v0'
            ]
        ),  # 此处设置需要训练的环境，可以同时训练多个环境，每个场景对应一个trial，
            # 每个场景使用num_rollout_workers数量的CPU和num_gpus数量的GPU

        env_config={
            'render_mode': 'rgb_array'
        },
        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=args.num_gpus_per_trial if args.num_gpus != 0 else 0,  # 此处设置每个trial使用的gpu数量
        train_batch_size=args.train_batch_size,
        num_rollout_workers=args.workers,  # 此处设置每个trial使用的rollout worker数量
        num_cpus_per_worker=args.num_cpus_per_worker,  # 此处设置每个rollout worker使用的cpu数量
    )

    train(
        PPO,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=DefaultCallbacks,
        checkpoint_freq=50,
        local_mode=False
    )
