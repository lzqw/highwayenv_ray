import gymnasium as gym
from matplotlib import pyplot as plt

# get_rllib_compatible_env
env = gym.make('highway-v0', render_mode='rgb_array')
config = {
       "observation": {
           "type": "GrayscaleObservation",
           "observation_shape": (128, 64),
           "stack_size": 1,
           "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
           "scaling": 1.75,
       },
       "policy_frequency": 2,
        "action": {
            "type": "ContinuousAction"
        },
        'offroad_terminal':True,

   }

env.configure(config)
obs, info = env.reset()
for _ in range(100):
    # action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step([0,0.1])
    print(env.vehicle.on_road)
    if done:
        obs, info = env.reset()


    env.render()

plt.imshow(env.render())