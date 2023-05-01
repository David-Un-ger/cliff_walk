# https://gymnasium.farama.org/environments/toy_text/cliff_walking/

from typing import Tuple
from gymnasium import Env
import gymnasium as gym
from strategy_tdl import SimpleQLearningStrategy, QLearningStrategy, TDLStrategy


def prepare_env(render: bool = False) -> Tuple[Env, int, int]:
    '''
    Returns a reset cliff walk environment either (with or without rendering)
    with its initial obsevation.
    '''
    if render:
        env = gym.make("CliffWalking-v0", render_mode="human")
    else:
        env = gym.make("CliffWalking-v0")

    init_observation, info = env.reset()

    return env, init_observation


env, init_observation = prepare_env(render=False)
observation = init_observation

# strategy = SimpleQLearningStrategy(env.observation_space.n, env.action_space.n)
strategy = QLearningStrategy(env.observation_space.n,
                             env.action_space.n,
                             alpha=0.5,
                             gamma=1.0)

# Training loop
for _ in range(10000):
    env, observation = strategy.step(env, observation, train=True)
print("finale Q-table:\n", strategy)

# Test loop
env, init_observation = prepare_env(render=True)
observation = init_observation
env, observation = strategy.step(env, observation, train=False)
while observation != init_observation:
    env, observation = strategy.step(env, observation, train=False)

env.close()
