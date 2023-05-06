# https://gymnasium.farama.org/environments/toy_text/cliff_walking/

from typing import Tuple
from gymnasium import Env
import gymnasium as gym
from strategy_tdl import SimpleQLearningStrategy, QLearningStrategy
import matplotlib.pyplot as plt


def prepare_env(render: bool = False) -> Tuple[Env, int, int]:
    '''
    Returns a reset cliff walk environment either (with or without rendering)
    with its initial obsevation.
    '''
    if render:
        env = gym.make("CliffWalking-v0", render_mode="human")
    else:
        env = gym.make("CliffWalking-v0")

    return env


# create environments
env_train = prepare_env(render=False)
env_run = prepare_env(render=True)

# create strategies
strategies = list()
strategies.append(
    SimpleQLearningStrategy(env_train.observation_space.n,
                            env_train.action_space.n))
strategies.append(
    QLearningStrategy(env_train.observation_space.n,
                      env_train.action_space.n,
                      alpha=0.5,
                      gamma=1.0))

# create figure
fig, axs = plt.subplots(len(strategies), 1, sharex=True, sharey=True)

# train and test strategies
for strategy, ax in zip(strategies, axs):
    strategy.train(env_train, max_nr_steps=2000)
    strategy.plot_training(ax)
    strategy.run(env_run)

env_train.close()
env_run.close()
plt.show()