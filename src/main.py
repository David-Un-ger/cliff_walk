# https://gymnasium.farama.org/environments/toy_text/cliff_walking/

import time

import gymnasium as gym
import numpy as np

env = gym.make("CliffWalking-v0")# , render_mode="human")#  , render_mode="rgb_array")

observation, info = env.reset()

q_table = np.zeros((env.observation_space.n, env.action_space.n))
steps_till_goal = 0
render = False
for _ in range(10000):
    #action = env.action_space.sample()  # agent policy that uses the observation and info
    action = np.argmax(q_table[observation])
    new_observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        reward = 100

    q_table[observation, action] = reward + np.max(q_table[new_observation])

    observation = new_observation
    # env.render()
    steps_till_goal += 1
    print(q_table)

    if terminated or truncated or reward == - 100:
        observation, info = env.reset()
        print(f"terminated in {steps_till_goal} steps", reward, observation, terminated)
        steps_till_goal = 0
        
    
    
        #env = gym.make("CliffWalking-v0", render_mode="human")
        #observation, info = env.reset()


env.close()