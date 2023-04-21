# https://gymnasium.farama.org/environments/toy_text/cliff_walking/

import gymnasium as gym

env = gym.make("CliffWalking-v0", render_mode="rgb_array")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    
    #env.render()
    print(observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()