from __future__ import annotations
import numpy as np
import gymnasium as gym

env = gym.make("Hopper-v4", render_mode="human")
num_episodes = 10

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()  # random: 0=left, 1=right
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

    print(f"Episode {episode + 1}: steps={step_count}, reward={total_reward}")
 
env.close()
