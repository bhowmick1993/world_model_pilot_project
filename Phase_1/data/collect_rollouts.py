from __future__ import annotations
import numpy as np
import gymnasium as gym
from tqdm import tqdm


def collect_rollouts(*,num_episodes: int):
    states = []
    actions = []
    next_states = []
    rewards = []

    env = gym.make("CartPole-v1")
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            states.append(obs)
            action = env.action_space.sample()  # random: 0=left, 1=right
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_states.append(obs)
            rewards.append(reward)
    return np.array(states), np.array(actions), np.array(next_states), np.array(rewards)

if __name__ == "__main__":
    states, actions, next_states, rewards = collect_rollouts(num_episodes=500)
    print("States shape: ", states.shape)
    print("Actions shape: ", actions.shape)
    print("Next states shape: ", next_states.shape)
    print("Rewards shape: ", rewards.shape)
    # save as npz file
    np.savez("transitions.npz", states=states, actions=actions, next_states=next_states, rewards=rewards)

