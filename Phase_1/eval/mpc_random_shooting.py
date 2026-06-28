from __future__ import annotations
import numpy as np
import gymnasium as gym
import torch
from Phase_1.models.transition_mlp import SimpleTransitionMLP
from Phase_1.eval.rollout_divergence import predict_next_state
from Phase_1.models.reward import get_reward, is_terminal
import matplotlib.pyplot as plt
import tqdm
# load the transition model
try:
    model = torch.load("Phase_1/deep_learning_models/transition_mlp.pth", weights_only=False)
    print("Transition model loaded from Phase_1/deep_learning_models/transition_mlp.pth")
except:
    raise FileNotFoundError("Transition model not found")

# load the mean and std
try:
    mean_std_data = np.load("Phase_1/data/mean_std.npz")
    mean = mean_std_data["mean"]
    std = mean_std_data["std"]
    print("Mean: ", mean)
    print("Std: ", std)
except:
    raise FileNotFoundError("Mean and std not found")


def sample_action_sequences(K: int, H: int):
    """
    Sample action sequences from the action space
    Parameters:
    ----------
    K: int
        The number of sequences to sample
    H: int
        The length of the sequences
    Returns:
    -------
    action_sequences: np.ndarray
        The action sequences
    """
    action_sequences = []
    for i in range(K):
        action_sequence = []
        for j in range(H):
            action_sequence.append(np.random.randint(0, 2)) # randomly pick 0 or 1
        action_sequences.append(action_sequence)
    return np.array(action_sequences)

def rollout_action_sequence(s0, action_sequence: np.ndarray):
    """
    Rollout the action sequence
    Parameters:
    ----------
    action_sequence: np.ndarray
        The action sequence
    Returns:
    -------
    total_reward: float
        The total reward
    """
    state = s0.copy()
    total_reward = 0
    for i in range(len(action_sequence)):
        if is_terminal(state):
            break
        state = predict_next_state(state=state, action=action_sequence[i], model=model, mean=mean, std=std)
        total_reward += get_reward(state)
        if is_terminal(state): # stop if the state is terminal. Stop imagining
            break
    return total_reward


def mpc_control(s0, K, H):
    """
    MPC control
    Parameters:
    ----------
    s0: np.ndarray
        The initial state
    K: int
    H: int
    Returns:
    -------
    action: int
        The action
    """
    action_sequences = sample_action_sequences(K, H)
    best_action_sequence = None
    best_reward = -np.inf

    for action_sequence in action_sequences:
        total_reward = rollout_action_sequence(s0, action_sequence)
        if total_reward > best_reward:
            best_reward = total_reward
            best_action_sequence = action_sequence
    best_action = best_action_sequence[0]
    return best_action, best_action_sequence, best_reward

def random_env_control(env, s0):
    """
    Random environment control
    Parameters:
    ----------
    env: gym.Env
        The environment
    s0: np.ndarray
        The initial state
    Returns:
    """
    total_reward = 0
    step_count = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()  # random: 0=left, 1=right
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
    return total_reward, step_count

if __name__ == "__main__":
    num_of_episodes = 50
    step_count_env_list = []
    step_count_mpc_list = []
    for episode in tqdm.tqdm(range(num_of_episodes)):
        env = gym.make("CartPole-v1")
        s0, _ = env.reset()

        total_reward_env, step_count_env = random_env_control(env, s0)
        step_count_env_list.append(step_count_env)
        env.close()

        env = gym.make("CartPole-v1")
        s0, _ = env.reset(seed=episode)
        K = 200
        H = 15
        total_reward = 0
        step_count = 0
        while True:
            best_action, best_action_sequence, best_reward = mpc_control(s0, K, H)
            obs, reward, terminated, truncated, info = env.step(best_action)
            total_reward += reward
            step_count += 1
            if terminated or truncated:
                break
            s0 = obs
        step_count_mpc_list.append(step_count)
        
    print("Step count environment: ", np.mean(step_count_env_list), np.std(step_count_env_list))
    print("Step count MPC: ", np.mean(step_count_mpc_list), np.std(step_count_mpc_list))

    # plot the step count of environment and MPC
    plt.plot(step_count_env_list, label="Environment")
    plt.plot(step_count_mpc_list, label="MPC")
    plt.xlabel("Episode")
    plt.ylabel("Step count")
    plt.title("Step count of environment and MPC")
    plt.legend()
    plt.savefig("Phase_1/images/step_count_environment_and_mpc.png")
    plt.close()