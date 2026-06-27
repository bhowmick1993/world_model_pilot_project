from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from Phase_1.models.transition_mlp import SimpleTransitionMLP

def generate_data():
    """
    Generate data for rollout divergence experiment
    Returns:
    -------
    states: np.ndarray
        The states
    actions: np.ndarray
        The actions
    next_states: np.ndarray
        The next states
    """
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()

    actions = []
    states = []
    next_states = []

    terminated = truncated = False

    while not (terminated or truncated):
        states.append(obs)
        action = env.action_space.sample()  # random: 0=left, 1=right
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        next_states.append(obs)

    print("States shape: ", np.array(states).shape)
    print("Actions shape: ", np.array(actions).shape)
    print("Next states shape: ", np.array(next_states).shape)
    # save as npz file
    np.savez("rollout_divergence.npz", states=np.array(states), actions=np.array(actions), next_states=np.array(next_states))
    print("Rollout divergence saved to rollout_divergence.npz")
    return np.array(states), np.array(actions), np.array(next_states)

def predict_next_state(*, state: np.ndarray, action: int, model: torch.nn.Module, mean: np.ndarray, std: np.ndarray):
    """
    Predict the next state using the model
    Parameters:
    ----------
    state: np.ndarray
        The state
    action: int
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    Returns:
    -------
    next_state_unnormalized: np.ndarray
        The predicted next state
    """
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state)
    if isinstance(action, (int, np.integer)):
        action = torch.tensor(int(action), dtype=torch.long)
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean)
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std)
        
    state_norm = (state - mean) / (std + 1e-8)
    one_hot_action = torch.nn.functional.one_hot(action, num_classes=2)

    input  = torch.concatenate([state_norm, one_hot_action.float()], axis=0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        next_state = model(input).squeeze(0)
    
    next_state_unnormalized = next_state * std + mean
    return next_state_unnormalized.detach().numpy()

if __name__ == "__main__":
    print("Generating data for rollout divergence experiment...")
    states, actions, next_states = generate_data()
    print("Data generated successfully")
    try:
        model = torch.load("deep_learning_models/transition_mlp.pth", weights_only=False)
        print("Model loaded from deep_learning_models/transition_mlp.pth")
    except:
        print("Model not found, training model...")
        raise FileNotFoundError("Model not found")

    mean_std_data = np.load("Phase_1/data/mean_std.npz")
    mean = mean_std_data["mean"]
    std = mean_std_data["std"]
    print("Mean: ", mean)
    print("Std: ", std)

    predicted_next_states = []
    s_hat = states[0].copy()

   # Create the true trajectory (s_hat, s_1, s_2, ..., s_T)
    true_trajectory = np.vstack([s_hat, next_states])

    predicted_next_states.append(s_hat) # start with s_hat as the first predicted state
    print("Predicting next states...")
    for action in actions:
        s_hat = predict_next_state(state=s_hat, action=action, model=model, mean=mean, std=std)
        predicted_next_states.append(s_hat)

    print("Predicted next states shape: ", np.array(predicted_next_states).shape)
    print("True next states shape: ", true_trajectory.shape)

    # Error vs step k
    errors = np.linalg.norm(true_trajectory - predicted_next_states, axis=1)  # (T+1,)
    # or per dimension:
    errors_per_dim = (true_trajectory - predicted_next_states) ** 2
    print("Errors per dimension: ", errors_per_dim)

    # plot the errors per dimension
    dimensions = ["car_pos", "car_vel", "pole_angle", "pole_ang_vel"]
    for i in range(4):
        plt.plot(errors_per_dim[:, i], label=f"Error {dimensions[i]}")
        plt.xlabel("Step")
        plt.ylabel("Error")
        plt.title(f"Error vs Step for {dimensions[i]}")
        plt.legend()
        plt.savefig(f"Phase_1/images/rollout_divergence_error_vs_step_{dimensions[i]}.png")
        plt.close()

    # plot the errors
    plt.plot(errors, label="Error")
    plt.xlabel("Step")
    plt.ylabel("Error")
    plt.title("Error vs Step")
    plt.legend()
    plt.show()