from __future__ import annotations
import numpy as np

CART_POS_THRESHOLD = 2.4
# Match Gymnasium CartPole-v1: 12 * 2 * pi / 360
POLE_ANGLE_THRESHOLD = 12 * 2 * np.pi / 360

def is_terminal(state: np.ndarray) -> bool:
    """
    Check if the state is terminal
    Parameters:
    ----------
    state: np.ndarray
        The state
    Returns:
    -------
    bool
        True if the state is terminal, False otherwise
    """
    x, _, theta, _ = state
    return abs(x) > CART_POS_THRESHOLD or abs(theta) > POLE_ANGLE_THRESHOLD

def get_reward(state, action = None) -> float:
    """
    Calculate the reward for the state and action
    Parameters:
    ----------
    state: np.ndarray
        The state
    action: int, optional
        The action, if not provided, the reward is 1.0 if the state is not terminal, 0.0 otherwise
    Returns:
    -------
    float
        The reward
    """
    if action is not None:
        return 1.0 if not is_terminal(state) else 0.0
    else:
        return 1.0 if not is_terminal(state) else 0.0
