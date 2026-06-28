from __future__ import annotations

import numpy as np
import gymnasium as gym

from Phase_1.models.reward import (
    CART_POS_THRESHOLD,
    POLE_ANGLE_THRESHOLD,
    is_terminal,
    reward,
)


def get_cartpole_thresholds(env: gym.Env) -> tuple[float, float]:
    """Read failure thresholds from the underlying Gymnasium CartPole env."""
    unwrapped = env.unwrapped
    return unwrapped.x_threshold, unwrapped.theta_threshold_radians


def env_is_failure_state(
    state: np.ndarray,
    *,
    x_threshold: float,
    theta_threshold: float,
) -> bool:
    """Match Gymnasium CartPole out-of-bounds check applied to state s."""
    x, _, theta, _ = state
    return (
        x < -x_threshold
        or x > x_threshold
        or theta < -theta_threshold
        or theta > theta_threshold
    )


def hand_picked_states(
    *,
    x_threshold: float,
    theta_threshold: float,
) -> list[tuple[str, np.ndarray]]:
    """States chosen to cover safe, clearly failed, and near-boundary cases."""
    return [
        ("safe interior", np.array([0.0, 0.0, 0.05, 0.0])),
        ("safe negative angle", np.array([0.5, -0.2, -0.1, 0.3])),
        ("pole fallen (+)", np.array([0.0, 0.0, 0.25, 0.0])),
        ("pole fallen (-)", np.array([0.0, 0.0, -0.3, 0.0])),
        ("cart off track (+)", np.array([2.5, 0.0, 0.05, 0.0])),
        ("cart off track (-)", np.array([-3.0, 0.0, 0.05, 0.0])),
        ("boundary cart (at limit)", np.array([x_threshold, 0.0, 0.05, 0.0])),
        ("boundary pole (at limit)", np.array([0.0, 0.0, theta_threshold, 0.0])),
        ("just inside cart", np.array([x_threshold - 0.01, 0.0, 0.05, 0.0])),
        ("just inside pole", np.array([0.0, 0.0, theta_threshold - 1e-4, 0.0])),
    ]


def verify_reward_sanity() -> None:
    safe = np.array([0.0, 0.0, 0.05, 0.0])
    fallen = np.array([0.0, 0.0, 0.5, 0.0])

    assert reward(safe) == 1.0, "Alive state should give reward 1.0"
    assert reward(fallen) == 0.0, "Terminal state should give reward 0.0"
    print("Reward sanity checks passed.")


def verify_terminal_alignment(
    *,
    x_threshold: float,
    theta_threshold: float,
    states: np.ndarray,
    labels: list[str] | None = None,
) -> tuple[int, int, list[str]]:
    """
    Compare is_terminal(s) against Gymnasium thresholds for each state.

    Returns (matches, total, mismatch_messages).
    """
    matches = 0
    mismatches: list[str] = []

    for i, state in enumerate(states):
        predicted = is_terminal(state)
        actual = env_is_failure_state(
            state,
            x_threshold=x_threshold,
            theta_threshold=theta_threshold,
        )
        label = labels[i] if labels is not None else f"sample {i}"

        if predicted == actual:
            matches += 1
        else:
            mismatches.append(
                f"  {label}: state={state.tolist()} "
                f"is_terminal={predicted}, env failure={actual}"
            )

    return matches, len(states), mismatches


def _print_results(title: str, matches: int, total: int, mismatches: list[str]) -> None:
    print(f"\n=== {title} ===")
    print(f"Matched {matches}/{total}")
    if mismatches:
        print("Mismatches:")
        print("\n".join(mismatches[:10]))
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    x_threshold, theta_threshold = get_cartpole_thresholds(env)
    print(
        f"Gymnasium thresholds: x={x_threshold}, "
        f"theta={theta_threshold}"
    )

    if CART_POS_THRESHOLD != x_threshold or POLE_ANGLE_THRESHOLD != theta_threshold:
        print(
            "WARNING: reward.py thresholds differ from Gymnasium:\n"
            f"  reward.py: x={CART_POS_THRESHOLD}, theta={POLE_ANGLE_THRESHOLD}\n"
            f"  gymnasium: x={x_threshold}, theta={theta_threshold}"
        )

    print("\n=== Reward sanity ===")
    verify_reward_sanity()

    picked = hand_picked_states(
        x_threshold=x_threshold,
        theta_threshold=theta_threshold,
    )
    picked_states = np.array([state for _, state in picked])
    picked_labels = [name for name, _ in picked]
    picked_matches, picked_total, picked_mismatches = verify_terminal_alignment(
        x_threshold=x_threshold,
        theta_threshold=theta_threshold,
        states=picked_states,
        labels=picked_labels,
    )
    _print_results("Hand-picked terminal cases", picked_matches, picked_total, picked_mismatches)

    data = np.load("Phase_1/data/transitions.npz")
    dataset_states = data["states"]
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(dataset_states), size=100, replace=False)
    sample_states = dataset_states[sample_idx]
    sample_matches, sample_total, sample_mismatches = verify_terminal_alignment(
        x_threshold=x_threshold,
        theta_threshold=theta_threshold,
        states=sample_states,
    )
    _print_results(
        "Random sample from transitions.npz (n=100)",
        sample_matches,
        sample_total,
        sample_mismatches,
    )

    print("\n=== Summary ===")
    print(f"Hand-picked: {picked_matches}/{picked_total} matched")
    print(f"Dataset sample: {sample_matches}/{sample_total} matched")

    if picked_mismatches or sample_mismatches:
        print("\nFAILED: terminal detector does not match Gymnasium.")
        raise SystemExit(1)

    print("\nPASSED: is_terminal aligns with Gymnasium on all checked states.")
