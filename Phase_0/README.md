# Phase 0 — Prerequisites

**Goal:** Set up tooling and MDP vocabulary before building a world model.

Phase 0 has no dynamics network and no training loop. You run the real environment, watch the agent–environment loop, and confirm your stack works. Everything in [Phase 1](../Phase_1/) builds on this foundation.

For the full curriculum, see [THEORY_README.md](../THEORY_README.md). For a longer write-up of concepts and reflections, see [notes/phase0.md](../notes/phase0.md).

---

## What you are learning

A **world model** is a learned simulator: given state and action, it predicts what happens next. Before approximating dynamics with \(f_\theta\), you need to understand what the real environment gives you at each timestep.

| Concept | In CartPole |
|---------|-------------|
| **State** \(s_t\) | 4D observation vector (see below) |
| **Action** \(a_t\) | Discrete `{0, 1}` — push cart left / right |
| **Reward** \(r_t\) | +1 per step while pole stays upright |
| **Episode** | One run from `reset()` until `terminated` or `truncated` |
| **Policy** | Here: uniform random via `env.action_space.sample()` |

**Theory anchors**

- MDP: \((\mathcal{S}, \mathcal{A}, P, R, \gamma)\)
- Model-based RL: learn \(\hat{P}(s' \mid s, a)\) or \(\hat{s}' = f_\theta(s, a)\), then plan or augment data

**Reading:** Sutton & Barto, *Reinforcement Learning: An Introduction* — **Ch. 3** (finite MDPs), **Ch. 8.1–8.3** (planning with a model).

---

## Setup

From the repository root:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Verify Gymnasium imports:

```bash
python -c "import gymnasium; print(gymnasium.__version__)"
```

**Requirements:** Python 3.10+, `gymnasium[classic-control]`, `torch`, `numpy`, `matplotlib` (see [requirements.txt](../requirements.txt)).

---

## Run random CartPole

```bash
python Phase_0/run_cartpole_random.py
```

The script:

- Creates `CartPole-v1` with `render_mode="human"`
- Runs **10 episodes** with random actions
- Prints `steps` and `total_reward` per episode

Random episodes typically last **20–30 steps**. The pole falls quickly because actions are not meaningful — that is expected.

To run headless (no window), change `render_mode="human"` to `render_mode=None` in the script.

---

## CartPole state vector

Observation \(s \in \mathbb{R}^4\):

| Index | Name | Meaning |
|-------|------|---------|
| 0 | `cart_pos` | Cart position on track |
| 1 | `cart_vel` | Cart velocity |
| 2 | `pole_angle` | Pole angle from vertical (rad) |
| 3 | `pole_ang_vel` | Pole angular velocity |

CartPole is **fully observable**: this 4D vector is the Markov state. Phase 1 will learn \(\hat{s}_{t+1} = f_\theta(s_t, a_t)\) from transitions collected with the same random policy.

---

## Steps & done-when checklist

| Step | Task | Done when |
|------|------|-----------|
| 0.1 | Python 3.10+, venv, install dependencies | `import gymnasium` works |
| 0.2 | Read MDP basics: state, action, reward, policy, episode | You can define each term |
| 0.3 | Run `CartPole-v1` with random actions for 10 episodes | You have seen `obs`, `action`, `reward`, `terminated` |
| 0.4 | Read Sutton & Barto Ch. 3 (skim), Ch. 8.1–8.3 | You can explain model-based planning in one paragraph |

**Milestone checklist**

- [ ] Environment runs
- [ ] Sutton & Barto Ch. 8.1–8.3 read

There is **no code deliverable** beyond a working environment and the random-policy script.

---

## What's next

**Phase 1 — CartPole state world model:** collect \((s_t, a_t, s_{t+1}, r_t)\) transitions, train a small MLP for one-step prediction, measure multi-step rollout divergence, and plan with the learned model.

Suggested pace for Phase 0: about **1 week**.
