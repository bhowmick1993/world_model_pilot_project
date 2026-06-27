# World Models — Theory-First Learning Path

A staged, hands-on curriculum for learning **world models** by building them. This repository is designed to be published as a **GitHub learning platform**: each phase is a module you implement, evaluate, and document before moving on.

**Primary focus:** understanding theory (dynamics, rollouts, planning, then latent generative models).  
**Secondary focus:** reproducible code, clear metrics, and visual proof that each idea works.

---

## Table of contents

1. [What you are learning](#what-you-are-learning)
2. [How to use this repo](#how-to-use-this-repo)
3. [Publishing as a GitHub learning platform](#publishing-as-a-github-learning-platform)
4. [Prerequisites (Phase 0)](#phase-0-prerequisites)
5. [Phase 1 — CartPole state model](#phase-1--cartpole-state-model)
6. [Phase 2 — MuJoCo continuous control](#phase-2--mujoco-continuous-control)
7. [Phase 3 — CarRacing latent generative model](#phase-3--carracing-latent-generative-model)
8. [Phase 4 — Modern latent world models](#phase-4--modern-latent-world-models)
9. [Repository structure](#repository-structure)
10. [Reading list](#reading-list)
11. [Glossary](#glossary)
12. [Milestone checklist](#milestone-checklist)

---

## What you are learning

A **world model** is a learned simulator: given current state (or observations) and actions, it predicts what happens next. Agents use it to **imagine** trajectories without acting in the real environment.

### Core loop (conceptual)

```text
         ┌─────────────┐
obs ───► │  Encoder    │ ───► latent z   (Phase 3+; skipped in Phase 1)
         └─────────────┘
                │
                ▼
         ┌─────────────┐
state ──►│  Dynamics f │ ───► predicted next state / z
action ─►└─────────────┘
                │
                ▼
         ┌─────────────┐
         │  Planner    │ ───► choose action (MPC, policy, etc.)
         └─────────────┘
```

### Three ideas you will prove in code (every phase)

| Idea | Question you answer |
|------|---------------------|
| **One-step prediction** | Does \(\hat{s}_{t+1} = f_\theta(s_t, a_t)\) fit the data? |
| **Multi-step divergence** | How fast does error grow when you chain predictions? |
| **Planning in imagination** | Can you pick good actions using only \(f_\theta\)? |

Phase 1 teaches these **without** vision or VAEs. Phase 3 adds **representation learning** on top of the same three ideas.

---

## How to use this repo

1. **Work phase by phase.** Do not skip evaluation scripts or milestone criteria.
2. **One theory note per phase.** After each phase, write a short `notes/phase-N.md` explaining what worked, what broke, and why.
3. **Tag releases** when a phase is complete (see [Publishing](#publishing-as-a-github-learning-platform)).
4. **Keep diffs small.** Each step should be one PR or one focused commit.

### Suggested pace

| Phase | Duration (guide) | Environment |
|-------|------------------|-------------|
| 0 | 1 week | — |
| 1 | 2–3 weeks | CartPole-v1 (state) |
| 2 | 2–3 weeks | MuJoCo (state obs) |
| 3 | 4–6 weeks | CarRacing-v2 (pixels) |
| 4 | Ongoing | Paper reproduction |

---

## Publishing as a GitHub learning platform

Structure this repository so others can follow the same path.

### 1. Top-level README.md (landing page)

Create a short `README.md` that points here:

- One-paragraph project goal
- Link to **THEORY_README.md** (this file) as the curriculum
- Quickstart: install, run Phase 1 data collection
- Table of phases with status badges (optional)
- License (MIT recommended for learning repos)

### 2. Branch or folder per phase (optional)

| Approach | When to use |
|----------|-------------|
| **Folders** `phase1/`, `phase2/`, … | Single repo, all history visible (recommended) |
| **Branches** `phase-1-cartpole`, … | Learners check out one branch at a time |
| **Tags** `v0.1-phase1-complete` | Mark completed milestones |

### 3. GitHub Issues as lesson cards

Create one issue per step (templates below). Close each when the step passes its success criteria.

**Issue title format:** `[Phase 1.2] Train one-step transition model`

**Issue body template:**

```markdown
## Goal
<one sentence>

## Theory
<2–3 bullet points, equations optional>

## Implementation
- [ ] file/script to add or run
- [ ] ...

## Success criteria
- [ ] metric or plot

## Further reading
- link
```

### 4. GitHub Releases

When a phase is done:

- Release name: `Phase 1 — CartPole World Model`
- Attach: example plots (rollout error curve, MPC vs random bar chart)
- Release notes: what you learned in 3–5 bullets

### 5. Discussions / Wiki (optional)

- **Q&A:** “Why does rollout error compound?”
- **Wiki:** Copy the glossary and key equations from this file

### 6. CI (later)

- Smoke test: `collect → train → eval` on CartPole with few episodes
- Keeps the learning platform runnable for new visitors

---

## Phase 0: Prerequisites

**Goal:** Minimal tooling and vocabulary before building.

### Steps

| Step | Task | Done when |
|------|------|-----------|
| 0.1 | Python 3.10+, venv, install `gymnasium`, `torch`, `numpy`, `matplotlib` | `import gymnasium` works |
| 0.2 | Read MDP basics: state, action, reward, policy, episode | Can define each term |
| 0.3 | Run `CartPole-v1` with random actions for 10 episodes | You have seen `obs`, `action`, `reward`, `terminated` |
| 0.4 | Read Sutton & Barto **Ch. 3** (finite MDPs) skim, **Ch. 8.1–8.3** (planning with a model) | You can explain “model-based planning” in one paragraph |

### Theory anchors

- MDP: \((\mathcal{S}, \mathcal{A}, P, R, \gamma)\)
- Model-based RL: learn \(\hat{P}(s' \mid s, a)\) or \(\hat{s}' = f_\theta(s,a)\), then plan or augment data

**No code deliverable** for Phase 0 beyond a working environment.

---

## Phase 1 — CartPole state model

**Goal:** Learn dynamics and planning **without** representation learning.  
**Environment:** `Gymnasium` → `CartPole-v1`  
**State:** \(s \in \mathbb{R}^4\) — `[cart_pos, cart_vel, pole_angle, pole_ang_vel]`  
**Action:** Discrete `{0, 1}` — left / right

### Why CartPole first

- Fully observable: \(s_t\) is Markov
- Low-dimensional: fit \(f_\theta\) with a small MLP
- Failures are about **dynamics and planning**, not blurry reconstructions

---

### Step 1.1 — Collect transition data

**Theory:** You need i.i.d. samples of \((s_t, a_t, s_{t+1}, r_t)\) from the environment distribution.

**Implementation:**

- Script: `phase1/data/collect_rollouts.py`
- Policy: uniform random over actions
- Target: 500–2000 episodes (~50k–200k transitions)
- Save: `phase1/data/transitions.npz` (or `.pt`)

**Success criteria:**

- [ ] Dataset loads and shapes are `(N, 4)`, `(N,)`, `(N, 4)`
- [ ] Histograms of each state dimension look reasonable (not collapsed)

**Notes to write:** Why random policy first? What happens if you only collect from a narrow policy?

---

### Step 1.2 — Train one-step transition model

**Theory:** Supervised learning for

\[
\hat{s}_{t+1} = f_\theta(s_t, a_t), \quad
\mathcal{L} = \frac{1}{N} \sum \| s_{t+1} - f_\theta(s_t, a_t) \|^2
\]

**Implementation:**

- Model: `phase1/models/transition_mlp.py` — MLP on `[one_hot(a); s]`
- Train: `phase1/train/train_transition.py`
- Normalize states (mean/std from train set) before training
- Train/val split: 90/10

**Success criteria:**

- [ ] Validation MSE reported per dimension
- [ ] Val MSE “low enough” (order 1e-3–1e-2 after normalization)
- [ ] Plot: predicted vs true \(s'\) for a few samples

**Optional extension:** Residual parameterization \(\hat{s}_{t+1} = s_t + \Delta_\theta(s_t, a_t)\)

---

### Step 1.3 — Multi-step rollout divergence

**Theory:** Good one-step error does **not** imply good long rollouts. Open-loop:

\[
\hat{s}_0 = s_0,\quad \hat{s}_{k+1} = f_\theta(\hat{s}_k, a_k)
\]

Error compounds because the model visits **off-manifold** states (covariate shift).

**Implementation:**

- Script: `phase1/eval/rollout_divergence.py`
- Take real episode: actions \(a_0,\ldots,a_{T-1}\), true states \(s_k\)
- Roll out \(\hat{s}_k\) from \(\hat{s}_0 = s_0\) with same actions
- Plot \(\|s_k - \hat{s}_k\|\) vs \(k\) for horizons 1, 5, 10, 20, 50

**Success criteria:**

- [ ] Error ≈ 0 at \(k=1\), grows with \(k\)
- [ ] Short written answer: *Why does error compound?*

**Experiments (document in notes):**

| Experiment | Question |
|------------|----------|
| Different start states | Where does the model fail first? |
| Teacher forcing vs free rolling | Same actions; error still grows — why? |

---

### Step 1.4 — Reward and termination model

**Theory:** Planning needs \(\hat{r}(s,a)\) and terminal signal aligned with the env.

CartPole: +1 per alive step. Terminal when:

- \(|\text{pole\_angle}| > 0.2095\) rad (12°)
- \(|\text{cart\_pos}| > 2.4\)

**Implementation:**

- `phase1/models/reward.py` — `is_terminal(s)`, `reward(s) -> 1 or 0`

**Success criteria:**

- [ ] Terminal detector matches env on **true** states (spot-check 100 states)

---

### Step 1.5 — Planning with learned model (MPC)

**Theory:** **Model predictive control (MPC)** with receding horizon:

1. Observe real \(s_t\)
2. Optimize action sequence \(a_{t:t+H-1}\) in the **learned** model
3. Execute only \(a_t\), replan at \(t+1\)

This is the continuous-state cousin of Sutton & Barto Ch. 8’s planning with a model.

**Implementation — Random shooting (start here):**

- Script: `phase1/eval/mpc_random_shooting.py`
- Each step: sample \(K\) action sequences of length \(H\)
- Roll out with \(f_\theta\), score sum of \(\hat{r}\) until terminal
- Pick best sequence, apply first action
- Hyperparams: \(K=100\)–\(1000\), \(H=10\)–\(25\)

**Success criteria:**

- [ ] Mean episode length **beats random policy** (random ≈ 20–30 steps)
- [ ] Table: random vs MPC (mean ± std over 50 episodes)
- [ ] Note: planning uses **only** \(f_\theta\), not env gradients

---

### Step 1.6 — Cross-entropy method (CEM) — optional upgrade

**Theory:** Refine a distribution over action sequences instead of pure random search.

**Implementation:**

- `phase1/eval/mpc_cem.py`
- Compare sample efficiency vs random shooting

**Success criteria:**

- [ ] CEM ≥ random shooting at same budget \(K\), or longer episodes with same compute

---

### Step 1.7 — Phase 1 wrap-up

**Deliverables:**

- [ ] `notes/phase1.md` — one-step fit, divergence, MPC results
- [ ] Plots in `phase1/figures/`
- [ ] Git tag: `v0.1-phase1-complete`
- [ ] GitHub Release with summary

**Phase 1 complete when:** You can explain all three core ideas (one-step, divergence, planning) using CartPole only.

---

## Phase 2 — MuJoCo continuous control

**Goal:** Same three ideas in **higher-dimensional, continuous** dynamics.  
**Environment:** e.g. `HalfCheetah-v4` or `Hopper-v4` (Gymnasium + MuJoCo)  
**State:** Proprioceptive observation vector (not pixels)

### Why MuJoCo after CartPole

| CartPole | MuJoCo |
|----------|--------|
| 4D state, discrete action | 10–30+ D state, continuous action |
| Simple nonlinear dynamics | Contacts, harder \(f_\theta\) |
| Discrete action MPC (enumerate or sample) | Continuous action MPC (sample or CEM) |

### Steps

| Step | Task | Success criteria |
|------|------|------------------|
| 2.1 | Collect random (or noisy) rollouts | Same pipeline as Phase 1 |
| 2.2 | Train \(f_\theta(s, a) \to s'\) — MLP or ensemble | One-step val MSE tracked |
| 2.3 | Rollout divergence vs horizon | Error curves; compare to Phase 1 |
| 2.4 | MPC with continuous actions (CEM recommended) | Beat random baseline on return |
| 2.5 | `notes/phase2.md` + release `v0.2-phase2-complete` | Document what got harder |

### Theory focus

- **Scaling:** input normalization, wider MLP, more data
- **Continuous actions:** planning in \(\mathbb{R}^{d_a}\)
- **Optional:** ensemble of models for uncertainty-aware MPC (literature touchpoint)

**Reading:** Skim Dreamer paper introduction — notice the same latent rollout loop, but without building Dreamer yet.

---

## Phase 3 — CarRacing latent generative model

**Goal:** Full **VAE + sequence model** world model on pixels.  
**Environment:** `CarRacing-v2` (Box2D, 96×96 RGB)  
**This phase matches your interest in latent generative models.**

### Architecture (World Models 2018 style)

```text
Phase A — Vision VAE (static)
  o_t ──► Encoder ──► z_t ──► Decoder ──► ô_t
  Loss: reconstruction + KL (ELBO)

Phase B — Latent dynamics (teacher forcing)
  (z_t, a_t, h_t) ──► RNN ──► p(z_{t+1})
  Loss: NLL in latent space (Gaussian or MDN)

Phase C — Imagination rollouts
  z_{t+1} ~ model, decode frames — measure error vs horizon

Phase D — Controller in latent space (optional)
  Train policy only in imagination (ES or small RL)
```

### Sub-phases and steps

#### Step 3.1 — Collect pixel rollouts

- Random policy on CarRacing
- Store `(o_t, a_t, o_{t+1})` — actions are continuous `[steer, gas, brake]`
- **Success:** ≥500 episodes; save sample frames for documentation

#### Step 3.2 — Frame VAE (no time yet)

**Theory — ELBO:**

\[
\mathcal{L} = \mathbb{E}_{q(z|o)}[\log p(o|z)] - D_{\mathrm{KL}}(q(z|o)\|p(z))
\]

- Conv encoder/decoder, latent dim 32–64
- Tune \(\beta\) in \(\beta\)-VAE; watch blur vs KL

**Success:**

- [ ] Reconstructions recognizable as track/car
- [ ] Explain blur: unimodal Gaussian decoder averages modes

**Read:** Kingma & Welling, VAE

#### Step 3.3 — Encode trajectories; train latent dynamics

**Theory:**

\[
h_t = \mathrm{RNN}(z_{1:t-1}, a_{1:t-1}), \quad
p(z_{t+1} \mid h_t, a_t)
\]

- Freeze VAE (or fine-tune later)
- Teacher forcing: \(z_t\) from encoder during training

**Success:**

- [ ] One-step latent NLL decreases on val set
- [ ] Compare deterministic MLP head vs Gaussian head

#### Step 3.4 — MDN-RNN (mixture density head)

**Theory:**

\[
p(z_{t+1} \mid h_t, a_t) = \sum_k \pi_k \mathcal{N}(z_{t+1}; \mu_k, \sigma_k)
\]

Multi-modal futures (e.g. left vs right turn) need mixtures.

**Success:**

- [ ] Document when MDN beats single Gaussian
- [ ] Sample multiple \(z_{t+1}\) from same \((z_t, a_t)\) — different futures

**Read:** World Models paper (V + M sections); Bishop, Mixture Density Networks

#### Step 3.5 — Imagination rollouts (no teacher forcing)

\[
\hat{z}_{t+1} \sim p(\cdot \mid \hat{z}_t, a_t, \hat{h}_t), \quad \hat{o}_t = \mathrm{Decode}(\hat{z}_t)
\]

**Success:**

- [ ] GIF: teacher-forced vs free-running side by side
- [ ] Plot: reconstruction / latent error vs horizon (1, 5, 10, 20)
- [ ] Written note: **exposure bias** (train on encoded \(z\), roll on predicted \(\hat{z}\))

#### Step 3.6 — Partial observability (optional)

- Stack 4 frames **or** RNN encoder
- Theory: single frame is not Markov; \(h_t\) carries velocity-like information

#### Step 3.7 — Latent controller (optional)

- Small policy \(\pi(a \mid z, h)\) trained only in imagination
- Evaluate on **real** CarRacing without env backprop

**Success:** Non-trivial lap progress vs random

#### Step 3.8 — Phase 3 wrap-up

- [ ] `notes/phase3.md`
- [ ] Release `v0.3-phase3-complete`

---

## Phase 4 — Modern latent world models

**Goal:** Connect your built intuition to current research.

### Suggested order

| Step | Topic | Action |
|------|-------|--------|
| 4.1 | Dreamer / RSSM | Read paper; map VAE+RNN to deterministic \(h\) + stochastic \(z\) |
| 4.2 | DreamerV3 | Read for unified recipe across domains |
| 4.3 | Reproduce subset | e.g. Dreamer on same MuJoCo task as Phase 2 |
| 4.4 | `notes/phase4.md` | Compare RSSM vs your Phase 3 pipeline |

**Theory question to answer:** What problem does RSSM solve that staged VAE + MDN-RNN does not?

---

## Repository structure

Target layout as you implement:

```text
world_model_pilot_project/
├── README.md                 # Landing page (link to THEORY_README.md)
├── THEORY_README.md          # This curriculum
├── requirements.txt
├── notes/
│   ├── phase1.md
│   ├── phase2.md
│   └── phase3.md
├── phase1/
│   ├── configs/cartpole.yaml
│   ├── data/collect_rollouts.py
│   ├── models/
│   │   ├── transition_mlp.py
│   │   └── reward.py
│   ├── train/train_transition.py
│   ├── eval/
│   │   ├── one_step_metrics.py
│   │   ├── rollout_divergence.py
│   │   ├── mpc_random_shooting.py
│   │   └── mpc_cem.py
│   └── figures/
├── phase2/
│   └── ...                   # Same pattern, MuJoCo
├── phase3/
│   ├── models/vae.py
│   ├── models/mdn_rnn.py
│   ├── train/train_vae.py
│   ├── train/train_dynamics.py
│   └── eval/imagination_rollout.py
└── .github/
    └── ISSUE_TEMPLATE/
        └── lesson.md         # Optional lesson card template
```

---

## Reading list

### Phase 0–1

| Resource | Chapters / focus |
|----------|------------------|
| Sutton & Barto, *Reinforcement Learning: An Introduction* | Ch. 3 (MDP), Ch. 8.1–8.3 (planning) |
| MPC / random shooting | Any short lecture notes on receding-horizon control |

### Phase 3 (latent generative)

| Resource | Focus |
|----------|--------|
| Kingma & Welling, *Auto-Encoding Variational Bayes* | ELBO, reparametrization |
| Ha & Schmidhuber, [World Models](https://worldmodels.github.io/) (2018) | VAE + MDN-RNN + imagination |
| Bishop, *Mixture Density Networks* | MDN head |
| Doersch, *Tutorial on Variational Autoencoders* | Optional intuition |

### Phase 4

| Resource | Focus |
|----------|--------|
| Hafner et al., Dreamer (2020) | RSSM, latent planning |
| Hafner et al., DreamerV3 (2023) | Scaled recipe |

---

## Glossary

| Term | Meaning |
|------|---------|
| **World model** | Learned approximation of environment dynamics (and optionally reward) |
| **One-step model** | \(\hat{s}_{t+1} = f_\theta(s_t, a_t)\) |
| **Open-loop rollout** | Chain model predictions without correcting with real observations |
| **Compounding error** | Rollout drift because inputs to \(f_\theta\) are model outputs, not real states |
| **MPC** | Optimize finite action sequence, apply first action, replan |
| **Teacher forcing** | Train sequence model with ground-truth \(z_t\) from encoder |
| **Exposure bias** | Train/inference mismatch when rolling with predicted \(\hat{z}_t\) |
| **ELBO** | Evidence lower bound; VAE training objective |
| **MDN** | Mixture density network; multimodal \(p(z_{t+1})\) |
| **RSSM** | Recurrent state-space model (Dreamer family) |

---

## Milestone checklist

Use this as the master tracker (copy into a GitHub Project or Issues).

### Phase 0

- [ ] Environment runs
- [ ] SB Ch. 8.1–8.3 read

### Phase 1 — CartPole

- [ ] 1.1 Data collected
- [ ] 1.2 One-step \(f_\theta\) trained
- [ ] 1.3 Rollout divergence plotted
- [ ] 1.4 Terminal/reward model verified
- [ ] 1.5 MPC beats random
- [ ] 1.6 (Optional) CEM
- [ ] 1.7 Notes + release `v0.1-phase1-complete`

### Phase 2 — MuJoCo

- [ ] 2.1–2.4 Pipeline ported
- [ ] 2.5 Notes + release `v0.2-phase2-complete`

### Phase 3 — CarRacing

- [ ] 3.1 Pixel rollouts
- [ ] 3.2 VAE
- [ ] 3.3 Latent dynamics (GRU)
- [ ] 3.4 MDN-RNN
- [ ] 3.5 Imagination rollouts
- [ ] 3.6 (Optional) Frame stack / partial observability
- [ ] 3.7 (Optional) Latent controller
- [ ] 3.8 Notes + release `v0.3-phase3-complete`

### Phase 4

- [ ] Dreamer / RSSM read and mapped to your code
- [ ] `notes/phase4.md`

---

## What to do right now

1. Create `README.md` (short landing page linking here).
2. Add `requirements.txt` with `gymnasium`, `torch`, `numpy`, `matplotlib`.
3. Open GitHub Issues for **Phase 1.1** through **1.5** using the lesson template above.
4. Implement **Step 1.1** — collect CartPole transitions.

When Phase 1 code exists, this repo becomes a walkthrough others can clone and follow phase by phase.

---

## License and attribution

When publishing: add `LICENSE` (MIT), cite Sutton & Barto and World Models paper in README, and note Gymnasium/MuJoCo licenses for envs.

---

*Last updated: curriculum version 1.0 — theory-first path, CartPole → MuJoCo → CarRacing (VAE + sequence) → Dreamer.*
