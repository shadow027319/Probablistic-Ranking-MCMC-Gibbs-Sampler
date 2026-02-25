# Probabilistic Ranking via Approximate Bayesian Inference

Bayesian skill estimation for professional tennis players using Gibbs sampling and Expectation Propagation on the TrueSkill factor graph. Built for Cambridge Engineering Part IIB, 4F13 Machine Learning (Rasmussen & Ge, 2024).

## Problem

Given 1801 binary match outcomes between 107 ATP players from the 2011 season, infer a posterior distribution over latent player skills and derive probabilistic predictions for pairwise matchups.

The generative model assumes each game outcome is determined by a noisy comparison of skill:

$$t_g = w_{I_g} - w_{J_g} + \epsilon, \quad \epsilon \sim \mathcal{N}(0,1), \quad w_i \sim \mathcal{N}(0, \sigma^2)$$

Player $i$ wins when $t_g > 0$. The inference task is to recover the posterior $p(\mathbf{w} \mid \text{outcomes})$.

## Methods

### Gibbs Sampling

Block Gibbs sampler alternating between:

- **Performance differences** $t_g \mid \mathbf{w}$: truncated Gaussian (rejection sampling)
- **Skills** $\mathbf{w} \mid \mathbf{t}$: conjugate Gaussian posterior via Cholesky decomposition of the precision matrix $A^\top A + \sigma^{-2}I$

Per-iteration complexity is $O(N + M^3)$, dominated by the Cholesky solve. Convergence diagnostics include autocorrelation analysis, integrated autocorrelation time (via `emcee`), and multi-chain initialisation to verify stationarity.

### Expectation Propagation

Message passing on the TrueSkill factor graph, iterating cavity computations and moment matching through the probit likelihood. Deterministic, with convergence assessed by monitoring the Gaussian natural parameters $(\mu, \tau)$ across iterations. Typically converges in 5–10 iterations.

## Key Results

- **Skill posteriors** recovered for all 107 players under both inference schemes, with Spearman rank correlations > 0.99 between Gibbs and EP rankings
- **Pairwise win probabilities** computed analytically from EP marginals and via Monte Carlo from Gibbs samples, accounting for both skill uncertainty and game noise
- **Comparison of skill estimation strategies** from Gibbs samples: independent Gaussian fits, joint Gaussian fits, and direct MC estimates — demonstrating that MC estimates are more robust to heavy-tailed sample distributions
- **Three-way ranking comparison** (empirical win rates, Gibbs predictions, EP predictions) showing that model-based approaches regularise effectively against sparse matchup data

## Structure

```
.
├── main.ipynb          # Full analysis notebook
├── gibbsrank.py        # Gibbs sampler implementation
├── eprank.py           # EP message passing implementation
├── tennis_data.mat     # 2011 ATP match data (107 players, 1801 games)
└── report.pdf          # Writeup
```

## Technical Details

| | Gibbs | EP |
|---|---|---|
| **Inference type** | MCMC (exact in the limit) | Deterministic approximate |
| **Convergence object** | Stationary distribution $p(\mathbf{w} \mid \mathbf{y})$ | Fixed-point Gaussian marginals |
| **Per-iteration cost** | $O(N + M^3)$ | $O(N + M)$ |
| **Burn-in** | ~50 iterations | N/A |
| **Thinning** | $\lfloor \tau_{\text{int}} \rfloor + 1$ | N/A |

## Dependencies

```
numpy, scipy, matplotlib, seaborn, pandas, emcee, tqdm
```

## Context

Part of the Cambridge Engineering Tripos (Information & Computer Engineering), covering probabilistic inference on graphical models. The TrueSkill model is a foundational approach to Bayesian skill rating, originally developed at Microsoft Research for Xbox Live matchmaking.
