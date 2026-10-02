# Hybrid Ensemble for Multi-Arm CATE Estimation: Technical Details

**Author:** Andrew McEachin

---

## Abstract

This document gives a self-contained technical description of the Hybrid ensemble implemented in this repository. The method is a 2-component ensemble that combines a Horseshoe prior Bayesian linear model (Component 1) with a multi-arm causal forest from the `grf` package (Component 2). Point estimates are combined by inverse-RMSE weighting (weights w_m = 1/RMSE_m, normalized to sum to 1; Horseshoe 47.6%, GRF 52.4%). Confidence intervals are blended with the same weights and then conformally calibrated (multiplier 1.10) to target 95% coverage. For each dataset the method estimates individual CATEs (iCATE), the sample CATE (sCATE), subgroup CATEs stratified by `x12` (subCATE), and the population ATE (PATE) for 4 treatment contrasts. Aggregate standard errors come from Horseshoe posterior draws, with z-multipliers tuned on simulated data. Treatment selection is provided both as the argmax of point estimates and as the arm with the highest posterior probability of being best (P(best)) under the Horseshoe joint posterior. All tuning and all results reported here use simulated data only.

---

## Table of Contents

1. [Problem Setup](#1-problem-setup)
2. [Covariate Preparation](#2-covariate-preparation)
3. [Component 1: Horseshoe Prior Linear Model](#3-component-1-horseshoe-prior-linear-model)
4. [Component 2: Multi-Arm Causal Forest (GRF)](#4-component-2-multi-arm-causal-forest-grf)
5. [Ensemble Combination](#5-ensemble-combination)
6. [Aggregate Estimands](#6-aggregate-estimands)
7. [Treatment Selection](#7-treatment-selection)
8. [Running the Code](#8-running-the-code)
9. [Validation and Calibration (Simulated Data)](#9-validation-and-calibration-simulated-data)
10. [Example Results on the Bundled Data](#10-example-results-on-the-bundled-data)
11. [Known Limitations](#11-known-limitations)
12. [References](#references)

---

## 1. Problem Setup

The method was developed for the ACIC 2026 Data Challenge. The task is defined per dataset: each dataset contains n = 1,000 observations, and each observation includes an identifier `ID`, an outcome y, a treatment assignment z in {a, b, c, d, e} (5 arms, with a as the control), and 40 covariates: 6 categorical (x1-x6), 19 binary (x7-x25), 12 continuous on [0, 1] (x26-x37), and 3 zero-inflated continuous on [0, 1] (x38-x40). Each dataset is analyzed independently; the method fits and predicts on the same file.

Four estimands are required for each of the four treatment contrasts w in {b, c, d, e} versus control arm a:

1. **iCATE** (individualized Conditional Average Treatment Effect): tau_w(x_i) = E[Y_i(w) - Y_i(a) | X_i = x_i] for each individual i, with point estimate and 95% confidence interval.
2. **sCATE** (sample CATE): the sample mean of the iCATEs, tau_bar_w = (1/n) * sum(tau_w(x_i)), with point estimate and 95% CI.
3. **subCATE** (subgroup CATE): the CATE stratified by binary covariate x12. The point estimate is the subgroup mean: subCATE_w(x) = (1/n_x) * sum_{i: x12_i=x}(tau_w(x_i)). The SE and CI construction use the n-denominator formula from the task definition (see Section 6.2).
4. **PATE** (Population Average Treatment Effect): PATE_w = E_X[tau_w(X)], estimated using population covariate means, with point estimate and 95% CI.

The task also asks for best-treatment selection for each individual: z*_i = argmax_w tau_hat_w(x_i) (or control a if all estimated effects are non-positive).

For each dataset, the code produces 5 CSV files:

- `dataset_<id>_icates.csv`: n x 12 matrix of iCATE estimates with CIs (4 contrasts x 3 columns)
- `dataset_<id>_aggregates.csv`: sCATE, subCATE, PATE estimates with CIs, plus best-treatment selections
- `dataset_<id>_best_icates.csv`: argmax treatment selection per individual
- `dataset_<id>_best_icates_pbest.csv`: P(best) treatment selection per individual
- `dataset_<id>_pbest_probs.csv`: full n x 5 posterior probability matrix

---

## 2. Covariate Preparation

All components share a canonical covariate encoding implemented in `R/prepare_covariates.R`. The 40 raw covariates are expanded to an n x 54 numeric matrix:

| Type | Variables | Columns | Encoding |
|------|-----------|---------|----------|
| Categorical | x1-x6 | 20 | One-hot, drop level 1 as reference |
| Binary | x7-x25 | 19 | Numeric 0/1 (as-is) |
| Continuous | x26-x40 | 15 | Numeric (as-is) |
| **Total** | | **54** | |

The categorical variable level counts are: x1 (3 levels, 2 dummies), x2 (4 levels, 3 dummies), x3 (3 levels, 2 dummies), x4 (5 levels, 4 dummies), x5 (4 levels, 3 dummies), x6 (7 levels, 6 dummies), totaling 20 dummy columns.

---

## 3. Component 1: Horseshoe Prior Linear Model

### 3.1 Model Specification

The Horseshoe component fits a Bayesian linear regression on the full interaction model:

> Y_i = X_i' * beta + sum_w(D_iw * gamma_w) + sum_w((X_i * D_iw)' * delta_w) + epsilon_i

where D_iw = 1[Z_i = w] is a treatment dummy. The outcome is centered (Y_centered = Y - mean(Y)) to eliminate the need for an intercept term. The design matrix has p = 274 columns:

- Columns 1-54: covariate main effects X (54 columns)
- Columns 55-58: treatment dummies D_b, D_c, D_d, D_e (4 columns)
- Columns 59-274: treatment-covariate interactions X x D (216 columns: 54 x 4)

The design matrix columns are standardized (divided by column SDs) before sampling to improve MCMC mixing; posterior draws are rescaled to the original parameterization before CATE extraction.

### 3.2 Prior Structure

The horseshoe prior (Carvalho, Polson & Scott, 2010) is applied selectively:

- Covariate main effects (X, 54 columns): **unpenalized** (flat/improper prior)
- Treatment dummies (D, 4 columns): **unpenalized** (flat/improper prior)
- Interactions (X x D, 216 columns): **penalized** (horseshoe prior)

For each penalized coefficient j, the horseshoe prior hierarchy is:

- beta_j | sigma^2, lambda_j, tau ~ N(0, sigma^2 * tau^2 * lambda_j^2)
- lambda_j ~ C+(0, 1)   (local shrinkage, half-Cauchy)
- tau ~ C+(0, 1)         (global shrinkage, half-Cauchy)

The local shrinkage parameters lambda_j allow individual interaction terms to escape global shrinkage when the data supports them, while the global parameter tau controls overall sparsity. Unpenalized coefficients receive no prior penalty, giving them OLS-like posteriors conditional on the penalized parameters.

This structure leaves the average treatment effects (gamma_w) unshrunk while regularizing the heterogeneous interaction terms (delta_w) in the n/p ~ 3.6 regime (n = 1,000, p_penalized = 216).

### 3.3 CATE Extraction

For each MCMC draw s and contrast w:

> tau_w^(s)(x_i) = gamma_w^(s) + x_i' * delta_w^(s)

Point estimates are posterior means; credible intervals are the 2nd and 98th percentiles (96% credible intervals, moderately widened from 95% to compensate for center mismatch between Horseshoe point estimates and the ensemble average).

### 3.4 Gibbs Sampler

The sampler implements the auxiliary-variable scheme of Makalic & Schmidt (2015) with the key modification that the horseshoe hierarchy (local shrinkage lambda_j, global shrinkage tau, and their auxiliary variables nu_j, xi) governs only the penalized subset of coefficients. The six conditional posteriors are all conjugate:

1. **beta | rest** ~ N_p(A^{-1} X'y, sigma^2 * A^{-1}) where A = X'X + Lambda*^{-1}
2. **sigma^2 | rest** ~ IG((n + p_pen)/2, [||y - X*beta||^2 + sum_pen(beta_j^2 / (tau^2 * lambda_j^2))] / 2)
3. **lambda_j^2 | rest** ~ IG(1, 1/nu_j + beta_j^2 / (2 * tau^2 * sigma^2)) for penalized j
4. **tau^2 | rest** ~ IG((p_pen + 1)/2, 1/xi + sum_pen(beta_j^2 / lambda_j^2) / (2 * sigma^2))
5. **nu_j | rest** ~ IG(1, 1/lambda_j^2 + 1) for penalized j
6. **xi | rest** ~ IG(1, 1/tau^2 + 1)

The diagonal penalty matrix Lambda*^{-1} has entry 1/(tau^2 * lambda_j^2) for penalized coefficients and 0 for unpenalized coefficients.

Beta is initialized at the ridge estimate (lambda = 1) for numerical stability, avoiding Cholesky failure at iteration 1 if X'X is singular.

### 3.5 MCMC Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Post-burnin draws | 3,000 | 3x base for stable 2%/98% quantiles |
| Burn-in | 1,000 | Discarded |
| Thinning | 1 | No thinning |
| Total iterations | 4,000 | |
| lambda_scale | 1 | Standard horseshoe |
| tau_scale | 1 | Standard horseshoe |

### 3.6 Strengths and Weaknesses

**Strengths:** Principled uncertainty quantification via the full posterior; interpretable linear coefficients; joint posterior across all 4 contrasts (shared MCMC chain) enables P(best) computation and model-based PATE estimation via the linearity property; out-of-sample RMSE 0.419 on simulation design v2 (Section 9.2).

**Weaknesses:** Assumes a linear CATE function tau_w(x) = gamma_w + x' * delta_w; systematic shrinkage of interaction terms produces negative bias; struggles with step-function and nonlinear CATEs.

**Implementation:** Custom R code in `R/horseshoe_gibbs.R` (sampler) and `R/fit_horseshoe.R` (wrapper).

---

## 4. Component 2: Multi-Arm Causal Forest (GRF)

### 4.1 Method

The GRF component uses `grf::multi_arm_causal_forest()` (Athey, Tibshirani & Wager, 2019), which estimates all four contrasts simultaneously via a single forest. The forest handles the multi-arm structure natively by splitting on a vector-valued treatment effect.

### 4.2 Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Number of trees | 2,000 | |
| Minimum node size | 20 | Tuned (default 5); p = 0.015 vs. default |
| Honesty | enabled | Default in `grf` |
| Threads | 1 | Single-threaded (parallelism at dataset level) |
| Seed | 42 | Fixed for reproducibility |

The `min.node.size = 20` was selected via a 2x2x2 factorial experiment (private research code) on 100 datasets from simulation design v2. A paired t-test showed p = 0.015 for the main effect of MNS = 20 vs. MNS = 5, with GRF standalone RMSE improving from 0.390 to 0.364 (8% gain).

### 4.3 Predictions and CIs

Point estimates are out-of-bag (OOB) predictions: calling `predict(cf, estimate.variance = TRUE)` without a `newdata` argument returns predictions based on trees where each observation was held out. (OOB is the forest's internal held-out mechanism, distinct from the out-of-sample evaluation on held-out simulated datasets used for ensemble weight calibration.) Confidence intervals are:

> tau_hat_w(x_i) +/- 1.96 * sigma_hat_w(x_i)

where sigma_hat^2_w(x_i) is the estimated asymptotic variance of the forest prediction.

### 4.4 Strengths and Weaknesses

**Strengths:** Nonparametric; handles complex CATE surfaces (interactions, nonlinearities); near-zero mean bias; OOB predictions avoid overfitting.

**Weaknesses:** Variance-based CIs can be too narrow in this setting; the forest's per-contrast predictions do not come with a joint posterior across contrasts, preventing direct P(best) computation; out-of-sample RMSE 0.381 on simulation design v2.

GRF also returns per-unit variance estimates, which are available for combined SE computation (Approach 3 in `R/compute_aggregate_se.R`), though the default configuration uses Approach 2 (Horseshoe posterior draws only) for aggregate SEs.

**Implementation:** R package `grf`, called directly in `R/fit_hybrid.R`.

---

## 5. Ensemble Combination

### 5.1 Point Estimates

The ensemble point estimate is an inverse-RMSE weighted average of the two components:

> tau_hat_hybrid(x_i) = (w_HS * tau_hat_HS(x_i) + w_GRF * tau_hat_GRF(x_i)) / (w_HS + w_GRF)

where w_m = 1/RMSE_m, normalized so they sum to 1. The RMSE values come from out-of-sample evaluation on 500 dataset pairs from simulation design v2 (private research code). "Out-of-sample" here means evaluation on held-out test data from the same simulated DGP (distinct from the GRF's internal out-of-bag mechanism described in Section 4.3):

| Component | Out-of-sample RMSE | Effective Weight |
|-----------|--------------------|------------------|
| Horseshoe | 0.419 | 47.6% |
| GRF | 0.381 | 52.4% |

### 5.2 Confidence Intervals

Confidence intervals are constructed in two steps:

**Step 1: Weighted blend.** The component CIs are blended with the same inverse-RMSE weights:

> tau_L_blend = (w_HS * tau_L_HS + w_GRF * tau_L_GRF) / (w_HS + w_GRF)
> tau_U_blend = (w_HS * tau_U_HS + w_GRF * tau_U_GRF) / (w_HS + w_GRF)

**Step 2: Conformal calibration.** The blended half-widths are scaled by a conformal calibration multiplier. The center is computed as the midpoint of the blended CI bounds, (tau_L_blend + tau_U_blend) / 2, which may differ slightly from the blended point estimate when component CIs are asymmetric (as the Horseshoe 2%/98% quantiles can be):

> tau_{L,U}_final = ci_center_blend -/+ CONFORMAL_MULTIPLIER * half_width_blend

The conformal multiplier was calibrated on 500 dataset pairs from simulation design v2 using 2,000,000 conformity scores. The empirical 95th percentile was 1.0450, which was conservatively bumped to **1.10** to insure against CATE-type distribution mismatch. Coverage is heterogeneous across CATE types (constant: 98.8%, interaction: 87.1%, smooth nonlinear: 76.4%); if the target data has fewer constant CATEs than the tuning simulations (70%), the empirical multiplier would undercover. At 1.10, the ensemble achieves approximately 96% coverage at 70% constant CATEs and 93-94% at 55% constant, with only 5.3% wider CIs.

### 5.3 Fallback Handling

If the Horseshoe fails (numerical issues in the Cholesky decomposition), the ensemble falls back to GRF-only with GRF CIs.

---

## 6. Aggregate Estimands

### 6.1 sCATE (Sample CATE)

The sCATE point estimate is the sample mean of iCATEs:

> sCATE_hat_w = (1/n) * sum(tau_hat_hybrid_w(x_i))

The standard error is computed from Horseshoe posterior draws (Approach 2). For each MCMC draw m, the sCATE is computed as tau_bar_w^(m) = (1/n) * sum_i(tau_w^(m)(x_i)), and the SE is the standard deviation across draws:

> SE(sCATE_hat_w) = sd({tau_bar_w^(m)}, m = 1..M)

Confidence intervals use a tuned z-multiplier: sCATE_hat_w +/- z * SE, where **z = 2.0** achieves 95.3% coverage on 100 datasets from simulation design v3 (private research code).

If Horseshoe posterior draws are unavailable (e.g., Horseshoe component failure), the SE falls back to Approach 1, which extracts per-unit variance from the calibrated iCATE confidence intervals: var_i = (half_width_i / 1.96)^2, then SE = sqrt(mean(var_i) / n).

### 6.2 subCATE (Subgroup CATE)

The subCATE is stratified by binary covariate x12. The **point estimate** is the subgroup mean (n_x denominator):

> subCATE_hat_w(x) = (1/n_x) * sum_{i: x12_i=x}(tau_hat_hybrid_w(x_i)),  x in {0, 1}

The **SE** uses the n-denominator formula from the task definition (full sample size n, not the subgroup size n_x), which treats subCATE as a population-level quantity. The SE is computed from Horseshoe posterior draws with the subgroup sum divided by n:

> SE(subCATE_hat_w(x)) = sd({(1/n) * sum_{i: x12_i = x}(tau_w^(m)(x_i))}, m = 1..M)

The subCATE z-multiplier is **z = 2.5**, wider than the sCATE multiplier because the posterior for subgroup means is slightly underdispersed relative to the full-sample mean (fewer observations per stratum). This achieves 95.4% average coverage across x12 strata on 100 datasets from simulation design v3.

### 6.3 PATE (Population Average Treatment Effect)

The PATE exploits the linearity of the Horseshoe model. Since tau_w(x) = gamma_w + x' * delta_w, the population average is:

> PATE_w = E_X[tau_w(X)] = gamma_w + x_bar_pop' * delta_w

where x_bar_pop is the 54-dimensional vector of population means of the encoded covariates (Section 2). These means are supplied by the user as a CSV of 54 values (one row per encoded column, with a `mean` column), passed as the optional third argument to `R/run_hybrid.R`. For the bundled simulated data, the means are computed exactly (analytically, not by sampling) from the simulator's parameters by `dgp/generate_data.R` and stored in `example/data/pop/x_pop_colmeans_<i>.csv`.

For each MCMC draw s, the PATE is computed as a single dot product:

> PATE_w^(s) = gamma_w^(s) + x_bar_pop' * delta_w^(s)

The hybrid PATE combines the Horseshoe PATE with a GRF proxy (the GRF sCATE, since E[sCATE] ~ PATE when the sample is drawn from the target population) using the same inverse-RMSE weights:

> PATE_hat_hybrid_w = (w_HS * PATE_hat_HS_w + w_GRF * sCATE_hat_GRF_w) / (w_HS + w_GRF)

The PATE SE propagates both Horseshoe and GRF variance through the ensemble weights:

> SE(PATE_hat_hybrid_w) = sqrt((w_HS/w_sum)^2 * SE_HS^2 + (w_GRF/w_sum)^2 * SE_GRF^2)

where SE_HS is the posterior SD of the PATE draws and SE_GRF is approximated by the sCATE SE.

PATE confidence intervals use **z = 2.0** (same as sCATE; not separately tuned, see Section 11). When population covariate means are not supplied, the code falls back to PATE = sCATE.

**Implementation:** `R/compute_pate.R` (Horseshoe PATE), `R/compute_aggregate_se.R` (SE computation).

---

## 7. Treatment Selection

The code provides two treatment selection methods.

### 7.1 Argmax Selection

For each individual, select the treatment with the largest estimated CATE, or control if no treatment improves on control:

> z*_i = argmax_w(tau_hat_w(x_i)) if max_w(tau_hat_w(x_i)) > 0, else 'a'

### 7.2 P(best) Selection

The posterior probability that each arm is optimal is computed from the Horseshoe joint posterior. For each MCMC draw m and individual i, we determine the winning arm among all 5 arms (with tau_a = 0 as the control baseline). The probability of each arm being best is:

> P(best_w | x_i) = (1/M) * sum_{m=1}^M 1[z*_i^(m) = w]

The selected treatment is z_hat_i^pbest = argmax_w P(best_w | x_i), with ties broken in favor of control.

**Why Horseshoe only for P(best):** Computing P(best) requires a joint posterior over all 4 contrasts simultaneously, because the "best" determination involves comparing draws from the same MCMC iteration. The Horseshoe provides this through its shared Gibbs chain. The GRF predictions for different contrasts come with per-contrast variance estimates only, so they lack the joint distributional structure needed for this calculation.

**Implementation:** `R/compute_p_best.R`.

---

## 8. Running the Code

All commands are run from the repository root. Dependencies are installed by `setup.sh`.

### 8.1 One dataset

```bash
Rscript R/run_hybrid.R <data_csv> <output_dir> [pop_means_csv] [seed]
```

- `data_csv`: input file in the task layout (`ID`, `y`, `z`, `x1`-`x40`). If it is named `data_<id>.csv`, `<id>` is used in output file names and as the default seed.
- `output_dir`: directory for the output CSVs (created if missing).
- `pop_means_csv` (optional): the 54 population covariate means (Section 6.3). Without it, PATE falls back to the sCATE.
- `seed` (optional): integer RNG seed (default: the dataset id).

Example:

```bash
Rscript R/run_hybrid.R example/data/train/data_1.csv example/output \
  example/data/pop/x_pop_colmeans_1.csv
```

Outputs (in `output_dir`):

| File | Contents |
|------|----------|
| `dataset_<id>_icates.csv` | iCATE estimates and 95% CIs for the 4 contrasts |
| `dataset_<id>_aggregates.csv` | sCATE, subCATE (by x12), PATE with CIs, and best arms |
| `dataset_<id>_best_icates.csv` | Best arm per unit (argmax of point estimates) |
| `dataset_<id>_best_icates_pbest.csv` | Best arm per unit (highest posterior P(best)) |
| `dataset_<id>_pbest_probs.csv` | P(each arm is best) per unit (n x 5) |

The script also prints a `TIMING|<id>|hybrid|<seconds>|<mean CI width>` line on stdout. Runtime is about 15 to 20 seconds per dataset on a single core.

### 8.2 Bundled example

```bash
bash run_example.sh [n_datasets] [workers]
```

Runs the ensemble on the first `n_datasets` (default 50, all) bundled simulated datasets in `example/data/train/`, passing the matching population means, with `workers` parallel R processes (default 4). Datasets that already have output are skipped. Outputs go to `example/output/`, and the script finishes by scoring them against the true effects.

### 8.3 Scoring against truth

```bash
Rscript R/evaluate.R <data_dir> <pred_dir>
```

Compares predictions with the ground truth written by the simulator (`truth/` and `metadata.csv` in `data_dir`). It reports iCATE RMSE, bias, 95% CI coverage, CI width, treatment assignment accuracy and regret, and absolute error and CI coverage for sCATE and PATE. Per-dataset results are written to `<pred_dir>/evaluation_by_dataset.csv`.

### 8.4 Generating more simulated data

```bash
Rscript dgp/generate_data.R [out_dir] [n_datasets] [preset]
```

Generates train/test dataset pairs that share one set of DGP parameters, plus true iCATEs, exact population covariate means, and a metadata file with the true PATE. Defaults: `out_dir = example/data`, `n_datasets = 50`, `preset = mostly_constant`. The preset sets the mix of CATE surface types across datasets:

| Preset | constant | sparse_linear | smooth_nonlinear | step_function | interaction |
|--------|----------|---------------|------------------|---------------|-------------|
| `mostly_constant` (the mix used for tuning) | 70% | 10% | 2% | 2% | 16% |
| `balanced` | 20% | 25% | 25% | 15% | 15% |

---

## 9. Validation and Calibration (Simulated Data)

All calibration and benchmarking used simulated datasets with known ground-truth CATEs.

### 9.1 Tuning Parameters

| Parameter | Symbol | Value | Provenance |
|-----------|--------|-------|------------|
| GRF trees | `NUM_TREES` | 2,000 | Default |
| GRF min node size | `MIN_NODE_SIZE` | 20 | Factorial experiment (see Section 4.2) |
| HS post-burnin draws | `N_MCMC` | 3,000 | 3x base for stable quantiles |
| HS burn-in | `BURNIN` | 1,000 | Standard practice |
| HS RMSE weight | `W_HS_RMSE` | 0.419 | 500 v2 pairs, out-of-sample |
| GRF RMSE weight | `W_GRF_RMSE` | 0.381 | 500 v2 pairs, out-of-sample |
| Conformal multiplier | `CONFORMAL_MULT` | 1.10 | 2M scores on v2, bumped from 1.045 |
| sCATE z-multiplier | `SCATE_Z_MULT` | 2.0 | 95.3% coverage on 100 v3 datasets |
| subCATE z-multiplier | `SUBCATE_Z_MULT` | 2.5 | 95.4% coverage on 100 v3 datasets |
| PATE z-multiplier | `PATE_Z_MULT` | 2.0 | Initial value, same as sCATE (not tuned) |

### 9.2 Simulation Designs

Three simulation designs were used (code comments in `R/` refer to them as "mirror DGP" v1, v2, and v3):

- **v1:** a balanced mix of CATE surface types; used for initial calibration and method development.
- **v2:** a mostly-constant mix (70% constant CATEs, matching the `mostly_constant` preset in Section 8.4); used for ensemble weight calibration and conformal multiplier tuning (500 train/test pairs).
- **v3:** a design with heterogeneity driven by x12; used for sCATE and subCATE z-multiplier calibration (100 datasets).

The tuning simulations drew the outcome scale and R² from distributions that differ from the parametric defaults in the public simulator (`dgp/generate_data.R`). The numbers in this section are therefore not directly comparable to the example results in Section 10.

### 9.3 2-Component vs. 3-Component Benchmark

A 500-pair benchmark on simulation design v2 (private research code) showed that the 2-component ensemble (Horseshoe + GRF) significantly outperforms a 3-component version (Horseshoe + GRF + X-Learner) on all metrics:

| Metric | 2-comp | 3-comp | p-value |
|--------|--------|--------|---------|
| RMSE | 0.376 | 0.394 | <0.0001 |
| Bias | -0.004 | +0.002 | 0.0002 |
| Assignment accuracy | 0.875 | 0.870 | 0.015 |
| Regret | 0.042 | 0.045 | 0.005 |

The 2-component version wins 79% of datasets on RMSE. The X-Learner (standalone RMSE 0.539) dilutes the two stronger estimators (Horseshoe = 0.419, GRF = 0.381), hurting all metrics.

---

## 10. Example Results on the Bundled Data

The repository ships 50 simulated datasets in `example/data/`, generated by `dgp/generate_data.R` with the `mostly_constant` preset and the simulator's parametric defaults. The table below reports in-sample results (fit and predict on `train/data_<i>.csv`, scored against `truth/train_truth_<i>.csv`) as produced by `bash run_example.sh`. Metrics are averaged over the 4 contrasts within each dataset, then summarized across the 50 datasets.

| Metric | Mean | Median |
|--------|------|--------|
| iCATE RMSE | 0.428 | 0.379 |
| iCATE bias | -0.013 | -0.024 |
| iCATE 95% CI coverage | 0.943 | 0.970 |
| iCATE CI width | 1.950 | 1.814 |
| Assignment accuracy | 0.886 | 1.000 |
| Regret | 0.053 | 0.000 |
| sCATE absolute error | 0.250 | 0.223 |
| sCATE 95% CI coverage | 0.940 | 1.000 |
| PATE absolute error | 0.252 | 0.233 |
| PATE 95% CI coverage | 0.850 | 1.000 |

iCATE and sCATE coverage are close to the 95% target. PATE coverage is below target (Section 11).

---

## 11. Known Limitations

- **PATE intervals are not calibrated.** `PATE_Z_MULT = 2.0` was copied from the sCATE multiplier and never tuned. On the bundled example, PATE intervals cover only 85% of the time against a 95% target.
- **The PATE SE is approximate.** SE_HS comes from the Horseshoe posterior, but the GRF part of the hybrid PATE uses the GRF sCATE as a proxy for its PATE, and its SE is approximated by the sCATE SE.
- **Coverage varies by CATE type.** iCATE coverage is strong for constant CATEs and weaker for smooth nonlinear and interaction surfaces (Section 5.2), where the linear Horseshoe component is misspecified.
- **Tuned on a mostly-constant mix.** Ensemble weights and the conformal multiplier were tuned on simulations with 70% constant CATEs. On data with more heterogeneous effects, the weights may not be optimal and iCATE intervals may undercover.

---

## References

1. Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. *Biometrika*, 97(2), 465-480.

2. Makalic, E. & Schmidt, D. F. (2015). A simple sampler for the horseshoe estimator. *arXiv:1508.03884v4*.

3. Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of Statistics*, 47(2), 1148-1178.

4. Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020). Bayesian regression tree models for causal inference: regularization, confounding, and heterogeneous effects. *Bayesian Analysis*, 15(3), 965-1056.
