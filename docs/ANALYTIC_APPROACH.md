# Analytic Approach: ACIC 2026 Data Challenge

## Overview

This method was developed for the ACIC 2026 Data Challenge, which asks for conditional average treatment effects (CATEs) of four treatments (b, c, d, e) relative to a control (a), in datasets of 1,000 units with 40 covariates. We estimate them with a 2-component Hybrid ensemble that combines Bayesian linear regression with machine learning-based causal forests. This approach balances parametric interpretability with nonparametric flexibility, leveraging the complementary strengths of each estimator.

## Primary Estimator: 2-Component Hybrid Ensemble

Our estimator fuses two base learners: Horseshoe Bayesian regression and Generalized Random Forest (GRF).

### Component 1: Horseshoe Bayesian Regression

The Horseshoe component is a Bayesian linear model that posits a linear specification for the conditional average treatment effect:

$$\text{CATE}_w(x) = \gamma_w + x' \delta_w$$

where $w$ denotes the treatment contrast (e.g., treatment $b$ vs. reference treatment $a$), $\gamma_w$ is the intercept, and $\delta_w$ is a vector of covariate slopes with horseshoe priors. The horseshoe prior shrinks small effects toward zero while preserving the magnitude of large effects, providing automatic variable selection without explicit thresholding.

We implement a custom Gibbs sampler that produces 3,000 post-burn-in MCMC draws (following a 1,000-iteration burn-in period). Each draw yields posterior samples of $(\gamma_w, \delta_w)$, which we use to compute point estimates (posterior means) and posterior credible intervals. The linearity of this specification enables efficient computation of population-level effects (PATE) via integration over the empirical covariate distribution.

### Component 2: Generalized Random Forest

The GRF component employs a multi-arm causal forest (`grf::multi_arm_causal_forest`) with the following tuning parameters, determined via factorial experiment:

- **Number of trees**: 2,000
- **Minimum node size**: 20 (calibrated for 8% RMSE improvement over the GRF default of 5, p=0.015)

GRF provides heterogeneous treatment effect estimates that adapt locally to the covariate space, yielding both point estimates and variance-based confidence intervals without distributional assumptions.

### Ensemble Combination

**Point estimates** are computed as inverse-RMSE weighted averages of the Horseshoe and GRF estimates:

$$\hat{\tau}^{\text{hybrid}}_w(x) = w_{\text{GRF}} \cdot \hat{\tau}^{\text{GRF}}_w(x) + w_{\text{HS}} \cdot \hat{\tau}^{\text{HS}}_w(x)$$

where weights are set to $w_{\text{GRF}} = 0.524$ and $w_{\text{HS}} = 0.476$. These weights were calibrated using out-of-sample performance on 500 mirror DGP v2 datasets.

**Confidence intervals** employ a weighted blend of component-specific intervals with conformal calibration. We apply a conformal multiplier (CONFORMAL_MULTIPLIER = 1.10) to calibrate interval coverage to the nominal 95% level, providing insurance against potential misspecification in either component or against distributional mismatch between the simulated tuning data and the data the method is applied to.

The 2-component ensemble replaced earlier 3-component variants (HS+GRF+Bayesian Causal Forest, later HS+GRF+X-Learner). In a 500-dataset mirror DGP v2 benchmark it significantly outperformed the HS+GRF+X-Learner ensemble on all validation metrics: RMSE 0.376 vs. 0.394 (p<0.0001).

## Key Estimands

The approach targets four estimands for each treatment contrast $w$:

1. **Individual Conditional Average Treatment Effect (iCATE)**: $\hat{\tau}_w(x_i)$ for each unit $i=1,\ldots,1000$
2. **Sample Conditional Average Treatment Effect (sCATE)**: $n^{-1}\sum_{i=1}^n \hat{\tau}_w(x_i)$
3. **Subgroup Conditional Average Treatment Effect (subCATE)**: sCATE stratified by covariate $x_{12}$
4. **Population Average Treatment Effect (PATE)**: $\mathbb{E}_X[\hat{\tau}_w(X)]$, estimated by exploiting the linearity of the Horseshoe specification

### PATE Computation

A key innovation is proper estimation of PATE using the linear structure of the Horseshoe posterior:

$$\text{PATE}_w = \mathbb{E}_X[\gamma_w + X'\delta_w] = \gamma_w + \mathbb{E}_X[X]' \delta_w$$

For each MCMC draw $s$, PATE computation reduces to a single dot product, $\text{PATE}_w^{(s)} = \gamma_w^{(s)} + \bar{x}_{\text{pop}}'\delta_w^{(s)}$, of that draw's coefficient vector with the population covariate means. Population covariate means are supplied as an input (a vector of 54 means); for the bundled simulated data they are computed exactly from the simulator. The hybrid PATE combines the Horseshoe-derived PATE with the GRF sample average using the same inverse-RMSE weights as the iCATE ensemble.

## Covariate Processing

We construct a 54-dimensional design matrix from the raw 40 covariates as follows:

- **Categorical variables** ($x_1$–$x_6$): One-hot encoded with level 1 dropped as the reference category (20 indicator columns)
- **Binary variables** ($x_7$–$x_{25}$): Used as-is (19 columns)
- **Continuous variables** ($x_{26}$–$x_{40}$, including the zero-inflated $x_{38}$–$x_{40}$): Used as-is on their native [0, 1] scale (15 columns)

The same encoding is shared by both components. Inside the Horseshoe Gibbs sampler, design matrix columns are standardized to improve MCMC mixing, and posterior draws are rescaled to the original parameterization before CATE extraction.

## Validation Framework

We developed three simulated data-generating process (DGP) designs, each comprising 500 train/test dataset pairs with ground-truth iCATEs, to validate and tune the ensemble:

1. **Mirror DGP v1** (n=500): Uniform distribution over treatment effect types (constant, sparse linear, smooth nonlinear, step function, interaction)
2. **Mirror DGP v2** (n=500): a mostly-constant mix of treatment effect types (70% constant)
3. **Mirror DGP v3** (n=500): $x_{12}$-driven heterogeneity via additive shift and continuous interaction, designed to validate subCATE accuracy (the sCATE, subCATE, and PATE z-multipliers were tuned on 100 of these datasets)

All tuning and weighting was performed on these simulated datasets. In a 500-dataset out-of-sample benchmark on the v2 design, the 2-component Hybrid reached an iCATE RMSE of 0.376, roughly 95% interval coverage, and 87.5% treatment assignment accuracy (GRF alone: 0.381 RMSE, about 93% coverage, 85.2% accuracy).

## Example Results

The repository ships 50 simulated datasets with known true effects (`example/data/`). Running `bash run_example.sh` fits the ensemble to each one and scores it (in-sample, as in the contest task):

| Metric | Mean | Median |
|--------|------|--------|
| iCATE RMSE | 0.428 | 0.379 |
| iCATE 95% CI coverage | 0.943 | 0.970 |
| Treatment assignment accuracy | 0.886 | 1.000 |
| sCATE 95% CI coverage | 0.940 | 1.000 |
| PATE 95% CI coverage | 0.850 | 1.000 |

The example simulator draws outcome scale and R² from simple parametric defaults, so these numbers are not directly comparable to the tuning benchmark above. PATE intervals undercover: the PATE z-multiplier (2.0) was never tuned. See `docs/METHOD_DETAILS.md` for full details and limitations.

## Software

The implementation is in R and depends only on `grf`, `ranger`, and `sandwich`. See the repository README for setup and usage.
