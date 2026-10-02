# =============================================================================
# mirror_dgp.R: Simulated data-generating process (DGP) for multi-arm CATE
#
# This file contains all functions needed to generate simulated datasets in
# the ACIC 2026 Data Challenge file layout (ID, y, z, x1-x40), together with
# the true individual treatment effects. It is fully self-contained: base R
# only, zero external dependencies. No contest data is needed or used.
#
# Key features of the simulator:
#   - 5 treatment arms (a, b, c, d, e) with Dirichlet-distributed arm probs
#   - 40 covariates (6 categorical, 19 binary, 12 continuous, 3 zero-inflated)
#   - 4 CATE contrasts (b-a, c-a, d-a, e-a) with heterogeneous functional forms
#     (constant, sparse linear, smooth nonlinear, step function, interaction)
#   - Noise calibrated to hit a target R²
#   - Outcome location and scale (y_mean, y_sd) drawn per dataset
#   - Treatment effects proportional to outcome scale
#   - Treatment assignment independent of covariates (Z ⊥ X)
#
# Usage:
#   source("dgp/mirror_dgp.R")
#   params <- draw_acic_params_v2(seed = 42)     # parametric defaults
#   sim <- generate_acic_mirror_v2(n = 1000, params = params, seed = 100)
#   df  <- format_acic_csv(sim)                  # ID, y, z, x1-x40
#   tau <- sim$tau_true                          # N x 4 true CATEs
#
# See dgp/generate_data.R for a batch generator with train/test/truth output.
# =============================================================================


# =============================================================================
# COLUMN INDEX HELPERS
#
# The internal covariate matrix X has 54 columns:
#   Columns 1-20:  One-hot encoded categoricals (6 variables, drop-first encoding)
#   Columns 21-39: Binary covariates (x7-x25 in the ACIC format)
#   Columns 40-51: Continuous [0,1] covariates (x26-x37 in ACIC format)
#   Columns 52-54: Zero-inflated covariates (x38-x40 in ACIC format)
#
# These indices are used by compute_cate_surface() and other functions to
# select covariates by type.
# =============================================================================
IDX_CAT  <- 1:20     # One-hot encoded categorical columns
IDX_BIN  <- 21:39    # Binary columns
IDX_CONT <- 40:51    # Continuous [0,1] columns
IDX_ZI   <- 52:54    # Zero-inflated columns


# =============================================================================
# generate_acic_covariates(): Generate the 54-column covariate matrix X
#                             plus the raw categorical values for CSV export.
#
# The covariate structure follows the ACIC 2026 file layout:
#   - x1 (3 levels), x2 (4 levels), x3 (3 levels), x4 (5 levels),
#     x5 (4 levels), x6 (7 levels) — one-hot encoded into 20 columns
#   - x7-x25: 19 binary (Bernoulli with prob ∈ [0.2, 0.8])
#   - x26-x37: 12 continuous Uniform(0,1)
#   - x38-x40: 3 zero-inflated (point mass at 0 + Uniform(0,1))
#
# Args:
#   n          — sample size (default 1000)
#   bin_probs  — Bernoulli probabilities for x7-x25 (19-vector, or NULL for random)
#   zero_fracs — zero-inflation fractions for x38-x40 (3-vector, or NULL for random)
#
# Returns: list with:
#   X        — n×54 numeric matrix (one-hot + binary + continuous + ZI)
#   raw_cats — list(x1, x2, x3, x4, x5, x6) of integer vectors (raw levels)
# =============================================================================
generate_acic_covariates <- function(n, bin_probs = NULL, zero_fracs = NULL) {
  # Default random values if not provided
  if (is.null(bin_probs))  bin_probs  <- runif(19, 0.2, 0.8)
  if (is.null(zero_fracs)) zero_fracs <- runif(3, 0.25, 0.40)

  # Categorical covariates (6 variables, varying number of levels)
  x1 <- sample(1:3, n, replace = TRUE)    # x1: 3 levels
  x2 <- sample(1:4, n, replace = TRUE)    # x2: 4 levels
  x3 <- sample(1:3, n, replace = TRUE)    # x3: 3 levels
  x4 <- sample(1:5, n, replace = TRUE)    # x4: 5 levels
  x5 <- sample(1:4, n, replace = TRUE)    # x5: 4 levels
  x6 <- sample(1:7, n, replace = TRUE)    # x6: 7 levels

  # One-hot encode (drop level 1 as reference category)
  encode_onehot <- function(x, n_levels) {
    sapply(2:n_levels, function(lev) as.numeric(x == lev))
  }
  X_cat <- cbind(
    encode_onehot(x1, 3),  # 2 cols
    encode_onehot(x2, 4),  # 3 cols
    encode_onehot(x3, 3),  # 2 cols
    encode_onehot(x4, 5),  # 4 cols
    encode_onehot(x5, 4),  # 3 cols
    encode_onehot(x6, 7)   # 6 cols
  )  # Total: 20 columns

  # Binary covariates (x7-x25): 19 variables
  X_bin <- sapply(bin_probs, function(p) rbinom(n, 1, p))

  # Continuous [0,1] covariates (x26-x37): 12 variables
  X_cont <- matrix(runif(n * 12), nrow = n, ncol = 12)

  # Zero-inflated covariates (x38-x40): 3 variables
  X_zi <- matrix(0, nrow = n, ncol = 3)
  for (j in 1:3) {
    nonzero <- rbinom(n, 1, 1 - zero_fracs[j])
    X_zi[, j] <- nonzero * runif(n)
  }

  # Assemble full numeric matrix (54 columns)
  X <- cbind(X_cat, X_bin, X_cont, X_zi)
  colnames(X) <- c(
    paste0("cat1_L", 2:3), paste0("cat2_L", 2:4), paste0("cat3_L", 2:3),
    paste0("cat4_L", 2:5), paste0("cat5_L", 2:4), paste0("cat6_L", 2:7),
    paste0("bin", 1:19), paste0("cont", 1:12), paste0("zi", 1:3)
  )

  list(
    X = X,
    raw_cats = list(x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6)
  )
}


# =============================================================================
# compute_cate_surface(): Compute the N×4 matrix of true CATEs from covariates.
#
# Each of the 4 contrasts (b-a, c-a, d-a, e-a) has an independently drawn
# CATE type and coefficient set. The 5 CATE types are:
#   - constant:         tau_j(x) = effect
#   - sparse_linear:    tau_j(x) = intercept + beta1*x_cont + beta2*x_cont + beta3*x_bin
#   - smooth_nonlinear: tau_j(x) = sin(2*pi*x) + x² + x1*x2 + intercept
#   - step_function:    tau_j(x) = {hi if x > threshold, lo otherwise} + beta*x_bin
#   - interaction:      tau_j(x) = beta1*x_cat*x_cont + beta2*x_bin*x_cont + beta3*x_cont + intercept
#
# Args:
#   X      — n×54 covariate matrix (from generate_acic_covariates)
#   params — list with $cate_types (named char[4]) and $cate_coefs (list of 4)
#
# Returns: n×4 matrix with columns "b","c","d","e"
# =============================================================================
compute_cate_surface <- function(X, params) {
  n <- nrow(X)
  tau_true <- matrix(0, nrow = n, ncol = 4)
  colnames(tau_true) <- c("b", "c", "d", "e")

  for (j in 1:4) {
    cc <- params$cate_coefs[[j]]
    ct <- params$cate_types[j]

    # Select covariate indices for this contrast — each contrast uses
    # different covariates to avoid collinearity between contrasts.
    ci1 <- IDX_CONT[min(1 + (j - 1) * 2, 12)]  # continuous index 1
    ci2 <- IDX_CONT[min(2 + (j - 1) * 2, 12)]  # continuous index 2
    ci3 <- IDX_CONT[min(3 + (j - 1) * 2, 12)]  # continuous index 3
    bi1 <- IDX_BIN[min(j, 19)]                   # binary index

    if (ct == "constant") {
      tau_true[, j] <- cc$effect

    } else if (ct == "sparse_linear") {
      tau_true[, j] <- cc$intercept +
        cc$coef_cont1 * X[, ci1] +
        cc$coef_cont2 * X[, ci2] +
        cc$coef_bin1  * X[, bi1]

    } else if (ct == "smooth_nonlinear") {
      tau_true[, j] <- cc$coef_sin  * sin(2 * pi * X[, ci1]) +
        cc$coef_quad * X[, ci2]^2 +
        cc$coef_interact * X[, ci1] * X[, ci2] +
        cc$intercept

    } else if (ct == "step_function") {
      tau_true[, j] <- ifelse(X[, ci1] > cc$threshold,
                               cc$hi_effect, cc$lo_effect) +
        cc$coef_bin1 * X[, bi1]

    } else if (ct == "interaction") {
      tau_true[, j] <- cc$coef1 * X[, IDX_CAT[min(j, 20)]] * X[, ci1] +
        cc$coef2 * X[, bi1] * X[, ci2] +
        cc$coef3 * X[, ci3] +
        cc$intercept
    }
  }
  tau_true
}


# =============================================================================
# draw_complex_mu_a_spec(): Draws random mu_a specification parameters.
#
# Creates a complex prognostic surface using 6-10 randomly selected covariates
# with linear + quadratic terms and random coefficients. This produces
# var(mu_a) in a realistic range (varies by draw) instead of a fixed low value.
#
# Args:
#   sub_seed — seed for reproducibility (derived from the rep's param_seed)
#
# Returns: list(cols, coefs, quad_idx, intercept) — the mu_a_spec structure
#   cols      — which of the 54 covariate columns to use
#   coefs     — linear coefficients for each covariate
#   quad_idx  — which of the selected covariates get quadratic terms
#   intercept — baseline outcome level
# =============================================================================
draw_complex_mu_a_spec <- function(sub_seed) {
  set.seed(sub_seed)

  # Number of covariates in the outcome surface (6-10)
  n_mu_covs <- sample(6:10, 1)

  # Select covariate indices from the full set (1:54)
  mu_cov_cols <- sample(1:54, n_mu_covs)

  # Linear coefficients (random signs and magnitudes)
  mu_coefs <- rnorm(n_mu_covs, sd = 1.0)

  # Nonlinear terms: randomly select 1-3 covariates for quadratic effects
  n_quad <- sample(1:min(3, n_mu_covs), 1)
  quad_idx <- sample(seq_len(n_mu_covs), n_quad)

  # Intercept (base level of outcome)
  mu_intercept <- rnorm(1, mean = 5, sd = 3)

  list(
    cols      = mu_cov_cols,
    coefs     = mu_coefs,
    quad_idx  = quad_idx,
    intercept = mu_intercept
  )
}


# =============================================================================
# compute_complex_mu_a(): Builds a complex random outcome surface from
# a mu_a_spec parameter structure.
#
# The mu_a_spec contains:
#   cols      — which of the 54 covariate columns to use
#   coefs     — linear coefficients for each covariate
#   quad_idx  — which of the selected covariates get quadratic terms
#   intercept — baseline outcome level
#
# Returns: length-n vector of mu_a values.
# =============================================================================
compute_complex_mu_a <- function(X, mu_a_spec) {
  n <- nrow(X)
  mu_a <- rep(mu_a_spec$intercept, n)
  for (k in seq_along(mu_a_spec$cols)) {
    mu_a <- mu_a + mu_a_spec$coefs[k] * X[, mu_a_spec$cols[k]]
  }
  for (k in mu_a_spec$quad_idx) {
    mu_a <- mu_a + 0.5 * mu_a_spec$coefs[k] * X[, mu_a_spec$cols[k]]^2
  }
  mu_a
}


# =============================================================================
# draw_acic_params_v2(): Draws ALL DGP-level parameters for one dataset.
#
# This function is called ONCE per dataset. It determines the "structure" of
# the dataset: arm probabilities, CATE types, CATE coefficients, prognostic
# surface, outcome scale, and R². All observation-level randomness is deferred
# to generate_acic_mirror_v2().
#
# The separation of "structure" (this function) from "observations"
# (generate_acic_mirror_v2) is critical: calling this function once with seed=k
# and then calling generate_acic_mirror_v2 twice with different seeds produces
# a train/test pair with the SAME underlying DGP but DIFFERENT observations.
#
# Args:
#   seed        — RNG seed for reproducibility
#   real_r2     — optional numeric vector of R² values to resample from.
#                 NULL (default) uses the parametric default described below.
#   real_y_mean — optional numeric vector of y_mean values to resample from
#   real_y_sd   — optional numeric vector of y_sd values to resample from
#   cate_probs  — probabilities of the 5 CATE types (constant, sparse_linear,
#                 smooth_nonlinear, step_function, interaction)
#
# Parametric defaults (used when the vectors above are NULL):
#   R²     ~ 0.05 + 0.8 * Beta(5, 5)    (range 0.05 to 0.85, mean 0.45)
#   y_mean ~ Normal(8.5, 6)
#   y_sd   ~ max(0.5, |Normal(4, 2)|)
#
# Returns: list of DGP-level parameters (passed to generate_acic_mirror_v2)
# =============================================================================
draw_acic_params_v2 <- function(seed, real_r2 = NULL, real_y_mean = NULL,
                                 real_y_sd = NULL,
                                 cate_probs = c(0.20, 0.25, 0.25, 0.15, 0.15)) {
  set.seed(seed)

  # --- Treatment assignment (Z ⊥ X) ---
  # Arm probabilities from Dirichlet(20,...,20).
  # alpha=20 keeps arms moderately unbalanced: the ratio of the largest to
  # the smallest arm share has a median of about 1.7.
  # Dirichlet(alpha,...,alpha) is simulated as normalized Gamma(alpha,1).
  arm_probs <- rgamma(5, shape = 20, rate = 1)
  arm_probs <- arm_probs / sum(arm_probs)   # Simplex normalization

  # --- Prognostic surface (random complex mu_a) ---
  # Uses draw_complex_mu_a_spec() which selects 6-10 covariates with
  # random linear + quadratic coefficients. Produces much higher var(mu_a)
  # than a fixed formula.
  mu_a_spec <- draw_complex_mu_a_spec(seed + 22222)

  # --- Target R² ---
  # Resample from a user-supplied vector if given; otherwise draw from the
  # parametric default. Noise is calibrated at generation time to hit this R².
  if (!is.null(real_r2)) {
    target_r2 <- sample(real_r2, 1)
  } else {
    # Parametric default
    target_r2 <- rbeta(1, shape1 = 5, shape2 = 5) * 0.8 + 0.05
  }

  # --- Outcome scale (joint sampling) ---
  # If user-supplied vectors are given, draw y_mean and y_sd from the SAME
  # index so their joint distribution is preserved. Otherwise draw each from
  # its parametric default.
  # Signal scaling (in generate_acic_mirror_v2) will calibrate the raw
  # signal to achieve these targets without a post-hoc affine transform.
  if (!is.null(real_y_mean) && !is.null(real_y_sd) &&
      length(real_y_mean) == length(real_y_sd)) {
    joint_idx <- sample(length(real_y_mean), 1)
    target_y_mean <- real_y_mean[joint_idx]
    target_y_sd   <- real_y_sd[joint_idx]
  } else {
    # Parametric defaults (independent draws)
    if (!is.null(real_y_mean)) {
      target_y_mean <- sample(real_y_mean, 1)
    } else {
      target_y_mean <- rnorm(1, mean = 8.5, sd = 6)
    }
    if (!is.null(real_y_sd)) {
      target_y_sd <- sample(real_y_sd, 1)
    } else {
      target_y_sd <- max(0.5, abs(rnorm(1, mean = 4, sd = 2)))  # floor avoids degenerate scale
    }
  }

  # --- CATE surface ---
  # 4 independent contrasts (b-a, c-a, d-a, e-a), each with independently
  # drawn CATE type and coefficients. The 5 CATE types (constant,
  # sparse_linear, smooth_nonlinear, step_function, interaction) are
  # reasonable approximations of real heterogeneity patterns.
  cate_types <- sapply(1:4, function(j) {
    sample(c("constant", "sparse_linear", "smooth_nonlinear",
             "step_function", "interaction"),
           1, prob = cate_probs)
  })
  names(cate_types) <- c("b", "c", "d", "e")

  # Draw CATE coefficients independently for each contrast.
  #
  # V3b calibration notes:
  #   - Coefficient SDs boosted by ~1.3× to increase raw CATE magnitude.
  #     Combined with signal scaling, this brings the typical largest
  #     absolute treatment effect to about 1.8.
  #   - Positive TE bias: intercepts/effects are shifted upward so CATEs
  #     are not centered at 0 (mean effect around +0.4).
  draw_cate_coefs <- function(cate_type) {
    cc <- list()
    if (cate_type == "constant") {
      cc$effect <- runif(1, -0.5, 1.5)           # positive bias (was -1,1)
    } else if (cate_type == "sparse_linear") {
      cc$intercept  <- runif(1, -0.15, 0.45)     # shifted +0.15
      cc$coef_cont1 <- rnorm(1, sd = 0.52)       # 1.3× boost
      cc$coef_cont2 <- rnorm(1, sd = 0.39)       # 1.3× boost
      cc$coef_bin1  <- rnorm(1, sd = 0.325)      # 1.3× boost
    } else if (cate_type == "smooth_nonlinear") {
      cc$coef_sin      <- rnorm(1, sd = 0.52)    # 1.3× boost
      cc$coef_quad     <- rnorm(1, sd = 0.39)    # 1.3× boost
      cc$coef_interact <- rnorm(1, sd = 0.26)    # 1.3× boost
      cc$intercept     <- runif(1, -0.05, 0.35)  # shifted +0.15
    } else if (cate_type == "step_function") {
      cc$threshold <- runif(1, 0.3, 0.7)
      cc$hi_effect <- rnorm(1, mean = 0.65, sd = 0.39)  # shifted +0.15, 1.3× sd
      cc$lo_effect <- rnorm(1, mean = -0.05, sd = 0.26) # shifted +0.15, 1.3× sd
      cc$coef_bin1 <- rnorm(1, sd = 0.325)               # 1.3× boost
    } else if (cate_type == "interaction") {
      cc$coef1     <- rnorm(1, sd = 0.52)         # 1.3× boost
      cc$coef2     <- rnorm(1, sd = 0.39)         # 1.3× boost
      cc$coef3     <- rnorm(1, sd = 0.26)         # 1.3× boost
      cc$intercept <- runif(1, -0.05, 0.35)       # shifted +0.15
    }
    cc
  }

  cate_coefs <- lapply(1:4, function(j) draw_cate_coefs(cate_types[j]))
  names(cate_coefs) <- c("b", "c", "d", "e")

  # --- Covariate distribution parameters ---
  # Shared across train/test to ensure structural consistency.
  bin_probs  <- runif(19, 0.2, 0.8)
  zero_fracs <- runif(3, 0.25, 0.40)

  list(
    arm_probs      = arm_probs,
    mu_a_spec      = mu_a_spec,
    target_r2      = target_r2,
    target_y_mean  = target_y_mean,
    target_y_sd    = target_y_sd,
    cate_types     = cate_types,
    cate_coefs     = cate_coefs,
    bin_probs      = bin_probs,
    zero_fracs     = zero_fracs,
    heteroscedastic = FALSE   # Homoscedastic for clean R² calibration
  )
}


# =============================================================================
# generate_acic_mirror_v2(): Generates observation-level data from v2 params.
#
# Key design features:
#   1. Treatment: random multinomial Z ~ Mult(arm_probs), Z ⊥ X
#   2. Propensities: constant pi_matrix (same arm_probs for all units)
#   3. mu_a: complex random surface from mu_a_spec (6-10 covariates)
#   4. Noise: calibrated to achieve target_r2
#   5. Two-step signal scaling:
#      - Step A: tau ∝ target_y_sd (R²-independent)
#      - Step B: mu_a absorbs the remaining signal budget for target R²
#      - Step C: noise sd = target_y_sd * sqrt(1 - R²)
#
# Args:
#   n        — sample size (default 1000)
#   params   — output of draw_acic_params_v2()
#   seed     — RNG seed for observation-level randomness
#   tau_frac — fraction of target_y_sd allocated to CATE signal (via
#              ate_rms normalization). Default 0.30 gives a median
#              largest absolute treatment effect of about 1.8.
#
# Returns: list with Y, Z, X, tau_true, raw_cats, pi_matrix, noise_sd,
#          sigma_x, cate_types, target_r2, target_y_mean, target_y_sd,
#          tau_scale, mu_a_scale, signal_scale
# =============================================================================
generate_acic_mirror_v2 <- function(n = 1000, params = NULL, seed = NULL,
                                     tau_frac = 0.30) {
  if (is.null(params)) {
    stop("generate_acic_mirror_v2 requires explicit params from draw_acic_params_v2()")
  }
  if (!is.null(seed)) set.seed(seed)

  # ---- 1. Covariates ----
  # 54 columns (20 one-hot + 19 binary + 12 continuous + 3 ZI)
  cov_data <- generate_acic_covariates(n, params$bin_probs, params$zero_fracs)
  X <- cov_data$X

  # ---- 2. Treatment assignment: Z ⊥ X (pure randomization) ----
  # arm_probs vary across DGP draws (Dirichlet) but are CONSTANT within
  # a dataset — every unit has the same probability of each arm.
  Z <- sample(c("a", "b", "c", "d", "e"), n, replace = TRUE,
              prob = params$arm_probs)

  # Propensity matrix: constant (same arm_probs for all units)
  pi_matrix <- matrix(rep(params$arm_probs, each = n), nrow = n, ncol = 5)
  colnames(pi_matrix) <- c("a", "b", "c", "d", "e")

  # ---- 3. CATE surfaces (4 independent contrasts) ----
  tau_true <- compute_cate_surface(X, params)

  # ---- 4. Prognostic surface: complex random mu_a ----
  mu_a <- compute_complex_mu_a(X, params$mu_a_spec)

  # ---- 5. Two-step signal scaling (V3b) ----
  #
  # Step A: Scale tau proportional to target_y_sd (R²-independent).
  #   This ensures te_max_abs ∝ target_y_sd (larger-scale outcomes get larger effects).
  #
  # Step B: Scale mu_a to fill the remaining signal budget for R².
  #   var(mu_a_scaled) = target_y_sd² * R² - var(tau_expected_scaled).
  #
  # Step C: Noise sd = target_y_sd * sqrt(1 - R²).
  #
  # Edge case: when tau_var > total_signal_var (tau alone exceeds the R²
  # signal budget), mu_a_scale = 0 and R² will be slightly higher than
  # target. Rare (~5% at very low R² < 0.08). Acceptable.

  target_r2 <- max(0.01, min(0.99, params$target_r2))

  # Step A: Scale tau proportional to target_y_sd
  # Normalizer: RMS of arm-level ATEs = sqrt(mean(E[tau_j(X)]²)).
  # Bounded ratio max|ATE|/ate_rms ∈ [1, 2] for 4 contrasts → natural
  # variation in te_max_abs without extreme outliers.
  arm_means_raw <- colMeans(tau_true)   # 4-vector: E[tau_j(X)] per contrast
  ate_rms <- sqrt(mean(arm_means_raw^2))
  if (ate_rms < 0.01) ate_rms <- 0.01   # floor for near-zero ATEs

  tau_scale <- params$target_y_sd * tau_frac / ate_rms
  tau_true <- tau_true * tau_scale

  # Step B: Scale mu_a to fill the remaining signal budget for R²
  tau_expected_scaled <- as.vector(tau_true %*% params$arm_probs[2:5])
  tau_var <- var(tau_expected_scaled)

  total_signal_var <- params$target_y_sd^2 * target_r2
  mu_a_budget <- max(0, total_signal_var - tau_var)

  mu_a_centered <- mu_a - mean(mu_a)
  mu_a_var_raw <- var(mu_a_centered)

  if (mu_a_var_raw > 1e-10 && mu_a_budget > 0) {
    mu_a_scale <- sqrt(mu_a_budget / mu_a_var_raw)
  } else {
    mu_a_scale <- 0  # tau dominates signal; mu_a contributes nothing
  }
  mu_a <- mu_a_centered * mu_a_scale

  # ---- 6. Noise ----
  # noise_sd is derived directly from the target variance decomposition.
  noise_sd <- params$target_y_sd * sqrt(1 - target_r2)
  sigma_x  <- rep(noise_sd, n)
  epsilon  <- rnorm(n, sd = noise_sd)

  # ---- 7. Outcome ----
  # Y = scaled_mu_a + scaled_tau*Z + noise + target_y_mean.
  # By construction: mean(Y) ≈ target_y_mean, sd(Y) ≈ target_y_sd,
  # R² ≈ target_r2, and CATEs ∝ target_y_sd.
  Y <- mu_a + epsilon + params$target_y_mean
  Y <- Y + tau_true[, 1] * as.numeric(Z == "b")
  Y <- Y + tau_true[, 2] * as.numeric(Z == "c")
  Y <- Y + tau_true[, 3] * as.numeric(Z == "d")
  Y <- Y + tau_true[, 4] * as.numeric(Z == "e")

  list(
    Y              = Y,
    Z              = Z,
    X              = X,
    tau_true       = tau_true,
    raw_cats       = cov_data$raw_cats,   # <-- NEW: for format_acic_csv()
    pi_matrix      = pi_matrix,
    noise_sd       = noise_sd,
    sigma_x        = sigma_x,
    cate_types     = params$cate_types,
    target_r2      = params$target_r2,
    target_y_mean  = params$target_y_mean,
    target_y_sd    = params$target_y_sd,
    tau_scale      = tau_scale,
    mu_a_scale     = mu_a_scale,
    signal_scale   = tau_scale,  # backward compat
    heteroscedastic = FALSE
  )
}


# =============================================================================
# format_acic_csv(): Convert internal simulation output to ACIC CSV format.
#
# The ACIC 2026 file layout has 43 columns: ID, y, z, x1-x40.
# Categorical variables (x1-x6) are stored as "level 1", "level 2", etc.
# Binary (x7-x25), continuous (x26-x37), and zero-inflated (x38-x40) are
# stored as raw numeric values.
#
# This function maps the internal representation (54-col one-hot X matrix +
# raw_cats list) back to the 40-column ACIC covariate format.
#
# Args:
#   sim — output of generate_acic_mirror_v2() (must contain Y, Z, X, raw_cats)
#
# Returns: data.frame with columns ID, y, z, x1, x2, ..., x40
# =============================================================================
format_acic_csv <- function(sim) {
  n <- length(sim$Y)

  # Categorical variables: convert integer levels to "level N" strings
  # (matching the ACIC 2026 file layout)
  x1 <- paste("level", sim$raw_cats$x1)
  x2 <- paste("level", sim$raw_cats$x2)
  x3 <- paste("level", sim$raw_cats$x3)
  x4 <- paste("level", sim$raw_cats$x4)
  x5 <- paste("level", sim$raw_cats$x5)
  x6 <- paste("level", sim$raw_cats$x6)

  # Binary covariates: x7-x25 (columns 21-39 of X matrix)
  X_bin <- sim$X[, IDX_BIN]

  # Continuous covariates: x26-x37 (columns 40-51 of X matrix)
  X_cont <- sim$X[, IDX_CONT]

  # Zero-inflated covariates: x38-x40 (columns 52-54 of X matrix)
  X_zi <- sim$X[, IDX_ZI]

  # Assemble the ACIC-format data.frame
  df <- data.frame(
    ID = 1:n,
    y  = sim$Y,
    z  = sim$Z,
    x1 = x1,
    x2 = x2,
    x3 = x3,
    x4 = x4,
    x5 = x5,
    x6 = x6,
    stringsAsFactors = FALSE
  )

  # Add binary columns (x7-x25)
  bin_df <- as.data.frame(X_bin)
  colnames(bin_df) <- paste0("x", 7:25)
  df <- cbind(df, bin_df)

  # Add continuous columns (x26-x37)
  cont_df <- as.data.frame(X_cont)
  colnames(cont_df) <- paste0("x", 26:37)
  df <- cbind(df, cont_df)

  # Add zero-inflated columns (x38-x40)
  zi_df <- as.data.frame(X_zi)
  colnames(zi_df) <- paste0("x", 38:40)
  df <- cbind(df, zi_df)

  df
}
