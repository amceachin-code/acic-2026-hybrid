# ===========================================================================
# R/horseshoe_gibbs.R — Custom horseshoe Gibbs sampler with
#                              per-parameter penalization control
#
# Implements the auxiliary-variable Gibbs sampler from:
#   Makalic & Schmidt (2015), "A simple sampler for the horseshoe estimator"
#   arXiv:1508.03884v4
#
# Key feature: the `penalized` argument is a logical vector of length p.
#   penalized[j] = TRUE  --> beta_j gets the horseshoe prior (shrinkage)
#   penalized[j] = FALSE --> beta_j gets a flat (improper) prior (no shrinkage)
#
# This lets you leave treatment main effects (gamma_w) unpenalized while
# shrinking the high-dimensional interaction terms (delta_w).
#
# All conditional posteriors are conjugate:
#   beta       ~ N_p(A^{-1} X^T y, sigma^2 * A^{-1})
#   sigma^2    ~ InvGamma(...)
#   lambda_j^2 ~ InvGamma(1, ...)    [only for penalized j]
#   tau^2      ~ InvGamma(...)        [driven only by penalized betas]
#   nu_j       ~ InvGamma(1, ...)    [auxiliary for lambda_j, penalized only]
#   xi         ~ InvGamma(1, ...)    [auxiliary for tau]
#
# References:
#   [1] Carvalho, Polson & Scott (2010). The horseshoe estimator for sparse
#       signals. Biometrika 97(2):465-480.
#   [2] Makalic & Schmidt (2015). arXiv:1508.03884v4.
#   [3] Bhattacharya, Chakraborty & Mallick (2016). Fast sampling with
#       Gaussian scale-mixture priors in high-dimensional regression.
#       Biometrika 103(4):985-991.
#
# Dependencies: none (base R only)
#
# Usage:
#   source("R/horseshoe_gibbs.R")
#   fit <- horseshoe_gibbs(y, X, penalized = c(rep(FALSE, 58), rep(TRUE, 216)))
# ===========================================================================


#' Horseshoe Gibbs sampler with selective penalization
#'
#' Samples from the posterior of Bayesian linear regression with the horseshoe
#' prior applied only to a user-specified subset of coefficients. Unpenalized
#' coefficients receive a flat (improper) prior — equivalent to an OLS-like
#' posterior for those parameters, conditional on the penalized parameters.
#'
#' @param y            Numeric vector of length n. The outcome (should be centered
#'                     if no intercept column is included in X).
#' @param X            Numeric matrix n x p. Design matrix (no intercept column
#'                     needed — include one explicitly if you want an unpenalized
#'                     intercept).
#' @param penalized    Logical vector of length p. TRUE = horseshoe prior on
#'                     beta_j; FALSE = flat prior (no shrinkage). Default: all TRUE
#'                     (standard horseshoe on every coefficient).
#' @param lambda_scale Positive scalar. Scale of the half-Cauchy prior on the
#'                     local shrinkage parameters: lambda_j ~ C+(0, lambda_scale).
#'                     Larger values allow individual coefficients to escape
#'                     shrinkage more easily (wider local tails). Default: 1.
#' @param tau_scale    Positive scalar. Scale of the half-Cauchy prior on the
#'                     global shrinkage parameter: tau ~ C+(0, tau_scale).
#'                     Smaller values enforce more aggressive overall shrinkage.
#'                     Default: 1.
#' @param n_mcmc       Integer. Number of post-burnin MCMC draws to store.
#'                     Default: 1000.
#' @param burnin       Integer. Number of initial iterations to discard.
#'                     Default: 500.
#' @param thin         Integer. Keep every thin-th draw after burnin.
#'                     Default: 1 (no thinning).
#' @param verbose      Logical. Print progress every 500 iterations. Default: TRUE.
#'
#' @return Named list:
#'   \item{beta_draws}{p x n_mcmc matrix of posterior draws for beta}
#'   \item{sigma2_draws}{n_mcmc vector of posterior draws for sigma^2}
#'   \item{tau2_draws}{n_mcmc vector of posterior draws for tau^2}
#'   \item{lambda2_draws}{p_pen x n_mcmc matrix of posterior draws for lambda_j^2
#'         (only for penalized coefficients, in the order they appear in X)}
#'   \item{pen_idx}{Integer vector of which columns of X are penalized}
#'   \item{free_idx}{Integer vector of which columns of X are unpenalized}
#'
#' @details
#' The sampler follows Makalic & Schmidt (2015) with the modification that
#' the horseshoe hierarchy (local shrinkage lambda_j, global shrinkage tau,
#' and their auxiliary variables nu_j, xi) only governs the penalized subset
#' of coefficients. Unpenalized coefficients enter the posterior of beta
#' through the likelihood (X^T X) term only, with no prior penalty.
#'
#' Concretely, the prior precision matrix Lambda_star^{-1} has:
#'   - Entry 1/(tau^2 * lambda_j^2) at position (j,j) for penalized j
#'   - Entry 0 at position (j,j) for unpenalized j (flat prior)
#'
#' The sigma^2 and tau^2 conditionals are adjusted to sum only over the
#' penalized coefficients, since unpenalized betas contribute no prior
#' information to these hyperparameters.
#'
#' The half-Cauchy scale parameters (lambda_scale, tau_scale) control the
#' prior breadth. Using the scale-mixture representation from Wand et al.
#' (2011), x ~ C+(0, A) iff x^2|a ~ IG(1/2, 1/a) and a ~ IG(1/2, 1/A^2).
#' Only the auxiliary variable conditionals (nu_j, xi) are affected:
#'   nu_j | . ~ IG(1, 1/lambda_j^2 + 1/lambda_scale^2)
#'   xi   | . ~ IG(1, 1/tau^2 + 1/tau_scale^2)
#' The lambda_j^2 and tau^2 conditionals are unchanged because they depend
#' on (nu_j, xi) rather than on the scale directly.
#'
#' Propriety requirement: the posterior is proper when n > p_free (number
#' of unpenalized parameters) and the columns of X corresponding to free
#' parameters have full column rank. The horseshoe prior on the penalized
#' parameters provides the regularization for the remaining columns.
horseshoe_gibbs <- function(y, X, penalized = NULL,
                            lambda_scale = 1, tau_scale = 1,
                            n_mcmc = 1000, burnin = 500, thin = 1,
                            verbose = TRUE) {

  # -------------------------------------------------------------------------
  # Input validation
  # -------------------------------------------------------------------------
  n <- nrow(X)
  p <- ncol(X)

  if (length(y) != n) {
    stop("Length of y (", length(y), ") must equal nrow(X) (", n, ")")
  }

  # Default: penalize everything (standard horseshoe)
  if (is.null(penalized)) {
    penalized <- rep(TRUE, p)
  }
  if (length(penalized) != p) {
    stop("Length of penalized (", length(penalized),
         ") must equal ncol(X) (", p, ")")
  }

  # Indices for penalized vs free parameters
  pen_idx  <- which(penalized)
  free_idx <- which(!penalized)
  p_pen    <- length(pen_idx)
  p_free   <- length(free_idx)

  if (p_pen == 0) {
    stop("No penalized parameters — use OLS instead of the horseshoe sampler.")
  }

  # Check propriety: need n > p_free for the flat prior on free params
  if (p_free > 0 && n <= p_free) {
    warning("n (", n, ") <= p_free (", p_free,
            "). Posterior may be improper for unpenalized parameters. ",
            "Consider penalizing more columns or reducing p_free.")
  }

  # Validate scale parameters
  if (lambda_scale <= 0) stop("lambda_scale must be positive, got ", lambda_scale)
  if (tau_scale <= 0) stop("tau_scale must be positive, got ", tau_scale)

  # -------------------------------------------------------------------------
  # Precompute quantities that don't change across iterations
  # -------------------------------------------------------------------------
  XtX <- crossprod(X)        # p x p, symmetric positive semi-definite
  Xty <- crossprod(X, y)     # p x 1

  # Precompute 1/A^2 for the half-Cauchy scale-mixture auxiliary variables.
  # These appear in the conditional posteriors for nu_j and xi:
  #   nu_j | . ~ IG(1, 1/lambda_j^2 + 1/lambda_scale^2)
  #   xi   | . ~ IG(1, 1/tau^2     + 1/tau_scale^2)
  inv_lambda_scale_sq <- 1 / lambda_scale^2   # e.g. 1/400 when lambda_scale=20
  inv_tau_scale_sq    <- 1 / tau_scale^2       # e.g. 400   when tau_scale=1/20

  # Total iterations: burnin + (n_mcmc draws * thinning interval)
  total_iter <- burnin + n_mcmc * thin

  # -------------------------------------------------------------------------
  # Initialize parameters
  # -------------------------------------------------------------------------
  # Beta: start at the ridge estimate (lambda = 1) for numerical stability.
  # This avoids the Cholesky failing at iteration 1 if X^T X is singular.
  beta <- as.numeric(solve(XtX + diag(1, p), Xty))

  # Residual variance: initialize from residuals
  sigma2 <- as.numeric(crossprod(y - X %*% beta)) / max(n - p, 1)

  # Local shrinkage: lambda_j^2 for penalized params only.
  # Initialize at 1 (no shrinkage). Length = p_pen.
  lambda2 <- rep(1, p_pen)

  # Global shrinkage: tau^2. Initialize at 1.
  tau2 <- 1

  # Auxiliary variables for the half-Cauchy decomposition:
  #   lambda_j ~ C+(0, lambda_scale) iff lambda_j^2 | nu_j ~ IG(1/2, 1/nu_j)
  #                                      and nu_j ~ IG(1/2, 1/lambda_scale^2)
  #   tau ~ C+(0, tau_scale)         iff tau^2 | xi ~ IG(1/2, 1/xi)
  #                                      and xi ~ IG(1/2, 1/tau_scale^2)
  # Initialize auxiliaries at 1.
  nu <- rep(1, p_pen)   # one per penalized param
  xi <- 1               # one for the global shrinkage

  # -------------------------------------------------------------------------
  # Storage for posterior draws
  # -------------------------------------------------------------------------
  beta_store    <- matrix(NA_real_, p, n_mcmc)
  sigma2_store  <- numeric(n_mcmc)
  tau2_store    <- numeric(n_mcmc)
  lambda2_store <- matrix(NA_real_, p_pen, n_mcmc)

  save_idx <- 0L  # counter for stored draws

  # -------------------------------------------------------------------------
  # Gibbs sampler main loop
  # -------------------------------------------------------------------------
  for (iter in 1:total_iter) {

    # --- Step 1: Sample beta | rest  [eq. 9 of Makalic & Schmidt] ---
    #
    # beta | . ~ N_p(A^{-1} X^T y, sigma^2 * A^{-1})
    # where A = X^T X + Lambda_star_inv
    #
    # Lambda_star_inv is a p x p diagonal matrix:
    #   diagonal[j] = 1/(tau^2 * lambda_j^2)  for penalized j
    #   diagonal[j] = 0                        for unpenalized j (flat prior)
    #
    # We build Lambda_star_inv as a vector and add to the diagonal of X^T X.
    diag_penalty <- numeric(p)
    diag_penalty[pen_idx] <- 1 / (tau2 * lambda2)
    A <- XtX + diag(diag_penalty)

    # Sample from N(mu, sigma^2 * A^{-1}) using Cholesky:
    #   A = R^T R  (R is upper triangular, from chol())
    #   mu = A^{-1} X^T y = solve(R, solve(t(R), Xty))
    #   beta = mu + sqrt(sigma2) * solve(R, z),  z ~ N(0, I_p)
    R <- chol(A)  # R is upper triangular: A = t(R) %*% R
    mu_beta <- backsolve(R, forwardsolve(t(R), Xty))
    z <- rnorm(p)
    beta <- as.numeric(mu_beta + sqrt(sigma2) * backsolve(R, z))

    # --- Step 2: Sample sigma^2 | rest  [eq. 10, modified] ---
    #
    # Only penalized betas contribute to the prior term.
    # sigma^2 | . ~ IG( (n + p_pen)/2,
    #                    [ ||y - X*beta||^2 + sum_{j in pen} beta_j^2/(tau^2*lambda_j^2) ] / 2 )
    resid <- y - X %*% beta
    rss <- as.numeric(crossprod(resid))
    prior_penalty <- sum(beta[pen_idx]^2 / (tau2 * lambda2))
    shape_sigma <- (n + p_pen) / 2
    rate_sigma  <- (rss + prior_penalty) / 2
    sigma2 <- 1 / rgamma(1, shape = shape_sigma, rate = rate_sigma)

    # --- Step 3: Sample lambda_j^2 | rest  [eq. 11, penalized only] ---
    #
    # lambda_j^2 | . ~ IG(1, 1/nu_j + beta_j^2 / (2 * tau^2 * sigma^2))
    # for j in pen_idx.
    # Note: shape parameter is 1 = (1/2 + 1/2), combining the IG(1/2,...) prior
    # with the N(0, lambda_j^2*tau^2*sigma^2) likelihood contribution.
    rate_lambda <- 1 / nu + beta[pen_idx]^2 / (2 * tau2 * sigma2)
    lambda2 <- 1 / rgamma(p_pen, shape = 1, rate = rate_lambda)

    # --- Step 4: Sample tau^2 | rest  [eq. 12, summing over penalized only] ---
    #
    # tau^2 | . ~ IG( (p_pen + 1)/2,
    #                  1/xi + sum_{j in pen} beta_j^2/(lambda_j^2) / (2*sigma^2) )
    shape_tau <- (p_pen + 1) / 2
    rate_tau  <- 1 / xi + sum(beta[pen_idx]^2 / lambda2) / (2 * sigma2)
    tau2 <- 1 / rgamma(1, shape = shape_tau, rate = rate_tau)

    # --- Step 5: Sample auxiliary variables nu_j | rest  [penalized only] ---
    #
    # With lambda_j ~ C+(0, lambda_scale), the scale-mixture gives:
    #   nu_j prior: IG(1/2, 1/lambda_scale^2)
    # Combined with lambda_j^2 | nu_j ~ IG(1/2, 1/nu_j):
    #   nu_j | . ~ IG(1, 1/lambda_scale^2 + 1/lambda_j^2)
    #
    # When lambda_scale = 1:  rate = 1 + 1/lambda_j^2  (standard horseshoe)
    # When lambda_scale = 20: rate = 1/400 + 1/lambda_j^2  (wider local tails)
    nu <- 1 / rgamma(p_pen, shape = 1, rate = inv_lambda_scale_sq + 1 / lambda2)

    # --- Step 6: Sample auxiliary variable xi | rest ---
    #
    # With tau ~ C+(0, tau_scale), the scale-mixture gives:
    #   xi prior: IG(1/2, 1/tau_scale^2)
    # Combined with tau^2 | xi ~ IG(1/2, 1/xi):
    #   xi | . ~ IG(1, 1/tau_scale^2 + 1/tau^2)
    #
    # When tau_scale = 1:    rate = 1 + 1/tau^2  (standard horseshoe)
    # When tau_scale = 1/20: rate = 400 + 1/tau^2  (tighter global shrinkage)
    xi <- 1 / rgamma(1, shape = 1, rate = inv_tau_scale_sq + 1 / tau2)

    # --- Store draws (after burnin, respecting thinning) ---
    if (iter > burnin && (iter - burnin) %% thin == 0) {
      save_idx <- save_idx + 1L
      beta_store[, save_idx]    <- beta
      sigma2_store[save_idx]    <- sigma2
      tau2_store[save_idx]      <- tau2
      lambda2_store[, save_idx] <- lambda2
    }

    # --- Progress reporting ---
    if (verbose && iter %% 500 == 0) {
      cat(sprintf("  [Horseshoe Gibbs] iter %d / %d | tau = %.4f | sigma = %.4f\n",
                  iter, total_iter, sqrt(tau2), sqrt(sigma2)))
    }
  }

  # -------------------------------------------------------------------------
  # Return posterior draws and index maps
  # -------------------------------------------------------------------------
  list(
    beta_draws    = beta_store,       # p x n_mcmc
    sigma2_draws  = sigma2_store,     # n_mcmc
    tau2_draws    = tau2_store,       # n_mcmc
    lambda2_draws = lambda2_store,    # p_pen x n_mcmc
    pen_idx       = pen_idx,          # which columns were penalized
    free_idx      = free_idx          # which columns were unpenalized
  )
}
