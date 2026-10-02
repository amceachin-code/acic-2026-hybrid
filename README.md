# Hybrid Ensemble for Multi-Arm CATE Estimation

An R implementation of the **Hybrid ensemble** we developed for the [ACIC 2026 Data Challenge](https://acic2026.mathematica.org/). Given observational data with five treatment arms (`a` = control, `b` to `e`), it estimates individual conditional average treatment effects (iCATEs) for the four contrasts b-a, c-a, d-a and e-a, with 95% intervals. It also produces sample, subgroup and population averages, and recommends a treatment for each unit.

The repository contains the method code, a simulator, and **50 simulated example datasets with known true effects**, so you can run and check the method end to end. It contains no contest data.

## Quick start

```bash
git clone https://github.com/amceachin-code/acic-2026-hybrid.git
cd acic-2026-hybrid

bash setup.sh                  # install grf, ranger, sandwich (R >= 4.1)
Rscript tests/run_tests.R      # quick checks (~5 s)

bash run_example.sh 50 4       # fit all 50 example datasets on 4 cores (~5 min)
```

`run_example.sh` writes outputs to `example/output/` and prints a scorecard against the true effects:

| Metric (50 simulated datasets) | Mean | Median |
|--------------------------------|------|--------|
| iCATE RMSE | 0.428 | 0.379 |
| iCATE 95% CI coverage | 0.943 | 0.970 |
| Treatment assignment accuracy | 0.886 | 1.000 |
| sCATE 95% CI coverage | 0.940 | 1.000 |
| PATE 95% CI coverage | 0.850 | 1.000 |

## Using it on your own data

Your file needs the ACIC 2026 layout: one row per unit, with columns `ID`, `y` (outcome), `z` (treatment: `a` to `e`), and covariates `x1` to `x40`:

| Columns | Type |
|---------|------|
| `x1`-`x6` | Categorical, stored as `"level 1"`, `"level 2"`, ... (3, 4, 3, 5, 4 and 7 levels) |
| `x7`-`x25` | Binary 0/1 |
| `x26`-`x40` | Continuous in [0, 1] (`x38`-`x40` may contain many exact zeros) |

```bash
Rscript R/run_hybrid.R my_data/data_1.csv results/            # PATE falls back to sCATE
Rscript R/run_hybrid.R my_data/data_1.csv results/ pop.csv    # model-based PATE
```

`pop.csv` is optional. It holds the population means of the 54 encoded covariates (a `mean` column, in the column order of `prepare_covariates()`). See `example/data/pop/` for the format.

Outputs, for a file named `data_<id>.csv`:

| File | Contents |
|------|----------|
| `dataset_<id>_icates.csv` | iCATE estimates with 95% CIs for each contrast (`tau_b`, `tau_b_L95`, `tau_b_U95`, ...) |
| `dataset_<id>_aggregates.csv` | sCATE, subCATE (by `x12`), and PATE with 95% CIs, plus the best arm for each |
| `dataset_<id>_best_icates.csv` | Recommended arm per unit (largest estimated effect; `a` if none is positive) |
| `dataset_<id>_best_icates_pbest.csv` | Recommended arm per unit (highest posterior probability of being best) |
| `dataset_<id>_pbest_probs.csv` | Posterior probability that each arm is best, per unit |

## How the method works

The ensemble combines two estimators with complementary strengths:

| Component | Method | Weight | Role |
|-----------|--------|--------|------|
| **Horseshoe** | Bayesian linear model, horseshoe prior on treatment-by-covariate interactions, custom Gibbs sampler | 47.6% | Posterior intervals, joint draws for P(best arm), linear structure for PATE |
| **GRF** | `grf::multi_arm_causal_forest` (2,000 trees, `min.node.size = 20`) | 52.4% | Flexible nonlinear heterogeneity, variance-based intervals |

- **Point estimates** are an inverse-RMSE weighted average of the two components.
- **Intervals** blend the two components' intervals with the same weights, then widen them by a conformal multiplier (1.10) to target 95% coverage.
- **Aggregates**: sCATE and subCATE standard errors come from the Horseshoe posterior draws. PATE uses the Horseshoe's linear form, `PATE = gamma + x_bar_pop' delta`, evaluated draw by draw.
- **Treatment choice** uses the Horseshoe joint posterior to compute each unit's probability that each arm is best.

All weights and multipliers were tuned on simulated data. In a 500-dataset out-of-sample benchmark, the Hybrid reached an iCATE RMSE of 0.376, versus 0.381 for GRF alone, with about 95% coverage.

- [`docs/ANALYTIC_APPROACH.md`](docs/ANALYTIC_APPROACH.md): concise overview of the approach
- [`docs/METHOD_DETAILS.md`](docs/METHOD_DETAILS.md): full technical description, tuning, and limitations

## Simulated data

`dgp/mirror_dgp.R` is a self-contained simulator (base R only) that writes data in the same layout, together with the true effects. Each dataset draws its own arm probabilities, outcome surface, R², outcome scale, and CATE functions for each contrast. The functions come from five types: constant, sparse linear, smooth nonlinear, step function, and interaction.

```bash
Rscript dgp/generate_data.R my_sims 200 balanced    # 200 pairs, balanced CATE mix
```

Presets: `mostly_constant` (the mix used for tuning, and the default) and `balanced`. Each pair has a train and a test file from the same DGP, true iCATEs for both, exact population covariate means, and the true PATE (in `metadata.csv`). `R/evaluate.R <data_dir> <pred_dir>` scores any run.

## Repository layout

```
.
├── README.md
├── LICENSE                     GPL-3.0
├── setup.sh                    Install R dependencies
├── run_example.sh              Run + score the bundled example
├── R/
│   ├── run_hybrid.R            Command-line entry point (one dataset)
│   ├── fit_hybrid.R            Ensemble: GRF + Horseshoe, weights, conformal CIs, PATE
│   ├── horseshoe_gibbs.R       Horseshoe Gibbs sampler
│   ├── fit_horseshoe.R         Horseshoe wrapper (design matrix, CATE extraction)
│   ├── fit_ols.R               Linear design-matrix builder used by the Horseshoe
│   ├── prepare_covariates.R    54-column covariate encoding; output writer
│   ├── compute_aggregate_se.R  sCATE/subCATE standard errors
│   ├── compute_pate.R          PATE from posterior draws
│   ├── compute_p_best.R        Probability each arm is best
│   ├── evaluate_multi.R        Evaluation metrics
│   └── evaluate.R              Score a run against simulated truth
├── dgp/
│   ├── mirror_dgp.R            Simulator functions
│   └── generate_data.R         Batch generator (train/test/truth/pop/metadata)
├── example/
│   ├── data/                   50 simulated pairs (train, test, truth, pop, metadata.csv)
│   └── output/                 Created by run_example.sh (gitignored)
├── docs/
│   ├── ANALYTIC_APPROACH.md
│   └── METHOD_DETAILS.md
└── tests/
    └── run_tests.R             Base-R checks; add --full for an end-to-end fit
```

## Known limitations

- PATE intervals undercover (85% on the example), because the PATE z-multiplier was never tuned.
- Coverage is weaker for smooth nonlinear and interaction CATE surfaces than for constant ones.
- The covariate encoding is fixed at the 40-covariate ACIC layout (54 encoded columns).
- `fit_hybrid.R` still contains code for an optional third component (BCF or X-learner) used in earlier experiments. The default 2-component pipeline does not use it.

## Citation

If you use this code, please cite this repository and the ACIC 2026 Data Challenge.

## License

GPL-3.0. See [LICENSE](LICENSE).
