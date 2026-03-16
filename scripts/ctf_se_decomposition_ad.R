#!/usr/bin/env Rscript
# Variable-level decomposition of Ctf-SE (by confounder) and Ctf-IE (by mediator) for AD data
# Adapted from CFA_repo/scripts/section-6/compas-business-necessity.R and individual_effects.py
# Outcome: ventricular_volume

library(faircause)
library(ggplot2)

root <- rprojroot::find_root(rprojroot::is_git_root)
source(file.path(root, "scripts", "ad-preproc.R"))

tmp <- preproc_ad()
dat <- tmp[[1]]
sfm <- tmp[[2]]

# Use SFM from ad-preproc
X <- sfm$X
Y <- sfm$Y
Z <- sfm$Z
W <- sfm$W
x0 <- sfm$x0
x1 <- sfm$x1

# Standard fairness decomposition
fc <- fairness_cookbook(dat, X = X, Z = Z, W = W, Y = Y, x0 = x0, x1 = x1,
                       method = "medDML", model = "ranger", nboot2 = 200)

cat("TV =", round(100 * fc$measures$value[fc$measures$measure == "tv"], 2), "%\n")
cat("Ctf-DE =", round(100 * fc$measures$value[fc$measures$measure == "ctfde"], 2), "%\n")
cat("Ctf-IE =", round(100 * fc$measures$value[fc$measures$measure == "ctfie"], 2), "%\n")
cat("Ctf-SE =", round(100 * fc$measures$value[fc$measures$measure == "ctfse"], 2), "%\n\n")
# Write for paper pipeline
write.csv(fc$measures, file.path(root, "outputs", "fairness_decomposition_sfm.csv"), row.names = FALSE)

# Variable-level Ctf-SE decomposition (sequential attribution)
# Order Z: we vary one confounder at a time from P(·|x0) to P(·|x1)
# Contribution of Zj = change in E[Y|do(X=x0)] when Zj switches from x0 to x1 dist
Z_order <- Z

est_quant_ad <- function(data, z_levels, n_samp = 3000, seed = 22) {
  # z_levels: 0 = sample from P(Z|X=x0), 1 = from P(Z|X=x1)
  set.seed(seed)
  n_samp <- min(n_samp, nrow(data))

  # Sample Z1 from group with X = x0 or x1
  idx0 <- which(data[[X]] == x0)
  idx1 <- which(data[[X]] == x1)
  if (length(idx0) < 10 | length(idx1) < 10) stop("Need more data per group")

  est_dat <- data.frame(matrix(NA, n_samp, length(c(Z, W, Y))))
  names(est_dat) <- c(Z, W, Y)

  for (i in seq_along(Z_order)) {
    z_var <- Z_order[i]
    lev <- z_levels[i]
    idx <- if (lev == 0) idx0 else idx1
    est_dat[[z_var]] <- sample(data[[z_var]][idx], n_samp, replace = TRUE)
  }

  # W | X, Z
  for (w_var in W) {
    form <- as.formula(paste(w_var, "~", paste(c(X, Z), collapse = "+")))
    mod <- lm(form, data = data)
    est_dat[[X]] <- x1
    est_dat[[w_var]] <- predict(mod, est_dat)
  }

  # Y | X, Z, W
  form_y <- as.formula(paste(Y, "~", paste(c(X, Z, W), collapse = "+")))
  mod_y <- lm(form_y, data = data)
  est_dat[[X]] <- x1
  mean(predict(mod_y, est_dat))
}

nrep <- 30
nboot <- 200  # bootstrap samples for confidence intervals

# Bootstrap: resample data, recompute contributions, collect
contrib_boot <- matrix(NA, nboot, length(Z_order))
colnames(contrib_boot) <- Z_order
ctf_se_boot <- numeric(nboot)

for (b in seq_len(nboot)) {
  set.seed(1000 + b)
  idx <- sample(nrow(dat), nrow(dat), replace = TRUE)
  boot_dat <- dat[idx, ]

  baseline <- mean(sapply(seq_len(nrep), function(i)
    est_quant_ad(boot_dat, rep(0, length(Z_order)), n_samp = 2000, seed = i + b * 100)))
  full_se <- mean(sapply(seq_len(nrep), function(i)
    est_quant_ad(boot_dat, rep(1, length(Z_order)), n_samp = 2000, seed = i + b * 100 + 500)))
  ctf_se_boot[b] <- full_se - baseline

  prev <- baseline
  for (j in seq_along(Z_order)) {
    z_lev <- rep(0, length(Z_order))
    z_lev[seq_len(j)] <- 1
    curr <- mean(sapply(seq_len(nrep), function(i)
      est_quant_ad(boot_dat, z_lev, n_samp = 2000, seed = i + b * 100 + j * 1000)))
    contrib_boot[b, j] <- curr - prev
    prev <- curr
  }
}

contributions <- colMeans(contrib_boot)
ctf_se_est <- mean(ctf_se_boot)
contrib_sd <- apply(contrib_boot, 2, sd)
contrib_ci_lo <- apply(contrib_boot, 2, quantile, 0.025)
contrib_ci_hi <- apply(contrib_boot, 2, quantile, 0.975)
ctf_se_ci <- quantile(ctf_se_boot, c(0.025, 0.975))

cat("Ctf-SE (fairness_cookbook):", round(100 * fc$measures$value[fc$measures$measure == "ctfse"], 2), "%\n")
cat("Ctf-SE (estimated):", round(100 * ctf_se_est, 2), "%  [95% CI: ", round(100 * ctf_se_ci[1], 2), "%, ", round(100 * ctf_se_ci[2], 2), "%]\n\n", sep = "")
cat("Variable contributions to Ctf-SE (sequential, bootstrap 95% CI):\n")
for (j in seq_along(Z_order)) {
  cat(sprintf("  %s: %+.2f%% [95%% CI: %+.2f%%, %+.2f%%]\n",
              Z_order[j], 100 * contributions[j],
              100 * contrib_ci_lo[j], 100 * contrib_ci_hi[j]))
}

# Write tabulated results to CSV
se_tab <- data.frame(
  effect = "Ctf-SE",
  variable = Z_order,
  contribution_pct = round(100 * contributions, 2),
  ci_lo_pct = round(100 * contrib_ci_lo, 2),
  ci_hi_pct = round(100 * contrib_ci_hi, 2),
  stringsAsFactors = FALSE
)

# Variable-level Ctf-IE decomposition (sequential attribution)
# Ctf-IE = E[Y|do(X=x1), W~P(W|x1), Z~P(Z|x1)] - E[Y|do(X=x1), W~P(W|x0), Z~P(Z|x1)]
# Contribution of Wj = change when Wj switches from P(W|x0) to P(W|x1), holding Z from x1
W_order <- W

est_quant_ie_ad <- function(data, w_levels, n_samp = 3000, seed = 22) {
  # w_levels: 0 = sample W from P(W|X=x0), 1 = from P(W|X=x1); Z always from x1
  set.seed(seed)
  n_samp <- min(n_samp, nrow(data))

  idx0 <- which(data[[X]] == x0)
  idx1 <- which(data[[X]] == x1)
  if (length(idx0) < 10 | length(idx1) < 10) return(NA)

  est_dat <- data.frame(matrix(NA, n_samp, length(c(Z, W, Y))))
  names(est_dat) <- c(Z, W, Y)

  # Z: all from x1
  for (i in seq_along(Z)) {
    z_var <- Z[i]
    est_dat[[z_var]] <- sample(data[[z_var]][idx1], n_samp, replace = TRUE)
  }
  est_dat[[X]] <- x1

  # W: sequential, each from P(W|X=x0) or P(W|X=x1) based on w_levels
  for (i in seq_along(W_order)) {
    w_var <- W_order[i]
    lev <- w_levels[i]
    idx <- if (lev == 0) idx0 else idx1
    pa <- c(X, Z)
    if (i > 1) pa <- c(pa, W_order[seq_len(i - 1)])
    form <- as.formula(paste(w_var, "~", paste(pa, collapse = "+")))
    mod <- lm(form, data = data)
    est_dat[[X]] <- if (lev == 0) x0 else x1
    est_dat[[w_var]] <- predict(mod, est_dat)
  }
  est_dat[[X]] <- x1

  # Y | X, Z, W
  form_y <- as.formula(paste(Y, "~", paste(c(X, Z, W), collapse = "+")))
  mod_y <- lm(form_y, data = data)
  mean(predict(mod_y, est_dat))
}

if (length(W_order) >= 1) {
  ie_contrib_boot <- matrix(NA, nboot, length(W_order))
  colnames(ie_contrib_boot) <- W_order
  ctf_ie_boot <- numeric(nboot)

  for (b in seq_len(nboot)) {
    set.seed(2000 + b)
    idx <- sample(nrow(dat), nrow(dat), replace = TRUE)
    boot_dat <- dat[idx, ]

    # Baseline: all W from x0
    baseline_ie <- mean(sapply(seq_len(nrep), function(i)
      est_quant_ie_ad(boot_dat, rep(0, length(W_order)), n_samp = 2000, seed = i + b * 100)))
    # Full IE: all W from x1
    full_ie <- mean(sapply(seq_len(nrep), function(i)
      est_quant_ie_ad(boot_dat, rep(1, length(W_order)), n_samp = 2000, seed = i + b * 100 + 500)))
    ctf_ie_boot[b] <- full_ie - baseline_ie

    prev <- baseline_ie
    for (j in seq_along(W_order)) {
      w_lev <- rep(0, length(W_order))
      w_lev[seq_len(j)] <- 1
      curr <- mean(sapply(seq_len(nrep), function(i)
        est_quant_ie_ad(boot_dat, w_lev, n_samp = 2000, seed = i + b * 100 + j * 1000)))
      ie_contrib_boot[b, j] <- curr - prev
      prev <- curr
    }
  }

  ie_contributions <- colMeans(ie_contrib_boot)
  ctf_ie_est <- mean(ctf_ie_boot)
  ie_contrib_ci_lo <- apply(ie_contrib_boot, 2, quantile, 0.025)
  ie_contrib_ci_hi <- apply(ie_contrib_boot, 2, quantile, 0.975)
  ctf_ie_ci <- quantile(ctf_ie_boot, c(0.025, 0.975))

  cat("\n--- Ctf-IE (indirect effect) by mediator ---\n")
  cat("Ctf-IE (fairness_cookbook):", round(100 * fc$measures$value[fc$measures$measure == "ctfie"], 2), "%\n")
  cat("Ctf-IE (estimated):", round(100 * ctf_ie_est, 2), "%  [95% CI: ", round(100 * ctf_ie_ci[1], 2), "%, ", round(100 * ctf_ie_ci[2], 2), "%]\n\n", sep = "")
  cat("Variable contributions to Ctf-IE (sequential, bootstrap 95% CI):\n")
  for (j in seq_along(W_order)) {
    cat(sprintf("  %s: %+.2f%% [95%% CI: %+.2f%%, %+.2f%%]\n",
                W_order[j], 100 * ie_contributions[j],
                100 * ie_contrib_ci_lo[j], 100 * ie_contrib_ci_hi[j]))
  }

  ie_tab <- data.frame(
    effect = "Ctf-IE",
    variable = W_order,
    contribution_pct = round(100 * ie_contributions, 2),
    ci_lo_pct = round(100 * ie_contrib_ci_lo, 2),
    ci_hi_pct = round(100 * ie_contrib_ci_hi, 2),
    stringsAsFactors = FALSE
  )
  out_tab <- rbind(se_tab, ie_tab)
} else {
  out_tab <- se_tab
}

out_path <- file.path(root, "outputs", "individual_effects_ventricular.csv")
write.csv(out_tab, out_path, row.names = FALSE)
cat("\nResults written to", out_path, "\n")
