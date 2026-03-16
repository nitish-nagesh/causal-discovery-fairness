#!/usr/bin/env Rscript
# Fairness decomposition for Heart Failure Clinical Records
# Outcome: death_event (binary), Sensitive: gender
# Mediators: smoking, high_bp, diabetes, cpk, anaemia, serum_creatinine, ejection_fraction
# Confounders: age, platelets, serum_sodium, time

library(faircause)

root <- rprojroot::find_root(rprojroot::is_git_root)
source(file.path(root, "scripts", "hf-preproc.R"))

tmp <- preproc_hf()
dat <- tmp[[1]]
sfm <- tmp[[2]]

X <- sfm$X
Y <- sfm$Y
Z <- sfm$Z
W <- sfm$W
x0 <- sfm$x0
x1 <- sfm$x1

# Standard fairness decomposition (faircause handles binary Y)
fc <- fairness_cookbook(dat, X = X, Z = Z, W = W, Y = Y, x0 = x0, x1 = x1,
                       method = "medDML", model = "ranger", nboot2 = 200)

cat("TV =", round(100 * fc$measures$value[fc$measures$measure == "tv"], 2), "%\n")
cat("Ctf-DE =", round(100 * fc$measures$value[fc$measures$measure == "ctfde"], 2), "%\n")
cat("Ctf-IE =", round(100 * fc$measures$value[fc$measures$measure == "ctfie"], 2), "%\n")
cat("Ctf-SE =", round(100 * fc$measures$value[fc$measures$measure == "ctfse"], 2), "%\n\n")

write.csv(fc$measures, file.path(root, "outputs", "fairness_decomposition_hf.csv"), row.names = FALSE)

# Variable-level decomposition with CIs (binary Y: use glm for outcome)
Z_order <- Z
W_order <- W
y_binary <- all(dat[[Y]] %in% c(0, 1))

est_quant_hf <- function(data, z_levels, n_samp = 2000, seed = 22) {
  set.seed(seed)
  n_samp <- min(n_samp, nrow(data))
  idx0 <- which(data[[X]] == x0)
  idx1 <- which(data[[X]] == x1)
  if (length(idx0) < 10 | length(idx1) < 10) return(NA)

  est_dat <- data.frame(matrix(NA, n_samp, length(c(Z, W, Y))))
  names(est_dat) <- c(Z, W, Y)

  for (i in seq_along(Z_order)) {
    z_var <- Z_order[i]
    lev <- z_levels[i]
    idx <- if (lev == 0) idx0 else idx1
    est_dat[[z_var]] <- sample(data[[z_var]][idx], n_samp, replace = TRUE)
  }

  for (w_var in W) {
    form <- as.formula(paste(w_var, "~", paste(c(X, Z), collapse = "+")))
    mod <- lm(form, data = data)
    est_dat[[X]] <- x1
    est_dat[[w_var]] <- predict(mod, est_dat)
  }

  form_y <- as.formula(paste(Y, "~", paste(c(X, Z, W), collapse = "+")))
  if (y_binary) {
    mod_y <- glm(form_y, data = data, family = binomial)
    mean(predict(mod_y, est_dat, type = "response"))
  } else {
    mod_y <- lm(form_y, data = data)
    mean(predict(mod_y, est_dat))
  }
}

est_quant_ie_hf <- function(data, w_levels, n_samp = 2000, seed = 22) {
  set.seed(seed)
  n_samp <- min(n_samp, nrow(data))
  idx0 <- which(data[[X]] == x0)
  idx1 <- which(data[[X]] == x1)
  if (length(idx0) < 10 | length(idx1) < 10) return(NA)

  est_dat <- data.frame(matrix(NA, n_samp, length(c(Z, W, Y))))
  names(est_dat) <- c(Z, W, Y)

  for (i in seq_along(Z)) {
    est_dat[[Z[i]]] <- sample(data[[Z[i]]][idx1], n_samp, replace = TRUE)
  }
  est_dat[[X]] <- x1

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

  form_y <- as.formula(paste(Y, "~", paste(c(X, Z, W), collapse = "+")))
  if (y_binary) {
    mod_y <- glm(form_y, data = data, family = binomial)
    mean(predict(mod_y, est_dat, type = "response"))
  } else {
    mod_y <- lm(form_y, data = data)
    mean(predict(mod_y, est_dat))
  }
}

nrep <- 20
nboot <- 100

# Ctf-SE bootstrap
contrib_boot <- matrix(NA, nboot, length(Z_order))
colnames(contrib_boot) <- Z_order
ctf_se_boot <- numeric(nboot)

for (b in seq_len(nboot)) {
  set.seed(1000 + b)
  idx <- sample(nrow(dat), nrow(dat), replace = TRUE)
  boot_dat <- dat[idx, ]
  baseline <- mean(sapply(seq_len(nrep), function(i)
    est_quant_hf(boot_dat, rep(0, length(Z_order)), n_samp = 1500, seed = i + b * 100)))
  full_se <- mean(sapply(seq_len(nrep), function(i)
    est_quant_hf(boot_dat, rep(1, length(Z_order)), n_samp = 1500, seed = i + b * 100 + 500)))
  ctf_se_boot[b] <- full_se - baseline
  prev <- baseline
  for (j in seq_along(Z_order)) {
    z_lev <- rep(0, length(Z_order))
    z_lev[seq_len(j)] <- 1
    curr <- mean(sapply(seq_len(nrep), function(i)
      est_quant_hf(boot_dat, z_lev, n_samp = 1500, seed = i + b * 100 + j * 1000)))
    contrib_boot[b, j] <- curr - prev
    prev <- curr
  }
}

contributions <- colMeans(contrib_boot)
contrib_ci_lo <- apply(contrib_boot, 2, quantile, 0.025)
contrib_ci_hi <- apply(contrib_boot, 2, quantile, 0.975)
ctf_se_ci <- quantile(ctf_se_boot, c(0.025, 0.975))

cat("Ctf-SE [95% CI]:", round(100 * mean(ctf_se_boot), 2), "% [",
    round(100 * ctf_se_ci[1], 2), "%, ", round(100 * ctf_se_ci[2], 2), "%]\n\n", sep = "")
cat("Variable contributions to Ctf-SE (95% CI):\n")
for (j in seq_along(Z_order)) {
  cat(sprintf("  %s: %+.2f%% [%+.2f%%, %+.2f%%]\n",
              Z_order[j], 100 * contributions[j],
              100 * contrib_ci_lo[j], 100 * contrib_ci_hi[j]))
}

se_tab <- data.frame(
  effect = "Ctf-SE",
  variable = Z_order,
  contribution_pct = round(100 * contributions, 2),
  ci_lo_pct = round(100 * contrib_ci_lo, 2),
  ci_hi_pct = round(100 * contrib_ci_hi, 2),
  stringsAsFactors = FALSE
)

# Ctf-IE bootstrap
if (length(W_order) >= 1) {
  ie_contrib_boot <- matrix(NA, nboot, length(W_order))
  colnames(ie_contrib_boot) <- W_order
  ctf_ie_boot <- numeric(nboot)

  for (b in seq_len(nboot)) {
    set.seed(2000 + b)
    idx <- sample(nrow(dat), nrow(dat), replace = TRUE)
    boot_dat <- dat[idx, ]
    baseline_ie <- mean(sapply(seq_len(nrep), function(i)
      est_quant_ie_hf(boot_dat, rep(0, length(W_order)), n_samp = 1500, seed = i + b * 100)))
    full_ie <- mean(sapply(seq_len(nrep), function(i)
      est_quant_ie_hf(boot_dat, rep(1, length(W_order)), n_samp = 1500, seed = i + b * 100 + 500)))
    ctf_ie_boot[b] <- full_ie - baseline_ie
    prev <- baseline_ie
    for (j in seq_along(W_order)) {
      w_lev <- rep(0, length(W_order))
      w_lev[seq_len(j)] <- 1
      curr <- mean(sapply(seq_len(nrep), function(i)
        est_quant_ie_hf(boot_dat, w_lev, n_samp = 1500, seed = i + b * 100 + j * 1000)))
      ie_contrib_boot[b, j] <- curr - prev
      prev <- curr
    }
  }

  ie_contributions <- colMeans(ie_contrib_boot)
  ie_contrib_ci_lo <- apply(ie_contrib_boot, 2, quantile, 0.025)
  ie_contrib_ci_hi <- apply(ie_contrib_boot, 2, quantile, 0.975)
  ctf_ie_ci <- quantile(ctf_ie_boot, c(0.025, 0.975))

  cat("\nCtf-IE [95% CI]:", round(100 * mean(ctf_ie_boot), 2), "% [",
      round(100 * ctf_ie_ci[1], 2), "%, ", round(100 * ctf_ie_ci[2], 2), "%]\n\n", sep = "")
  cat("Variable contributions to Ctf-IE (95% CI):\n")
  for (j in seq_along(W_order)) {
    cat(sprintf("  %s: %+.2f%% [%+.2f%%, %+.2f%%]\n",
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

out_path <- file.path(root, "outputs", "individual_effects_hf.csv")
write.csv(out_tab, out_path, row.names = FALSE)
cat("\nResults written to", out_path, "\n")
