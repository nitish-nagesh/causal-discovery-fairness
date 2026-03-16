#' Preprocess Heart Failure Clinical Records for fairness decomposition
#' Sensitive: gender, Outcome: death_event
#' Mediators: smoking, high_bp, diabetes, cpk, anaemia, serum_creatinine, ejection_fraction
#' Confounders: age, platelets, serum_sodium, time
preproc_hf <- function(csv_path = NULL) {
  if (is.null(csv_path)) {
    root <- rprojroot::find_root(rprojroot::is_git_root)
    csv_path <- file.path(root, "outputs", "hf_data_linear.csv")
  }
  dat <- read.csv(csv_path, check.names = FALSE)

  # Binary gender: 0 = female, 1 = male (or use as-is if already 0/1)
  if (!all(unique(dat$gender) %in% c(0, 1))) {
    dat$gender <- as.integer(dat$gender == 1)  # male = 1
  }

  sfm <- list(
    X = "gender",
    Y = "death_event",
    W = c("smoking", "high_bp", "diabetes", "cpk", "anaemia", "serum_creatinine", "ejection_fraction"),
    Z = c("age", "platelets", "serum_sodium", "time"),
    x0 = 0,
    x1 = 1
  )

  list(dat, sfm)
}
