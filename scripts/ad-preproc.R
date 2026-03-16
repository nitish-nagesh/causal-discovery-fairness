
#' Preprocess AD (Alzheimer's) data for accuracy decomposition
#' Matches cfa_ad.R variable setup: sex_binary, mediators, confounders, outcome
preproc_ad <- function(csv_path = NULL) {

  if (is.null(csv_path)) {
    root <- rprojroot::find_root(rprojroot::is_git_root)
    csv_path <- file.path(root, "outputs", "ad_data_1000_linear.csv")
  }
  dat <- read.csv(csv_path, check.names = FALSE)

  # Rename columns with spaces to avoid formula issues
  names(dat)[names(dat) == "brain volume"] <- "brain_volume"
  names(dat)[names(dat) == "ventricular volume"] <- "ventricular_volume"

  # Binary gender: above median = 1 (dominant), else 0 (protected)
  # x0 = 0 (protected), x1 = 1 (dominant) per cfa_ad.R convention
  dat$sex_binary <- as.integer(dat$sex > median(dat$sex))

  sfm <- list(
    X = "sex_binary",
    Y = "ventricular_volume",
    W = c("moca", "brain_volume"),
    Z = c("education", "age", "apoe4", "av45", "tau"),
    x0 = 0, x1 = 1
  )

  list(dat, sfm)
}
