root <- rprojroot::find_root(rprojroot::is_git_root)
r_dir <- file.path(root, "causal-acc-decomp", "R")
invisible(lapply(list.files(r_dir, full.names = TRUE), source))
source(file.path(root, "scripts", "ad-preproc.R"))

dataset <- "ad"
loss <- "rmse"
tmp <- preproc(dataset)
data <- tmp[[1]]
sfm <- tmp[[2]]

out_dir <- file.path(root, "outputs")
dir.create(out_dir, showWarnings = FALSE)

# bootstrap the accuracy decomposition (pred="baseline" = ranger, fast)
acc_boot <- accuracy_decomposition_boot(data, sfm$X, sfm$Y, sfm$Z, sfm$W,
                                        loss = loss, x0 = 0, x1 = 1,
                                        pred = "baseline")

for (type in c("df_da", "loss_bars", "pareto", "tv_bar")) {

  plt <- vis_route(acc_boot, type, dataset)
  ggsave(filename = file.path(out_dir, paste0(dataset, "_", type, ".png")),
         plot = plt, width = 7 + 3 * (type == "pareto"),
         height = 4 + 2 * (type == "pareto"), bg = "white")
}
vis_route(acc_boot, "wtp")
