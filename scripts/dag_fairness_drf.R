#!/usr/bin/env Rscript
# DAG-based fairness estimation (ranger + residual sampling)
# Adapted from CFA_repo/scripts/section-6/drf-beyond-sfm.R
# Works for arbitrary discovered DAGs - not limited to SFM structure
#
# Data source: Run notebooks/cd_revamp.ipynb export cell first to create
# outputs/ad_drf_export/ with data.csv, node_names.txt, adj_*.csv

library(ranger)

root <- rprojroot::find_root(rprojroot::is_git_root)
out_dir <- file.path(root, "outputs", "ad_drf_export")

# Build topological order from adjacency matrix (parents before children)
# adj[i,j]=1 means edge from node i to node j (i is parent of j)
topo_order <- function(adj, nodes) {
  n <- nrow(adj)
  ord <- c()
  remaining <- seq_len(n)
  while (length(remaining) > 0) {
    for (i in remaining) {
      parents <- which(adj[, i] != 0)
      if (all(parents %in% ord)) {
        ord <- c(ord, i)
        remaining <- setdiff(remaining, i)
        break
      }
    }
  }
  nodes[ord]
}

# Monte Carlo intervention: ranger + residual sampling
do_mc <- function(data, adj, nodes, x_var, x_val, n_samp = 1000, seed = 42) {
  set.seed(seed)
  n_samp <- min(n_samp, nrow(data))

  ord <- topo_order(adj, nodes)
  int_data <- data[sample(nrow(data), n_samp, replace = TRUE), ]

  for (node in ord) {
    j <- match(node, nodes)
    pa_idx <- which(adj[, j] != 0)
    pa <- nodes[pa_idx]

    if (length(pa) == 0) next

    if (x_var %in% pa) int_data[[x_var]] <- x_val

    form <- as.formula(paste(node, "~", paste(pa, collapse = "+")))
    fit <- ranger(form, data = data, num.trees = 100)
    pred <- predict(fit, int_data)$predictions
    res <- data[[node]] - predict(fit, data)$predictions
    int_data[[node]] <- pred + sample(res, n_samp, replace = TRUE)
  }

  int_data
}

# Load exported data from notebook
data <- read.csv(file.path(out_dir, "data.csv"), check.names = FALSE)
nodes <- trimws(readLines(file.path(out_dir, "node_names.txt")))

# sex binarized in notebook export; ventricular_volume
x_var <- "sex"
y_var <- "ventricular_volume"
x0 <- 0
x1 <- 1

graphs <- c("ground_truth", "PC", "GES", "NOTEARS", "DAGMA", "DAG_GNN")
results <- data.frame(graph = character(), TV = numeric(), stringsAsFactors = FALSE)

for (g in graphs) {
  adj_path <- file.path(out_dir, paste0("adj_", g, ".csv"))
  if (!file.exists(adj_path)) next

  adj <- as.matrix(read.csv(adj_path, header = FALSE))

  nrep <- 3
  e0 <- mean(replicate(nrep, mean(do_mc(data, adj, nodes, x_var, x0, n_samp = 300, seed = sample.int(1e6, 1))[[y_var]])))
  e1 <- mean(replicate(nrep, mean(do_mc(data, adj, nodes, x_var, x1, n_samp = 300, seed = sample.int(1e6, 1))[[y_var]])))
  tv <- e1 - e0

  results <- rbind(results, data.frame(graph = g, TV = tv))
}

print(results)
# Write for paper pipeline
write.csv(results, file.path(root, "outputs", "fairness_tv_per_graph.csv"), row.names = FALSE)
