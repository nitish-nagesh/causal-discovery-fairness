#!/usr/bin/env python3
"""
Heart Failure Clinical Records pipeline.
- Fetches UCI dataset, renames columns
- Ground truth DAG (undirected edges resolved to directed)
- Causal discovery: PC, GES, FCI (NOTEARS, DAGMA, LINGAM omitted - sparse)
- Bootstrap CIs for all metrics (HF_BOOTSTRAP=30 by default)
- Fairness: composite, path-specific, CFUR (run_hf_fairness_analysis.py)
- R: ctf_se_decomposition_hf.R (skip dag_fairness_drf)
"""
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import subprocess
import shutil
import os
import sys

# Causal discovery (PC, GES, FCI only; NOTEARS, DAGMA, LINGAM omitted - sparse graphs)
from castle.algorithms import PC, GES
from castle.metrics import MetricsDAG

root = Path(__file__).resolve().parent.parent
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)
hf_export = out_dir / "hf_export"
hf_export.mkdir(exist_ok=True)

# Node order for adjacency matrices
NODE_NAMES = [
    "gender",
    "smoking",
    "high_bp",
    "diabetes",
    "cpk",
    "serum_sodium",
    "serum_creatinine",
    "ejection_fraction",
    "anaemia",
    "platelets",
    "age",
    "time",
    "death_event",
]

# UCI column -> our name
COL_MAP = {
    "sex": "gender",
    "high_blood_pressure": "high_bp",
    "creatinine_phosphokinase": "cpk",
    "DEATH_EVENT": "death_event",
}


def fetch_hf_data():
    """Fetch Heart Failure Clinical Records from UCI."""
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
    try:
        df = pd.read_csv(csv_url)
    except Exception:
        try:
            from ucimlrepo import fetch_ucirepo
            hf = fetch_ucirepo(id=519)
            df = pd.concat([hf.data.features, hf.data.targets], axis=1)
        except Exception:
            import urllib.request
            import zipfile
            import io
            url = "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip"
            req = urllib.request.urlopen(url)
            with zipfile.ZipFile(io.BytesIO(req.read())) as z:
                name = [n for n in z.namelist() if n.endswith(".csv")][0]
                df = pd.read_csv(z.open(name))
    return df


def preprocess_hf(df):
    """Rename columns to our convention."""
    df = df.copy()
    df = df.rename(columns=COL_MAP)
    # Ensure all our names exist
    for c in NODE_NAMES:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} after rename. Got: {list(df.columns)}")
    return df[NODE_NAMES]


def build_ground_truth_dag():
    """
    Build ground truth DAG. Per user spec:
    - Remove edge (high_bp - diabetes)
    - Convert undirected to directed: serum_sodium -> ejection_fraction, serum_creatinine -> ejection_fraction
    """
    edges = [
        ("gender", "smoking"),
        ("gender", "high_bp"),
        ("gender", "diabetes"),
        ("gender", "death_event"),
        ("smoking", "high_bp"),
        # high_bp - diabetes REMOVED
        ("high_bp", "serum_creatinine"),
        ("high_bp", "death_event"),
        ("high_bp", "ejection_fraction"),
        ("diabetes", "cpk"),
        ("diabetes", "serum_creatinine"),
        ("diabetes", "ejection_fraction"),
        ("serum_sodium", "ejection_fraction"),  # was undirected
        ("serum_creatinine", "ejection_fraction"),  # was undirected (was ef->sc, now sc->ef)
        ("ejection_fraction", "cpk"),
        ("ejection_fraction", "platelets"),
        ("ejection_fraction", "death_event"),
        ("anaemia", "ejection_fraction"),
        ("serum_creatinine", "death_event"),
        ("age", "ejection_fraction"),
        ("age", "death_event"),
        ("time", "death_event"),
    ]
    g = nx.DiGraph()
    g.add_nodes_from(NODE_NAMES)
    g.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(g), "Ground truth must be a DAG"
    adj = nx.to_numpy_array(g, nodelist=NODE_NAMES)
    return adj


def _pag_to_dag(G):
    """Convert causallearn PAG to DAG for MetricsDAG. Directed: arrow at j, tail at i -> i->j.
    Endpoint: TAIL=-1, ARROW=1, CIRCLE=2, NULL=0."""
    g = G.graph
    n = g.shape[0]
    B = np.zeros((n, n))
    TAIL, ARROW, NULL = -1, 1, 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if g[i, j] == ARROW and g[j, i] == TAIL:
                B[i, j] = 1
            elif g[i, j] != NULL and g[j, i] != NULL and B[j, i] == 0:
                if i < j:
                    B[i, j] = 1
    return B.astype(int)


def run_causal_discovery(data, seed=42):
    """Run PC, GES, FCI (NOTEARS, DAGMA, LINGAM omitted - sparse graphs)."""
    np.random.seed(seed)
    X = data.astype(np.float64)
    n, d = X.shape

    # Standardize for continuous methods
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    results = {}

    # PC
    pc = PC()
    pc.learn(X_std)
    pc_adj = pc.causal_matrix
    if pc_adj.shape != (d, d):
        pc_adj = np.zeros((d, d))
    results["PC"] = (pc_adj > 0).astype(int)

    # GES
    ges = GES(criterion="bic")
    ges.learn(X_std)
    ges_adj = ges.causal_matrix
    results["GES"] = (np.abs(ges_adj) > 0.05).astype(int)

    # FCI (causallearn)
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.cit import fisherz
        G_fci, _ = fci(dataset=X_std, alpha=0.05, independence_test_method=fisherz)
        results["FCI"] = _pag_to_dag(G_fci)
    except Exception:
        results["FCI"] = np.zeros((d, d)).astype(int)

    return results


def bootstrap_metrics(adj_gt, data, n_boot=100, seed=42):
    """Bootstrap causal discovery metrics for CIs."""
    rng = np.random.default_rng(seed)
    n = len(data)
    metrics_list = []

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_data = data[idx]
        adjs = run_causal_discovery(boot_data, seed=seed + b)
        for name, adj in adjs.items():
            try:
                m = MetricsDAG(B_est=adj, B_true=adj_gt)
                metrics_list.append({"boot": b, "algorithm": name, **m.metrics})
            except Exception:
                pass

    return pd.DataFrame(metrics_list)


def main():
    print("1. Fetching Heart Failure Clinical Records...")
    df_raw = fetch_hf_data()
    df = preprocess_hf(df_raw)
    data = df.values
    print(f"   Loaded {len(df)} rows, {len(NODE_NAMES)} variables")

    # Save for R
    df.to_csv(out_dir / "hf_data.csv", index=False)
    df.to_csv(out_dir / "hf_data_linear.csv", index=False)
    print(f"   Saved to {out_dir}/hf_data.csv")

    print("2. Building ground truth DAG...")
    adj_gt = build_ground_truth_dag()
    np.savetxt(hf_export / "adj_ground_truth.csv", adj_gt, fmt="%d", delimiter=",")
    with open(hf_export / "node_names.txt", "w") as f:
        f.write("\n".join(NODE_NAMES) + "\n")

    print("3. Running causal discovery (PC, GES, FCI)...")
    adjs = run_causal_discovery(data)
    for name, adj in adjs.items():
        np.savetxt(hf_export / f"adj_{name}.csv", adj, fmt="%d", delimiter=",")

    print("3b. Visualizing discovery outputs...")
    try:
        subprocess.run(
            [sys.executable, str(root / "scripts" / "visualize_hf_discovery.py")],
            cwd=str(root),
            capture_output=True,
            timeout=30,
        )
    except Exception as e:
        print(f"   Visualization failed: {e}. Run: python scripts/visualize_hf_discovery.py")

    ALGS = ["PC", "GES", "FCI"]
    print("4. Computing causal discovery metrics (point estimates)...")
    cd_results = []
    for name in ALGS:
        if name not in adjs:
            continue
        m = MetricsDAG(B_est=adjs[name], B_true=adj_gt)
        cd_results.append({"algorithm": name, **m.metrics})
    cd_df = pd.DataFrame(cd_results)
    print(cd_df.to_string())

    n_boot = int(os.environ.get("HF_BOOTSTRAP", "30"))
    skip_boot = str(os.environ.get("SKIP_BOOTSTRAP", "0")).lower() in ("1", "true", "yes")
    if skip_boot:
        print("5. Skipping bootstrap (SKIP_BOOTSTRAP=1). Using point estimates only.")
        boot_df = pd.DataFrame()
    else:
        print(f"5. Bootstrap for confidence intervals (n={n_boot})...")
        boot_df = bootstrap_metrics(adj_gt, data, n_boot=n_boot)
    ci_cols = ["fdr_lo", "fdr_hi", "tpr_lo", "tpr_hi", "shd_lo", "shd_hi", "F1_lo", "F1_hi"]
    if len(boot_df) > 0:
        ci = boot_df.groupby("algorithm").agg(
            fdr_lo=("fdr", lambda x: np.percentile(x, 2.5)),
            fdr_hi=("fdr", lambda x: np.percentile(x, 97.5)),
            tpr_lo=("tpr", lambda x: np.percentile(x, 2.5)),
            tpr_hi=("tpr", lambda x: np.percentile(x, 97.5)),
            shd_lo=("shd", lambda x: np.percentile(x, 2.5)),
            shd_hi=("shd", lambda x: np.percentile(x, 97.5)),
            F1_lo=("F1", lambda x: np.percentile(x, 2.5)),
            F1_hi=("F1", lambda x: np.percentile(x, 97.5)),
        ).reset_index()
        cd_with_ci = cd_df.merge(ci[["algorithm"] + ci_cols], on="algorithm")
        cd_with_ci.to_csv(out_dir / "paper_cd_metrics_hf.csv", index=False)
        print("   Saved paper_cd_metrics_hf.csv with CIs")
    else:
        for c in ci_cols:
            cd_df[c] = np.nan
        cd_df.to_csv(out_dir / "paper_cd_metrics_hf.csv", index=False)
        print("   Saved paper_cd_metrics_hf.csv (no bootstrap CIs)")

    # Re-run visualization to include metrics bar chart (needs CSV)
    try:
        subprocess.run(
            [sys.executable, str(root / "scripts" / "visualize_hf_discovery.py")],
            cwd=str(root),
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass

    print("5b. Running HF fairness analysis (composite, path-specific, CFUR)...")
    try:
        subprocess.run(
            [sys.executable, str(root / "scripts" / "run_hf_fairness_analysis.py")],
            cwd=str(root),
            capture_output=True,
            timeout=300,
        )
    except Exception as e:
        print(f"   Fairness analysis failed: {e}. Run: python scripts/run_hf_fairness_analysis.py")

    print("6. Running R fairness script (ctf_se_decomposition_hf.R)...")
    rscript = shutil.which("Rscript")
    if not rscript:
        conda_r = Path.home() / "anaconda3" / "envs" / "r_env" / "bin" / "Rscript"
        if conda_r.exists():
            rscript = str(conda_r)
    if rscript:
        r1 = subprocess.run(
            [rscript, str(root / "scripts" / "ctf_se_decomposition_hf.R")],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r1.returncode != 0:
            print("   ctf_se_decomposition_hf.R stderr:", (r1.stderr or "")[:800])
        else:
            print("   ctf_se_decomposition_hf.R OK")
            for line in (r1.stdout or "").strip().split("\n")[:25]:
                print("     ", line)
    else:
        print("   Rscript not found. Run manually: Rscript scripts/ctf_se_decomposition_hf.R")

    # Load fairness results
    fair_path = out_dir / "fairness_decomposition_hf.csv"
    if fair_path.exists():
        fair = pd.read_csv(fair_path)
        print("\n7. Fairness decomposition (with CIs from faircause bootstrap):")
        for m in ["tv", "ctfde", "ctfie", "ctfse"]:
            row = fair[fair["measure"] == m]
            if len(row):
                v = row["value"].values
                print(f"   {m}: {100*v.mean():.2f}% (n={len(v)} bootstrap)")

    # Combined paper results
    paper = cd_df.copy()
    if fair_path.exists():
        for m, col in [("tv", "TV"), ("ctfde", "Ctf_DE"), ("ctfie", "Ctf_IE"), ("ctfse", "Ctf_SE")]:
            r = fair[fair["measure"] == m]["value"]
            if len(r):
                paper[col] = 100 * r.mean()
    paper.to_csv(out_dir / "paper_results_hf.csv", index=False)
    print(f"\n8. Saved paper_results_hf.csv")
    print("Done.")


if __name__ == "__main__":
    main()
