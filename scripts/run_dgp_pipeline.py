#!/usr/bin/env python3
"""
Run full pipeline: custom DGP → causal discovery → metrics → fairness.
Saves data and combined results for the paper.
"""
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# Causal discovery
from castle.algorithms import PC, GES, Notears, DAG_GNN
from castle.metrics import MetricsDAG
from castle.common import GraphDAG

# DAGMA (separate package)
from dagma.linear import DagmaLinear
from dagma.utils import count_accuracy

root = Path(__file__).resolve().parent.parent
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)
drf_export = out_dir / "ad_drf_export"
drf_export.mkdir(exist_ok=True)


def generate_ad_ventricular(n=1000, seed=42):
    """Custom DGP for ventricular_volume outcome (no edge education → ventricular_volume)."""
    rng = np.random.default_rng(seed)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    U = rng.normal(0, 1, size=n)
    age = np.clip(rng.normal(70, 8, size=n), 50, 95)
    sex = rng.choice([0, 1], size=n, p=[0.5, 0.5])

    logit_apoe = -1.1 + 0.01 * (age - 70)
    apoe4 = (rng.random(n) < sigmoid(logit_apoe)).astype(int)

    education = np.clip(
        12.0 + 0.05 * (age - 70) - 0.6 * apoe4 + 0.5 * U + rng.normal(0, 2.0, size=n), 0, 25
    )
    av45 = 0.0005 * (age**2) + 0.8 * apoe4 + 0.5 * U + rng.normal(0, np.sqrt(0.4), size=n)
    tau = 0.00001 * (age**3) + 0.6 * av45 + 0.3 * U + rng.normal(0, np.sqrt(0.25), size=n)

    moca = (
        30.0
        - 0.03 * (age - 70)
        + 0.15 * (education - 12)
        + 0.5 * sex
        - 0.5 * apoe4
        - 0.5 * U
        + rng.normal(0, 2.0, size=n)
    )
    moca = np.clip(moca, 0, 30)

    brain_vol = (
        1400.0
        - 2.5 * (age - 70)
        - 40.0 * av45
        - 25.0 * tau
        + 30.0 * sex
        - 15.0 * apoe4
        + 2.0 * moca
        + 20.0 * U
        + rng.normal(0, 50.0, size=n)
    )
    brain_vol = np.clip(brain_vol, 800, 1600)

    ventricular_vol = (
        30.0
        + 0.5 * (age - 70)
        + 10.0 * tau
        + 5.0 * sex
        + 2.0 * apoe4
        + 3.0 * av45
        - 0.15 * moca
        + 5.0 * U
        + rng.normal(0, 15.0, size=n)
    )
    ventricular_vol = np.clip(ventricular_vol, 5, 200)

    node_names = [
        "sex",
        "education",
        "age",
        "apoe4",
        "moca",
        "av45",
        "tau",
        "brain volume",
        "ventricular volume",
    ]
    data = np.column_stack(
        [sex, education, age, apoe4, moca, av45, tau, brain_vol, ventricular_vol]
    )
    return data, node_names


def build_ground_truth():
    """Build ground truth graph (no education → ventricular_volume)."""
    node_names = [
        "sex",
        "education",
        "age",
        "apoe4",
        "moca",
        "av45",
        "tau",
        "brain volume",
        "ventricular volume",
    ]
    edges = [
        ("sex", "ventricular volume"),
        ("sex", "brain volume"),
        ("sex", "moca"),
        ("education", "moca"),
        ("age", "moca"),
        ("age", "av45"),
        ("age", "tau"),
        ("age", "brain volume"),
        ("age", "ventricular volume"),
        ("moca", "brain volume"),
        ("moca", "ventricular volume"),
        ("av45", "tau"),
        ("av45", "brain volume"),
        ("av45", "ventricular volume"),
        ("tau", "brain volume"),
        ("tau", "ventricular volume"),
        ("apoe4", "brain volume"),
        ("apoe4", "ventricular volume"),
    ]
    g = nx.DiGraph()
    g.add_nodes_from(node_names)
    g.add_edges_from(edges)
    return nx.to_numpy_array(g), node_names


def main():
    seed = 42
    n = 1000

    print("1. Generating data (custom DGP)...")
    data, node_names = generate_ad_ventricular(n=n, seed=seed)
    adj_gt, _ = build_ground_truth()

    # Save data to CSV (for R scripts)
    df = pd.DataFrame(data, columns=node_names)
    df.to_csv(out_dir / "ad_data_1000_dgp.csv", index=False)
    df.to_csv(out_dir / "ad_data_1000_linear.csv", index=False)  # for ad-preproc
    print(f"   Saved to {out_dir}/ad_data_1000_dgp.csv and ad_data_1000_linear.csv")

    # Export for dag_fairness_drf.R
    df_export = df.copy()
    df_export.columns = [c.replace(" ", "_") for c in df_export.columns]
    df_export["sex"] = (df_export["sex"] > df_export["sex"].median()).astype(int)
    df_export.to_csv(drf_export / "data.csv", index=False)
    with open(drf_export / "node_names.txt", "w") as f:
        f.write("\n".join(n.replace(" ", "_") for n in node_names) + "\n")
    for name, adj in [("ground_truth", adj_gt)]:
        np.savetxt(drf_export / f"adj_{name}.csv", adj, fmt="%d", delimiter=",")

    print("2. Running causal discovery algorithms...")
    # PC
    pc = PC()
    pc.learn(data)
    pc_adj = pc.causal_matrix

    # GES
    ges = GES(criterion="bic")
    ges.learn(data)
    ges_adj = ges.causal_matrix

    # NOTEARS
    notears = Notears()
    notears.learn(data)
    notears_adj = notears.causal_matrix

    # DAG-GNN (slow; run with SKIP_DAG_GNN=1 to skip)
    import os
    skip_dag_gnn = str(os.environ.get("SKIP_DAG_GNN", "0")).lower() in ("1", "true", "yes")
    if skip_dag_gnn:
        dag_gnn_adj = np.zeros_like(adj_gt)  # placeholder
        print("   Skipping DAG-GNN (SKIP_DAG_GNN=1)")
    else:
        dag_gnn = DAG_GNN()
        dag_gnn.learn(data)
        dag_gnn_adj = dag_gnn.causal_matrix

    # DAGMA
    model = DagmaLinear(loss_type="l2")
    W_est = model.fit(data.astype(np.float64), lambda1=0.1)
    dagma_adj = (np.abs(W_est) > 0.05).astype(int)

    # Save adjacency matrices for R
    adj_dict = {
        "ground_truth": adj_gt,
        "PC": pc_adj,
        "GES": ges_adj,
        "NOTEARS": notears_adj,
        "DAGMA": dagma_adj,
    }
    if not skip_dag_gnn:
        adj_dict["DAG_GNN"] = dag_gnn_adj
    for name, adj in adj_dict.items():
        np.savetxt(drf_export / f"adj_{name}.csv", adj, fmt="%d", delimiter=",")

    print("3. Computing causal discovery metrics...")
    results = []
    for name, adj in adj_dict.items():
        if name == "ground_truth":
            continue
        m = MetricsDAG(B_est=adj, B_true=adj_gt)
        results.append((name, m.metrics))

    cd_df = pd.DataFrame({name: m for name, m in results}).T
    print(cd_df.to_string())

    print("4. Running R fairness scripts...")
    import subprocess
    import shutil

    rscript = shutil.which("Rscript")
    if not rscript:
        # Try conda r_env (used by experiments_ad.R)
        conda_r = Path.home() / "anaconda3" / "envs" / "r_env" / "bin" / "Rscript"
        if conda_r.exists():
            rscript = str(conda_r)
        else:
            rscript = None
    if not rscript:
        print("   Rscript not found. Skipping R fairness scripts.")
        print("   Run manually: Rscript scripts/ctf_se_decomposition_ad.R")
        print("   Run manually: Rscript scripts/dag_fairness_drf.R")
        r1 = r2 = r3 = type("R", (), {"returncode": 1, "stdout": "", "stderr": "Rscript not found"})()
    else:
        # ctf_se_decomposition_ad.R (TV, Ctf-DE, Ctf-IE, Ctf-SE)
        r1 = subprocess.run(
            [rscript, str(root / "scripts" / "ctf_se_decomposition_ad.R")],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r1.returncode != 0:
            print("   ctf_se_decomposition_ad.R stderr:", (r1.stderr or "")[:500])
        else:
            print("   ctf_se_decomposition_ad.R output:")
            for line in (r1.stdout or "").strip().split("\n")[:20]:
                print("     ", line)

        # dag_fairness_drf.R (TV per graph) - can take several minutes
        r2 = subprocess.run(
            [rscript, str(root / "scripts" / "dag_fairness_drf.R")],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r2.returncode != 0:
            print("   dag_fairness_drf.R stderr:", (r2.stderr or "")[:500])
        else:
            print("   dag_fairness_drf.R output:")
            for line in (r2.stdout or "").strip().split("\n"):
                print("     ", line)

        # experiments_ad.R (CFUR, figures)
        r3 = subprocess.run(
            [rscript, str(root / "scripts" / "experiments_ad.R")],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r3.returncode != 0:
            print("   experiments_ad.R stderr:", (r3.stderr or "")[:500])

    # Load fairness results from R outputs
    fair_sfm_path = out_dir / "fairness_decomposition_sfm.csv"
    fair_tv_path = out_dir / "fairness_tv_per_graph.csv"

    if fair_sfm_path.exists():
        f_sfm = pd.read_csv(fair_sfm_path)
        tv_val = f_sfm.loc[f_sfm["measure"] == "tv", "value"]
        de_val = f_sfm.loc[f_sfm["measure"] == "ctfde", "value"]
        ie_val = f_sfm.loc[f_sfm["measure"] == "ctfie", "value"]
        se_val = f_sfm.loc[f_sfm["measure"] == "ctfse", "value"]
        fair_sfm = {
            "TV": float(tv_val.mean()) if len(tv_val) else None,
            "Ctf_DE": float(de_val.mean()) if len(de_val) else None,
            "Ctf_IE": float(ie_val.mean()) if len(ie_val) else None,
            "Ctf_SE": float(se_val.mean()) if len(se_val) else None,
        }
        print(f"   SFM fairness: TV={fair_sfm.get('TV')}, DE={fair_sfm.get('Ctf_DE')}, IE={fair_sfm.get('Ctf_IE')}, SE={fair_sfm.get('Ctf_SE')}")
    else:
        fair_sfm = {}

    if fair_tv_path.exists():
        fair_tv_df = pd.read_csv(fair_tv_path)
        print(f"   TV per graph: {fair_tv_df.to_dict('records')}")

    # Build combined results for paper
    paper_results = []
    for name in ["PC", "GES", "NOTEARS", "DAGMA"] + (["DAG_GNN"] if not skip_dag_gnn else []):
        row = {"algorithm": name}
        if name in cd_df.index:
            for col in cd_df.columns:
                row[f"cd_{col}"] = cd_df.loc[name, col]
        paper_results.append(row)

    cd_results_path = out_dir / "paper_results_dgp.csv"
    pd.DataFrame(paper_results).to_csv(cd_results_path, index=False)
    print(f"\n5. Saved causal discovery results to {cd_results_path}")

    # Save combined paper results (CD + fairness)
    combined = []
    for name in ["PC", "GES", "NOTEARS", "DAGMA"] + (["DAG_GNN"] if not skip_dag_gnn else []):
        row = {"algorithm": name}
        if name in cd_df.index:
            row.update({k: v for k, v in cd_df.loc[name].items()})
        if fair_tv_path.exists():
            ft = pd.read_csv(fair_tv_path)
            match = ft[ft["graph"] == name]
            if len(match):
                row["TV_drf"] = match["TV"].values[0]
        combined.append(row)
    combined_path = out_dir / "paper_results_combined_dgp.csv"
    pd.DataFrame(combined).to_csv(combined_path, index=False)
    print(f"   Saved combined results to {combined_path}")

    # SFM fairness summary
    if fair_sfm:
        pd.DataFrame([{"metric": k, "value": v} for k, v in fair_sfm.items()]).to_csv(
            out_dir / "paper_fairness_sfm_dgp.csv", index=False
        )
        print(f"   Saved SFM fairness to {out_dir}/paper_fairness_sfm_dgp.csv")

    cd_df.to_csv(out_dir / "paper_cd_metrics_dgp.csv")
    print(f"   Saved CD metrics table to {out_dir}/paper_cd_metrics_dgp.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
