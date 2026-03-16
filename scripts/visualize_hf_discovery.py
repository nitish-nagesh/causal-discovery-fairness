#!/usr/bin/env python3
"""
Visualize HF causal discovery outputs: ground truth, PC, GES, FCI.
Helps determine mediators and confounders for SFM mapping.
Includes metrics bar chart with 95% CIs.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent.parent
hf_export = root / "outputs" / "hf_export"
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)

NODE_NAMES = [
    "gender", "smoking", "high_bp", "diabetes", "cpk", "serum_sodium",
    "serum_creatinine", "ejection_fraction", "anaemia", "platelets", "age",
    "time", "death_event",
]

X_VAR = "gender"
Y_VAR = "death_event"


def load_adj(name):
    """Load adjacency matrix. adj[i,j]=1 means edge i->j."""
    path = hf_export / f"adj_{name}.csv"
    if not path.exists():
        return None
    adj = np.loadtxt(path, delimiter=",")
    if adj.ndim == 1:
        adj = adj.reshape(1, -1)
    return adj.astype(int)


def get_mediators_confounders(adj, nodes):
    """
    Mediators = nodes on directed path from X to Y (excluding X,Y).
    Confounders = ancestors of Y that are not X and not mediators.
    For CPDAG/cyclic: use path existence (X->node->Y).
    """
    n = len(nodes)
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(nodes[i])
    for i in range(n):
        for j in range(n):
            if adj[i, j] != 0:
                G.add_edge(nodes[i], nodes[j])

    mediators = set()
    confounders = set()

    if X_VAR not in G or Y_VAR not in G:
        return sorted(mediators), sorted(confounders)

    try:
        # Mediators: nodes on any simple path from X to Y
        paths = list(nx.all_simple_paths(G, X_VAR, Y_VAR, cutoff=15))
        for p in paths:
            for node in p[1:-1]:
                mediators.add(node)

        # Confounders: ancestors of Y not in {X} ∪ mediators
        ancestors_of_y = set(nx.ancestors(G, Y_VAR))
        for node in ancestors_of_y:
            if node != X_VAR and node not in mediators:
                confounders.add(node)
    except (nx.NetworkXError, nx.NodeNotFound, nx.NetworkXNoPath):
        # Fallback: nodes reachable from X that can reach Y
        try:
            for node in G.nodes():
                if node in (X_VAR, Y_VAR):
                    continue
                if nx.has_path(G, X_VAR, node) and nx.has_path(G, node, Y_VAR):
                    mediators.add(node)
        except nx.NetworkXError:
            pass

    return sorted(mediators), sorted(confounders)


def draw_graph(adj, nodes, title, ax, layout="shell"):
    """Draw DAG on given axes."""
    G = nx.DiGraph()
    for i, n in enumerate(nodes):
        G.add_node(n)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if adj[i, j] != 0:
                G.add_edge(nodes[i], nodes[j])

    # Layout: hierarchical if DAG, else spring
    try:
        if nx.is_directed_acyclic_graph(G):
            levels = {}
            for node in nx.topological_sort(G):
                preds = list(G.predecessors(node))
                levels[node] = max([levels.get(p, 0) for p in preds] or [0]) + 1
            by_level = {}
            for n, lvl in levels.items():
                by_level.setdefault(lvl, []).append(n)
            pos = {}
            for lvl in sorted(by_level.keys()):
                for i, n in enumerate(by_level[lvl]):
                    pos[n] = (i - len(by_level[lvl]) / 2, -lvl)
        else:
            pos = nx.spring_layout(G, k=1.2, seed=42)
    except (nx.NetworkXError, nx.NetworkXNoCycle):
        pos = nx.spring_layout(G, k=1.2, seed=42)

    # Highlight X and Y
    node_colors = []
    for n in G.nodes():
        if n == X_VAR:
            node_colors.append("#e74c3c")  # red
        elif n == Y_VAR:
            node_colors.append("#3498db")  # blue
        else:
            node_colors.append("#95a5a6")  # gray

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=15,
                          edge_color="#7f8c8d", connectionstyle="arc3,rad=0.1")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    graphs = ["ground_truth", "PC", "GES", "FCI"]
    adjs = {}
    for g in graphs:
        a = load_adj(g)
        if a is not None:
            adjs[g] = a

    if not adjs:
        print("No adjacency matrices found. Run: python scripts/run_hf_pipeline.py")
        return

    # Figure: 2x3 grid
    n_plots = len(adjs)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for idx, (name, adj) in enumerate(adjs.items()):
        ax = axes.flat[idx]
        draw_graph(adj, NODE_NAMES, name, ax)
        med, conf = get_mediators_confounders(adj, NODE_NAMES)
        m_str = ",".join(med[:4]) + ("..." if len(med) > 4 else "")
        c_str = ",".join(conf[:4]) + ("..." if len(conf) > 4 else "")
        ax.text(0.02, 0.98, f"M: {m_str or '-'}\nC: {c_str or '-'}",
                transform=ax.transAxes, fontsize=5, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for idx in range(len(adjs), nrows * ncols):
        axes.flat[idx].axis("off")

    plt.suptitle("HF Causal Discovery: Red=gender (X), Blue=death_event (Y)", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / "hf_discovery_graphs.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

    # Manuscript: 3-panel (PC, GES, FCI) as PDF for vector quality (no pixelation)
    manuscript_dir = root / "manuscript"
    if manuscript_dir.exists():
        for algo in ["PC", "GES", "FCI"]:
            if algo in adjs:
                fig3, ax3 = plt.subplots(1, 1, figsize=(5, 4))
                draw_graph(adjs[algo], NODE_NAMES, algo, ax3)
                plt.tight_layout()
                pdf_path = manuscript_dir / f"{algo.lower()}-hfcr-rv-01.pdf"
                plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
                plt.close()
                print(f"Saved {pdf_path}")

    # Tabulate mediators/confounders per graph
    rows = []
    for name, adj in adjs.items():
        med, conf = get_mediators_confounders(adj, NODE_NAMES)
        rows.append({
            "graph": name,
            "mediators": ",".join(med),
            "confounders": ",".join(conf),
            "n_edges": int(np.sum(adj)),
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = out_dir / "hf_sfm_mapping.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved SFM mapping to {csv_path}")
    print(df.to_string())

    # Metrics bar chart with 95% CIs
    metrics_path = out_dir / "paper_cd_metrics_hf.csv"
    if metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        # Filter to algorithms we have (exclude ground_truth)
        mdf = mdf[mdf["algorithm"].isin([g for g in graphs if g != "ground_truth"])]
        if len(mdf) > 0 and "F1_lo" in mdf.columns:
            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
            metrics_to_plot = [
                ("F1", "F1 Score", axes2[0, 0]),
                ("shd", "Structural Hamming Distance", axes2[0, 1]),
                ("fdr", "False Discovery Rate", axes2[1, 0]),
                ("tpr", "True Positive Rate", axes2[1, 1]),
            ]
            for col, title, ax in metrics_to_plot:
                x = mdf["algorithm"]
                y = mdf[col].fillna(0)
                lo = mdf[f"{col}_lo"] if f"{col}_lo" in mdf.columns else pd.Series([np.nan] * len(mdf))
                hi = mdf[f"{col}_hi"] if f"{col}_hi" in mdf.columns else pd.Series([np.nan] * len(mdf))
                valid = lo.notna() & hi.notna()
                yerr_lo = (y - lo).where(valid, 0).values
                yerr_hi = (hi - y).where(valid, 0).values
                yerr = np.array([yerr_lo, yerr_hi])
                ax.bar(x, y, color="steelblue", edgecolor="navy", alpha=0.8)
                if valid.any():
                    ax.errorbar(x, y, yerr=yerr, fmt="none", color="black", capsize=3)
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=45)
            plt.suptitle("HF Causal Discovery Metrics (95% CI)", fontsize=12)
            plt.tight_layout()
            metrics_plot_path = out_dir / "hf_cd_metrics_plot.png"
            plt.savefig(metrics_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved metrics plot to {metrics_plot_path}")


if __name__ == "__main__":
    main()
