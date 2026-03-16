#!/usr/bin/env python3
"""
HFCR Fairness Analysis: composite effects, path-specific effects, utility ratio (CFUR).

Runs for ground_truth, PC, GES, FCI using mediators and confounders from hf_sfm_mapping.csv.
All outputs include 95% bootstrap CIs. Generates plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "scripts"))

from cfa_fairness import cfa_decomposition
from cfur_fairness import compute_cfur
from individual_effects import individual_effects_decomposition

out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)

X_NAME = "gender"
Y_NAME = "death_event"
CONFIGS = ["ground_truth", "PC", "GES", "FCI"]
N_BOOT = int(__import__("os").environ.get("HF_FAIRNESS_BOOT", "100"))
N_REPS_CFUR = int(__import__("os").environ.get("HF_CFUR_REPS", "10"))
N_BOOT_CFUR = int(__import__("os").environ.get("HF_CFUR_BOOT", "50"))


def parse_sfm_mapping(csv_path: Path) -> dict:
    """Load hf_sfm_mapping.csv and return {graph: {mediators: [...], confounders: [...]}}."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        g = row["graph"]
        med = row["mediators"]
        conf = row["confounders"]
        mediators = [x.strip() for x in str(med).split(",") if x.strip()] if pd.notna(med) and str(med).strip() else []
        confounders = [x.strip() for x in str(conf).split(",") if x.strip()] if pd.notna(conf) and str(conf).strip() else []
        out[g] = {"W": mediators, "Z": confounders}
    return out


def run_one_config(df: pd.DataFrame, sfm: dict, label: str, n_boot: int = N_BOOT):
    """Run composite, path-specific, and CFUR for one SFM config."""
    sfm_full = {"X": X_NAME, "Y": Y_NAME, "Z": sfm["Z"], "W": sfm["W"]}

    # Composite effects with bootstrap CIs
    cfa = cfa_decomposition(
        df, adj=None, names=None,
        x_name=X_NAME, y_name=Y_NAME,
        sfm_override=sfm_full,
        n_bootstrap=n_boot,
        random_state=42,
    )

    # Path-specific (individual effects) with bootstrap CIs
    n_ie_boot = min(15, n_boot)
    rng = np.random.default_rng(42)
    ie_boot = []
    for b in range(n_ie_boot):
        idx = rng.integers(0, len(df), size=len(df))
        boot_df = df.iloc[idx].reset_index(drop=True)
        ie_b = individual_effects_decomposition(
            boot_df, adj=None, names=None,
            x_name=X_NAME, y_name=Y_NAME,
            sfm_override=sfm_full,
            n_rep=10,
            n_samp=1000,
            seed=42 + b,
        )
        if "error" not in ie_b:
            ie_boot.append(ie_b)
    ie = individual_effects_decomposition(
        df, adj=None, names=None,
        x_name=X_NAME, y_name=Y_NAME,
        sfm_override=sfm_full,
        n_rep=15,
        n_samp=1500,
        seed=42,
    )
    if ie_boot:
        ie["ctf_se_ci"] = {}
        ie["ctf_se_sd"] = {}
        ie["ctf_ie_ci"] = {}
        ie["ctf_ie_sd"] = {}
        for z in ie.get("ctf_se_by_z", {}):
            vals = [ib["ctf_se_by_z"].get(z, np.nan) for ib in ie_boot]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                arr = np.array(vals)
                ie["ctf_se_ci"][z] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
                ie["ctf_se_sd"][z] = float(np.std(arr))
        for w in ie.get("ctf_ie_by_w", {}):
            vals = [ib["ctf_ie_by_w"].get(w, np.nan) for ib in ie_boot]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                arr = np.array(vals)
                ie["ctf_ie_ci"][w] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
                ie["ctf_ie_sd"][w] = float(np.std(arr))

    # CFUR (utility ratio)
    cfur = compute_cfur(
        df, adj=None, names=None,
        x_name=X_NAME, y_name=Y_NAME,
        sfm_override=sfm_full,
        nreps=N_REPS_CFUR,
        nboot=N_BOOT_CFUR,
        seed=42,
    )

    return {"cfa": cfa, "ie": ie, "cfur": cfur, "label": label}


def main():
    mapping_path = out_dir / "hf_sfm_mapping.csv"
    data_path = out_dir / "hf_data.csv"

    if not mapping_path.exists() or not data_path.exists():
        print("Run python scripts/run_hf_pipeline.py first to generate hf_sfm_mapping.csv and hf_data.csv")
        return 1

    df = pd.read_csv(data_path)
    if "gender" not in df.columns:
        if "sex" in df.columns:
            df = df.rename(columns={"sex": "gender"})
        else:
            print("Missing gender column")
            return 1

    sfm_map = parse_sfm_mapping(mapping_path)

    results = {}
    for cfg in CONFIGS:
        if cfg not in sfm_map:
            print(f"Skipping {cfg} (not in SFM mapping)")
            continue
        sfm = sfm_map[cfg]
        # Skip if both empty (no paths to decompose)
        if not sfm["Z"] and not sfm["W"]:
            print(f"Skipping {cfg} (no mediators or confounders)")
            continue
        print(f"Running {cfg}: Z={len(sfm['Z'])} vars, W={len(sfm['W'])} vars")
        results[cfg] = run_one_config(df, sfm, cfg)
        print(f"  Done {cfg}")

    # Build summary tables with CIs
    composite_rows = []
    path_specific_rows = []
    cfur_rows = []

    for cfg, res in results.items():
        cfa = res["cfa"]
        ie = res["ie"]
        cfur = res["cfur"]

        pt = cfa.get("point", {})
        ci = cfa.get("ci", {})
        se = cfa.get("se", {})
        for key in ["TV", "Ctf-DE", "Ctf-IE", "Ctf-SE"]:
            v = pt.get(key, np.nan)
            c = ci.get(key, (np.nan, np.nan))
            sd = se.get(key, np.nan)
            composite_rows.append({
                "graph": cfg,
                "effect": key,
                "value": v,
                "sd": sd,
                "ci_lo": c[0],
                "ci_hi": c[1],
            })

        se_ci = ie.get("ctf_se_ci", {})
        se_sd = ie.get("ctf_se_sd", {})
        ie_ci = ie.get("ctf_ie_ci", {})
        ie_sd = ie.get("ctf_ie_sd", {})
        for z, contrib in ie.get("ctf_se_by_z", {}).items():
            c = se_ci.get(z, (np.nan, np.nan))
            path_specific_rows.append({
                "graph": cfg,
                "effect": "Ctf-SE",
                "variable": z,
                "contribution": contrib,
                "sd": se_sd.get(z, np.nan),
                "ci_lo": c[0],
                "ci_hi": c[1],
            })
        for w, contrib in ie.get("ctf_ie_by_w", {}).items():
            c = ie_ci.get(w, (np.nan, np.nan))
            path_specific_rows.append({
                "graph": cfg,
                "effect": "Ctf-IE",
                "variable": w,
                "contribution": contrib,
                "sd": ie_sd.get(w, np.nan),
                "ci_lo": c[0],
                "ci_hi": c[1],
            })
        if "ctf_de" in ie and not np.isnan(ie["ctf_de"]):
            path_specific_rows.append({
                "graph": cfg,
                "effect": "Ctf-DE",
                "variable": "(direct)",
                "contribution": ie["ctf_de"],
                "ci_lo": np.nan,
                "ci_hi": np.nan,
            })

        if "error" not in cfur and len(cfur.get("summary", [])) > 0:
            for _, r in cfur["summary"].iterrows():
                sd = r["cfur_sd"]
                ci_lo = r["cfur_mean"] - 1.96 * sd if not np.isnan(sd) else np.nan
                ci_hi = r["cfur_mean"] + 1.96 * sd if not np.isnan(sd) else np.nan
                cfur_rows.append({
                    "graph": cfg,
                    "path": r["path"],
                    "cfur_mean": r["cfur_mean"],
                    "cfur_sd": r["cfur_sd"],
                    "cfur_ci_lo": ci_lo,
                    "cfur_ci_hi": ci_hi,
                    "fair_gain_mean": r["fair_gain_mean"],
                    "fair_gain_sd": r["fair_gain_sd"],
                    "stat_drop_mean": r["stat_drop_mean"],
                    "stat_drop_sd": r["stat_drop_sd"],
                })

    # Save CSVs
    pd.DataFrame(composite_rows).to_csv(out_dir / "hf_composite_effects.csv", index=False)
    pd.DataFrame(path_specific_rows).to_csv(out_dir / "hf_path_specific_effects.csv", index=False)
    pd.DataFrame(cfur_rows).to_csv(out_dir / "hf_cfur_utility_ratio.csv", index=False)
    print(f"Saved hf_composite_effects.csv, hf_path_specific_effects.csv, hf_cfur_utility_ratio.csv")

    # Print summary for verification
    print("\n=== HFCR Fairness Summary (Final) ===")
    comp = pd.DataFrame(composite_rows)
    for g in comp["graph"].unique():
        sg = comp[comp["graph"] == g]
        print(f"\n{g}:")
        for _, r in sg.iterrows():
            ci_str = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]" if pd.notna(r["ci_lo"]) else "—"
            print(f"  {r['effect']}: {r['value']:.4f} (sd={r['sd']:.4f}, 95% CI {ci_str})")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Composite effects bar chart (ground_truth)
    ax = axes[0, 0]
    comp_df = pd.DataFrame(composite_rows)
    gt = comp_df[comp_df["graph"] == "ground_truth"]
    if len(gt) > 0:
        x = gt["effect"]
        y = gt["value"]
        yerr_lo = y - gt["ci_lo"]
        yerr_hi = gt["ci_hi"] - y
        yerr = np.array([yerr_lo.values, yerr_hi.values])
        bars = ax.bar(x, y, color="steelblue", edgecolor="navy", alpha=0.8)
        ax.errorbar(x, y, yerr=yerr, fmt="none", color="black", capsize=4)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("Effect (probability scale)")
    ax.set_title("Composite Effects (Ground Truth, 95% CI)")
    ax.tick_params(axis="x", rotation=45)

    # 2. Composite effects by graph (TV, Ctf-DE, Ctf-IE, Ctf-SE)
    ax = axes[0, 1]
    graphs_present = comp_df["graph"].unique().tolist()
    for eff in ["TV", "Ctf-DE", "Ctf-IE", "Ctf-SE"]:
        sub = comp_df[comp_df["effect"] == eff]
        if len(sub) > 0:
            x = sub["graph"]
            y = sub["value"]
            yerr_lo = y - sub["ci_lo"]
            yerr_hi = sub["ci_hi"] - y
            yerr = np.array([yerr_lo.values, yerr_hi.values])
            ax.errorbar(range(len(x)), y, yerr=yerr, fmt="o-", label=eff, capsize=3)
    ax.set_xticks(range(len(graphs_present)))
    ax.set_xticklabels(graphs_present, rotation=45)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("Effect")
    ax.set_title("Composite Effects by Graph (95% CI)")
    ax.legend(loc="best", fontsize=8)

    # 3. Path-specific effects (ground truth, Ctf-SE and Ctf-IE by variable) with CIs
    ax = axes[1, 0]
    ps_df = pd.DataFrame(path_specific_rows)
    gt_ps = ps_df[ps_df["graph"] == "ground_truth"]
    if len(gt_ps) > 0:
        vars_ = gt_ps["variable"].tolist()
        contribs = gt_ps["contribution"].tolist()
        colors = ["#e74c3c" if e == "Ctf-SE" else "#3498db" for e in gt_ps["effect"]]
        xerr_lo = (gt_ps["contribution"] - gt_ps["ci_lo"]).fillna(0).values
        xerr_hi = (gt_ps["ci_hi"] - gt_ps["contribution"]).fillna(0).values
        xerr = np.array([xerr_lo, xerr_hi])
        ax.barh(vars_, contribs, color=colors, alpha=0.8)
        if (xerr != 0).any():
            ax.errorbar(contribs, vars_, xerr=xerr, fmt="none", color="black", capsize=2)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Contribution (probability scale)")
    ax.set_title("Path-Specific Effects (Ground Truth)\nRed=Ctf-SE, Blue=Ctf-IE")

    # 4. CFUR by path and graph
    ax = axes[1, 1]
    cfur_df = pd.DataFrame(cfur_rows)
    if len(cfur_df) > 0:
        graphs = cfur_df["graph"].unique()
        paths = cfur_df["path"].unique()
        x = np.arange(len(graphs))
        width = 0.8 / max(len(paths), 1)
        for i, path in enumerate(paths):
            sub = cfur_df[cfur_df["path"] == path]
            vals = []
            sds = []
            for g in graphs:
                sg = sub[sub["graph"] == g]
                vals.append(sg["cfur_mean"].values[0] if len(sg) > 0 else 0)
                sds.append(sg["cfur_sd"].values[0] if len(sg) > 0 else 0)
            offset = (i - (len(paths) - 1) / 2) * width
            ax.bar(x + offset, vals, width, yerr=sds, label=path, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(graphs, rotation=45)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_ylabel("CFUR (fair_gain / stat_drop)")
        ax.set_title("Utility Ratio by Path (mean ± sd)")
        ax.legend(loc="best", fontsize=8)

    plt.suptitle("HFCR Fairness Analysis: Composite, Path-Specific, Utility Ratio", fontsize=12)
    plt.tight_layout()
    plot_path = out_dir / "hf_fairness_analysis_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")

    # Additional: CFUR bar chart for ground truth only
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    gt_cfur = cfur_df[cfur_df["graph"] == "ground_truth"]
    if len(gt_cfur) > 0:
        paths = gt_cfur["path"].tolist()
        means = gt_cfur["cfur_mean"].tolist()
        sds = gt_cfur["cfur_sd"].tolist()
        ax2.bar(paths, means, yerr=sds, color="steelblue", edgecolor="navy", capsize=4)
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_ylabel("CFUR")
    ax2.set_title("Utility Ratio by Path (Ground Truth, 95% CI)")
    plt.tight_layout()
    plt.savefig(out_dir / "hf_cfur_ground_truth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved hf_cfur_ground_truth.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
