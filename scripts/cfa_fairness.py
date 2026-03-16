"""
Causal Fairness Analysis (CFA) — Python implementation.

Implements the decomposition from Plečko & Bareinboim (2024):
    TV = Ctf-DE - Ctf-IE - Ctf-SE

where:
    TV    = P(Y=1|X=x1) - P(Y=1|X=x0)               [total variation]
    Ctf-DE = E[Y_{x1,W(x0)}|X=x0] - E[Y|X=x0]       [direct effect]
    Ctf-IE = E[Y_{x1,W(x0)}|X=x0] - E[Y_{x1}|X=x0]  [indirect effect]
    Ctf-SE = E[Y_{x1}|X=x0] - E[Y|X=x1]              [spurious effect]

Uses imputation-based estimation with cross-fitting for valid inference.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold


def _fit_outcome_model(X_train, y_train):
    """Fit binary outcome model P(Y|features)."""
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def _fit_mediator_model(X_train, y_train):
    """Fit continuous mediator model E[W|features]."""
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def extract_sfm_variables(adj: np.ndarray, names: List[str],
                          x_name: str = "sex", y_name: str = "moca"):
    """
    Extract Standard Fairness Model variables from the DAG.

    X: protected attribute
    Z: confounders (not descendants of X, not mediators)
    W: mediators (on directed X→Y paths, caused by X)
    Y: outcome
    """
    import networkx as nx

    n = len(names)
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                G.add_edge(names[i], names[j])

    if x_name not in G or y_name not in G:
        return {"X": x_name, "Z": [], "W": [], "Y": y_name}

    descendants_of_x = nx.descendants(G, x_name)

    try:
        all_paths = list(nx.all_simple_paths(G, x_name, y_name))
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        all_paths = []

    mediators = set()
    for path in all_paths:
        for node in path[1:-1]:
            mediators.add(node)

    confounders = []
    for node in names:
        if node == x_name or node == y_name:
            continue
        if node in mediators:
            continue
        if node not in descendants_of_x:
            confounders.append(node)

    return {
        "X": x_name,
        "Z": sorted(confounders),
        "W": sorted(mediators),
        "Y": y_name,
        "all_paths": [[n for n in p] for p in all_paths],
    }


def cfa_decomposition(df: pd.DataFrame, adj: Optional[np.ndarray] = None,
                      names: Optional[List[str]] = None,
                      x_name: str = "sex", y_name: str = "moca_binary",
                      x0: int = 0, x1: int = 1,
                      n_folds: int = 5, n_bootstrap: int = 200,
                      random_state: int = 42,
                      sfm_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute CFA decomposition: TV = Ctf-DE - Ctf-IE - Ctf-SE.

    Uses imputation with K-fold cross-fitting for valid inference.
    Bootstrap CIs are computed over n_bootstrap resamples.

    If sfm_override is provided (dict with X, Y, Z, W), use it instead of
    extracting from adj. Use for HF when Z, W come from SFM mapping.
    """
    if sfm_override is not None:
        sfm = sfm_override
        Z_vars = list(sfm.get("Z", []))
        W_vars = list(sfm.get("W", []))
    else:
        sfm = extract_sfm_variables(adj, names, x_name,
                                    y_name.replace("_binary", ""))
        Z_vars = sfm["Z"]
        W_vars = sfm["W"]

    exclude_from_Z = {"brain_mri", "slice_number"}
    Z_vars = [z for z in Z_vars if z not in exclude_from_Z]

    outcome_features = [x_name] + Z_vars + W_vars
    mediator_features = [x_name] + Z_vars

    for col in outcome_features + [y_name]:
        if col not in df.columns:
            return _nan_result(sfm, f"Missing column: {col}")

    def _estimate_once(data):
        x0_mask = data[x_name] == x0
        x1_mask = data[x_name] == x1
        n0, n1 = x0_mask.sum(), x1_mask.sum()
        if n0 < 10 or n1 < 10:
            return None

        ey_x0 = data.loc[x0_mask, y_name].mean()
        ey_x1 = data.loc[x1_mask, y_name].mean()
        tv = ey_x1 - ey_x0

        if not W_vars:
            return {"TV": tv, "Ctf-DE": tv, "Ctf-IE": 0.0, "Ctf-SE": 0.0}

        X_out = data[outcome_features].values
        y_out = data[y_name].values

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        ey_x1_w_x0 = np.zeros(n0)
        ey_x1_w_x1_x0 = np.zeros(n0)

        x0_indices = np.where(x0_mask.values)[0]
        fold_preds_direct = {}
        fold_preds_indirect = {}

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_out)):
            out_model = _fit_outcome_model(X_out[train_idx], y_out[train_idx])

            X_med_train = data.iloc[train_idx][mediator_features].values
            for w_idx, w_var in enumerate(W_vars):
                w_train = data.iloc[train_idx][w_var].values
                med_model = _fit_mediator_model(X_med_train, w_train)

                test_x0_in_fold = np.intersect1d(test_idx, x0_indices)
                if len(test_x0_in_fold) == 0:
                    continue

                x0_data = data.iloc[test_x0_in_fold]

                cf_direct = x0_data[outcome_features].copy()
                cf_direct[x_name] = x1
                pred_direct = out_model.predict_proba(cf_direct.values)[:, 1]

                cf_med_input = x0_data[mediator_features].copy()
                cf_med_input[x_name] = x1
                w_counterfactual = med_model.predict(cf_med_input.values)

                cf_indirect = cf_direct.copy()
                cf_indirect[w_var] = w_counterfactual
                pred_indirect = out_model.predict_proba(cf_indirect.values)[:, 1]

                for i, global_idx in enumerate(test_x0_in_fold):
                    local_idx = np.searchsorted(x0_indices, global_idx)
                    if local_idx < n0 and x0_indices[local_idx] == global_idx:
                        fold_preds_direct[global_idx] = pred_direct[i]
                        fold_preds_indirect[global_idx] = pred_indirect[i]

        for i, idx in enumerate(x0_indices):
            ey_x1_w_x0[i] = fold_preds_direct.get(idx, np.nan)
            ey_x1_w_x1_x0[i] = fold_preds_indirect.get(idx, np.nan)

        valid = ~(np.isnan(ey_x1_w_x0) | np.isnan(ey_x1_w_x1_x0))
        if valid.sum() < 10:
            return None

        mean_ey_x1_w_x0 = ey_x1_w_x0[valid].mean()
        mean_ey_x1_w_x1_x0 = ey_x1_w_x1_x0[valid].mean()

        ctf_de = mean_ey_x1_w_x0 - ey_x0
        ctf_ie = mean_ey_x1_w_x0 - mean_ey_x1_w_x1_x0
        ctf_se = mean_ey_x1_w_x1_x0 - ey_x1

        return {"TV": tv, "Ctf-DE": ctf_de, "Ctf-IE": ctf_ie, "Ctf-SE": ctf_se}

    point = _estimate_once(df)
    if point is None:
        return _nan_result(sfm, "Estimation failed (too few samples per group)")

    rng = np.random.default_rng(random_state)
    boot_results = {"TV": [], "Ctf-DE": [], "Ctf-IE": [], "Ctf-SE": []}
    for b in range(n_bootstrap):
        boot_df = df.sample(n=len(df), replace=True, random_state=rng.integers(0, 2**31))
        boot_df = boot_df.reset_index(drop=True)
        res_b = _estimate_once(boot_df)
        if res_b is not None:
            for key in boot_results:
                boot_results[key].append(res_b[key])

    ci = {}
    se = {}
    for key in boot_results:
        arr = np.array(boot_results[key])
        if len(arr) > 10:
            ci[key] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
            se[key] = float(np.std(arr))
        else:
            ci[key] = (np.nan, np.nan)
            se[key] = np.nan

    check = point["Ctf-DE"] - point["Ctf-IE"] - point["Ctf-SE"]

    return {
        "sfm": sfm,
        "point": point,
        "se": se,
        "ci": ci,
        "n_boot": len(boot_results["TV"]),
        "decomposition_check": {
            "TV": point["TV"],
            "DE-IE-SE": check,
            "residual": abs(point["TV"] - check),
        },
    }


def _nan_result(sfm, error_msg):
    return {
        "sfm": sfm,
        "point": {"TV": np.nan, "Ctf-DE": np.nan, "Ctf-IE": np.nan, "Ctf-SE": np.nan},
        "se": {"TV": np.nan, "Ctf-DE": np.nan, "Ctf-IE": np.nan, "Ctf-SE": np.nan},
        "ci": {k: (np.nan, np.nan) for k in ["TV", "Ctf-DE", "Ctf-IE", "Ctf-SE"]},
        "error": error_msg,
    }


def format_cfa_result(result: Dict[str, Any], label: str = "") -> str:
    """Format CFA result for display."""
    lines = []
    if label:
        lines.append(f"=== {label} ===")

    if "error" in result:
        lines.append(f"  ERROR: {result['error']}")
        return "\n".join(lines)

    sfm = result.get("sfm", {})
    lines.append(f"  SFM: X={sfm.get('X')}, Y={sfm.get('Y')}")
    lines.append(f"       Z (confounders) = {sfm.get('Z', [])}")
    lines.append(f"       W (mediators)   = {sfm.get('W', [])}")

    paths = sfm.get("all_paths", [])
    if paths:
        lines.append(f"       Causal paths X→Y: {len(paths)}")
        for p in paths[:5]:
            lines.append(f"         {' → '.join(p)}")

    pt = result["point"]
    se_dict = result.get("se", {})
    ci_dict = result.get("ci", {})

    lines.append("")
    for key in ["TV", "Ctf-DE", "Ctf-IE", "Ctf-SE"]:
        val = pt.get(key, np.nan)
        se_val = se_dict.get(key, np.nan)
        ci_val = ci_dict.get(key, (np.nan, np.nan))
        sig = ""
        if not np.isnan(ci_val[0]) and not np.isnan(ci_val[1]):
            if ci_val[0] > 0 or ci_val[1] < 0:
                sig = " *"
        lines.append(f"  {key:8s} = {val:+.6f}  (SE={se_val:.4f}, "
                     f"95%CI=[{ci_val[0]:+.4f}, {ci_val[1]:+.4f}]){sig}")

    chk = result.get("decomposition_check", {})
    lines.append(f"\n  Decomposition check: TV={chk.get('TV', '?'):.6f}, "
                 f"DE-IE-SE={chk.get('DE-IE-SE', '?'):.6f}, "
                 f"residual={chk.get('residual', '?'):.6f}")

    return "\n".join(lines)


def fairness_utility_table(cfa_results: Dict[str, Dict],
                           perf_metrics: Dict[str, Dict],
                           cfur_results: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    """
    Build a combined fairness-utility table across algorithms.

    cfa_results:  {algorithm_label: cfa_decomposition result}
    perf_metrics: {algorithm_label: {F1, SHD, TPR, FPR, ...}}
    cfur_results: optional {algorithm_label: compute_cfur result}
    """
    rows = []
    for label in cfa_results:
        cfa = cfa_results[label]
        perf = perf_metrics.get(label, {})
        pt = cfa.get("point", {})

        tv = pt.get("TV", np.nan)
        de = pt.get("Ctf-DE", np.nan)
        ie = pt.get("Ctf-IE", np.nan)
        se_val = pt.get("Ctf-SE", np.nan)

        f1 = perf.get("F1", np.nan)
        shd = perf.get("shd", np.nan)
        tpr = perf.get("tpr", np.nan)

        unfair_total = abs(de) + abs(ie) if not (np.isnan(de) or np.isnan(ie)) else np.nan

        eps = 1e-6
        fu_ratio = f1 / (unfair_total + eps) if not (np.isnan(f1) or np.isnan(unfair_total)) else np.nan

        row = {
            "algorithm": label,
            "TV": tv,
            "Ctf-DE": de,
            "Ctf-IE": ie,
            "Ctf-SE": se_val,
            "|unfair|": unfair_total,
            "F1": f1,
            "SHD": shd,
            "TPR": tpr,
            "FU-ratio": fu_ratio,
        }

        if cfur_results and label in cfur_results:
            cfur = cfur_results[label]
            if "error" not in cfur:
                summary = cfur["summary"]
                for _, r in summary.iterrows():
                    row[f"CFUR_{r['path']}"] = r["cfur_mean"]

        rows.append(row)

    return pd.DataFrame(rows)
