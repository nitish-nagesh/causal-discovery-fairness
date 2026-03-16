"""
CFUR (Causal Fairness-Utility Ratio) for the AD dataset.

Adapts the methodology from CHASE2026-FAIRY (Plečko & Bareinboim, 2024)
for the Alzheimer's Disease dataset.

CFUR = fair_gain / stat_drop per causal path {DE, IE, SE}.

  fair_gain : reduction in TV when blocking a causal path
  stat_drop : increase in loss (1-AUC) when blocking the same path

The decomposition is done via Shapley-value–style path attribution over
all 8 subsets S ⊆ {DE, IE, SE}.

Pipeline (mirrors FAIRY experiments_hfcr.R):
  1. For each S, construct a "fair predictor" that blocks paths in S.
  2. Compute loss and TV for each subset.
  3. walk_over_paths() computes per-path stat_drop and fair_gain.
  4. CFUR = fair_gain / stat_drop.
"""

import numpy as np
import pandas as pd
from math import comb
from typing import Dict, List, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from cfa_fairness import extract_sfm_variables


def _fit_classifier(X, y, seed=42):
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=seed,
    )
    model.fit(X, y)
    return model


def _fit_regressor(X, y, seed=42):
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=seed,
    )
    model.fit(X, y)
    return model


def _bit_select(i: int) -> List[bool]:
    """Map 1-based index (1..8) to boolean mask [DE, IE, SE]."""
    val = i - 1
    return [bool((val >> j) & 1) for j in range(3)]


_PATH_NAMES = ["DE", "IE", "SE"]


def _compute_tv(yhat: np.ndarray, x_vals: np.ndarray,
                x0: int = 0, x1: int = 1) -> float:
    m1 = x_vals == x1
    m0 = x_vals == x0
    if m1.sum() == 0 or m0.sum() == 0:
        return 0.0
    return float(yhat[m1].mean() - yhat[m0].mean())


def _compute_loss(y_true: np.ndarray, y_pred: np.ndarray,
                  loss: str = "auc") -> float:
    if loss == "auc":
        try:
            return 1.0 - roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.5
    elif loss == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    elif loss == "bce":
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    return 0.0


def _bootstrap_loss(y_true, y_pred, loss, nboot, rng):
    n = len(y_true)
    losses = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n, size=n)
        losses[b] = _compute_loss(y_true[idx], y_pred[idx], loss)
    return losses


def _construct_fair_predictions(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    x_name: str,
    y_name: str,
    z_vars: List[str],
    w_vars: List[str],
    blocked: List[str],
    x0: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """
    Construct fair predictions by blocking specified causal paths.

    Blocking mechanism (intervention-based, follows FAIRY logic):
      Block DE : set X = x0 in outcome model (removes direct X->Y channel)
      Block IE : replace W with counterfactual W(X=x0) (removes X->W->Y)
      Block SE : marginalize over Z (breaks Z<->X confounding)
    """
    features = [x_name] + z_vars + w_vars
    outcome_model = _fit_classifier(
        train_df[features].values, train_df[y_name].values, seed=seed
    )

    med_models = {}
    if w_vars:
        med_features = [x_name] + z_vars
        for w in w_vars:
            med_models[w] = _fit_regressor(
                train_df[med_features].values, train_df[w].values, seed=seed
            )

    cf = eval_df[features].copy()

    if "DE" in blocked:
        cf[x_name] = x0

    if "IE" in blocked and w_vars:
        med_input = eval_df[[x_name] + z_vars].copy()
        med_input[x_name] = x0
        for w in w_vars:
            cf[w] = med_models[w].predict(med_input.values)

    if "SE" in blocked:
        rng = np.random.default_rng(seed + 999)
        for z in z_vars:
            cf[z] = rng.permutation(cf[z].values)

    yhat = outcome_model.predict_proba(cf.values)[:, 1]
    return yhat


def _accuracy_decomposition(
    df: pd.DataFrame,
    x_name: str,
    y_name: str,
    z_vars: List[str],
    w_vars: List[str],
    x0: int = 0,
    x1: int = 1,
    loss: str = "auc",
    nboot: int = 50,
    seed: int = 42,
) -> Dict[int, Dict]:
    """
    Accuracy decomposition for all 8 subsets of {DE, IE, SE}.
    Mirrors accuracy_decomposition() from FAIRY causal-acc-decomp.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    mid = n // 2
    train_df = df.iloc[perm[:mid]].reset_index(drop=True)
    eval_df = df.iloc[perm[mid:]].reset_index(drop=True)

    x_eval = eval_df[x_name].values
    y_eval = eval_df[y_name].values

    decomp = {}
    for i in range(1, 9):
        bits = _bit_select(i)
        blocked = [p for p, b in zip(_PATH_NAMES, bits) if b]

        yhat = _construct_fair_predictions(
            train_df, eval_df, x_name, y_name,
            z_vars, w_vars, blocked, x0, seed=seed + i,
        )

        boot_losses = _bootstrap_loss(y_eval, yhat, loss, nboot, rng)
        tv = _compute_tv(yhat, x_eval, x0, x1)

        decomp[i] = {
            "blocked": blocked,
            "yhat": yhat,
            "stat": boot_losses,
            "tv": tv,
        }

    return decomp, x_eval


def _walk_over_paths(decomp: Dict[int, Dict], x_eval: np.ndarray):
    """
    Shapley-style path attribution.
    For each ordered pair (i, j) where j adds exactly one blocked path:
      stat_drop = mean(stat_j - stat_i)
      fair_gain  = -(tv_j - tv_i)
    """
    results = []
    for i in range(1, 9):
        for j in range(2, 9):
            ibit = _bit_select(i)
            jbit = _bit_select(j)

            if any(ib > jb for ib, jb in zip(ibit, jbit)):
                continue
            if sum(jb - ib for ib, jb in zip(ibit, jbit)) != 1:
                continue

            wgh = 1.0 / (3 * comb(2, sum(ibit)))
            diff_idx = next(k for k in range(3) if jbit[k] > ibit[k])

            stat_drop = float(np.mean(decomp[j]["stat"] - decomp[i]["stat"]))
            fair_gain = -(decomp[j]["tv"] - decomp[i]["tv"])

            results.append({
                "sA": i, "sB": j,
                "path": _PATH_NAMES[diff_idx],
                "stat_drop": stat_drop,
                "fair_gain": fair_gain,
                "wgh": wgh,
            })

    return pd.DataFrame(results)


def compute_cfur(
    df: pd.DataFrame,
    adj: Optional[np.ndarray] = None,
    names: Optional[List[str]] = None,
    x_name: str = "sex",
    y_name: str = "moca_binary",
    x0: int = 0,
    x1: int = 1,
    loss: str = "auc",
    nreps: int = 5,
    nboot: int = 20,
    seed: int = 42,
    sfm_override: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Full CFUR computation for a given causal graph.

    If sfm_override is provided (dict with Z, W lists), use it instead of adj.
    Returns:
      sfm       : Standard Fairness Model variable categorisation
      summary   : per-path CFUR (mean ± sd)
      detailed  : per-path, per-rep aggregated results
      all_paths : raw Shapley-step table across all reps
    """
    if sfm_override is not None:
        sfm = {"X": x_name, "Y": y_name, "Z": sfm_override.get("Z", []),
               "W": sfm_override.get("W", [])}
        z_vars = [z for z in sfm["Z"] if z not in {"brain_mri", "slice_number"}]
        w_vars = sfm["W"]
    else:
        sfm = extract_sfm_variables(adj, names, x_name,
                                    y_name.replace("_binary", ""))
        z_vars = [z for z in sfm["Z"] if z not in {"brain_mri", "slice_number"}]
        w_vars = sfm["W"]

    for col in [x_name, y_name] + z_vars + w_vars:
        if col not in df.columns:
            return {"sfm": sfm, "error": f"Missing column: {col}",
                    "summary": pd.DataFrame(), "detailed": pd.DataFrame()}

    all_path_dfs = []
    for rep in range(nreps):
        decomp, x_eval = _accuracy_decomposition(
            df, x_name, y_name, z_vars, w_vars,
            x0, x1, loss, nboot, seed=seed + rep * 1000,
        )
        path_df = _walk_over_paths(decomp, x_eval)
        path_df["rep"] = rep
        all_path_dfs.append(path_df)

    all_df = pd.concat(all_path_dfs, ignore_index=True)

    agg = (
        all_df
        .groupby(["path", "rep"])
        .apply(lambda g: pd.Series({
            "stat_drop": (g["stat_drop"] * g["wgh"]).sum(),
            "fair_gain": (g["fair_gain"] * g["wgh"]).sum(),
        }), include_groups=False)
        .reset_index()
    )

    eps = 1e-10
    agg["cfur"] = agg["fair_gain"] / (agg["stat_drop"].abs() + eps)

    summary = (
        agg
        .groupby("path")
        .agg(
            stat_drop_mean=("stat_drop", "mean"),
            stat_drop_sd=("stat_drop", "std"),
            fair_gain_mean=("fair_gain", "mean"),
            fair_gain_sd=("fair_gain", "std"),
            cfur_mean=("cfur", "mean"),
            cfur_sd=("cfur", "std"),
        )
        .reset_index()
    )

    return {
        "sfm": sfm,
        "summary": summary,
        "detailed": agg,
        "all_paths": all_df,
    }


def cfur_comparison_table(
    cfur_results: Dict[str, Dict],
    perf_metrics: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Build a comparison table of CFUR across algorithms, optionally merged
    with performance metrics (F1, SHD).
    """
    rows = []
    for label, res in cfur_results.items():
        if "error" in res:
            rows.append({"algorithm": label, "note": res["error"]})
            continue
        summary = res["summary"]
        row = {"algorithm": label}
        for _, r in summary.iterrows():
            p = r["path"]
            row[f"fair_gain_{p}"] = r["fair_gain_mean"]
            row[f"stat_drop_{p}"] = r["stat_drop_mean"]
            row[f"CFUR_{p}"] = r["cfur_mean"]
            row[f"CFUR_{p}_sd"] = r["cfur_sd"]

        if perf_metrics and label in perf_metrics:
            pm = perf_metrics[label]
            row["F1"] = pm.get("F1", np.nan)
            row["SHD"] = pm.get("shd", np.nan)
            row["TPR"] = pm.get("tpr", np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def format_cfur_result(result: Dict[str, Any], label: str = "") -> str:
    lines = []
    if label:
        lines.append(f"=== CFUR: {label} ===")

    sfm = result.get("sfm", {})
    lines.append(f"  SFM: X={sfm.get('X')}, Y={sfm.get('Y')}")
    lines.append(f"       Z={sfm.get('Z', [])}")
    lines.append(f"       W={sfm.get('W', [])}")

    if "error" in result:
        lines.append(f"  ERROR: {result['error']}")
        return "\n".join(lines)

    summary = result["summary"]
    lines.append("")
    lines.append(f"  {'Path':<4}  {'fair_gain':>12}  {'stat_drop':>12}  {'CFUR':>12}")
    lines.append(f"  {'----':<4}  {'----------':>12}  {'----------':>12}  {'----':>12}")
    for _, r in summary.iterrows():
        fg = f"{r['fair_gain_mean']:+.4f}±{r['fair_gain_sd']:.4f}"
        sd = f"{r['stat_drop_mean']:+.4f}±{r['stat_drop_sd']:.4f}"
        cf = f"{r['cfur_mean']:+.4f}±{r['cfur_sd']:.4f}"
        lines.append(f"  {r['path']:<4}  {fg:>12}  {sd:>12}  {cf:>12}")

    lines.append("")
    lines.append("  CFUR = fair_gain / |stat_drop|")
    lines.append("  fair_gain > 0 : blocking this path reduces disparity")
    lines.append("  stat_drop > 0 : blocking this path increases loss")
    lines.append("  CFUR > 0      : net fairness gain per unit accuracy cost")

    return "\n".join(lines)
