"""
Individual path effects: Ctf-SE by confounder (Z), Ctf-IE by mediator (W).

Adapted from FAIRY faircause_hfcr_clinical_necessity.R and individual_effects_tables.R.
Decomposes the composite Ctf-SE into per-confounder contributions and Ctf-IE into
per-mediator contributions, supporting clinical necessity and intervention prioritization.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from cfa_fairness import extract_sfm_variables


def _fit_classifier(X, y, **kwargs):
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42, **kwargs
    )
    model.fit(X, y)
    return model


def _fit_regressor(X, y, **kwargs):
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42, **kwargs
    )
    model.fit(X, y)
    return model


def _est_quant(
    df: pd.DataFrame,
    z_ord: List[str],
    w_ord: List[str],
    x_name: str,
    y_name: str,
    xz_config: List[int],
    xw_config: List[int],
    x_y: int,
    n_samp: int = 2000,
    seed: int = 42,
) -> float:
    """
    Estimate E[Y] under sequential interventions on Z and W.

    xz_config[i] = 0 or 1: sample Z_i from group x0 (0) or x1 (1)
    xw_config[i] = 0 or 1: sample W_i from group x0 (0) or x1 (1)
    x_y: value of X for the outcome model (0 or 1)
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    x0, x1 = 0, 1

    int_dat = pd.DataFrame(index=range(n_samp))

    for i, zi in enumerate(z_ord):
        x_cond = x1 if xz_config[i] == 1 else x0
        mask = df[x_name] == x_cond
        if mask.sum() < 5:
            return np.nan
        idx = rng.choice(np.where(mask)[0], size=n_samp, replace=True)
        int_dat[zi] = df[zi].values[idx]

    int_dat[x_name] = x_y

    for i, wi in enumerate(w_ord):
        x_cond = x1 if xw_config[i] == 1 else x0
        features = [x_name] + z_ord
        X_train = df[features].values
        y_train = df[wi].values
        is_binary = df[wi].nunique() <= 2
        if is_binary:
            model = _fit_classifier(X_train, y_train)
            int_dat[wi] = model.predict(int_dat[features].values)
        else:
            model = _fit_regressor(X_train, y_train)
            int_dat[wi] = model.predict(int_dat[features].values)

    features_y = [x_name] + z_ord + w_ord
    X_train = df[features_y].values
    y_train = df[y_name].values
    model = _fit_classifier(X_train, y_train)
    prob = model.predict_proba(int_dat[features_y].values)[:, 1]
    return float(prob.mean())


def individual_effects_decomposition(
    df: pd.DataFrame,
    adj: Optional[np.ndarray] = None,
    names: Optional[List[str]] = None,
    x_name: str = "sex",
    y_name: str = "moca_binary",
    x0: int = 0,
    x1: int = 1,
    n_rep: int = 20,
    n_samp: int = 2000,
    seed: int = 42,
    sfm_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Decompose Ctf-SE by confounder and Ctf-IE by mediator.

    If sfm_override is provided (dict with Z, W), use it instead of adj.
    Returns:
        sfm: Standard Fairness Model
        ctf_se_by_z: {z_var: contribution to Ctf-SE}
        ctf_ie_by_w: {w_var: contribution to Ctf-IE}
        ctf_de: composite direct effect (no decomposition)
        composite: {Ctf-SE, Ctf-IE, Ctf-DE} from CFA for validation
    """
    if sfm_override is not None:
        sfm = sfm_override
        z_vars = list(sfm.get("Z", []))
        w_vars = list(sfm.get("W", []))
    else:
        sfm = extract_sfm_variables(adj, names, x_name, y_name.replace("_binary", ""))
        z_vars = [z for z in sfm["Z"] if z not in {"brain_mri", "slice_number"}]
        w_vars = sfm["W"]

    for col in [x_name, y_name] + z_vars + w_vars:
        if col not in df.columns:
            return {"error": f"Missing column: {col}"}

    from cfa_fairness import cfa_decomposition
    cfa = cfa_decomposition(df, adj=adj, names=names, x_name=x_name, y_name=y_name,
                            n_bootstrap=30, random_state=seed, sfm_override=sfm_override)
    composite = cfa.get("point", {})
    ctf_de = composite.get("Ctf-DE", np.nan)
    ctf_ie_composite = composite.get("Ctf-IE", np.nan)
    ctf_se_composite = composite.get("Ctf-SE", np.nan)

    ctf_se_by_z = {}
    if z_vars:
        e_se = np.zeros(len(z_vars) + 1)
        for i in range(len(z_vars) + 1):
            xz_config = [0] * i + [1] * (len(z_vars) - i)
            xw_config = [1] * len(w_vars)
            vals = []
            for r in range(n_rep):
                v = _est_quant(df, z_vars, w_vars, x_name, y_name,
                              xz_config, xw_config, x1, n_samp, seed + r * 1000)
                if not np.isnan(v):
                    vals.append(v)
            e_se[i] = np.mean(vals) if vals else np.nan

        se_contrib = -np.diff(e_se)
        for i, z in enumerate(z_vars):
            ctf_se_by_z[z] = float(se_contrib[i]) if not np.isnan(se_contrib[i]) else np.nan

    ctf_ie_by_w = {}
    if w_vars:
        if len(w_vars) == 1:
            ctf_ie_by_w[w_vars[0]] = ctf_ie_composite
        else:
            e_ie = np.zeros(len(w_vars) + 1)
            for i in range(len(w_vars) + 1):
                xz_config = [1] * len(z_vars)
                xw_config = [0] * i + [1] * (len(w_vars) - i)
                vals = []
                for r in range(n_rep):
                    v = _est_quant(df, z_vars, w_vars, x_name, y_name,
                                  xz_config, xw_config, x1, n_samp, seed + r * 1000 + 5000)
                    if not np.isnan(v):
                        vals.append(v)
                e_ie[i] = np.mean(vals) if vals else np.nan

            ie_contrib = np.diff(e_ie)
            for i, w in enumerate(w_vars):
                ctf_ie_by_w[w] = float(ie_contrib[i]) if not np.isnan(ie_contrib[i]) else np.nan

    return {
        "sfm": sfm,
        "ctf_se_by_z": ctf_se_by_z,
        "ctf_ie_by_w": ctf_ie_by_w,
        "ctf_de": ctf_de,
        "composite": composite,
    }


def individual_effects_table(result: Dict[str, Any]) -> pd.DataFrame:
    """Build a table of individual effects for display."""
    rows = []
    for z, contrib in result.get("ctf_se_by_z", {}).items():
        rows.append({"Effect": "Ctf-SE", "Variable": z, "Contribution": contrib})
    for w, contrib in result.get("ctf_ie_by_w", {}).items():
        rows.append({"Effect": "Ctf-IE", "Variable": w, "Contribution": contrib})
    if "ctf_de" in result and not np.isnan(result["ctf_de"]):
        rows.append({"Effect": "Ctf-DE", "Variable": "(direct)", "Contribution": result["ctf_de"]})
    return pd.DataFrame(rows)


def format_individual_effects(result: Dict[str, Any], label: str = "") -> str:
    lines = []
    if label:
        lines.append(f"=== Individual Effects: {label} ===")

    if "error" in result:
        lines.append(f"  ERROR: {result['error']}")
        return "\n".join(lines)

    sfm = result.get("sfm", {})
    lines.append(f"  Z (confounders): {sfm.get('Z', [])}")
    lines.append(f"  W (mediators):   {sfm.get('W', [])}")
    lines.append("")

    lines.append("  Ctf-SE by confounder (contribution to spurious effect):")
    for z, c in result.get("ctf_se_by_z", {}).items():
        lines.append(f"    {z}: {c:+.4f} ({100*c:.2f}%)")

    lines.append("")
    lines.append("  Ctf-IE by mediator (contribution to indirect effect):")
    for w, c in result.get("ctf_ie_by_w", {}).items():
        lines.append(f"    {w}: {c:+.4f} ({100*c:.2f}%)")

    lines.append("")
    lines.append(f"  Ctf-DE (direct): {result.get('ctf_de', np.nan):+.4f}")

    comp = result.get("composite", {})
    lines.append("")
    lines.append("  Composite (validation):")
    lines.append(f"    Ctf-SE = {comp.get('Ctf-SE', np.nan):+.4f}")
    lines.append(f"    Ctf-IE = {comp.get('Ctf-IE', np.nan):+.4f}")
    lines.append(f"    Ctf-DE = {comp.get('Ctf-DE', np.nan):+.4f}")

    return "\n".join(lines)
