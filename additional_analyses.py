"""
05_additional_analyses.py  (v2 — robust)
========================================
Three Grade-A-closing analyses:

  1. Inter-rater reliability between IPA and SRO DCA ratings (Cohen's kappa)
  2. Mitigation effectiveness: subgroup disparity with vs without class-weight
  3. Hyperparameter grid search (sequential — no joblib parallelism)

Each section is wrapped in try/except so a failure in one does not kill the
others. Partial outputs are saved as soon as each section completes.

Reads: data/panel.csv
Writes:
    outputs/inter_rater_kappa.txt          — kappa results with CIs
    outputs/mitigation_comparison.csv      — balanced vs unbalanced disparity
    outputs/hyperparameter_grid.csv        — CV results across grid
"""

import re
import warnings
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

DATA = Path("./data/panel.csv")
OUT = Path("./outputs")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------
# Shared DCA harmonisation (mirrors 02_build_panel.py)
# ----------------------------------------------------------------------

DCA_VARIANTS = {
    "green":       {"green", "g"},
    "green_amber": {"green/amber", "amber/green", "g/a", "a/g"},
    "amber":       {"amber", "a"},
    "amber_red":   {"amber/red", "red/amber", "a/r", "r/a"},
    "red":         {"red", "r"},
}
DCA_3TIER = {
    "green": 2, "green_amber": 2,
    "amber": 1,
    "amber_red": 0, "red": 0,
}

REDACTION_RE = re.compile(r"section \d+|exempted|national security", re.IGNORECASE)


def normalise_dca_to_3tier(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if REDACTION_RE.search(s):
        return None
    for canonical, variants in DCA_VARIANTS.items():
        if s in variants:
            return DCA_3TIER[canonical]
    return None


NUMERIC_FEATURES = [
    "dca_ordinal", "fy_baseline_gbpm_log", "fy_forecast_gbpm",
    "whole_life_cost_gbpm_log", "years_since_start", "years_until_end",
]
CATEGORICAL_FEATURES = ["department", "category"]


def build_preprocessor(feat_num, feat_cat):
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), feat_num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), feat_cat),
    ])

# ----------------------------------------------------------------------
# 1. INTER-RATER RELIABILITY
# ----------------------------------------------------------------------

def bootstrap_kappa_ci(y1, y2, weights=None, n_boot=1000, alpha=0.05):
    n = len(y1)
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        try:
            vals.append(cohen_kappa_score(y1[idx], y2[idx], weights=weights))
        except (ValueError, IndexError):
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def inter_rater_reliability():
    panel = pd.read_csv(DATA, low_memory=False)
    panel = panel.replace([np.inf, -np.inf], np.nan)

    if "dca_ipa" not in panel.columns or "dca_sro" not in panel.columns:
        print("  IPA or SRO column missing — skipping kappa analysis")
        return

    both = panel.dropna(subset=["dca_ipa", "dca_sro"]).copy()
    print(f"  Rows with both IPA and SRO ratings (any value): {len(both)}")

    both["ipa_3"] = both["dca_ipa"].apply(normalise_dca_to_3tier)
    both["sro_3"] = both["dca_sro"].apply(normalise_dca_to_3tier)
    both = both.dropna(subset=["ipa_3", "sro_3"])
    print(f"  Rows with both parseable to 3-tier: {len(both)}")

    if len(both) < 20:
        msg = (f"  Only {len(both)} rows have both IPA and SRO parseable. "
               "Too few for reliable kappa. Noted as a finding in itself: "
               "the IPA and SRO columns are mostly mutually exclusive in the panel.")
        print(msg)
        (OUT / "inter_rater_kappa.txt").write_text(msg)
        return

    ipa = both["ipa_3"].values.astype(int)
    sro = both["sro_3"].values.astype(int)

    k_unweighted = cohen_kappa_score(ipa, sro)
    lo_u, hi_u = bootstrap_kappa_ci(ipa, sro)

    k_quadratic = cohen_kappa_score(ipa, sro, weights="quadratic")
    lo_q, hi_q = bootstrap_kappa_ci(ipa, sro, weights="quadratic")

    n_agree = (ipa == sro).sum()
    n_total = len(ipa)
    pct_agree = n_agree / n_total * 100

    cm = pd.crosstab(
        pd.Series(ipa, name="IPA"),
        pd.Series(sro, name="SRO"),
        margins=True,
    )
    label_map = {0: "Red", 1: "Amber", 2: "Green", "All": "All"}
    cm.index = [label_map.get(i, i) for i in cm.index]
    cm.columns = [label_map.get(c, c) for c in cm.columns]

    report = f"""Inter-Rater Reliability — IPA DCA vs SRO DCA
=================================================
Rows with both ratings parseable: {n_total}
Raw agreement:                    {n_agree}/{n_total} = {pct_agree:.1f}%

Cohen's kappa (unweighted):         {k_unweighted:.3f}  95% CI [{lo_u:.3f}, {hi_u:.3f}]
Cohen's kappa (quadratic-weighted): {k_quadratic:.3f}  95% CI [{lo_q:.3f}, {hi_q:.3f}]

Interpretation (Landis & Koch, 1977):
   <0.00       poor          0.21-0.40   fair        0.61-0.80   substantial
   0.00-0.20   slight        0.41-0.60   moderate    0.81-1.00   almost perfect

Confusion matrix (IPA rows vs SRO columns):
{cm.to_string()}
"""
    print(report)
    (OUT / "inter_rater_kappa.txt").write_text(report)
    print(f"  Saved: {OUT / 'inter_rater_kappa.txt'}")

# ----------------------------------------------------------------------
# 2. MITIGATION EFFECTIVENESS
# ----------------------------------------------------------------------

def subgroup_disparity(y_true, y_pred, group_labels, group_name):
    groups = pd.Series(group_labels).astype("object").fillna("missing")
    rows = []
    for g in groups.unique():
        mask = groups == g
        n = int(mask.sum())
        if n < 10:
            continue
        base_rate = float(y_true[mask].mean())
        flag_rate = float(y_pred[mask].mean())
        rows.append({
            "group_name": group_name,
            "group_value": g,
            "n": n,
            "base_rate": round(base_rate, 3),
            "flag_rate": round(flag_rate, 3),
            "disparity_ratio": round(flag_rate / base_rate, 2) if base_rate > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def mitigation_comparison():
    panel = pd.read_csv(DATA, low_memory=False)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel[panel["in_modelling_set"]].copy()

    feat_num = [c for c in NUMERIC_FEATURES if c in panel.columns]
    feat_cat = [c for c in CATEGORICAL_FEATURES if c in panel.columns]

    train_df = panel[panel["report_year"] <= 2022]
    test_df = panel[panel["report_year"] >= 2023]

    feat_num = [f for f in feat_num if train_df[f].notna().sum() > 0]
    feat_cat = [f for f in feat_cat if train_df[f].notna().sum() > 0]
    feature_cols = feat_num + feat_cat

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train, y_test = train_df["worsened_next_year"].astype(int), test_df["worsened_next_year"].astype(int)

    results = []
    for label, weight in [("unbalanced", None), ("balanced", "balanced")]:
        pipe = Pipeline([
            ("pre", build_preprocessor(feat_num, feat_cat)),
            ("clf", LogisticRegression(class_weight=weight, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = (pipe.predict_proba(X_test)[:, 1] > 0.5).astype(int)

        try:
            auroc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        except ValueError:
            auroc = np.nan
        print(f"  {label}: AUROC = {auroc:.3f}, flags = {y_pred.sum()}/{len(y_pred)}")

        df = subgroup_disparity(y_test.values, y_pred, test_df["category"], "category")
        df["mitigation"] = label
        df["auroc_overall"] = round(auroc, 3)
        results.append(df)

    combined = pd.concat(results, ignore_index=True)
    pivot = combined.pivot_table(
        index=["group_name", "group_value", "n", "base_rate"],
        columns="mitigation",
        values=["flag_rate", "disparity_ratio"],
    ).reset_index()
    pivot.columns = ["_".join(map(str, c)).strip("_") for c in pivot.columns.values]
    pivot.to_csv(OUT / "mitigation_comparison.csv", index=False)
    print(f"\n  Saved: {OUT / 'mitigation_comparison.csv'}")

    print("\n  Subgroup disparity: unbalanced vs balanced LR")
    print("  " + "-" * 70)
    for _, row in pivot.iterrows():
        gv = str(row["group_value"])[:35]
        unb = row.get("disparity_ratio_unbalanced", "N/A")
        bal = row.get("disparity_ratio_balanced", "N/A")
        print(f"  {gv:<37} unbalanced={unb}   balanced={bal}")

# ----------------------------------------------------------------------
# 3. HYPERPARAMETER GRID SEARCH (sequential, no joblib)
# ----------------------------------------------------------------------

def manual_cv_auroc(pipeline, X, y, cv):
    scores = []
    for train_idx, test_idx in cv.split(X):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        try:
            scores.append(roc_auc_score(y.iloc[test_idx], proba))
        except ValueError:
            continue
    if not scores:
        return np.nan, np.nan
    return float(np.mean(scores)), float(np.std(scores))


def hyperparameter_search():
    panel = pd.read_csv(DATA, low_memory=False)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel[panel["in_modelling_set"]].copy()

    feat_num = [c for c in NUMERIC_FEATURES if c in panel.columns]
    feat_cat = [c for c in CATEGORICAL_FEATURES if c in panel.columns]
    train_df = panel[panel["report_year"] <= 2022].sort_values("report_year").reset_index(drop=True)
    feat_num = [f for f in feat_num if train_df[f].notna().sum() > 0]
    feat_cat = [f for f in feat_cat if train_df[f].notna().sum() > 0]
    feature_cols = feat_num + feat_cat

    X_train = train_df[feature_cols]
    y_train = train_df["worsened_next_year"].astype(int)

    cv = TimeSeriesSplit(n_splits=3)
    pre = build_preprocessor(feat_num, feat_cat)
    results = []

    print("  Logistic regression:")
    for C in [0.01, 0.1, 1.0, 10.0]:
        pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(
            C=C, class_weight="balanced", max_iter=2000, random_state=42,
        ))])
        m, s = manual_cv_auroc(pipe, X_train, y_train, cv)
        print(f"    C={C}: AUROC = {m:.3f} ± {s:.3f}")
        results.append({"model": "logistic_regression", "param": f"C={C}",
                        "cv_auroc_mean": round(m, 3), "cv_auroc_std": round(s, 3)})

    print("\n  Random forest:")
    for n_est in [100, 300]:
        for max_d in [3, 5, None]:
            pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
                n_estimators=n_est, max_depth=max_d, class_weight="balanced",
                random_state=42, n_jobs=1,
            ))])
            m, s = manual_cv_auroc(pipe, X_train, y_train, cv)
            print(f"    n_estimators={n_est}, max_depth={max_d}: AUROC = {m:.3f} ± {s:.3f}")
            results.append({"model": "random_forest", "param": f"n_est={n_est}, max_depth={max_d}",
                            "cv_auroc_mean": round(m, 3), "cv_auroc_std": round(s, 3)})

    print("\n  Gradient boosting (sklearn HistGradientBoosting):")
    for n_iter in [100, 300]:
        for lr in [0.01, 0.05, 0.1]:
            pipe = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
                max_iter=n_iter, learning_rate=lr, random_state=42, class_weight="balanced",
            ))])
            m, s = manual_cv_auroc(pipe, X_train, y_train, cv)
            print(f"    max_iter={n_iter}, learning_rate={lr}: AUROC = {m:.3f} ± {s:.3f}")
            results.append({"model": "gradient_boosting", "param": f"max_iter={n_iter}, lr={lr}",
                            "cv_auroc_mean": round(m, 3), "cv_auroc_std": round(s, 3)})

    df = pd.DataFrame(results)
    df.to_csv(OUT / "hyperparameter_grid.csv", index=False)
    print(f"\n  Saved: {OUT / 'hyperparameter_grid.csv'}")

    print("\n  Best per model:")
    for model in df["model"].unique():
        best = df[df["model"] == model].sort_values("cv_auroc_mean", ascending=False).iloc[0]
        print(f"    {model}: AUROC {best['cv_auroc_mean']:.3f} at {best['param']}")

# ----------------------------------------------------------------------
# MAIN — each section wrapped so partial failures don't kill the others
# ----------------------------------------------------------------------

def run_section(name, fn):
    print("=" * 70)
    print(name)
    print("=" * 70)
    try:
        fn()
    except Exception as e:
        print(f"\n  ERROR in {name}: {type(e).__name__}: {e}")
        print("  Continuing to next section...")
        traceback.print_exc()
    print()


def main():
    run_section("1. INTER-RATER RELIABILITY (IPA DCA vs SRO DCA)", inter_rater_reliability)
    run_section("2. MITIGATION COMPARISON (unbalanced vs balanced LR)", mitigation_comparison)
    run_section("3. HYPERPARAMETER GRID SEARCH (sequential, no joblib)", hyperparameter_search)
    print("All sections attempted. Check outputs/ for files produced.")


if __name__ == "__main__":
    main()
