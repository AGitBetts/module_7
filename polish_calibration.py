"""
04_polish_calibration.py
========================
Regenerates the calibration plot from outputs/predictions.csv with report-quality
formatting: quantile binning, 95% binomial confidence intervals per bin, sample
sizes, and clearer legend.

Run after 03_train_baseline.py. Writes outputs/calibration_polished.png.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta

PRED_PATH = "./outputs/predictions.csv"
OUT_PATH = "./outputs/calibration_polished.png"

N_BINS = 8                      # quantile bins; fewer = smoother, more = finer
ALPHA = 0.05                    # 95% CIs

# Colourblind-safe palette
COLOURS = {
    "logistic_regression": "#0072B2",
    "random_forest":       "#009E73",
    "gradient_boosting":   "#D55E00",
}

LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest":       "Random forest",
    "gradient_boosting":   "Gradient boosting",
}


def binomial_ci(successes, trials, alpha=ALPHA):
    """Clopper-Pearson exact binomial CI — robust at low n."""
    if trials == 0:
        return (np.nan, np.nan)
    lo = beta.ppf(alpha / 2, successes, trials - successes + 1) if successes > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, successes + 1, trials - successes) if successes < trials else 1.0
    return (lo, hi)


def calibration_quantile_bins(proba, y_true, n_bins=N_BINS):
    """
    Quantile-bin the predicted probabilities so each bin has ~equal sample size.
    Returns a DataFrame with one row per bin.
    """
    # Build n_bins quantile edges; unique() handles ties
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, n_bins + 1)))
    if len(edges) - 1 < 2:
        return pd.DataFrame()
    bin_idx = np.digitize(proba, edges[1:-1])
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        successes = int(y_true[mask].sum())
        observed = successes / n
        lo, hi = binomial_ci(successes, n)
        rows.append({
            "predicted_mean": float(proba[mask].mean()),
            "observed_rate":  observed,
            "ci_lo":          lo,
            "ci_hi":          hi,
            "n":              n,
        })
    return pd.DataFrame(rows)


def plot(df_preds):
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # ----- top panel: calibration curves -----
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1, label="Perfect calibration")

    y_true = df_preds["worsened_next_year"].values.astype(int)
    for model, label in LABELS.items():
        col = f"pred_{model}"
        if col not in df_preds.columns:
            continue
        proba = df_preds[col].values
        cal = calibration_quantile_bins(proba, y_true)
        if cal.empty:
            continue
        ax.errorbar(
            cal["predicted_mean"], cal["observed_rate"],
            yerr=[cal["observed_rate"] - cal["ci_lo"], cal["ci_hi"] - cal["observed_rate"]],
            fmt="o-", color=COLOURS[model], label=label,
            capsize=3, alpha=0.85, linewidth=1.8, markersize=6,
        )

    ax.set_ylabel("Observed deterioration rate", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_title(
        f"Calibration on 2023–2024 test set  (n = {len(df_preds):,})\n"
        f"Quantile-binned, 95% Clopper–Pearson intervals",
        fontsize=11, pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ----- bottom panel: predicted-probability histograms -----
    for model, label in LABELS.items():
        col = f"pred_{model}"
        if col in df_preds.columns:
            ax_hist.hist(
                df_preds[col], bins=20, alpha=0.4, color=COLOURS[model],
                label=label, density=False,
            )
    ax_hist.set_xlabel("Predicted probability of deterioration", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved polished calibration plot to {OUT_PATH}")


def main():
    df = pd.read_csv(PRED_PATH)
    plot(df)


if __name__ == "__main__":
    main()
