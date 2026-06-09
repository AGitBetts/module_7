# GMPP Deterioration Prediction

**A machine learning pipeline for predicting next-year delivery confidence deterioration in UK Government Major Projects.**

Submitted as part of Multiverse Advanced Data Fellowship Module 7: *Leveraging Machine Learning to Improve Efficiency*. Business sponsor: Turner & Townsend, Portfolio Assurance practice.

[![Open Government Licence](https://img.shields.io/badge/data-OGL%20v3.0-blue)](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
[![License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

## What this project does

Builds and evaluates a binary classifier that predicts whether a UK Government Major Projects Portfolio (GMPP) project will see its Delivery Confidence Assessment (DCA) worsen by the next annual reporting cycle. The intended deployment is portfolio assurance triage: rank projects by predicted deterioration probability so that scarce expert-review capacity is directed first to projects most likely to need intervention.

### Headline results

| Metric | Value | Interpretation |
|---|---|---|
| AUROC | 0.710 [0.576, 0.816] | Better than chance with 95% confidence |
| AUPRC | 0.200 | 2.2× lift over the 0.090 base rate |
| Recall@25% | 0.537 | Flagging top 25% of projects catches 54% of true deteriorations |

### What this project does *not* do

- It does **not** replace expert assurance review — it triages where expert attention should go first.
- It does **not** quote trustworthy probabilities — see the calibration discussion in the report; ranking is the intended use.
- It does **not** generalise to Infrastructure and Construction projects — the model is anti-predictive for that category (AUROC 0.43); deploy only on the remaining categories.

---

## Repository structure

```
.
├── README.md                       # this file
├── LICENSE                         # MIT
├── requirements.txt                # pinned dependencies
├── .gitignore                      # excludes data/, outputs/, .venv, etc.
│
├── 01_download_gmpp_data.py        # scrapes gov.uk for GMPP annual reports 2013-2024
├── 02_build_panel.py               # harmonises schemas, builds the modelling panel
├── 03_train_baseline.py            # trains 4 models, runs bias audit, generates calibration plot
├── 04_polish_calibration.py        # report-quality calibration figure
├── 05_additional_analyses.py       # kappa, mitigation comparison, hyperparameter grid
│
└── docs/
    ├── METHODOLOGY.md              # full methodology document (Milestone 1 evidence pack)
    ├── REPORT.md                   # final 4,000-word report
    └── REPORT_OUTLINE.md           # section-by-section rubric mapping
```

Folders generated at runtime (gitignored):
```
data/raw/{year}/                    # downloaded gov.uk files
data/panel.csv                      # harmonised modelling dataset
outputs/                            # metrics, plots, predictions
```

---

## Quick start

### Prerequisites

- Python 3.10 or later
- Approximately 100MB of disk space for raw downloads

### Setup

```bash
# clone
git clone https://github.com/AGitBetts/module_7.git
cd module_7

# virtual environment
python3 -m venv .venv
source .venv/bin/activate           # macOS/Linux
# .venv\Scripts\activate            # Windows

# install
pip install -r requirements.txt
```

### Run the pipeline

```bash
# 1. Download 195 GMPP files from gov.uk (15-30 minutes, depending on connection)
python download_gmpp_data.py

# 2. Harmonise into a modelling panel
python build_panel.py

# 3. Train and evaluate models
python train_baseline.py

# 4. Generate report-quality calibration figure
python polish_calibration.py

# 5. Run inter-rater reliability, mitigation comparison, hyperparameter grid
python additional_analyses.py
```

All artifacts land in `outputs/`. The main results are in `outputs/metrics.csv`, `outputs/subgroup_metrics.csv`, `outputs/mitigation_comparison.csv`, `outputs/hyperparameter_grid.csv`, and `outputs/calibration_polished.png`.

---

## Methodology in brief

| Element | Choice | Justification |
|---|---|---|
| Task type | Binary classification | Aligns with binary intervene/don't decision |
| Temporal scope | 2021–2024 only | Stable 3-tier DCA rating regime; pre-2021 used 5-tier system |
| Validation | Temporal hold-out (train ≤2022, test ≥2023) | Forecasting realism; random k-fold would leak future into training |
| Algorithms | Dummy, Logistic Regression, Random Forest, Gradient Boosting | Span baseline + linear + tree-based ensembles; deep nets rejected on sample size |
| Primary metric | Recall@25% with bootstrap CI | Maps directly to assurance capacity decision |
| Bias audit | Per-department, per-category, per-scale subgroup performance | Quantifies disparity, supports PSED compliance |
| Mitigation | Unbalanced LR with threshold tuning preferred over class-weight balancing | Empirical finding; class weighting inflates disparity without improving AUROC |


---

## Limitations

- **Sample size** — 254 training and 203 test observations. Restricted to post-2021 stable-regime data.
- **Covariate shift** — deterioration rate fell from 20.1% (training period) to 8.9% (test period). Model probabilities require recalibration for face-value interpretation.
- **Target subjectivity** — DCA is assigned by IPA or SRO with departmental input; known optimism bias (Flyvbjerg, 2014). Direct inter-rater reliability could not be computed because IPA and SRO ratings are mutually exclusive in the GMPP convention.
- **Category exclusion** — model is anti-predictive for Infrastructure and Construction projects; deploy only on the remaining categories.
- **Survivorship bias** — only HMT-approved projects appear in GMPP; generalisation to earlier planning-stage decisions not supported.

---

## Data licence and attribution

All training data is sourced from gov.uk GMPP transparency releases (2013–2024) and the NISTA Annual Report 2024–25, available under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). The raw data is not committed to this repository; rerun `01_download_gmpp_data.py` to reproduce it from source.

---

## Code licence

This repository's code is released under the [MIT License](LICENSE). You are free to reuse, modify, and redistribute with attribution.

---
