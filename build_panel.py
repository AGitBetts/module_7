"""
02_build_panel.py  (v2 — encoding fallback + flexible column matching)
======================================================================
Reads raw per-department GMPP files, harmonises schemas across years and
departments, normalises project names, constructs the binary deterioration
target, and writes a clean modelling panel.

v2 changes:
  - CSV loader tries UTF-8 then falls back to Windows-1252 then Latin-1.
    Pre-2021 gov.uk publications use cp1252 (en-dashes, smart quotes).
  - Column matching strips embedded parenthetical definitions and newlines
    before comparing. From 2021 the gov.uk files put the column definition
    *inside* the column header itself.
  - 2-word-or-longer variants match as substrings, not just exact.
"""

import re
import logging
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("./data/raw")
OUT_DIR = Path("./data")
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=OUT_DIR / "harmonisation.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# DCA HARMONISATION
# ----------------------------------------------------------------------

DCA_VARIANTS = {
    "green":       {"green", "g"},
    "green_amber": {"green/amber", "amber/green", "g/a", "a/g", "green - amber", "amber - green"},
    "amber":       {"amber", "a"},
    "amber_red":   {"amber/red", "red/amber", "a/r", "r/a", "amber - red", "red - amber"},
    "red":         {"red", "r"},
    "exempt":      {"exempt", "exempted"},
}

DCA_ORDINAL = {
    "green": 4, "green_amber": 3, "amber": 2, "amber_red": 1, "red": 0,
}

DCA_3TIER = {
    "green": "Green", "green_amber": "Green",
    "amber": "Amber",
    "amber_red": "Red", "red": "Red",
}

REDACTION_PATTERNS = re.compile(
    r"section \d+|exempted|national security|commercial interest|future publication|formulation of government policy",
    re.IGNORECASE,
)


def normalise_dca(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if REDACTION_PATTERNS.search(s):
        return None
    for canonical, variants in DCA_VARIANTS.items():
        if s in variants:
            return canonical
    return None

# ----------------------------------------------------------------------
# COLUMN NAME HARMONISATION  (v2: tolerant matching)
# ----------------------------------------------------------------------

def normalise_colname(c):
    """Strip parenthetical definitions, newlines, punctuation; lower + collapse."""
    s = str(c)
    s = re.sub(r"\s+", " ", s)                  # newlines + whitespace -> single space
    s = re.sub(r"\([^)]*\)", "", s)             # remove parenthetical content
    s = re.sub(r"[^\w\s]", " ", s)              # strip punctuation
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


COLUMN_VARIANTS = {
    "project_name": [
        "project name", "gmpp project name", "project programme name",
        "project name as displayed in gmpp", "project title",
    ],
    "department": ["department", "dept", "organisation"],
    "category": [
        "annual report category", "category", "project category",
        "ipa category", "departmental annual report category",
    ],
    "dca_ipa": [
        # Post-2020 IPA-prefixed (exact)
        "ipa delivery confidence assessment",
        "ipa delivery confidence",
        "ipa rag rating",
        "ipa rag",
        # Pre-2021 generic — picked up when no IPA-specific column exists.
        # Specificity-wins matching ensures these don't override SRO columns.
        "delivery confidence assessment",
        "delivery confidence",
        "rag rating",
        "rag status",
        "dca",
    ],
    "dca_sro": [
        "sro delivery confidence assessment",
        "sro delivery confidence",
        "sro dca",
        "sro quarterly delivery confidence assessment",
        "sro rag rating",
        "sro rag",
    ],
    "start_date": ["start date", "project start date", "approved start date"],
    "end_date": ["end date", "project end date", "current end date", "approved end date"],
    "fy_baseline_gbpm": [
        "financial year baseline",
        "fy baseline",
        "departmental baseline",
        "annual budget baseline",
        "total baseline annual budget",
    ],
    "fy_forecast_gbpm": [
        "financial year forecast",
        "fy forecast",
        "forecast cost",
    ],
    "fy_variance_pct": [
        "financial year variance", "fy variance", "variance percentage",
    ],
    "whole_life_cost_gbpm": [
        "whole life cost",
        "whole life costs",
        "total baseline whole life cost",
        "total baseline whole life costs",
        "total project cost",
    ],
    "benefits_gbpm": [
        "monetised benefits",
        "total baseline benefits",
        "total benefits",
    ],
    "evaluation_plan": [
        "does the project have an evaluation plan",
        "evaluation plan",
        "has evaluation plan",
    ],
    "gmpp_id": ["gmpp id", "gmpp id number", "id", "project id"],
}


def harmonise_columns(df, source_label):
    """
    Match each raw column to a canonical name.

    Strategy — specificity wins:
      For every (column, variant) pair, compute a match score:
        - exact match on normalised forms: score = len(variant) * 10
        - substring match for 2+ word variants:    score = len(variant)
        - otherwise:                                no match
      Each column is assigned to the canonical whose variant scored highest.
      This lets a generic variant like "delivery confidence assessment" exist
      alongside specific ones like "sro delivery confidence assessment" without
      collision — the specific one always wins when both apply.
    """
    cols_normed = {c: normalise_colname(c) for c in df.columns}

    # Pre-normalise all variants
    variants_normed = {
        canonical: [normalise_colname(v) for v in variants if normalise_colname(v)]
        for canonical, variants in COLUMN_VARIANTS.items()
    }

    rename_map = {}
    for original, normed in cols_normed.items():
        if not normed:
            continue
        best_canonical = None
        best_score = 0
        for canonical, v_list in variants_normed.items():
            for v in v_list:
                if normed == v:
                    score = len(v) * 10                       # exact: big boost
                elif len(v.split()) >= 2 and v in normed:
                    score = len(v)                            # substring: by length
                else:
                    continue
                if score > best_score:
                    best_score = score
                    best_canonical = canonical
        if best_canonical:
            rename_map[original] = best_canonical

    unmapped = [c for c in df.columns if c not in rename_map]
    if unmapped:
        # Log a short version to keep the log readable
        short = [c[:60].replace("\n", " ") + ("..." if len(c) > 60 else "") for c in unmapped]
        log.info(f"{source_label}: unmapped columns: {short}")
    df = df.rename(columns=rename_map)

    # Defensive: if two original columns mapped to the same canonical name,
    # keep the first occurrence and warn. Prevents downstream `df[col]` returning
    # a DataFrame instead of a Series.
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        log.info(f"{source_label}: duplicate canonical columns after rename, keeping first: {dups}")
        df = df.loc[:, ~df.columns.duplicated()]

    # Defragment to silence pandas PerformanceWarning on subsequent .insert() calls
    return df.copy()

# ----------------------------------------------------------------------
# PROJECT NAME NORMALISATION
# ----------------------------------------------------------------------

BOILERPLATE = re.compile(r"\b(the|of|for|and|&)\b", re.IGNORECASE)

def normalise_project_name(name):
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = BOILERPLATE.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------------------------------------------------------
# DATE / NUMERIC COERCION
# ----------------------------------------------------------------------

def parse_date(value):
    if pd.isna(value):
        return pd.NaT
    s = str(value)
    if "Mid:" in s:
        m = re.search(r"Mid:\s*(\d{1,2}/\d{1,2}/\d{4})", s)
        if m:
            return pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")
    # format='mixed' silences the dayfirst warning and handles ISO + dd/mm/yyyy
    return pd.to_datetime(s, dayfirst=True, format="mixed", errors="coerce")


def parse_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if REDACTION_PATTERNS.search(s):
        return np.nan
    if "Mid:" in s:
        m = re.search(r"Mid:\s*([\d,]+\.?\d*)", s)
        if m:
            s = m.group(1)
    s = re.sub(r"[£,\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan

# ----------------------------------------------------------------------
# LOAD AND PROCESS  (v2: encoding fallback)
# ----------------------------------------------------------------------

def load_raw_file(fp):
    """CSV loader with encoding fallback. Pre-2021 gov.uk files use cp1252."""
    try:
        if fp.suffix == ".csv":
            last_err = None
            for encoding in ("utf-8", "cp1252", "latin-1"):
                try:
                    return pd.read_csv(fp, encoding=encoding, on_bad_lines="warn")
                except UnicodeDecodeError as e:
                    last_err = e
                    continue
            log.warning(f"all encodings failed for {fp}: {last_err}")
            return None
        elif fp.suffix in {".xlsx", ".xls"}:
            return pd.read_excel(fp)
        elif fp.suffix == ".ods":
            return pd.read_excel(fp, engine="odf")
        else:
            return None
    except Exception as e:
        log.warning(f"failed to load {fp}: {e}")
        return None


def process_one_file(fp, report_year, source_dept):
    df = load_raw_file(fp)
    if df is None or df.empty:
        return pd.DataFrame()
    df = harmonise_columns(df, source_label=fp.name)

    # Coerce types
    for c in ["fy_baseline_gbpm", "fy_forecast_gbpm", "fy_variance_pct",
              "whole_life_cost_gbpm", "benefits_gbpm"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_numeric)
    for c in ["start_date", "end_date"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_date)

    # Harmonise DCA: prefer IPA where present, fall back to SRO
    if "dca_ipa" in df.columns:
        df["dca_canonical"] = df["dca_ipa"].apply(normalise_dca)
    else:
        df["dca_canonical"] = None
    if "dca_sro" in df.columns:
        sro = df["dca_sro"].apply(normalise_dca)
        df["dca_canonical"] = df["dca_canonical"].fillna(sro)

    df["dca_ordinal"] = df["dca_canonical"].map(DCA_ORDINAL)
    df["dca_3tier"] = df["dca_canonical"].map(DCA_3TIER)

    if "project_name" in df.columns:
        df["project_key"] = df["project_name"].apply(normalise_project_name)
    else:
        df["project_key"] = ""

    df["report_year"] = report_year
    if "department" not in df.columns and source_dept:
        df["department"] = source_dept.upper()

    return df


def build_panel():
    frames = []
    files_processed = 0
    files_failed = 0
    for year_dir in sorted(RAW_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            report_year = int(year_dir.name)
        except ValueError:
            log.warning(f"Skipping non-year directory: {year_dir.name}")
            continue
        for fp in year_dir.glob("*"):
            dept = fp.stem.split("_")[0] if "_" in fp.stem else None
            frame = process_one_file(fp, report_year, dept)
            if not frame.empty:
                frames.append(frame)
                files_processed += 1
            else:
                files_failed += 1

    panel = pd.concat(frames, ignore_index=True)
    log.info(f"Files processed: {files_processed}, failed/empty: {files_failed}")
    log.info(f"Combined panel: {len(panel):,} rows")
    log.info(f"Rows with parseable DCA: {panel['dca_canonical'].notna().sum():,}")

    # Target: did this project worsen by next year?
    panel = panel.sort_values(["project_key", "department", "report_year"])
    panel["next_year_ordinal"] = (
        panel.groupby(["project_key", "department"])["dca_ordinal"].shift(-1)
    )
    panel["next_year_report_year"] = (
        panel.groupby(["project_key", "department"])["report_year"].shift(-1)
    )

    panel["worsened_next_year"] = np.where(
        panel["next_year_ordinal"].notna() & panel["dca_ordinal"].notna(),
        (panel["next_year_ordinal"] < panel["dca_ordinal"]).astype(float),
        np.nan,
    )

    modelling_mask = (
        panel["worsened_next_year"].notna()
        & (panel["dca_canonical"] != "red")
        & panel["dca_ordinal"].notna()
    )
    panel["in_modelling_set"] = modelling_mask

    log.info(f"Rows in modelling set: {modelling_mask.sum():,}")
    if modelling_mask.sum() > 0:
        log.info(f"Deterioration rate: {panel.loc[modelling_mask, 'worsened_next_year'].mean():.3f}")

    return panel


def derive_features(panel):
    p = panel.copy()
    for c in ["fy_baseline_gbpm", "whole_life_cost_gbpm", "benefits_gbpm"]:
        if c in p.columns:
            p[f"{c}_log"] = np.log1p(p[c].clip(lower=0))
    if "start_date" in p.columns:
        p["years_since_start"] = (
            (pd.to_datetime(p["report_year"].astype(str) + "-03-31") - p["start_date"])
            .dt.days / 365.25
        )
    if "end_date" in p.columns:
        p["years_until_end"] = (
            (p["end_date"] - pd.to_datetime(p["report_year"].astype(str) + "-03-31"))
            .dt.days / 365.25
        )
    for c in ["sro_name", "senior_responsible_owner_(sro)_name", "senior responsible owner (sro) name"]:
        if c in p.columns:
            p = p.drop(columns=[c])
    return p


def main():
    panel = build_panel()
    panel = derive_features(panel)
    panel.to_csv(OUT_DIR / "panel.csv", index=False)
    log.info(f"Wrote {len(panel):,} rows to data/panel.csv")

    feature_dict = pd.DataFrame({
        "column": panel.columns,
        "dtype": panel.dtypes.astype(str).values,
        "n_non_null": panel.notna().sum().values,
        "pct_non_null": (panel.notna().mean() * 100).round(1).values,
    })
    feature_dict.to_csv(OUT_DIR / "feature_dict.csv", index=False)

    print(f"Panel: {len(panel):,} rows, {len(panel.columns)} columns")
    print(f"Rows with parseable DCA: {panel['dca_canonical'].notna().sum():,}")
    print(f"Modelling set: {panel['in_modelling_set'].sum():,} rows")
    if panel['in_modelling_set'].sum() > 0:
        print(f"Deterioration base rate: "
              f"{panel.loc[panel['in_modelling_set'], 'worsened_next_year'].mean():.3f}")


if __name__ == "__main__":
    main()
