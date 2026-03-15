# ---------------------------------------------
# 📊 Interactive EDA Dashboard (Modular)
# Subject 
# ---------------------------------------------

import os
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import gaussian_kde
from pandas.api.types import is_numeric_dtype, is_string_dtype

# ----------------------------
# Global plotting preferences
# ----------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# ----------------------------
# Constants & Theme
# ----------------------------
MAX_ROWS_PAIRPLOT = 2000
MAX_COLS_PAIRPLOT = 6


# ====================================
# 0) Page Config & Helper Functions
# ====================================

def setup_page() -> None:
    # Configure Streamlit page and header.
    st.set_page_config(page_title="EDA Dashboard", layout="wide")
    st.markdown("""### 📊 Interactive EDA Dashboard""", unsafe_allow_html=True)
    st.markdown("---")
    
# -----------------------------------------------------------------------------------

def get_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # Return dataframe containing only numeric columns.
    return df.select_dtypes(include="number")
    
# -----------------------------------------------------------------------------------

def list_csv_files(dirpath: str = ".") -> List[str]:
    # List .csv files in a directory.
    return [f for f in os.listdir(dirpath) if f.lower().endswith(".csv")]
    
# -----------------------------------------------------------------------------------

@st.cache_data
def load_data(file_name: str) -> pd.DataFrame:
    # Fix nullable integer types (Int64) for Arrow compatibility
    # Convert nullable integers to float64 (Because float can handle NaN)
    df = pd.read_csv(file_name)
    for col in df.select_dtypes("Int64").columns:
        df[col] = df[col].astype("float64")
    return df
    
# -----------------------------------------------------------------------------------

def pretty_index_df(df: pd.DataFrame, index_name: str = "R.No.") -> pd.DataFrame:
    # Display helper: 1-based (not 0) index with a custom left header.
    tmp = df.reset_index(drop=True).copy()
    tmp.index = tmp.index + 1
    tmp.index.name = index_name
    return tmp
    
# -----------------------------------------------------------------------------------

def data_quality_score(df: pd.DataFrame) -> Tuple[float, float, float]:

    # -------------------------------------------------
    # Compute a simple data quality score (0-100) using missing & duplicate penalties.
    # * 50% weight for missing ratio
    # * 50% weight for duplicate ratio
    # -------------------------------------------------
    missing_ratio = df.isna().sum().sum() / df.size  # fraction of total cells
    duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0.0
    penalty = (missing_ratio * 50) + (duplicate_ratio * 50)
    score = max(0, round(100 - penalty, 2))
    return score, round(missing_ratio * 100, 2), round(duplicate_ratio * 100, 2)

# -----------------------------------------------------------------------

def count_missing_cells(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())

# -----------------------------------------------------------------------

def count_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())

# -----------------------------------------------------------------------


# ================================
# 1) Sidebar Controls
# ================================

def sidebar_controls() -> Tuple[str, Optional[str], int]:
    
    # -------------------------------------------------------------------------
    # Returns: (selected_file, selected_numeric_column_from_RAW_or_None, bins)
    #
    # Behavior:
    # - Remembers last chosen CSV across reruns using st.session_state['selected_file_preselect'].
    # - If that file exists in the refreshed list, it remains selected.
    # -------------------------------------------------------------------------

    st.sidebar.header("Controls")

    # List CSVs in current directory (absolute paths for stable matching)
    entries = [f for f in list_csv_files(".")]
    abs_paths = [os.path.abspath(p) for p in entries]

    if not abs_paths:
        st.sidebar.error("No CSV files found in the current directory.")
        st.stop()

    # Build display labels (just the base names) but keep absolute paths as the true values
    labels = [os.path.basename(p) for p in abs_paths]

    # Determine preselection index from session (if present and still exists)
    preselect = st.session_state.get("selected_file_preselect")
    if preselect and os.path.abspath(preselect) in abs_paths:
        idx = abs_paths.index(os.path.abspath(preselect))
    else:
        idx = 0  # default to first if no previous or no longer exists

    # Show selectbox; store absolute path as the returned value
    selected_label = st.sidebar.selectbox(
        "Select Dataset",
        options=labels,
        index=idx,
        key="select_dataset_label",  # UI key only
    )
    
    # Map back to absolute path
    selected_file = abs_paths[labels.index(selected_label)]

    # Persist current choice for next rerun
    st.session_state["selected_file_preselect"] = selected_file

    # Load raw and build the numeric column selector from this selected file
    df = load_data(selected_file)
    numeric_columns = list(df.select_dtypes(include="number").columns)
    selected_column = None
    if numeric_columns:
        selected_column = st.sidebar.selectbox("Select Numeric Column", numeric_columns, key="select_numeric_column")

    bins = st.sidebar.slider("Number of Bins", min_value=5, max_value=50, value=20)
    return selected_file, selected_column, bins


# -----------------------------------------------------------------------


# ================================
# 2) Tabs
# ================================

# ------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------------

def render_overview(df: pd.DataFrame) -> None:
    
    # Overview tab: ONLY the Dataset Overview / Summary (KPIs + Quality Score)
    
    st.write("### Dataset Overview")

    # --- Quick metrics ---
    rows, cols = df.shape
    num_cols = len(df.select_dtypes(include="number").columns)
    missing_cells = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Numeric Columns", num_cols)
    c4.metric("Missing Values (cells)", missing_cells)
    c5.metric("Duplicate Rows", dup_rows)

    
    # --- First 5 Rows ---
    st.markdown("---")
    st.subheader("First 5 Rows")
    st.dataframe(pretty_index_df(df.head(), "R.No."))
    st.markdown("---")    


# -------------------------------------------------
# TAB 2 — DISTRIBUTION (Histogram)
# -------------------------------------------------

def render_distribution(df: pd.DataFrame, selected_column: Optional[str], bins: int) -> None:

    if selected_column is None or selected_column not in df.columns or not is_numeric_dtype(df[selected_column]):
        # Graceful fallback: pick the first numeric column if the sidebar choice is missing or invalid
        numeric_cols = list(get_numeric_df(df).columns)
        if not numeric_cols:
            st.warning("No numeric columns found in the selected file.")
            return
        selected_column = numeric_cols[0]

    st.subheader(f"Distribution of {selected_column}")

    # Toggle KDE density curve
    show_density = st.checkbox("Show Density Curve", value=True)

    # Remove NaNs (plots can't handle NaN)
    data = df[selected_column].dropna()

    # Plot Histogram with Specified No. of Bins
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, density=show_density, edgecolor="black")

    # Plot KDE density curve
    if show_density and len(data) > 1:
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_vals, kde(x_vals), color="darkorange", linewidth=2)

    ax.set_xlabel(selected_column, fontsize=10, color="darkblue")
    ax.set_ylabel("Density" if show_density else "Frequency", fontsize=10, color="darkblue")
    ax.set_title(f"Histogram of Column = {selected_column}", fontsize=12, color="purple")
    st.pyplot(fig)

    st.markdown("---")
    # Summary stats
    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.markdown("---")


# --------------------------------------------------------------
# TAB 3 — DATA QUALITY
# --------------------------------------------------------------

def render_data_quality(df: pd.DataFrame) -> None:
    st.subheader("Data Quality Analysis")

    indent = "\u00A0"  # non-breaking space

    # --- Quick health computations ---
    missing_cells = count_missing_cells(df)
    duplicate_rows = count_duplicate_rows(df)

    # --- Top metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Columns", len(get_numeric_df(df).columns))
    col4.metric("Missing Values (cells)", missing_cells)
    col5.metric("Duplicate Rows", duplicate_rows)

    # --- Data Quality Score banner ---
    score, miss_pct, dup_pct = data_quality_score(df)
    quality_str = (
        f"**Missing ratio** = {miss_pct}%{indent*10} **Duplicate ratio** = {dup_pct}%\n\n"
        f"**QUALITY SCORE** (QS) = **{score}%**{indent*10} "
        f"i.e. QS = 100 - (Missing% × 0.5 + Duplicate% × 0.5)"
    )
    st.info(quality_str)


    st.markdown("---")
    st.subheader("Column Summary (compact)")

    n_rows = max(len(df), 1)
    col_summary_long = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique_values": df.nunique(dropna=True),
        "missing_values": df.isnull().sum(),
        "missing %": (df.isnull().sum() / n_rows * 100).round(2),
    }).sort_values("missing_values", ascending=False)

    # Present transposed so dataset columns are horizontal
    st.dataframe(col_summary_long.T)

    st.markdown("---")


# -------------------------------------------------
# TAB 4 — CORRELATION HEATMAP
# -------------------------------------------------

def render_correlation(df: pd.DataFrame) -> None:
    st.subheader("Pearson Correlation Heatmap")

    numeric_df = get_numeric_df(df)
    numeric_cols = list(numeric_df.columns)  # use a list for stable indexing

    # Need at least 2 numeric columns to compute correlation
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns to compute correlation.")
        return

    # Toggle to show/hide annotations on the heatmap
    show_ann = st.checkbox("Show numeric annotations", value=True)

    # Compute Pearson correlation once (reused for r in explorer)
    corr_matrix = numeric_df.corr(numeric_only=True)

    # Mask upper triangle above the diagonal to avoid duplicate info
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Heatmap (fixed annotation font size = 8)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr_matrix, mask=mask, annot=show_ann, annot_kws={"size": 8},
        cmap="coolwarm", fmt=".2f", linewidths=0.5, center=0, square=True, ax=ax,
    )
    # X-axis (column names across the top/bottom)
    ax.tick_params(axis="x", labelsize=9, colors="steelblue")
    # Y-axis (column names down the left)
    ax.tick_params(axis="y", labelsize=9, colors="steelblue")

    # Optional rotations
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")

    for lbl in ax.get_yticklabels():
        lbl.set_rotation(0)
        lbl.set_ha("right")

    ax.set_xlabel("Features", fontsize=10, color="darkblue")
    ax.set_title("Feature Correlation Heatmap", fontsize=10, color="purple")
    st.pyplot(fig, clear_figure=True)

    st.markdown("---")


# -------------------------------------------------
# TAB 5 — Clean: Pipeline (All-in-One)
# -------------------------------------------------

def render_clean_pipeline(df: pd.DataFrame, selected_file: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean: Pipeline (All-in-One)  — file-scoped session state

    Steps:
      1) Categorical standardization (trim + lowercase)
      2) Date parsing (acceptance ≥ 0.60)
      3) Numeric conversion (acceptance ≥ 0.70)
      4) Outlier detection (3×IQR) — flag only (no modification)
      5) Median imputation for numeric NaNs
      5b) Categorical imputation (selector: None / Constant 'Unknown' / Mode / Group-aware + fallback)
      6) Duplicate removal (full-row, keep='first')

    Behavior:
      - Always starts from the currently selected CSV (no cross-file leakage).
      - Preview is stored per file key in st.session_state.
      - Apply & Save writes to <same folder>/clean_<original_name>.csv
        * If the clean file exists → asks overwrite confirmation; else creates silently.
      - After saving, st.rerun() and preselects the saved file path.
    """
    st.subheader("🛠️ Clean: Pipeline (All-in-One)")

    # ---------- File-scoped keys ----------
    file_key = os.path.abspath(selected_file).replace("\\", "/").lower()
    key_df  = f"pipeline_preview_df::{file_key}"
    key_log = f"pipeline_preview_log::{file_key}"

    # ---------- Source: ALWAYS the selected CSV ----------
    # (Prevents accidentally starting from a different file's cleaned frame)
    source = df.copy()
    rows_before, cols_before = source.shape

    # -------------------------
    # Pipeline Settings
    # -------------------------
    st.markdown("### ⚙️ Pipeline Settings")

    def checkbox_tile(container, key: str, label: str, default: bool = True, caption: Optional[str] = None) -> bool:
        val = container.checkbox(label, value=default, key=key)
        if caption:
            container.caption(caption)
        return val

    row1 = st.columns(2, gap="large")
    do_cat_std    = checkbox_tile(row1[0], "do_cat_std",    "Categorical standardization (trim + lower)", True)
    do_date_parse = checkbox_tile(row1[1], "do_date_parse", "Date parsing (accept ≥60%)", True)

    row2 = st.columns(2, gap="large")
    do_num_parse   = checkbox_tile(row2[0], "do_num_parse",  "Numeric conversion (accept ≥70%)", True)
    do_outlier_flag= checkbox_tile(row2[1], "do_outlier_flag","Outlier detection (3×IQR) — flag only", True)

    row3 = st.columns(1, gap="large")
    do_num_median  = checkbox_tile(row3[0], "do_num_median",  "Median imputation (numeric)", True)

    # Categorical missing strategy
    cat_strategy = st.selectbox(
        "Categorical missing strategy",
        ["None (leave as NA)", "Constant: 'Unknown'", "Mode per column", "Group-aware (recommended)"],
        index=3,
        key="cat_missing_strategy"
    )

    st.divider()
    st.info(
        "### Defaults used by the pipeline\n"
        "- **Duplicate removal:** full-row deduplication with `keep='first'` (applied at the end).\n"
        "- **Numeric missing values:** **Median** imputation (robust to skew/outliers).\n"
        "- **Categorical/Text missing values:** controlled by the selector above."
    )
    st.divider()

    # Thresholds
    DATE_ACCEPT_THR = 0.60
    NUM_ACCEPT_THR  = 0.70

    run_btn = st.button("🚀 Run Cleaning (Preview)", key="btn_run_pipeline")

    # Return placeholders (file-scoped)
    last_df:  pd.DataFrame = st.session_state.get(key_df,  source)
    last_log: List[str]    = st.session_state.get(key_log, [])

    # ---------- Helpers ----------
    def log(msg: str, op_log: List[str]):
        op_log.append(msg)

    def iqr_bounds(series: pd.Series, k: float = 3.0) -> Tuple[float, float]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return (float("-inf"), float("inf"))
        return (q1 - k * iqr, q3 + k * iqr)

    # Group-aware categorical imputation helper
    def fill_by_group_mode(df_in: pd.DataFrame, target: str, by: str) -> pd.DataFrame:
        if target not in df_in.columns or by not in df_in.columns:
            return df_in
        group_mode = df_in.groupby(by)[target].apply(
            lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else np.nan
        )
        def mapper(row):
            if pd.isna(row[target]):
                gm = group_mode.get(row[by], np.nan)
                return gm if pd.notna(gm) else row[target]
            return row[target]
        df_out = df_in.copy()
        df_out[target] = df_out.apply(mapper, axis=1)
        # Global fallback
        if df_out[target].isna().any():
            m = df_out[target].mode(dropna=True)
            df_out[target] = df_out[target].fillna(m.iloc[0] if not m.empty else "Unknown")
        return df_out

    # =================
    # RUN THE PIPELINE
    # =================
    if run_btn:
        cleaned = source.copy()
        op_log: List[str] = []
        outlier_report = {}

        # 1) Categorical standardization
        if do_cat_std:
            text_cols = [c for c in cleaned.columns if is_string_dtype(cleaned[c]) or cleaned[c].dtype.name == "category"]
            changes = 0
            for c in text_cols:
                s = cleaned[c]
                s_notna = s.notna()
                s_str = s.astype(str)
                s_new = s_str.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
                mask = s_notna & (s_str != s_new)
                if mask.any():
                    cleaned.loc[mask, c] = s_new[mask]
                    changes += int(mask.sum())
            log(f"Categorical standardization: trimmed/normalized {changes} cell(s) across {len(text_cols)} column(s).", op_log)

        # 2) Date parsing (≥ 0.60)
        if do_date_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_datetime(s, errors="coerce", utc=False)
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= DATE_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log(
                    "Date parsing: converted columns (acceptance≥{:.0%}): {}."
                    .format(DATE_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])),
                    op_log
                )
            if skipped_cols:
                log(
                    "Date parsing: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])),
                    op_log
                )

        # 3) Numeric conversion (≥ 0.70)
        if do_num_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_numeric(s.str.replace(",", ""), errors="coerce")
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= NUM_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log(
                    "Numeric conversion: converted columns (acceptance≥{:.0%}): {}."
                    .format(NUM_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])),
                    op_log
                )
            if skipped_cols:
                log(
                    "Numeric conversion: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])),
                    op_log
                )

        # 4) Outlier detection (3×IQR) — flag only
        if do_outlier_flag:
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    s = cleaned[c].dropna()
                    if len(s) < 4:
                        continue
                    low, high = iqr_bounds(s, k=3.0)
                    mask = (cleaned[c] < low) | (cleaned[c] > high)
                    count = int(mask.sum())
                    if count > 0:
                        outlier_report[c] = {"lower": float(low), "upper": float(high), "count": count}
            if outlier_report:
                total = sum(v["count"] for v in outlier_report.values())
                log(f"Outlier detection (3×IQR): flagged {total} value(s) across {len(outlier_report)} numeric column(s).", op_log)
            else:
                log("Outlier detection (3×IQR): no outliers flagged.", op_log)

        # 5) Median imputation (numeric)
        if do_num_median:
            imputed_cells = 0
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    na_before = int(cleaned[c].isna().sum())
                    if na_before > 0:
                        med = cleaned[c].median()
                        cleaned[c] = cleaned[c].fillna(med)
                        na_after = int(cleaned[c].isna().sum())
                        imputed_cells += (na_before - na_after)
            log(f"Median imputation (numeric): filled {imputed_cells} missing numeric cell(s).", op_log)

        # 5b) Categorical imputation
        cat_cols = [c for c in cleaned.columns if (cleaned[c].dtype == "object" or str(cleaned[c].dtype) == "category")]
        if cat_strategy == "Constant: 'Unknown'":
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (constant 'Unknown'): filled {filled_cells_cat} cell(s).", op_log)

        elif cat_strategy == "Mode per column":
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    mode_vals = cleaned[c].mode(dropna=True)
                    if not mode_vals.empty:
                        cleaned[c] = cleaned[c].fillna(mode_vals.iloc[0])
                    else:
                        cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (mode/Unknown fallback): filled {filled_cells_cat} cell(s).", op_log)

        elif cat_strategy == "Group-aware (recommended)":
            # Example group-aware policy for penguins-like schema
            if set(["species", "island"]).issubset(cleaned.columns):
                cleaned = fill_by_group_mode(cleaned, target="island", by="species")
            if set(["species", "sex"]).issubset(cleaned.columns):
                cleaned = fill_by_group_mode(cleaned, target="sex", by="species")
            # Global safety net
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (group-aware + 'Unknown' fallback): filled {filled_cells_cat} cell(s).", op_log)

        else:
            log("Categorical imputation: None (left as NA).", op_log)

        # 6) Duplicate removal — full-row, keep='first'
        rows_pre = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first").copy()
        removed = rows_pre - len(cleaned)
        log(f"Duplicate removal: full-row; keep='first'. Rows removed: {removed}.", op_log)

        # ----- Preview -----
        st.success(
            f"Preview complete — rows before: {rows_before:,}, after: {len(cleaned):,}. "
            f"Columns: {cols_before} → {cleaned.shape[1]}."
        )
        st.markdown("**Head (5) of cleaned preview**")
        st.dataframe(pretty_index_df(cleaned.head(), "R.No."), use_container_width=True)

        # Persist preview for Apply step (FILE-SCOPED)
        st.session_state[key_df]  = cleaned.copy()
        st.session_state[key_log] = op_log.copy()

        last_df  = cleaned
        last_log = op_log

    # --- Apply & Save (visible after a preview for THIS FILE) ---
    if st.session_state.get(key_df) is not None:
        if st.button("✅ Apply & Save to Session (Pipeline)", key=f"apply_pipeline_trigger::{file_key}"):
            # Target path: clean_<original>.csv
            base_name  = os.path.basename(selected_file)
            dir_name   = os.path.dirname(os.path.abspath(selected_file)) or os.path.abspath(".")
            clean_name = f"clean_{base_name}"
            clean_path = os.path.join(dir_name, clean_name)

            file_exists = os.path.exists(clean_path)

            if file_exists:
                st.session_state["show_confirm_overwrite"] = True
                st.warning(
                    "You are about to **overwrite** the existing cleaned file:\n\n"
                    f"**{clean_path}**\n\n"
                    "This action cannot be undone. Proceed?"
                )
                c1, c2 = st.columns(2)
                proceed = c1.button("🛑 Overwrite now", type="primary", key=f"btn_overwrite_confirm::{file_key}")
                cancel  = c2.button("Cancel", key=f"btn_overwrite_cancel::{file_key}")

                if proceed:
                    cleaned = st.session_state[key_df].copy()
                    op_log  = st.session_state.get(key_log, [])

                    # Save to session (optional, file-agnostic)
                    st.session_state["clean_df_pipeline"] = cleaned
                    st.session_state.setdefault("cleaning_log", [])
                    st.session_state["cleaning_log"].extend(op_log)

                    try:
                        cleaned.to_csv(clean_path, index=False)
                        st.success(f"Overwritten: {clean_path}")
                    except Exception as e:
                        st.error(f"Failed to write cleaned CSV: {e}")
                    finally:
                        st.session_state["show_confirm_overwrite"] = False
                        # Clear only THIS file's preview buffers
                        st.session_state.pop(key_df,  None)
                        st.session_state.pop(key_log, None)
                        # Preselect the saved file after rerun (if your sidebar reads this)
                        st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                        st.rerun()

                if cancel:
                    st.session_state["show_confirm_overwrite"] = False
                    st.info("Overwrite canceled.")
            else:
                # First-time create: save directly (no confirmation)
                cleaned = st.session_state[key_df].copy()
                op_log  = st.session_state.get(key_log, [])

                st.session_state["clean_df_pipeline"] = cleaned
                st.session_state.setdefault("cleaning_log", [])
                st.session_state["cleaning_log"].extend(op_log)

                try:
                    cleaned.to_csv(clean_path, index=False)
                    st.success(f"Saved cleaned CSV: {clean_path}")
                except Exception as e:
                    st.error(f"Failed to write cleaned CSV: {e}")
                finally:
                    # Clear only THIS file's preview buffers
                    st.session_state.pop(key_df,  None)
                    st.session_state.pop(key_log, None)
                    st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                    st.rerun()
    else:
        st.info("Run **Cleaning (Preview)** first to enable Apply & Save.")

    return last_df, last_log
    
    
def render_clean_pipeline_old(df: pd.DataFrame, selected_file: str) -> Tuple[pd.DataFrame, List[str]]:

    # =============================================================
    # Clean: Pipeline (All-in-One)
    #
    # Steps:
    #   1) Categorical standardization (trim + lowercase)
    #   2) Date parsing (acceptance ≥ 0.60)
    #   3) Numeric conversion (acceptance ≥ 0.70)
    #   4) Outlier detection (3×IQR) — flag only (no modification)
    #   5) Median imputation for numeric NaNs
    #     - Categorical imputation (select: None/Constant 'Unknown'/ Mode /Group-aware + fallback)
    #   6) Duplicate removal (full-row, keep='first')
    # =============================================================

    # Behavior:
    #   - No download buttons.
    #   - Apply & Save writes to <same folder>/clean_<original_name>.csv
    #     ** If the clean file exists → asks overwrite confirmation; else creates silently.
    #   - After saving, st.rerun() so the app refreshes; preselects the saved file path.
    # =============================================================

    st.subheader("🛠️ Clean: Pipeline (All-in-One)")

    # Use last saved pipeline DF if present; else the incoming df
    source = st.session_state.get("clean_df_pipeline", df).copy()
    rows_before, cols_before = source.shape

    # -------------------------
    # Pipeline Settings
    # -------------------------
    st.markdown("##### ⚙️ Pipeline Settings")

    def checkbox_tile(container, key: str, label: str, default: bool = True, caption: Optional[str] = None) -> bool:
        val = container.checkbox(label, value=default, key=key)
        if caption:
            container.caption(caption)
        return val

    row1 = st.columns(2, gap="large")
    do_cat_std = checkbox_tile(row1[0], "do_cat_std",    "Categorical standardization (trim + lower)", True)
    do_date_parse = checkbox_tile(row1[1], "do_date_parse", "Date parsing (accept ≥60%)", True)

    row2 = st.columns(2, gap="large")
    do_num_parse = checkbox_tile(row2[0], "do_num_parse",  "Numeric conversion (accept ≥70%", True)
    do_outlier_flag = checkbox_tile(row2[1], "do_outlier_flag", "Outlier detection (3×IQR) — flag only", True)

    row3 = st.columns(1, gap="large")
    do_num_median = checkbox_tile(row3[0], "do_num_median",   "Median imputation (numeric)", True)


    # Categorical missing strategy
    cat_strategy = st.selectbox(
        "Categorical missing strategy",
        ["None (leave as NA)", "Constant: 'Unknown'", "Mode per column", "Group-aware (recommended)"],
        index=3,
        key="cat_missing_strategy"
    )

    st.divider()
    st.info(
        "##### Defaults used by the pipeline\n"
        "- **Duplicate removal:** full-row deduplication with `keep='first'` (applied at the end).\n"
        "- **Numeric missing values:** **Median** imputation (robust to skew/outliers).\n"
        "- **Categorical/Text missing values:** controlled by the selector above."
    )
    st.divider()

    # Thresholds
    DATE_ACCEPT_THR = 0.60
    NUM_ACCEPT_THR  = 0.70

    run_btn = st.button("🚀 Run Cleaning (Preview)", key="btn_run_pipeline")

    # Return placeholders
    last_df: pd.DataFrame = st.session_state.get("pipeline_preview_df", source)
    last_log: List[str] = st.session_state.get("pipeline_preview_log", [])

    # ---------- Helpers ----------
    def log(msg: str, op_log: List[str]):
        op_log.append(msg)

    def iqr_bounds(series: pd.Series, k: float = 3.0) -> Tuple[float, float]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return (float("-inf"), float("inf"))
        return (q1 - k * iqr, q3 + k * iqr)

    # Group-aware categorical imputation helper
    def fill_by_group_mode(df_in: pd.DataFrame, target: str, by: str) -> pd.DataFrame:
        if target not in df_in.columns or by not in df_in.columns:
            return df_in
        group_mode = df_in.groupby(by)[target].apply(
            lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else np.nan
        )
        def mapper(row):
            if pd.isna(row[target]):
                gm = group_mode.get(row[by], np.nan)
                return gm if pd.notna(gm) else row[target]
            return row[target]
        df_out = df_in.copy()
        df_out[target] = df_out.apply(mapper, axis=1)
        # Global fallback
        if df_out[target].isna().any():
            m = df_out[target].mode(dropna=True)
            df_out[target] = df_out[target].fillna(m.iloc[0] if not m.empty else "Unknown")
        return df_out

    # =================================
    # RUN THE DATA CLEANING PIPELINE
    # =================================
    if run_btn:
        cleaned = source.copy()
        op_log: List[str] = []
        outlier_report = {}

        # ---- 1) Categorical standardization
        if do_cat_std:
            text_cols = [c for c in cleaned.columns if is_string_dtype(cleaned[c]) or cleaned[c].dtype.name == "category"]
            changes = 0
            for c in text_cols:
                s = cleaned[c]
                s_notna = s.notna()
                s_str = s.astype(str)
                s_new = s_str.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
                mask = s_notna & (s_str != s_new)
                if mask.any():
                    cleaned.loc[mask, c] = s_new[mask]
                    changes += int(mask.sum())
            log(f"Categorical standardization: trimmed/normalized {changes} cell(s) across {len(text_cols)} column(s).", op_log)

        # ---- 2) Date parsing (≥ 0.60)
        if do_date_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_datetime(s, errors="coerce", utc=False)
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= DATE_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log(
                    "Date parsing: converted columns (acceptance≥{:.0%}): {}."
                    .format(DATE_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])),
                    op_log
                )
            if skipped_cols:
                log(
                    "Date parsing: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])),
                    op_log
                )

        # ---- 3) Numeric conversion (≥ 0.70)
        if do_num_parse:
            candidates = [c for c in cleaned.columns if is_string_dtype(cleaned[c])]
            converted_cols, skipped_cols = [], []
            for c in candidates:
                s = cleaned[c]
                nonnull = s.notna().sum()
                if nonnull == 0:
                    continue
                parsed = pd.to_numeric(s.str.replace(",", ""), errors="coerce")
                ok = parsed.notna().sum()
                ratio = ok / nonnull if nonnull else 0.0
                if ratio >= NUM_ACCEPT_THR:
                    cleaned[c] = parsed
                    converted_cols.append((c, ratio))
                else:
                    skipped_cols.append((c, ratio))
            if converted_cols:
                log(
                    "Numeric conversion: converted columns (acceptance≥{:.0%}): {}."
                    .format(NUM_ACCEPT_THR, ", ".join([f"{c} ({r:.0%})" for c, r in converted_cols])),
                    op_log
                )
            if skipped_cols:
                log(
                    "Numeric conversion: skipped columns (below threshold): {}."
                    .format(", ".join([f"{c} ({r:.0%})" for c, r in skipped_cols])),
                    op_log
                )

        # ---- 4) Outlier detection (3×IQR) — flag only
        if do_outlier_flag:
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    s = cleaned[c].dropna()
                    if len(s) < 4:
                        continue
                    low, high = iqr_bounds(s, k=3.0)
                    mask = (cleaned[c] < low) | (cleaned[c] > high)
                    count = int(mask.sum())
                    if count > 0:
                        outlier_report[c] = {"lower": float(low), "upper": float(high), "count": count}
            if outlier_report:
                total = sum(v["count"] for v in outlier_report.values())
                log(f"Outlier detection (3×IQR): flagged {total} value(s) across {len(outlier_report)} numeric column(s).", op_log)
            else:
                log("Outlier detection (3×IQR): no outliers flagged.", op_log)

        # ---- 5) Median imputation (numeric)
        if do_num_median:
            imputed_cells = 0
            for c in cleaned.columns:
                if is_numeric_dtype(cleaned[c]):
                    na_before = int(cleaned[c].isna().sum())
                    if na_before > 0:
                        med = cleaned[c].median()
                        cleaned[c] = cleaned[c].fillna(med)
                        na_after = int(cleaned[c].isna().sum())
                        imputed_cells += (na_before - na_after)
            log(f"Median imputation (numeric): filled {imputed_cells} missing numeric cell(s).", op_log)

        # ---- 5b) Categorical imputation
        cat_cols = [c for c in cleaned.columns if (cleaned[c].dtype == "object" or str(cleaned[c].dtype) == "category")]
        if cat_strategy == "Constant: 'Unknown'":
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (constant 'Unknown'): filled {filled_cells_cat} cell(s).", op_log)

        elif cat_strategy == "Mode per column":
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    mode_vals = cleaned[c].mode(dropna=True)
                    if not mode_vals.empty:
                        cleaned[c] = cleaned[c].fillna(mode_vals.iloc[0])
                    else:
                        cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (mode/Unknown fallback): filled {filled_cells_cat} cell(s).", op_log)

        elif cat_strategy == "Group-aware (recommended)":
            # Example group-aware policy for penguins-like schema
            if set(["species", "island"]).issubset(cleaned.columns):
                cleaned = fill_by_group_mode(cleaned, target="island", by="species")
            if set(["species", "sex"]).issubset(cleaned.columns):
                cleaned = fill_by_group_mode(cleaned, target="sex", by="species")
            # Global safety net
            filled_cells_cat = 0
            for c in cat_cols:
                na_before = int(cleaned[c].isna().sum())
                if na_before > 0:
                    cleaned[c] = cleaned[c].fillna("Unknown")
                    filled_cells_cat += na_before
            log(f"Categorical imputation (group-aware + 'Unknown' fallback): filled {filled_cells_cat} cell(s).", op_log)

        else:
            log("Categorical imputation: None (left as NA).", op_log)

        # ---- 6) Duplicate removal — full-row, keep='first'
        rows_pre = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first").copy()
        removed = rows_pre - len(cleaned)
        log(f"Duplicate removal: full-row; keep='first'. Rows removed: {removed}.", op_log)

        # =========== Preview ===========
        st.success(
            f"Preview complete — rows before: {rows_before:,}, after: {len(cleaned):,}. "
            f"Columns: {cols_before} → {cleaned.shape[1]}."
        )
        st.markdown("**Head (5) of cleaned preview**")
        st.dataframe(pretty_index_df(cleaned.head(), "R.No."), use_container_width=True)

        # Persist preview for Apply step
        st.session_state["pipeline_preview_df"]  = cleaned.copy()
        st.session_state["pipeline_preview_log"] = op_log.copy()

        last_df  = cleaned
        last_log = op_log

    # --- Apply & Save (always visible after a preview) ---
    if st.session_state.get("pipeline_preview_df") is not None:
        if st.button("✅ Apply & Save Cleaned Data", key="apply_pipeline_trigger"):
            # Target path: clean_<original>.csv
            base_name  = os.path.basename(selected_file)
            dir_name   = os.path.dirname(os.path.abspath(selected_file)) or os.path.abspath(".")
            clean_name = f"clean_{base_name}"
            clean_path = os.path.join(dir_name, clean_name)

            file_exists = os.path.exists(clean_path)

            if file_exists:
                st.session_state["show_confirm_overwrite"] = True
                st.warning(
                    "You are about to **overwrite** the existing cleaned file:\n\n"
                    f"**{clean_path}**\n\n"
                    "This action cannot be undone. Proceed?"
                )
                c1, c2 = st.columns(2)
                proceed = c1.button("🛑 Overwrite now", type="primary", key="btn_overwrite_confirm")
                cancel  = c2.button("Cancel", key="btn_overwrite_cancel")

                if proceed:
                    cleaned = st.session_state["pipeline_preview_df"].copy()
                    op_log  = st.session_state.get("pipeline_preview_log", [])

                    # Save to session
                    st.session_state["clean_df_pipeline"] = cleaned
                    st.session_state.setdefault("cleaning_log", [])
                    st.session_state["cleaning_log"].extend(op_log)

                    try:
                        cleaned.to_csv(clean_path, index=False)
                        st.success(f"Overwritten: {clean_path}")
                    except Exception as e:
                        st.error(f"Failed to write cleaned CSV: {e}")
                    finally:
                        st.session_state["show_confirm_overwrite"] = False
                        st.session_state.pop("pipeline_preview_df", None)
                        st.session_state.pop("pipeline_preview_log", None)
                        # Keep the saved file selected after rerun (if your sidebar reads this)
                        st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                        st.rerun()

                if cancel:
                    st.session_state["show_confirm_overwrite"] = False
                    st.info("Overwrite canceled.")
            else:
                # First-time create: save directly (no confirmation)
                cleaned = st.session_state["pipeline_preview_df"].copy()
                op_log  = st.session_state.get("pipeline_preview_log", [])

                st.session_state["clean_df_pipeline"] = cleaned
                st.session_state.setdefault("cleaning_log", [])
                st.session_state["cleaning_log"].extend(op_log)

                try:
                    cleaned.to_csv(clean_path, index=False)
                    st.success(f"Saved cleaned CSV: {clean_path}")
                except Exception as e:
                    st.error(f"Failed to write cleaned CSV: {e}")
                finally:
                    st.session_state.pop("pipeline_preview_df", None)
                    st.session_state.pop("pipeline_preview_log", None)
                    st.session_state["selected_file_preselect"] = os.path.abspath(clean_path)
                    st.rerun()
    else:
        st.info("Run **Cleaning (Preview)** first to enable Apply & Save.")

    return last_df, last_log
    
    
# ================================
# 4) Main
# ================================
def main() -> None:
    setup_page()

    # --- Sidebar controls ---
    selected_file, selected_numeric_col, bins = sidebar_controls()

    # --- Load the selected dataset (RAW for EDA tabs) ---
    df = load_data(selected_file)

    # --- Tabs layout ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📄 Overview",
            "📈 Distribution",
            "🔎 Data Quality",
            "📊 Correlation",
            "🛠️ Clean: Pipeline (All-in-One)",
        ]
    )

    with tab1:
        render_overview(df)
    with tab2:
        render_distribution(df, selected_numeric_col, bins)
    with tab3:
        render_data_quality(df)
    with tab4:
        render_correlation(df)
    with tab5:
        # pass selected_file so the pipeline can save to the same folder with prefix 'clean_'
        _cleaned_pipeline, _log_pipeline = render_clean_pipeline(df, selected_file)


if __name__ == "__main__":
    main()
    