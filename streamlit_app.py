# streamlit_app.py — OPTIMIZED VERSION
import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Safer: show a clear UI message if sklearn fails to import
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception as e:
    st.set_page_config(page_title="Weekly Forecast", layout="wide")
    st.error(
        "scikit-learn failed to import. This usually means the cloud build missed requirements. "
        "Please ensure requirements.txt is pinned and click Manage app → Reboot. "
        f"\n\nDetails: {type(e).__name__}: {e}"
    )
    st.stop()


st.set_page_config(page_title="Weekly Forecast (BASE / ML / BEST)", layout="wide")


# ----------------------------
# Generic helpers
# ----------------------------
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Weighted Absolute Percentage Error."""
    denom = float(np.abs(y_true).sum())
    return np.nan if denom == 0 else float(np.abs(y_true - y_pred).sum()) / denom


def monday_week_start(s: pd.Series) -> pd.Series:
    """Finds the Monday of the week for a given date series."""
    return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()


# ----------------------------
# Load Excel (Base sheet) — fast, only needed cols (+ detect P-9)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_base_excel(file) -> pd.DataFrame:
    """Loads and preprocesses the uploaded Excel file, focusing on necessary columns."""

    def _need(col):
        # keep base cols or P-9-like cols
        n = str(col).replace("\u00A0", " ").strip().upper().replace("-", "").replace("_", "")
        base_cols = {"CUSTOMER GROUP", "ITEMCODE", "UOM", "DELIVERY DATE", "QUANTITY"}
        return (n in base_cols) or (n in {"P9", "P09", "PERIOD9"})

    # Use a broader set of potential column names to handle variations
    col_map = {
        "Customer Group": ["Corporate Customer", "Customer", "CustomerGroup"],
        "Itemcode": ["Item Code", "Item", "SKU", "Item Code "],
    }

    df = pd.read_excel(file, sheet_name="Base", header=4, usecols=_need)

    # Normalize column names
    for standard_name, variations in col_map.items():
        if standard_name not in df.columns:
            for alt in variations:
                if alt in df.columns:
                    df = df.rename(columns={alt: standard_name})
                    break

    # Basic cleaning
    df["UOM"] = (
        df["UOM"].astype(str).str.replace("\u00A0", " ", regex=False).str.strip().str.upper()
    )
    df["Delivery Date"] = pd.to_datetime(df["Delivery Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Customer Group"] = df["Customer Group"].astype(str)
    df["Itemcode"] = df["Itemcode"].astype(str)
    return df


# ----------------------------
# Build weekly table at chosen aggregation level
# ----------------------------
def build_weekly(df: pd.DataFrame, series_cols: List[str]) -> pd.DataFrame:
    """Aggregates the daily data to a weekly level and fills in missing weeks with zeros."""
    # Filter: Jan–Aug 2025, UOM=CS
    mask = (df["UOM"].eq("CS")) & (df["Delivery Date"].between("2025-01-01", "2025-08-31"))
    base_cols = set(series_cols) | {"Delivery Date", "Quantity"}
    d = df.loc[mask, list(base_cols)].copy()
    d["week_start"] = monday_week_start(d["Delivery Date"])

    group_cols = series_cols + ["week_start"]
    wk = d.groupby(group_cols, dropna=False)["Quantity"].sum().rename("qty").reset_index()

    if wk.empty:
        return wk.assign(qty=0.0)

    # Complete weekly calendar per series (zero-fill)
    cal = pd.date_range(wk["week_start"].min(), wk["week_start"].max(), freq="W-MON")
    keys = wk[series_cols].drop_duplicates().sort_values(series_cols)
    full = (
        keys.assign(_k=1)
        .merge(pd.DataFrame({"week_start": cal, "_k": 1}), on="_k")
        .drop(columns="_k")
    )
    wk_full = (
        full.merge(wk, on=group_cols, how="left")
        .assign(qty=lambda d: d["qty"].fillna(0.0))
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return wk_full


# ----------------------------
# Feature engineering (OPTIMIZED AND VECTORIZED)
# ----------------------------
def add_lags_rolls_calendar_vectorized(
    wk_full: pd.DataFrame, series_cols: List[str]
) -> pd.DataFrame:
    """Creates lag, rolling, and calendar features efficiently using vectorized operations."""
    df = wk_full.sort_values(series_cols + ["week_start"]).copy()
    gb = df.groupby(series_cols, dropna=False)["qty"]

    # Lags
    for k in [1, 2, 3, 4]:
        df[f"lag{k}"] = gb.shift(k)

    # Rolling means
    for w in [4, 8, 12]:
        df[f"rollmean_{w}"] = gb.shift(1).rolling(w, min_periods=2).mean()

    # Calendar features
    woy = df["week_start"].dt.isocalendar().week.astype(int)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)

    wom = ((df["week_start"].dt.day - 1) // 7 + 1).astype(int).clip(1, 5)
    df["wom"] = wom
    df["wom_sin"] = np.sin(2 * np.pi * df["wom"] / 5.0)
    df["wom_cos"] = np.cos(2 * np.pi * df["wom"] / 5.0)

    # Momentum features
    df["slope_1"] = df["lag1"] - df["rollmean_4"]
    df["mom_4_8"] = df["rollmean_4"] - df["rollmean_8"]
    df["ratio_4_8"] = df["rollmean_4"] / np.where(
        df["rollmean_8"].abs() > 1e-9, df["rollmean_8"].abs(), 1e-9
    )
    return df


# ----------------------------
# GLOBAL MODEL PIPELINE (TRAIN, BACKTEST, FORECAST)
# ----------------------------
def run_global_model_pipeline(
    X: pd.DataFrame, feat_cols: List[str], series_cols: List[str], decay: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Trains a single global model, runs an August backtest, and generates a September forecast.
    """
    # --- 1. Define Time Splits ---
    TRAIN_END = pd.Timestamp("2025-07-31")
    VAL_START = pd.Timestamp("2025-08-01")
    VAL_END = pd.Timestamp("2025-08-31")
    SEPT_WEEKS = pd.date_range("2025-09-01", "2025-09-30", freq="W-MON")

    # --- 2. Prepare Data ---
    X["qty_clipped"] = X["qty"].clip(lower=0)
    categorical_features = [c for c in series_cols if X[c].dtype.name in ['category', 'object']]
    for c in categorical_features:
        X[c] = X[c].astype("category")

    # Split data
    train_df = X[X["week_start"] <= TRAIN_END].copy()
    val_df = X[(X["week_start"] >= VAL_START) & (X["week_start"] <= VAL_END)].copy()
    
    # Drop rows with missing features for training/validation
    train_df = train_df.dropna(subset=feat_cols)
    val_df = val_df.dropna(subset=feat_cols)

    # Add sample weights for recency
    age_weeks = ((TRAIN_END - train_df["week_start"]).dt.days // 7).clip(lower=0)
    train_df["sample_weight"] = (decay**age_weeks).astype(float)
    
    y_train = np.log1p(train_df["qty_clipped"])
    
    # --- 3. Train Global Model ---
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=5,
        learning_rate=0.06,
        max_iter=300,
        l2_regularization=0.0,
        categorical_features=categorical_features,
        random_state=42,
    )
    model.fit(train_df[feat_cols + series_cols], y_train, sample_weight=train_df["sample_weight"])
    
    # --- 4. August Backtest ---
    # Baseline predictions (rolling 4-week mean)
    val_df["pred_base"] = val_df["rollmean_4"]
    # ML predictions
    val_df["pred_ml"] = np.expm1(model.predict(val_df[feat_cols + series_cols])).clip(lower=0)

    # Calculate WAPE by series
    def get_wape(g, col):
        return wape(g["qty"].to_numpy(), g[col].to_numpy())
    
    base_wapes = val_df.groupby(series_cols, dropna=False).apply(get_wape, "pred_base").rename("WAPE_base")
    ml_wapes = val_df.groupby(series_cols, dropna=False).apply(get_wape, "pred_ml").rename("WAPE_ML")
    backtest_results = pd.concat([base_wapes, ml_wapes], axis=1).reset_index()
    
    # --- 5. September Forecast ---
    # Create future dataframe for September
    future_keys = X[series_cols].drop_duplicates()
    future_df = (
        future_keys.assign(_k=1)
        .merge(pd.DataFrame({"week_start": SEPT_WEEKS, "_k": 1}), on="_k")
        .drop(columns="_k")
    )
    
    # Iteratively forecast and update features
    all_preds_base = []
    all_preds_ml = []
    
    hist_df = X[X["week_start"] <= VAL_END].copy()
    
    for week_start in SEPT_WEEKS:
        # Create features for the current week
        latest_features = add_lags_rolls_calendar_vectorized(hist_df, series_cols)
        current_week_feats = latest_features[latest_features["week_start"] == week_start.normalize() - pd.DateOffset(weeks=1)].copy()
        current_week_feats['week_start'] = week_start

        # Baseline Prediction
        base_pred = current_week_feats['rollmean_4'].fillna(current_week_feats['lag1']).fillna(0)

        # ML Prediction
        for c in categorical_features:
            current_week_feats[c] = current_week_feats[c].astype('category').cat.set_categories(X[c].cat.categories)

        ml_pred = np.expm1(model.predict(current_week_feats[feat_cols + series_cols])).clip(lower=0)

        # Store predictions
        week_preds_base = current_week_feats[series_cols + ["week_start"]].copy()
        week_preds_base["pred_base"] = base_pred
        all_preds_base.append(week_preds_base)

        week_preds_ml = current_week_feats[series_cols + ["week_start"]].copy()
        week_preds_ml["pred_ml"] = ml_pred
        all_preds_ml.append(week_preds_ml)

        # Append predictions to history for next iteration's feature calculation
        new_rows = current_week_feats[series_cols + ["week_start"]].copy()
        new_rows["qty"] = ml_pred # Use ML pred to generate features for the next step
        hist_df = pd.concat([hist_df, new_rows], ignore_index=True)

    pred_base = pd.concat(all_preds_base, ignore_index=True)
    pred_ml = pd.concat(all_preds_ml, ignore_index=True)

    return backtest_results, pred_base, pred_ml, val_df


# ----------------------------
# S&OP monthly compare from Base (P-9)
# ----------------------------
def monthly_sop_from_base(base_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the September S&OP numbers (P-9) from the raw data."""
    norm = {
        c: str(c).replace("\u00A0", " ").strip().upper().replace("-", "").replace("_", "")
        for c in base_df.columns
    }
    p9_col = None
    for c, n in norm.items():
        if n in {"P9", "P09", "PERIOD9"}:
            p9_col = c
            break

    if p9_col is None:
        return pd.DataFrame(columns=["Customer Group", "sop_sept"])

    cols = [c for c in ["Customer Group", "Itemcode", p9_col] if c in base_df.columns]
    sop_p9 = base_df[cols].copy()
    sop_p9[p9_col] = pd.to_numeric(sop_p9[p9_col], errors="coerce")
    p9_by_item = (
        sop_p9.dropna(subset=[p9_col])
        .groupby(["Customer Group", "Itemcode"], as_index=False)[p9_col]
        .max()
    )
    sop_sep = (
        p9_by_item.groupby("Customer Group", as_index=False)[p9_col]
        .sum()
        .rename(columns={p9_col: "sop_sept"})
    )
    return sop_sep


def series_label(row: pd.Series, series_cols: List[str]) -> str:
    """Creates a display label for a time series."""
    if series_cols == ["Customer Group"]:
        return str(row["Customer Group"])
    return f"{row['Customer Group']} · {row['Itemcode']}"

# ----------------------------
# --- Streamlit UI Sidebar ---
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "Upload Excel (Jan-Aug_ SOP Sept-Dec.xlsx)", type=["xlsx"]
)

st.sidebar.header("Settings")
agg_level = st.sidebar.radio(
    "Aggregation level", ["Customer", "Customer × Item"], index=0
)
SERIES_COLS = (
    ["Customer Group"] if agg_level == "Customer" else ["Customer Group", "Itemcode"]
)

rel_improve = st.sidebar.slider(
    "ML must beat BASE by at least (relative)", 0.00, 0.20, 0.05, 0.01
)
decay = st.sidebar.slider(
    "Recency decay (per week)",
    0.90,
    0.999,
    0.97,
    0.001,
    help="Lower = more weight on recent weeks; higher = more uniform.",
)

# Top-N cap for Customer×Item to keep app responsive
topN = None
if agg_level == "Customer × Item":
    topN = st.sidebar.slider(
        "Limit to top-N Customer×Item series (by total qty Jan–Aug)",
        min_value=50,
        max_value=2000,
        value=300,
        step=50,
        help="Keeps the biggest series so the app runs fast. Raise later after testing.",
    )

# ----------------------------
# --- Main App Logic ---
# ----------------------------
st.title("Weekly Forecast — BASE / ML / BEST (September 2025)")

if not uploaded:
    st.info(
        "Upload the Excel file to run the app (sheet **Base**, daily actuals Jan–Aug 2025, S&OP P-9)."
    )
    st.stop()

# --- Full Pipeline Execution ---
try:
    base_df = load_base_excel(uploaded)

    with st.spinner("Building weekly data…"):
        wk_full = build_weekly(base_df, SERIES_COLS)

        # Fast mode for Customer×Item
        if topN is not None and SERIES_COLS == ["Customer Group", "Itemcode"]:
            totals = (
                wk_full.groupby(SERIES_COLS, as_index=False)["qty"]
                .sum()
                .sort_values("qty", ascending=False)
                .head(int(topN))[SERIES_COLS]
            )
            before = wk_full[SERIES_COLS].drop_duplicates().shape[0]
            wk_full = wk_full.merge(totals, on=SERIES_COLS, how="inner")
            after = wk_full[SERIES_COLS].drop_duplicates().shape[0]
            st.info(
                f"Fast mode: training on top {after}/{before} Customer×Item series (by total qty)."
            )

        st.info(
            f"Series: {wk_full[SERIES_COLS].drop_duplicates().shape[0]} | "
            f"Weeks: {wk_full['week_start'].nunique()} | Rows: {len(wk_full):,}"
        )

        if wk_full.empty:
            st.error(
                "No rows after filtering (UOM='CS' & Jan–Aug 2025). Check the file."
            )
            st.stop()

    with st.spinner("Engineering features…"):
        X = add_lags_rolls_calendar_vectorized(wk_full, SERIES_COLS)
        feat_cols = [
            "lag1", "lag2", "lag3", "lag4",
            "rollmean_4", "rollmean_8", "rollmean_12",
            "woy_sin", "woy_cos",
            "wom", "wom_sin", "wom_cos",
            "slope_1", "mom_4_8", "ratio_4_8",
        ]
        
    with st.spinner("Training Global Model, Backtesting, and Forecasting…"):
        backtest_results, pred_base, pred_ml, val_aug_df = run_global_model_pipeline(
            X, feat_cols, SERIES_COLS, decay
        )
    
    # Calculate overall WAPEs from August backtest
    overall_base_wape = wape(val_aug_df["qty"], val_aug_df["pred_base"])
    overall_ml_wape = wape(val_aug_df["qty"], val_aug_df["pred_ml"])

    # Pick winners using relative improvement threshold
    winners = backtest_results.assign(
        abs_gain=lambda d: d["WAPE_base"] - d["WAPE_ML"],
        rel_gain=lambda d: (d["WAPE_base"] - d["WAPE_ML"]) / d["WAPE_base"].clip(lower=1e-9),
        winner=lambda d: np.where(
            (d["WAPE_ML"] < d["WAPE_base"]) & (d["rel_gain"] >= rel_improve),
            "ML",
            "BASE",
        ),
    )

    # Combine September forecasts
    comb = pred_base.merge(
        pred_ml, on=SERIES_COLS + ["week_start"], how="left"
    ).merge(winners[SERIES_COLS + ["winner"]], on=SERIES_COLS, how="left")
    comb["winner"] = comb["winner"].fillna("BASE")
    comb["pred_best"] = np.where(
        comb["winner"].eq("ML") & comb["pred_ml"].notna(),
        comb["pred_ml"],
        comb["pred_base"],
    )

    # S&OP monthly compare
    sop_sep = monthly_sop_from_base(base_df)
    f_sept = (
        comb.groupby("Customer Group", as_index=False)["pred_best"]
        .sum()
        .rename(columns={"pred_best": "forecast_sept"})
    )
    cmp_month = f_sept.merge(sop_sep, on="Customer Group", how="left")
    cmp_month["delta"] = cmp_month["forecast_sept"] - cmp_month["sop_sept"]
    cmp_month["delta%"] = cmp_month["delta"] / cmp_month["sop_sept"].replace({0: np.nan})

    # ----------------------------
    # --- UI Tabs ---
    # ----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Forecast", "August Backtest", "S&OP Compare (monthly)", "README"]
    )

    with tab1:
        st.subheader(
            f"Forecast — September 2025 | Level: {'Customer' if SERIES_COLS==['Customer Group'] else 'Customer × Item'}"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "August WAPE — Baseline",
            f"{overall_base_wape:.3f}" if pd.notna(overall_base_wape) else "n/a",
        )
        c2.metric(
            "August WAPE — ML",
            f"{overall_ml_wape:.3f}" if pd.notna(overall_ml_wape) else "n/a",
        )
        wc = winners["winner"].value_counts() if not winners.empty else pd.Series(dtype=int)
        c3.metric("Winners (ML / BASE)", f"{int(wc.get('ML',0))} / {int(wc.get('BASE',0))}")

        mode = st.radio("Forecast Mode", ["BEST", "ML", "BASE"], horizontal=True)
        df_show = {
            "BEST": comb.rename(columns={"pred_best": "forecast_qty"}),
            "ML": pred_ml.rename(columns={"pred_ml": "forecast_qty"}),
            "BASE": pred_base.rename(columns={"pred_base": "forecast_qty"}),
        }[mode][SERIES_COLS + ["week_start", "forecast_qty"]].copy()
        
        # Series picker for charts
        sel_df = df_show.copy()
        sel_df["series_label"] = sel_df.apply(lambda r: series_label(r, SERIES_COLS), axis=1)
        
        unique_labels = sorted(sel_df["series_label"].unique())
        if not unique_labels:
            st.warning("No forecast data available to display for the selected mode.")
            st.stop()
            
        default_label = unique_labels[0]
        series_sel = st.selectbox(
            "Select series to view",
            unique_labels,
            index=unique_labels.index(default_label),
        )
        
        # Chart and table data
        sel_rows = sel_df[sel_df["series_label"] == series_sel]
        base_sel = pred_base.merge(sel_rows[SERIES_COLS].drop_duplicates(), on=SERIES_COLS, how="inner").rename(columns={"pred_base": "BASE"})
        ml_sel = pred_ml.merge(sel_rows[SERIES_COLS].drop_duplicates(), on=SERIES_COLS, how="inner").rename(columns={"pred_ml": "ML"})
        best_sel = comb.merge(sel_rows[SERIES_COLS].drop_duplicates(), on=SERIES_COLS, how="inner").rename(columns={"pred_best": "BEST"})

        joined = (
            base_sel.merge(ml_sel, on=SERIES_COLS + ["week_start"], how="outer")
            .merge(best_sel[SERIES_COLS + ["week_start", "BEST", "winner"]], on=SERIES_COLS + ["week_start"], how="outer")
            .sort_values("week_start")
        )
        joined["week_start"] = pd.to_datetime(joined["week_start"]).dt.date
        st.line_chart(joined.set_index("week_start")[["BASE", "ML", "BEST"]])
        st.dataframe(joined, use_container_width=True)
        
        # Download button
        def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                for name, df_ in sheets.items():
                    _df = df_.copy()
                    if "week_start" in _df.columns:
                        _df["week_start"] = pd.to_datetime(_df["week_start"]).dt.date
                    _df.to_excel(writer, sheet_name=name, index=False)
            buf.seek(0)
            return buf.read()
            
        st.download_button(
            "Download September forecasts (Excel)",
            data=to_excel_bytes({
                "sept_base": pred_base.rename(columns={"pred_base": "forecast_qty"}),
                "sept_ml": pred_ml.rename(columns={"pred_ml": "forecast_qty"}),
                "sept_best": comb[SERIES_COLS + ["week_start", "pred_best"]].rename(columns={"pred_best": "forecast_qty"}),
            }),
            file_name="forecast_sept_2025_optimized.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with tab2:
        st.subheader(
            f"August Backtest — WAPE & Winners | Level: {'Customer' if SERIES_COLS==['Customer Group'] else 'Customer × Item'}"
        )
        if winners.empty:
            st.info("Not enough data per series to compute the backtest.")
        else:
            comp = winners[SERIES_COLS + ["WAPE_base", "WAPE_ML", "abs_gain", "rel_gain", "winner"]] \
                .sort_values(["winner", "rel_gain"], ascending=[True, False]) \
                .reset_index(drop=True)
            st.dataframe(comp, use_container_width=True)

    with tab3:
        st.subheader("S&OP Compare — Monthly (P-9 vs September total)")
        if sop_sep.empty:
            st.info("P-9 (September S&OP) column not detected in Base sheet.")
        else:
            cmp_view = (
                cmp_month.sort_values("forecast_sept", ascending=False)
                .rename(columns={
                    "forecast_sept": "Forecast Sept (Best-of)",
                    "sop_sept": "S&OP Sept (P-9)",
                    "delta": "Δ (Forecast - S&OP)",
                    "delta%": "Δ% vs S&OP",
                })
            )
            st.dataframe(
                cmp_view,
                use_container_width=True,
                column_config={"Δ% vs S&OP": st.column_config.ProgressColumn(format="%.2f%%", min_value=-100, max_value=100)}
            )
            st.caption("S&OP is comparison-only (not a training input).")

    with tab4:
        st.subheader("README / Approach")
        st.markdown(f"""
**Goal.** Independent weekly forecast of **September 2025** per series  
(**Level:** {'Customer' if SERIES_COLS==['Customer Group'] else 'Customer × Item'}). We output **BASE**, **ML**, and **BEST (winner)**.

**Method (Optimized):**
- **BASE:** last 4-week mean, iterative.
- **ML (Global Model):** A *single* gradient boosting model is trained on *all series at once*.
  - **Categorical Features:** `Customer Group` and `Itemcode` are used as features so the model can learn per-series patterns.
  - **Train:** **Jan–Jul**, validate on **August**.
  - **Loss:** **MAE-like** (`absolute_error`), **log1p** target, **recency weights** (sidebar).
- **Best-of:** Use ML only if it beats BASE on the August backtest by ≥ threshold (sidebar), else keep BASE.
- **S&OP (P-9):** monthly comparison only (not used in training).

This global model approach is significantly **faster** and often **more accurate** than training hundreds of individual models.
""")

except Exception as e:
    st.error(
        "🚨 The app hit an error while processing the uploaded file. "
        "Please verify the file format (sheet ‘Base’, required columns) or share the logs."
    )
    st.exception(e)
    st.stop()
