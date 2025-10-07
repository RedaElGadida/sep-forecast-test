# app.py
import io
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Weekly Forecast (BASE / ML / BEST)", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_base_excel(file) -> pd.DataFrame:
    """
    Load the 'Base' sheet (headers start on row 5 -> header=4).
    Returns a DataFrame with at least:
    ['Customer Group','Itemcode','UOM','Quantity','Delivery Date', ... 'P-9' ...]
    """
    df = pd.read_excel(file, sheet_name="Base", header=4)
    # clean UOM
    df["UOM"] = (
        df["UOM"].astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.strip()
        .str.upper()
    )
    # parse date
    df["Delivery Date"] = pd.to_datetime(df["Delivery Date"], errors="coerce")
    return df

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    return np.nan if denom == 0 else float(np.abs(y_true - y_pred).sum()) / denom

def monday_week_start(s: pd.Series) -> pd.Series:
    # Normalize any date to its Monday start (00:00)
    return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()

def build_weekly(df: pd.DataFrame) -> pd.DataFrame:
    # Jan–Aug 2025 only, UOM=CS
    mask = (df["UOM"].eq("CS")) & (df["Delivery Date"].between("2025-01-01", "2025-08-31"))
    d = df.loc[mask, ["Customer Group", "Itemcode", "Delivery Date", "Quantity"]].copy()
    d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0.0)
    d["week_start"] = monday_week_start(d["Delivery Date"])

    # Aggregate to weekly by Corporate Customer (Customer Group)
    wk = (
        d.groupby(["Customer Group", "week_start"], dropna=False)["Quantity"]
        .sum()
        .rename("qty")
        .reset_index()
    )

    # Complete weekly calendar per group (zero-fill missing weeks)
    if wk.empty:
        return wk.assign(qty=0.0)

    cal = pd.date_range(wk["week_start"].min(), wk["week_start"].max(), freq="W-MON")
    groups = wk["Customer Group"].drop_duplicates().sort_values()
    full = pd.MultiIndex.from_product(
        [groups, cal], names=["Customer Group", "week_start"]
    ).to_frame(index=False)
    wk_full = (
        full.merge(wk, on=["Customer Group", "week_start"], how="left")
        .assign(qty=lambda d: d["qty"].fillna(0.0))
        .sort_values(["Customer Group", "week_start"])
        .reset_index(drop=True)
    )
    return wk_full

def add_lags_rolls_calendar(wk_full: pd.DataFrame) -> pd.DataFrame:
    def _add_feats(g):
        g = g.sort_values("week_start").copy()
        for k in [1, 2, 3, 4]:
            g[f"lag{k}"] = g["qty"].shift(k)
        for w in [4, 8, 12]:
            g[f"rollmean_{w}"] = g["qty"].shift(1).rolling(w, min_periods=2).mean()
        woy = g["week_start"].dt.isocalendar().week.astype(int)
        g["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
        g["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
        return g

    def _add_more_feats(g):
        g = g.sort_values("week_start").copy()
        wom = ((g["week_start"].dt.day - 1) // 7 + 1).astype(int).clip(1, 5)
        g["wom"] = wom
        g["wom_sin"] = np.sin(2 * np.pi * g["wom"] / 5.0)
        g["wom_cos"] = np.cos(2 * np.pi * g["wom"] / 5.0)
        prev = g["qty"].shift(1)
        g["slope_1"] = prev - g["rollmean_4"]
        g["mom_4_8"] = g["rollmean_4"] - g["rollmean_8"]
        g["ratio_4_8"] = g["rollmean_4"] / np.where(
            g["rollmean_8"].abs() > 1e-9, g["rollmean_8"].abs(), 1e-9
        )
        return g

    X = (
        wk_full.groupby("Customer Group", dropna=False, group_keys=False)
        .apply(_add_feats)
        .reset_index(drop=True)
    )
    X = (
        X.groupby("Customer Group", dropna=False, group_keys=False)
        .apply(_add_more_feats)
        .reset_index(drop=True)
    )
    return X

def split_masks(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    SPLIT_TRAIN_END = pd.Timestamp("2025-07-31")
    VAL_START = pd.Timestamp("2025-08-01")
    VAL_END = pd.Timestamp("2025-08-31")
    mask_train = X["week_start"] <= SPLIT_TRAIN_END
    mask_val = (X["week_start"] >= VAL_START) & (X["week_start"] <= VAL_END)
    return mask_train, mask_val

def august_baseline(X: pd.DataFrame, mask_val: pd.Series) -> Tuple[float, pd.DataFrame]:
    val = X.loc[mask_val, ["Customer Group", "week_start", "qty", "rollmean_4"]].copy()
    val = val.rename(columns={"rollmean_4": "pred_base"})
    overall = wape(val["qty"].to_numpy(), val["pred_base"].to_numpy())
    by_group = (
        val.groupby("Customer Group", dropna=False)
        .apply(lambda g: wape(g["qty"].to_numpy(), g["pred_base"].to_numpy()))
        .reset_index(name="WAPE_base")
        .dropna()
    )
    return overall, by_group

def recency_weights(X: pd.DataFrame, mask_train: pd.Series, decay: float = 0.97) -> pd.DataFrame:
    SPLIT_TRAIN_END = pd.Timestamp("2025-07-31")
    train_df = X.loc[mask_train].copy()
    age_weeks = ((SPLIT_TRAIN_END - train_df["week_start"]).dt.days // 7).clip(lower=0)
    train_df["sample_weight"] = (decay ** age_weeks).astype(float)
    return train_df

def ml_backtest_v2(X: pd.DataFrame, train_df: pd.DataFrame, mask_val: pd.Series, feat_cols: List[str]) -> Tuple[float, pd.DataFrame]:
    # clip negatives (returns/credits) to 0
    X = X.copy()
    X["qty"] = X["qty"].clip(lower=0)
    train_df = train_df.copy()
    train_df["qty"] = train_df["qty"].clip(lower=0)

    val_df = X.loc[mask_val].copy()
    rows, all_y, all_yhat = [], [], []

    for cg, gtr in train_df.groupby("Customer Group", dropna=False):
        gva = val_df[val_df["Customer Group"].eq(cg)].copy()
        if len(gtr) < 20 or len(gva) == 0:
            continue

        y_tr = np.log1p(gtr["qty"].to_numpy())
        w_tr = gtr["sample_weight"].to_numpy()

        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=6,
            learning_rate=0.05,
            max_iter=600,
            l2_regularization=0.0,
        )
        model.fit(gtr[feat_cols], y_tr, sample_weight=w_tr)

        yhat = np.expm1(model.predict(gva[feat_cols]))
        yhat = np.clip(yhat, 0.0, None)

        denom = float(np.abs(gva["qty"]).sum())
        w_ml = np.nan if denom == 0 else float(np.abs(gva["qty"] - yhat).sum()) / denom
        rows.append({"Customer Group": cg, "WAPE_ML_v2": w_ml})

        all_y.append(gva["qty"].to_numpy())
        all_yhat.append(yhat)

    if not all_y:
        return np.nan, pd.DataFrame(columns=["Customer Group", "WAPE_ML_v2"])

    all_y = np.concatenate(all_y)
    all_yhat = np.concatenate(all_yhat)
    overall = np.abs(all_y - all_yhat).sum() / max(np.abs(all_y).sum(), 1e-9)
    ml_by_group_v2 = pd.DataFrame(rows)
    return overall, ml_by_group_v2

def make_feat_from_history_v2(q_series: pd.Series, next_week_start: pd.Timestamp) -> Dict[str, float]:
    vals = q_series.values
    out: Dict[str, float] = {}
    for k in [1, 2, 3, 4]:
        out[f"lag{k}"] = vals[-k] if len(vals) >= k else np.nan
    for w in [4, 8, 12]:
        out[f"rollmean_{w}"] = float(pd.Series(vals).tail(w).mean()) if len(vals) >= 2 else np.nan
    woy = int(pd.Timestamp(next_week_start).isocalendar().week)
    out["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    out["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    wom = int(((pd.Timestamp(next_week_start).day - 1) // 7) + 1)
    wom = max(1, min(5, wom))
    out["wom"] = wom
    out["wom_sin"] = np.sin(2 * np.pi * wom / 5.0)
    out["wom_cos"] = np.cos(2 * np.pi * wom / 5.0)
    prev = vals[-1] if len(vals) >= 1 else np.nan
    rm4 = float(pd.Series(vals).tail(4).mean()) if len(vals) >= 2 else np.nan
    rm8 = float(pd.Series(vals).tail(8).mean()) if len(vals) >= 2 else np.nan
    out["slope_1"] = (prev - rm4) if (np.isfinite(prev) and np.isfinite(rm4)) else np.nan
    out["mom_4_8"] = (rm4 - rm8) if (np.isfinite(rm4) and np.isfinite(rm8)) else np.nan
    denom = rm8 if (rm8 is not None and np.isfinite(rm8) and abs(rm8) > 1e-9) else 1e-9
    out["ratio_4_8"] = (rm4 / denom) if np.isfinite(rm4) else np.nan
    return out

def september_weeks() -> pd.DatetimeIndex:
    return pd.date_range("2025-09-01", "2025-09-30", freq="W-MON")

def baseline_forecast_sept(wk_full: pd.DataFrame) -> pd.DataFrame:
    weeks = list(september_weeks())
    out = []
    hist_cutoff = pd.Timestamp("2025-08-31")
    for cg, g in wk_full.groupby("Customer Group", dropna=False):
        h = g[g["week_start"] <= hist_cutoff].sort_values("week_start")[["week_start", "qty"]]
        q_hist = h["qty"].tolist()
        for wk in weeks:
            if len(q_hist) >= 2:
                base = float(pd.Series(q_hist[-4:]).mean())
            elif len(q_hist) == 1:
                base = float(q_hist[-1])
            else:
                base = 0.0
            out.append({"Customer Group": cg, "week_start": wk, "pred_base": base})
            q_hist.append(base)
    return pd.DataFrame(out)

def ml_forecast_sept_v2(X: pd.DataFrame, wk_full: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    # train up to Aug 31 with recency weights, log1p target, MAE loss
    VAL_END = pd.Timestamp("2025-08-31")
    decay = 0.97
    train = X[X["week_start"] <= VAL_END].copy()
    age_weeks = ((VAL_END - train["week_start"]).dt.days // 7).clip(lower=0)
    train["sample_weight"] = (decay ** age_weeks).astype(float)
    train["qty"] = train["qty"].clip(lower=0)

    weeks = list(september_weeks())
    rows = []
    for cg, gtr in train.groupby("Customer Group", dropna=False):
        if len(gtr) < 20:
            continue

        y_tr = np.log1p(gtr["qty"].to_numpy())
        w_tr = gtr["sample_weight"].to_numpy()
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=6,
            learning_rate=0.05,
            max_iter=600,
            l2_regularization=0.0,
        )
        model.fit(gtr[feat_cols], y_tr, sample_weight=w_tr)

        h = (
            wk_full[(wk_full["Customer Group"].eq(cg)) & (wk_full["week_start"] <= VAL_END)]
            .sort_values("week_start")[["week_start", "qty"]]
            .copy()
        )
        q_hist = h["qty"].clip(lower=0).tolist()

        for wk in weeks:
            f = make_feat_from_history_v2(pd.Series(q_hist), wk)
            xrow = pd.DataFrame([f])[feat_cols]
            yhat = float(np.expm1(model.predict(xrow)[0]))
            yhat = max(0.0, yhat)
            rows.append({"Customer Group": cg, "week_start": wk, "pred_ml": yhat})
            q_hist.append(yhat)

    return pd.DataFrame(rows)

def monthly_sop_from_base(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an S&OP September table per 'Customer Group' from Base sheet P-9 column.
    We deduplicate by (Customer Group, Itemcode) and sum.
    """
    # find P-9 column robustly
    norm_map = {c: str(c).replace("\u00A0"," ").strip().upper().replace("-","").replace("_","") for c in base_df.columns}
    p9_col = None
    for c, n in norm_map.items():
        if n in {"P9", "P09", "PERIOD9"}:
            p9_col = c
            break
    if p9_col is None:
        return pd.DataFrame(columns=["Customer Group", "sop_sept"])

    sop_p9 = base_df[["Customer Group", "Itemcode", p9_col]].copy()
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

# ----------------------------
# UI: Sidebar
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Excel (Jan-Aug_ SOP Sept-Dec.xlsx)", type=["xlsx"])
rel_improve = st.sidebar.slider("ML must beat BASE by at least (relative)", 0.0, 0.20, 0.05, 0.01)
decay = st.sidebar.slider("Recency decay (per week)", 0.90, 0.999, 0.97, 0.001)

st.title("Weekly Forecast — BASE / ML / BEST (September 2025)")

if not uploaded:
    st.info("Upload the Excel file to run the app (sheet: **Base**, with daily actuals Jan–Aug 2025 and S&OP P-9).")
    st.stop()

# ----------------------------
# Pipeline
# ----------------------------
base_df = load_base_excel(uploaded)
wk_full = build_weekly(base_df)
if wk_full.empty:
    st.error("No data after filtering (UOM='CS' & Jan–Aug 2025). Please check the file.")
    st.stop()

X = add_lags_rolls_calendar(wk_full)

# select features + drop early NaNs
feat_cols = [
    "lag1", "lag2", "lag3", "lag4",
    "rollmean_4", "rollmean_8", "rollmean_12",
    "woy_sin", "woy_cos",
    "wom", "wom_sin", "wom_cos",
    "slope_1", "mom_4_8", "ratio_4_8",
]
X_model = X.dropna(subset=feat_cols).copy()

mask_train, mask_val = split_masks(X_model)

# August baseline
overall_base_wape, by_group = august_baseline(X_model, mask_val)

# ML backtest v2 with chosen recency decay
train_df = recency_weights(X_model, mask_train, decay=decay)
overall_ml_wape, ml_by_group_v2 = ml_backtest_v2(X_model, train_df, mask_val, feat_cols)

# winners by relative improvement threshold
winners = (
    ml_by_group_v2.merge(by_group, on="Customer Group", how="inner")
    .assign(
        abs_gain=lambda d: d["WAPE_base"] - d["WAPE_ML_v2"],
        rel_gain=lambda d: (d["WAPE_base"] - d["WAPE_ML_v2"]) / d["WAPE_base"].clip(lower=1e-9),
        winner=lambda d: np.where(
            (d["WAPE_ML_v2"] < d["WAPE_base"]) & (d["rel_gain"] >= rel_improve),
            "ML", "BASE"
        ),
    )
)

# September forecasts
pred_base = baseline_forecast_sept(wk_full)
pred_ml_sept_v2 = ml_forecast_sept_v2(X_model, wk_full, feat_cols)

comb = (
    pred_base.merge(pred_ml_sept_v2, on=["Customer Group", "week_start"], how="left")
    .merge(winners[["Customer Group", "winner"]], on="Customer Group", how="left")
)
comb["winner"] = comb["winner"].fillna("BASE")
comb["pred_best"] = np.where(
    (comb["winner"].eq("ML")) & (comb["pred_ml"].notna()),
    comb["pred_ml"],
    comb["pred_base"],
)

# S&OP monthly compare (optional)
sop_sep = monthly_sop_from_base(base_df)
f_sept = comb.groupby("Customer Group", as_index=False)["pred_best"].sum().rename(columns={"pred_best": "forecast_sept"})
cmp_month = f_sept.merge(sop_sep, on="Customer Group", how="left")
cmp_month["delta"] = cmp_month["forecast_sept"] - cmp_month["sop_sept"]
cmp_month["delta%"] = cmp_month["delta"] / cmp_month["sop_sept"].replace({0: np.nan})

# ----------------------------
# UI: Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "August Backtest", "S&OP Compare (monthly)", "README"])

with tab1:
    st.subheader("Forecast — September 2025")
    c1, c2, c3 = st.columns(3)
    c1.metric("August WAPE — Baseline", f"{overall_base_wape:.3f}")
    c2.metric("August WAPE — ML (v2)", f"{overall_ml_wape:.3f}")
    win_counts = winners["winner"].value_counts() if not winners.empty else pd.Series(dtype=int)
    c3.metric("Winners (ML / BASE)", f"{int(win_counts.get('ML',0))} / {int(win_counts.get('BASE',0))}")

    mode = st.radio("Forecast Mode", ["BEST", "ML", "BASE"], horizontal=True)
    df_show = {
        "BEST": comb.rename(columns={"pred_best": "forecast_qty"})[["Customer Group","week_start","forecast_qty"]],
        "ML":   pred_ml_sept_v2.rename(columns={"pred_ml": "forecast_qty"}),
        "BASE": pred_base.rename(columns={"pred_base": "forecast_qty"}),
    }[mode].copy()
    df_show["week_start"] = pd.to_datetime(df_show["week_start"])

    # Explorer
    top_totals = df_show.groupby("Customer Group")["forecast_qty"].sum().sort_values(ascending=False)
    default_cg = top_totals.index[0] if len(top_totals) else df_show["Customer Group"].iloc[0]
    cg = st.selectbox("Select Customer Group", sorted(df_show["Customer Group"].unique()), index=sorted(df_show["Customer Group"].unique()).index(default_cg))

    cg_base = pred_base[pred_base["Customer Group"].eq(cg)].rename(columns={"pred_base": "BASE"})
    cg_ml   = pred_ml_sept_v2[pred_ml_sept_v2["Customer Group"].eq(cg)].rename(columns={"pred_ml": "ML"})
    cg_best = comb[comb["Customer Group"].eq(cg)][["Customer Group","week_start","pred_best","winner"]].rename(columns={"pred_best":"BEST"})

    viz = (
        cg_base.merge(cg_ml, on=["Customer Group","week_start"], how="outer")
               .merge(cg_best, on=["Customer Group","week_start"], how="outer")
               .sort_values("week_start")
    )
    viz["week_start"] = pd.to_datetime(viz["week_start"]).dt.date

    st.line_chart(viz.set_index("week_start")[["BASE","ML","BEST"]])

    st.dataframe(viz.rename(columns={"week_start":"Week"}), use_container_width=True)

    # Downloads
    def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for name, df in sheets.items():
                _df = df.copy()
                if "week_start" in _df.columns:
                    _df["week_start"] = pd.to_datetime(_df["week_start"]).dt.date
                _df.to_excel(writer, sheet_name=name, index=False)
        buf.seek(0)
        return buf.read()

    st.download_button(
        "Download September forecasts (Excel)",
        data=to_excel_bytes({
            "sept_base": pred_base.rename(columns={"pred_base": "forecast_qty"}),
            "sept_ml_v2": pred_ml_sept_v2.rename(columns={"pred_ml": "forecast_qty"}),
            "sept_best": comb[["Customer Group","week_start","pred_best"]].rename(columns={"pred_best":"forecast_qty"}),
        }),
        file_name="forecast_sept_2025_bestof.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with tab2:
    st.subheader("August Backtest — WAPE and Winners")
    if winners.empty:
        st.info("Not enough data to compute backtest per customer.")
    else:
        comp = (
            winners[["Customer Group","WAPE_base","WAPE_ML_v2","abs_gain","rel_gain","winner"]]
            .sort_values(["winner","rel_gain"], ascending=[True, False])
            .reset_index(drop=True)
        )
        st.dataframe(comp, use_container_width=True)
        st.caption("Winner rule: ML must beat BASE by the chosen relative threshold (sidebar).")

with tab3:
    st.subheader("S&OP Compare — Monthly (P-9 vs September total)")
    if sop_sep.empty:
        st.info("P-9 (September S&OP) column not detected in Base sheet. This view is optional and for comparison only.")
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
        st.dataframe(cmp_view, use_container_width=True)
        st.caption("S&OP is used only for comparison (not as a training input).")

with tab4:
    st.subheader("README / Approach")
    st.markdown("""
**Goal.** Independent weekly forecast of **September 2025** per Corporate Customer (Customer Group) from **Jan–Aug actuals**.  
We deliver **BASE**, **ML**, and **BEST (per-customer winner)** for a clean comparison against client actuals and S&OP.

**Method (short):**
- **Baseline (BASE):** last 4-week mean, applied iteratively.
- **ML (v2):** gradient boosting on weekly features (lags/rolling means, week-of-year, week-of-month, momentum).
  - Training: **Jan–Jul**, validation on **August**.
  - Loss: **MAE-like** (`absolute_error`), **log1p target**, **recency weights** (weeks closer to July weighted more).
  - Safeguard: clip negatives to **0**.
- **Best-of (winner):** use ML only where it beats BASE on August by ≥ threshold (sidebar), else keep BASE.
- **S&OP (P-9):** shown **for comparison only** in the monthly view.

**August results (typical with 8 months of history & no promo/seasonality flags):**
- Baseline WAPE around **0.46**; ML around **0.41**; ML wins on ~**50%** of customers.
- Weekly, per-customer WAPE of **0.40–0.45** is a solid v1; lower requires more signal.

**To improve further:**
1) Add **2024** actuals (annual seasonality).
2) Provide **promo/price/stock/weekly market factor** signals.
3) Optional **per-customer bias calibration** (fit on August) and **intermittent demand** handling for sparse customers.
""")
