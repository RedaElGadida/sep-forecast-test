# streamlit_app.py
import io
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

st.set_page_config(page_title="Weekly Forecast (BASE / ML / BEST)", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    return np.nan if denom == 0 else float(np.abs(y_true - y_pred).sum()) / denom

def monday_week_start(s: pd.Series) -> pd.Series:
    return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()

@st.cache_data(show_spinner=False)
def load_base_excel(file) -> pd.DataFrame:
    # 'Base' sheet, headers start on row 5 (header=4)
    df = pd.read_excel(file, sheet_name="Base", header=4)
    # Normalize key columns
    if "Customer Group" not in df.columns:
        # Try common variants — adjust if your file differs
        for alt in ["Corporate Customer", "Customer", "CustomerGroup"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Customer Group"})
                break
    if "Itemcode" not in df.columns:
        for alt in ["Item Code", "Item", "SKU", "Item Code "]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Itemcode"})
                break
    df["UOM"] = (
        df["UOM"].astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.strip()
        .str.upper()
    )
    df["Delivery Date"] = pd.to_datetime(df["Delivery Date"], errors="coerce")
    # Make sure Quantity is numeric
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    return df

def build_weekly(df: pd.DataFrame, series_cols: List[str]) -> pd.DataFrame:
    # Filter to Jan–Aug 2025 and UOM=CS
    mask = (df["UOM"].eq("CS")) & (df["Delivery Date"].between("2025-01-01", "2025-08-31"))
    base_cols = set(series_cols) | {"Itemcode", "Delivery Date", "Quantity"}
    d = df.loc[mask, list(base_cols)].copy()
    d["Quantity"] = d["Quantity"].fillna(0.0)
    d["week_start"] = monday_week_start(d["Delivery Date"])

    group_cols = series_cols + ["week_start"]
    wk = d.groupby(group_cols, dropna=False)["Quantity"].sum().rename("qty").reset_index()

    if wk.empty:
        return wk.assign(qty=0.0)

    # Complete weekly calendar per series (zero-fill missing weeks)
    cal = pd.date_range(wk["week_start"].min(), wk["week_start"].max(), freq="W-MON")
    keys = wk[series_cols].drop_duplicates().sort_values(series_cols)
    full = keys.assign(_k=1).merge(
        pd.DataFrame({"week_start": cal, "_k": 1}), on="_k"
    ).drop(columns="_k")

    wk_full = (
        full.merge(wk, on=group_cols, how="left")
            .assign(qty=lambda d: d["qty"].fillna(0.0))
            .sort_values(group_cols)
            .reset_index(drop=True)
    )
    return wk_full

def add_lags_rolls_calendar(wk_full: pd.DataFrame, series_cols: List[str]) -> pd.DataFrame:
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
        wk_full.groupby(series_cols, dropna=False, group_keys=False)
               .apply(_add_feats)
               .reset_index(drop=True)
    )
    X = (
        X.groupby(series_cols, dropna=False, group_keys=False)
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

def august_baseline(X: pd.DataFrame, mask_val: pd.Series, series_cols: List[str]):
    val = X.loc[mask_val, series_cols + ["week_start", "qty", "rollmean_4"]].copy()
    val = val.rename(columns={"rollmean_4": "pred_base"})
    overall = wape(val["qty"].to_numpy(), val["pred_base"].to_numpy())
    by_series = (
        val.groupby(series_cols, dropna=False)
           .apply(lambda g: wape(g["qty"].to_numpy(), g["pred_base"].to_numpy()))
           .reset_index(name="WAPE_base").dropna()
    )
    return overall, by_series, val

def recency_weights(X: pd.DataFrame, mask_train: pd.Series, decay: float = 0.97) -> pd.DataFrame:
    SPLIT_TRAIN_END = pd.Timestamp("2025-07-31")
    train_df = X.loc[mask_train].copy()
    age_weeks = ((SPLIT_TRAIN_END - train_df["week_start"]).dt.days // 7).clip(lower=0)
    train_df["sample_weight"] = (decay ** age_weeks).astype(float)
    return train_df

def ml_backtest_v2(
    X: pd.DataFrame,
    train_df: pd.DataFrame,
    mask_val: pd.Series,
    feat_cols: List[str],
    series_cols: List[str],
) -> Tuple[float, pd.DataFrame]:
    X = X.copy()
    X["qty"] = X["qty"].clip(lower=0)
    train_df = train_df.copy()
    train_df["qty"] = train_df["qty"].clip(lower=0)

    val_df = X.loc[mask_val].copy()
    rows, all_y, all_yhat = [], [], []

    for keys, gtr in train_df.groupby(series_cols, dropna=False):
        gva = val_df.copy()
        # filter to this series
        if isinstance(keys, tuple):
            for c, v in zip(series_cols, keys):
                gva = gva[gva[c].eq(v)]
            keys_dict = dict(zip(series_cols, keys))
        else:
            gva = gva[gva[series_cols[0]].eq(keys)]
            keys_dict = {series_cols[0]: keys}

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

        row = {**keys_dict, "WAPE_ML_v2": w_ml}
        rows.append(row)

        all_y.append(gva["qty"].to_numpy())
        all_yhat.append(yhat)

    if not all_y:
        return np.nan, pd.DataFrame(columns=series_cols + ["WAPE_ML_v2"])

    all_y = np.concatenate(all_y)
    all_yhat = np.concatenate(all_yhat)
    overall = np.abs(all_y - all_yhat).sum() / max(np.abs(all_y).sum(), 1e-9)
    ml_by_series_v2 = pd.DataFrame(rows)
    return overall, ml_by_series_v2

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

def _mask_series(df: pd.DataFrame, series_cols: List[str], keys: Union[tuple, object]) -> pd.DataFrame:
    """Return subset of df matching the series keys."""
    out = df
    if isinstance(keys, tuple):
        for c, v in zip(series_cols, keys):
            out = out[out[c].eq(v)]
    else:
        out = out[out[series_cols[0]].eq(keys)]
    return out

def baseline_forecast_sept(wk_full: pd.DataFrame, series_cols: List[str]) -> pd.DataFrame:
    weeks = list(september_weeks())
    out = []
    hist_cutoff = pd.Timestamp("2025-08-31")
    group_cols = series_cols + ["week_start"]

    for keys, g in wk_full.groupby(series_cols, dropna=False):
        h = _mask_series(wk_full[wk_full["week_start"] <= hist_cutoff], series_cols, keys) \
                .sort_values("week_start")[["week_start", "qty"]]
        q_hist = h["qty"].tolist()
        keys_dict = dict(zip(series_cols, keys)) if isinstance(keys, tuple) else {series_cols[0]: keys}
        for wk in weeks:
            if len(q_hist) >= 2:
                base = float(pd.Series(q_hist[-4:]).mean())
            elif len(q_hist) == 1:
                base = float(q_hist[-1])
            else:
                base = 0.0
            out.append({**keys_dict, "week_start": wk, "pred_base": base})
            q_hist.append(base)
    return pd.DataFrame(out)

def ml_forecast_sept_v2(X: pd.DataFrame, wk_full: pd.DataFrame, feat_cols: List[str], series_cols: List[str]) -> pd.DataFrame:
    VAL_END = pd.Timestamp("2025-08-31")
    decay = 0.97
    train = X[X["week_start"] <= VAL_END].copy()
    age_weeks = ((VAL_END - train["week_start"]).dt.days // 7).clip(lower=0)
    train["sample_weight"] = (decay ** age_weeks).astype(float)
    train["qty"] = train["qty"].clip(lower=0)

    weeks = list(september_weeks())
    rows = []

    for keys, gtr in train.groupby(series_cols, dropna=False):
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

        # history up to Aug 31 for this series
        h = _mask_series(wk_full[wk_full["week_start"] <= VAL_END], series_cols, keys) \
                .sort_values("week_start")[["week_start", "qty"]]
        q_hist = h["qty"].clip(lower=0).tolist()

        keys_dict = dict(zip(series_cols, keys)) if isinstance(keys, tuple) else {series_cols[0]: keys}
        for wk in weeks:
            f = make_feat_from_history_v2(pd.Series(q_hist), wk)
            xrow = pd.DataFrame([f])[feat_cols]
            yhat = float(np.expm1(model.predict(xrow)[0]))
            yhat = max(0.0, yhat)
            rows.append({**keys_dict, "week_start": wk, "pred_ml": yhat})
            q_hist.append(yhat)

    return pd.DataFrame(rows)

def monthly_sop_from_base(base_df: pd.DataFrame) -> pd.DataFrame:
    """Build S&OP September table per Customer Group from Base sheet P-9 column."""
    # detect P-9 robustly
    norm = {c: str(c).replace("\u00A0"," ").strip().upper().replace("-","").replace("_","") for c in base_df.columns}
    p9_col = None
    for c, n in norm.items():
        if n in {"P9", "P09", "PERIOD9"}:
            p9_col = c
            break
    if p9_col is None:
        return pd.DataFrame(columns=["Customer Group", "sop_sept"])
    # Deduplicate by (Customer Group, Itemcode) then sum to customer
    cols = [c for c in ["Customer Group", "Itemcode", p9_col] if c in base_df.columns]
    sop_p9 = base_df[cols].copy()
    sop_p9[p9_col] = pd.to_numeric(sop_p9[p9_col], errors="coerce")
    p9_by_item = sop_p9.dropna(subset=[p9_col]).groupby(["Customer Group","Itemcode"], as_index=False)[p9_col].max()
    sop_sep = p9_by_item.groupby("Customer Group", as_index=False)[p9_col].sum().rename(columns={p9_col:"sop_sept"})
    return sop_sep

def series_label(row: pd.Series, series_cols: List[str]) -> str:
    if series_cols == ["Customer Group"]:
        return str(row["Customer Group"])
    cg = str(row["Customer Group"])
    it = str(row["Itemcode"])
    return f"{cg} · {it}"

# ----------------------------
# UI — Sidebar
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Excel (Jan-Aug_ SOP Sept-Dec.xlsx)", type=["xlsx"])

st.sidebar.header("Settings")
agg_level = st.sidebar.radio("Aggregation level", ["Customer", "Customer × Item"], index=0)
SERIES_COLS = ["Customer Group"] if agg_level == "Customer" else ["Customer Group", "Itemcode"]

rel_improve = st.sidebar.slider("ML must beat BASE by at least (relative)", 0.00, 0.20, 0.05, 0.01)
decay = st.sidebar.slider("Recency decay (per week)", 0.90, 0.999, 0.97, 0.001,
                          help="Lower = more weight on recent weeks; higher = more uniform.")

# --- limit to top-N (only for Customer × Item) ---
topN = None
if agg_level == "Customer × Item":
    topN = st.sidebar.slider(
        "Limit to top-N Customer×Item series (by total qty Jan–Aug)",
        min_value=50, max_value=2000, value=300, step=50,
        help="Keeps the biggest series so the app runs fast. Raise later after testing."
    )


st.title("Weekly Forecast — BASE / ML / BEST (September 2025)")

if not uploaded:
    st.info("Upload the Excel file to run the app (sheet **Base**, daily actuals Jan–Aug 2025, S&OP P-9).")
    st.stop()

# ----------------------------
# Pipeline
# ----------------------------
base_df = load_base_excel(uploaded)
wk_full = build_weekly(base_df, SERIES_COLS)

# --- fast mode filter (only applies in Customer × Item mode) ---
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
    st.info(f"Fast mode: training on top {after}/{before} Customer×Item series (by total qty).")

if wk_full.empty:
    st.error("No rows after filtering (UOM='CS' & Jan–Aug 2025). Check the file.")
    st.stop()

X = add_lags_rolls_calendar(wk_full, SERIES_COLS)

feat_cols = [
    "lag1","lag2","lag3","lag4",
    "rollmean_4","rollmean_8","rollmean_12",
    "woy_sin","woy_cos",
    "wom","wom_sin","wom_cos",
    "slope_1","mom_4_8","ratio_4_8",
]
X_model = X.dropna(subset=feat_cols).copy()

mask_train, mask_val = split_masks(X_model)

# August baseline
overall_base_wape, by_series, val_aug = august_baseline(X_model, mask_val, SERIES_COLS)

# ML backtest (v2)
train_df = recency_weights(X_model, mask_train, decay=decay)
overall_ml_wape, ml_by_series_v2 = ml_backtest_v2(X_model, train_df, mask_val, feat_cols, SERIES_COLS)

# Winners (per series)
winners = (
    ml_by_series_v2.merge(by_series, on=SERIES_COLS, how="inner")
    .assign(
        abs_gain=lambda d: d["WAPE_base"] - d["WAPE_ML_v2"],
        rel_gain=lambda d: (d["WAPE_base"] - d["WAPE_ML_v2"]) / d["WAPE_base"].clip(lower=1e-9),
        winner=lambda d: np.where(
            (d["WAPE_ML_v2"] < d["WAPE_base"]) & (d["rel_gain"] >= rel_improve), "ML", "BASE"
        ),
    )
)

# September forecasts
pred_base = baseline_forecast_sept(wk_full, SERIES_COLS)
pred_ml_sept_v2 = ml_forecast_sept_v2(X_model, wk_full, feat_cols, SERIES_COLS)

comb = (
    pred_base.merge(pred_ml_sept_v2, on=SERIES_COLS + ["week_start"], how="left")
    .merge(winners[SERIES_COLS + ["winner"]], on=SERIES_COLS, how="left")
)
comb["winner"] = comb["winner"].fillna("BASE")
comb["pred_best"] = np.where(
    comb["winner"].eq("ML") & comb["pred_ml"].notna(),
    comb["pred_ml"],
    comb["pred_base"],
)

# S&OP monthly compare (always at Customer level)
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
# UI — Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "August Backtest", "S&OP Compare (monthly)", "README"])

with tab1:
    st.subheader(f"Forecast — September 2025  |  Level: {agg_level}")
    c1, c2, c3 = st.columns(3)
    c1.metric("August WAPE — Baseline", f"{overall_base_wape:.3f}" if pd.notna(overall_base_wape) else "n/a")
    c2.metric("August WAPE — ML (v2)", f"{overall_ml_wape:.3f}" if pd.notna(overall_ml_wape) else "n/a")
    wc = winners["winner"].value_counts() if not winners.empty else pd.Series(dtype=int)
    c3.metric("Winners (ML / BASE)", f"{int(wc.get('ML',0))} / {int(wc.get('BASE',0))}")

    mode = st.radio("Forecast Mode", ["BEST", "ML", "BASE"], horizontal=True)
    df_show = {
        "BEST": comb.rename(columns={"pred_best": "forecast_qty"})[SERIES_COLS + ["week_start","forecast_qty"]],
        "ML":   pred_ml_sept_v2.rename(columns={"pred_ml": "forecast_qty"})[SERIES_COLS + ["week_start","forecast_qty"]],
        "BASE": pred_base.rename(columns={"pred_base": "forecast_qty"})[SERIES_COLS + ["week_start","forecast_qty"]],
    }[mode].copy()
    df_show["week_start"] = pd.to_datetime(df_show["week_start"])

    # Series selector
    sel_df = df_show.copy()
    sel_df["series_label"] = sel_df.apply(lambda r: series_label(r, SERIES_COLS), axis=1)
    totals = sel_df.groupby("series_label")["forecast_qty"].sum().sort_values(ascending=False)
    if len(totals) == 0:
        st.info("No series available.")
        st.stop()
    default_label = totals.index[0]
    series_sel = st.selectbox("Select series", sorted(sel_df["series_label"].unique()),
                              index=sorted(sel_df["series_label"].unique()).index(default_label))
    sel_rows = sel_df[sel_df["series_label"].eq(series_sel)].sort_values("week_start")
    viz = sel_rows.copy()
    viz["week_start"] = viz["week_start"].dt.date
    st.line_chart(viz.set_index("week_start")[["forecast_qty"]])

    # Show the three modes side-by-side for the selected series
    def subset_mode(dfm, label):
        t = dfm.copy()
        t["series_label"] = t.apply(lambda r: series_label(r, SERIES_COLS), axis=1)
        return t[t["series_label"].eq(label)][SERIES_COLS + ["week_start", dfm.columns[-1]]].rename(
            columns={dfm.columns[-1]: dfm.columns[-1].replace("pred_", "").replace("best", "BEST").upper()}
        )

    cg_mask = sel_df["series_label"].eq(series_sel)
    base_sel = pred_base.merge(sel_df[cg_mask][SERIES_COLS].drop_duplicates(), on=SERIES_COLS, how="inner") \
                        .rename(columns={"pred_base":"BASE"})
    ml_sel   = pred_ml_sept_v2.merge(sel_df[cg_mask][SERIES_COLS].drop_duplicates(), on=SERIES_COLS, how="inner") \
                              .rename(columns={"pred_ml":"ML"})
    best_sel = comb.merge(sel_df[cg_mask][SERIES_COLS].drop_duplicates(), on=SERIES_COLS, how="inner") \
                   .rename(columns={"pred_best":"BEST"})

    joined = (base_sel.merge(ml_sel,  on=SERIES_COLS + ["week_start"], how="outer")
                     .merge(best_sel[SERIES_COLS + ["week_start","BEST","winner"]],
                            on=SERIES_COLS + ["week_start"], how="outer")
                     .sort_values("week_start"))
    joined["week_start"] = pd.to_datetime(joined["week_start"]).dt.date
    st.dataframe(joined, use_container_width=True)

    # Downloads (Excel)
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
            "sept_base": pred_base.rename(columns={"pred_base":"forecast_qty"}),
            "sept_ml_v2": pred_ml_sept_v2.rename(columns={"pred_ml":"forecast_qty"}),
            "sept_best": comb[SERIES_COLS + ["week_start","pred_best"]].rename(columns={"pred_best":"forecast_qty"}),
        }),
        file_name="forecast_sept_2025_bestof.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with tab2:
    st.subheader(f"August Backtest — WAPE & Winners  |  Level: {agg_level}")
    if winners.empty:
        st.info("Not enough data per series to compute the backtest.")
    else:
        comp = winners[SERIES_COLS + ["WAPE_base","WAPE_ML_v2","abs_gain","rel_gain","winner"]] \
                     .sort_values(["winner","rel_gain"], ascending=[True, False]) \
                     .reset_index(drop=True)
        st.dataframe(comp, use_container_width=True)

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
        st.caption("S&OP is comparison-only (not a training input).")

with tab4:
    st.subheader("README / Approach")
    st.markdown(f"""
**Goal.** Independent weekly forecast of **September 2025** per series  
(**Level:** {agg_level}; toggle in the sidebar). We output **BASE**, **ML**, and **BEST (winner)**.

**Method (short):**
- **BASE:** last 4-week mean, iterative.
- **ML (v2):** gradient boosting on weekly features (lags/rolling means, week-of-year/month, momentum).
  - Train: **Jan–Jul**, validate on **August**.
  - Loss: **MAE-like** (`absolute_error`), **log1p target**, **recency weights** (slider).
  - Negatives clipped to **0**.
- **Best-of:** Use ML only if it beats BASE on August by ≥ threshold (sidebar), else keep BASE.
- **S&OP (P-9):** monthly comparison only (not used in training).

**Typical v1 accuracy with 8 months and no promo/seasonality:**  
Baseline WAPE ≈ 0.46; ML ≈ 0.40–0.42; ML wins on ~50% of series.

**To improve further:** add **2024** actuals (seasonality), **promo/price/stock/weekly market** signals,  
and optional per-series **bias calibration** + intermittent-demand handling.
""")
