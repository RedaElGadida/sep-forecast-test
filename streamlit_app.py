# streamlit_app.py — September Test (SOP baseline + ADP + comparison)
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="September Weekly Forecast – Test", layout="centered")

st.title("September Weekly Forecast – Test")
st.caption("Upload your Excel. The app builds two September forecasts: (1) S&OP-based weekly split, (2) Actuals-driven projection from Jan–Aug; then compares them. Accuracy is shown only if September actuals are present.")

# =========================
# Upload
# =========================
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Please upload the Excel file to continue.")
    st.stop()

try:
    xl = pd.ExcelFile(uploaded)
except Exception as e:
    st.error(f"Could not read workbook: {e}")
    st.stop()

with st.expander("Detected sheets", expanded=False):
    st.write(xl.sheet_names)

# =========================
# Helpers — auto header + column detection
# =========================
def find_header_row(xl_file, sheet, expected_keys, probe_rows=30, search_top=10):
    prv = pd.read_excel(xl_file, sheet_name=sheet, header=None, nrows=probe_rows)
    best_row, best_score = 0, -1
    for i in range(min(search_top, len(prv))):
        row = prv.iloc[i].astype(str).str.strip().str.lower().tolist()
        row_set = set(row)
        score = 0
        for cand_list in expected_keys.values():
            if any(c in row_set for c in cand_list):
                score += 1
        if score > best_score:
            best_row, best_score = i, score
    return best_row, best_score

def load_with_auto_header(xl_file, sheet, expected_keys, fallback_header=0):
    hdr, score = find_header_row(xl_file, sheet, expected_keys)
    df = pd.read_excel(xl_file, sheet_name=sheet, header=hdr if score > 0 else fallback_header)
    df.columns = [str(c).strip() for c in df.columns]
    return df, hdr, score

BASE_KEYS = {
    "date": ["delivery date","date","order date","ship date"],
    "item": ["itemcode","item code","sku"],
    "qty":  ["quantity","units","qty","total sales"],
    "month": ["month"],
    "corp": ["customer group","group","corporate customer","corp customer"]
}
FREQ_KEYS = {
    "corp": ["customers","customer group","group","corporate customer"],
    "seq":  ["sequence","days","pattern"]
}

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    clower = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in clower:
            return df.columns[clower.index(cand.lower())]
    return None

# =========================
# Part B — Parse Base & Frecuency, build S&OP monthly (Sep–Dec) and frequency weights
# =========================
st.header("Data Parsing")

try:
    base_df, base_hdr, base_score = load_with_auto_header(uploaded, "Base", BASE_KEYS)
    freq_df, freq_hdr, freq_score = load_with_auto_header(uploaded, "Frecuency", FREQ_KEYS)
except Exception as e:
    st.error(f"Could not parse required sheets: {e}")
    st.stop()

st.write(f"**Base** header row: {base_hdr}  ·  **Frecuency** header row: {freq_hdr}")

# Detect columns
DATE_COL  = find_col(base_df, BASE_KEYS["date"])
ITEM_COL  = find_col(base_df, BASE_KEYS["item"])
QTY_COL   = find_col(base_df, BASE_KEYS["qty"])
MONTH_COL = find_col(base_df, BASE_KEYS["month"])
CORP_COL  = find_col(base_df, BASE_KEYS["corp"])
FREQ_CORP_COL = find_col(freq_df, FREQ_KEYS["corp"])
FREQ_SEQ_COL  = find_col(freq_df, FREQ_KEYS["seq"])

missing = [name for name, col in [
    ("Date", DATE_COL), ("Item", ITEM_COL), ("Qty", QTY_COL),
    ("Month", MONTH_COL), ("Corp", CORP_COL),
    ("Freq.Corp", FREQ_CORP_COL), ("Freq.Seq", FREQ_SEQ_COL)
] if col is None]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

with st.expander("Detected columns", expanded=False):
    st.write("Base:", [DATE_COL, ITEM_COL, QTY_COL, MONTH_COL, CORP_COL])
    st.write("Frecuency:", [FREQ_CORP_COL, FREQ_SEQ_COL])

# Actuals (with real dates)
dates_all = pd.to_datetime(base_df[DATE_COL], errors="coerce")
actuals = base_df.loc[dates_all.notna(), [DATE_COL, ITEM_COL, QTY_COL, CORP_COL]].copy()
actuals[DATE_COL] = dates_all[dates_all.notna()]
actuals[QTY_COL]  = pd.to_numeric(actuals[QTY_COL], errors="coerce")
st.success(f"Actuals loaded: {len(actuals)} rows")
st.caption(f"Date range: {actuals[DATE_COL].min()} → {actuals[DATE_COL].max()}")

# Parse Month tokens (handles numbers, names, P-9/P-10…)
MONTH_NAME_MAP = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
    "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,
    "sep":9,"sept":9,"september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
}
def parse_month_token(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    m = pd.to_numeric(s, errors="coerce")
    if pd.notna(m):
        v = int(m)
        return v if 1 <= v <= 12 else np.nan
    if s in MONTH_NAME_MAP: return MONTH_NAME_MAP[s]
    if s[:3] in MONTH_NAME_MAP: return MONTH_NAME_MAP[s[:3]]
    m = re.search(r'(\d{1,2})$', s)  # captures 9 from 'P-9'
    return int(m.group(1)) if m and 1 <= int(m.group(1)) <= 12 else np.nan

base_df["_MonthNum"] = base_df[MONTH_COL].apply(parse_month_token)

# S&OP = rows without real date and Month in 9..12
sop_raw = base_df.loc[dates_all.isna() & base_df["_MonthNum"].isin([9,10,11,12]),
                      [CORP_COL, ITEM_COL, "_MonthNum", QTY_COL]].copy()
sop_raw[QTY_COL] = pd.to_numeric(sop_raw[QTY_COL], errors="coerce")

sop_monthly = (sop_raw.dropna(subset=[QTY_COL])
               .groupby([CORP_COL, ITEM_COL, "_MonthNum"], as_index=False)[QTY_COL]
               .sum()
               .rename(columns={CORP_COL:"corp_customer", ITEM_COL:"item", "_MonthNum":"Month", QTY_COL:"SOP_Monthly"}))

st.success(f"S&OP monthly (Sep–Dec) rows: {len(sop_monthly)}")
with st.expander("Preview S&OP monthly", expanded=False):
    st.dataframe(sop_monthly.head(12), use_container_width=True)

# Frequency weights
weekday_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_cols = [c for c in freq_df.columns if any(w in c.lower() for w in weekday_names)]
dot1 = [c for c in weekday_cols if ".1" in c]
use_cols = dot1 if dot1 else weekday_cols

freq_use = freq_df[[FREQ_CORP_COL, FREQ_SEQ_COL] + use_cols].copy()
if use_cols:
    row_sums = freq_use[use_cols].replace({np.nan: 0}).sum(axis=1)
    for c in use_cols:
        freq_use[c] = np.where(row_sums > 0, freq_use[c].fillna(0) / row_sums, 0.0)

seq_to_weekday = {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday",7:"Saturday"}
def parse_sequence(seq):
    allowed = set()
    if pd.isna(seq): return allowed
    for ch in str(seq):
        if ch.isdigit():
            d = int(ch)
            if d in seq_to_weekday: allowed.add(seq_to_weekday[d])
    return allowed

freq_use["AllowedDays"] = freq_use[FREQ_SEQ_COL].apply(parse_sequence)

def build_weights(row):
    allowed = row["AllowedDays"]
    if use_cols:
        weights = {col.split(".")[0].split()[0].capitalize(): float(row[col]) if pd.notna(row[col]) else 0.0
                   for col in use_cols}
        if allowed:
            for wd in weekday_names:
                if wd not in allowed: weights[wd] = 0.0
        total = sum(weights.get(wd,0.0) for wd in weekday_names)
        if total > 0:
            return {wd: weights.get(wd,0.0)/total for wd in weekday_names}
        elif allowed:
            eq = 1.0/len(allowed); return {wd: (eq if wd in allowed else 0.0) for wd in weekday_names}
        else:
            return {wd: 0.0 for wd in weekday_names}
    else:
        if allowed:
            eq = 1.0/len(allowed); return {wd: (eq if wd in allowed else 0.0) for wd in weekday_names}
        return {wd:0.0 for wd in weekday_names}

freq_use["Weights"] = freq_use.apply(build_weights, axis=1)
freq_weights_lookup = dict(zip(freq_use[FREQ_CORP_COL], freq_use["Weights"]))
st.success(f"Frequency profiles loaded: {len(freq_weights_lookup)}")

# =========================
# Part C — September allocation (S&OP baseline)
# =========================
st.header("September Weekly Forecast (S&OP Baseline)")

sep_start, sep_end = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-30")
cal = pd.DataFrame({"date": pd.date_range(sep_start, sep_end, freq="D")})
cal["weekday_name"] = cal["date"].dt.day_name()
cal["week_start"]   = cal["date"] - pd.to_timedelta(cal["date"].dt.weekday, unit="D")
weekday_counts = cal["weekday_name"].value_counts().to_dict()

def get_weights_for_corp(corp):
    w = freq_weights_lookup.get(corp)
    if w is None:
        eq = 1.0/5.0
        w = {d: (eq if d in ["Monday","Tuesday","Wednesday","Thursday","Friday"] else 0.0)
             for d in weekday_names}
    w = {d: float(w.get(d,0.0)) for d in weekday_names}
    total = sum(w.values())
    if total > 0:
        w = {d: v/total for d, v in w.items()}
    return w

sop_sep = sop_monthly[sop_monthly["Month"] == 9].copy()

# Monthly -> daily -> weekly
alloc_rows = []
for _, r in sop_sep.iterrows():
    corp = r["corp_customer"]; item = r["item"]
    monthly_total = float(r["SOP_Monthly"]) if pd.notna(r["SOP_Monthly"]) else 0.0
    if monthly_total <= 0: continue
    w = get_weights_for_corp(corp)
    per_wd_month = {wd: monthly_total * w.get(wd, 0.0) for wd in weekday_names}
    per_wd_day   = {wd: (per_wd_month[wd] / weekday_counts.get(wd, 1)) for wd in weekday_names}
    for _, c in cal.iterrows():
        wd = c["weekday_name"]; qty = per_wd_day.get(wd, 0.0)
        if qty > 0:
            alloc_rows.append({
                "corp_customer": corp,
                "item": item,
                "date": c["date"],
                "weekday": wd,
                "qty_allocated": qty,
                "week_start": c["week_start"]
            })
alloc_daily_sep = pd.DataFrame(alloc_rows)
sep_weekly = (alloc_daily_sep.groupby(["corp_customer","item","week_start"], as_index=False)["qty_allocated"]
              .sum().rename(columns={"qty_allocated":"forecast_weekly"}))

# QA
alloc_sum = sep_weekly.groupby(["corp_customer","item"], as_index=False)["forecast_weekly"].sum()
cmp = sop_sep.merge(alloc_sum, on=["corp_customer","item"], how="left")
cmp["diff"] = cmp["SOP_Monthly"] - cmp["forecast_weekly"]
cmp["abs_diff"] = cmp["diff"].abs()
total_sop   = cmp["SOP_Monthly"].sum()
total_alloc = cmp["forecast_weekly"].sum()
total_abs   = cmp["abs_diff"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("Total S&OP September", f"{total_sop:,.2f}")
c2.metric("Total Allocated September", f"{total_alloc:,.2f}")
c3.metric("Total |Diff|", f"{total_abs:,.6f}")
if (cmp["abs_diff"] > 1e-6).any():
    st.error("Allocation mismatch detected (see sample):")
    st.dataframe(cmp[cmp["abs_diff"] > 1e-6].head(10), use_container_width=True)
else:
    st.success("Allocation integrity ✅ (S&OP total equals allocated total).")

st.dataframe(sep_weekly.sort_values(["corp_customer","item","week_start"]).head(20),
             use_container_width=True)

# =========================
# Accuracy (only if Sept actuals exist)
# =========================
st.subheader("Accuracy (September)")
has_sep_actuals = (dates_all.dropna().between(sep_start, sep_end)).any()

accuracy_table = None
overall_wape = overall_smape = None

def WAPE(y, yhat):
    denom = np.sum(np.abs(y));  return np.nan if denom == 0 else np.sum(np.abs(y - yhat)) / denom
def sMAPE(y, yhat):
    denom = (np.abs(y) + np.abs(yhat)); mask = denom > 0
    return np.nan if mask.sum() == 0 else np.mean(2.0 * np.abs(y - yhat)[mask] / denom[mask])

weekly_actuals = pd.DataFrame(columns=["corp_customer","item","week_start","Actual"])
if has_sep_actuals:
    actuals_sep = base_df.loc[dates_all.between(sep_start, sep_end), [DATE_COL, ITEM_COL, QTY_COL, CORP_COL]].copy()
    actuals_sep[DATE_COL] = pd.to_datetime(actuals_sep[DATE_COL], errors="coerce")
    actuals_sep[QTY_COL]  = pd.to_numeric(actuals_sep[QTY_COL], errors="coerce").fillna(0)
    actuals_sep["week_start"] = actuals_sep[DATE_COL] - pd.to_timedelta(actuals_sep[DATE_COL].dt.weekday, unit="D")
    weekly_actuals = (actuals_sep.groupby([CORP_COL, ITEM_COL, "week_start"], as_index=False)[QTY_COL]
                      .sum().rename(columns={CORP_COL:"corp_customer", ITEM_COL:"item", QTY_COL:"Actual"}))
    joined = sep_weekly.merge(weekly_actuals, on=["corp_customer","item","week_start"], how="inner")
    if len(joined):
        overall_wape  = WAPE(joined["Actual"].values, joined["forecast_weekly"].values)
        overall_smape = sMAPE(joined["Actual"].values, joined["forecast_weekly"].values)
        m1, m2 = st.columns(2)
        m1.metric("WAPE (Sep)",  f"{overall_wape:.3f}")
        m2.metric("sMAPE (Sep)", f"{overall_smape:.3f}")
        accuracy_table = (joined.groupby(["corp_customer","item"], as_index=False)
                          .apply(lambda g: pd.Series({
                              "WAPE":  WAPE(g["Actual"].values, g["forecast_weekly"].values),
                              "sMAPE": sMAPE(g["Actual"].values, g["forecast_weekly"].values),
                              "Vol":   g["Actual"].sum()
                          })).sort_values("Vol", ascending=False).reset_index(drop=True))
        with st.expander("Accuracy by (Corporate, Item)", expanded=False):
            st.dataframe(accuracy_table.head(30), use_container_width=True)
else:
    st.caption("Forecast built from S&OP + Frequency. Download below to compare offline.")

# =========================
# Mini chart — weekly Forecast vs Actual (selected pair)
# =========================
st.subheader("Mini Chart (Weekly: Forecast vs Actual)")
pairs_for_chart = sep_weekly[["corp_customer","item"]].drop_duplicates().sort_values(["corp_customer","item"])
if len(pairs_for_chart) == 0:
    st.info("No September forecast pairs to visualize.")
else:
    corp_list = pairs_for_chart["corp_customer"].unique().tolist()
    sel_corp = st.selectbox("Select Corporate Customer", corp_list)
    item_list = pairs_for_chart.loc[pairs_for_chart["corp_customer"] == sel_corp, "item"].unique().tolist()
    sel_item = st.selectbox("Select Item", item_list)

    f_sel = sep_weekly[(sep_weekly["corp_customer"] == sel_corp) & (sep_weekly["item"] == sel_item)].copy()
    plot_df = f_sel[["week_start","forecast_weekly"]].set_index("week_start").sort_index()
    plot_df.rename(columns={"forecast_weekly":"Forecast (S&OP Split)"}, inplace=True)

    if len(weekly_actuals):
        a_sel = weekly_actuals[(weekly_actuals["corp_customer"] == sel_corp) & (weekly_actuals["item"] == sel_item)]
        if len(a_sel):
            a_sel = a_sel[["week_start","Actual"]].set_index("week_start").sort_index()
            plot_df = plot_df.join(a_sel, how="left")

    st.line_chart(plot_df)
    st.dataframe(plot_df.reset_index().rename(columns={"week_start":"Week Start"}),
                 use_container_width=True)

# =========================
# Actuals-driven Forecast (ADP) for September + Comparison vs S&OP
# =========================
st.header("September Weekly Forecast (Actuals-Driven) + Comparison")
# Weekly actuals up to Aug-31
act = base_df.loc[dates_all.notna(), [DATE_COL, ITEM_COL, QTY_COL, CORP_COL]].copy()
act[DATE_COL] = dates_all[dates_all.notna()]
act[QTY_COL]  = pd.to_numeric(act[QTY_COL], errors="coerce").fillna(0)
act["week_start"] = act[DATE_COL] - pd.to_timedelta(act[DATE_COL].dt.weekday, unit="D")
weekly_all = (act.groupby([CORP_COL, ITEM_COL, "week_start"], as_index=False)[QTY_COL]
              .sum().rename(columns={CORP_COL:"corp_customer", ITEM_COL:"item", QTY_COL:"actual"}))
train_cutoff = pd.Timestamp("2025-08-31")
train_wk = weekly_all[weekly_all["week_start"] <= train_cutoff].copy()

# September week grid
sep_weeks = sorted(cal["week_start"].unique())
k_weeks = len(sep_weeks)

# Item-level fallback: mean of last 8 weeks across all customers for that item
item_week = (train_wk.groupby(["item","week_start"], as_index=False)["actual"].sum()
             .sort_values("week_start"))
item_last8_mean = (item_week.groupby("item")["actual"]
                   .apply(lambda s: s.tail(8).mean() if len(s) else 0.0)).to_dict()

def forecast_next_weeks(series_df, item, k=k_weeks):
    s = series_df.sort_values("week_start")["actual"].to_numpy()
    n = len(s)
    if n >= 8:
        last8 = s[-8:]
        level = float(last8.mean())
        x = np.arange(len(last8))
        slope = float(np.polyfit(x, last8, 1)[0])
        return [max(0.0, level + slope*(i+1)) for i in range(k)]
    elif n >= 4:
        level = float(s[-4:].mean())
        return [max(0.0, level)] * k
    elif n >= 1:
        level = float(s.mean())
        return [max(0.0, level)] * k
    else:
        level = float(item_last8_mean.get(item, 0.0))
        return [max(0.0, level)] * k

# Forecast only pairs present in S&OP Sep (so comparison is aligned)
pairs = sop_sep[["corp_customer","item"]].drop_duplicates()
adp_rows = []
for _, p in pairs.iterrows():
    cc, it = p["corp_customer"], p["item"]
    hist = train_wk[(train_wk["corp_customer"] == cc) & (train_wk["item"] == it)][["week_start","actual"]]
    preds = forecast_next_weeks(hist, it, k=k_weeks)
    for wk, yhat in zip(sep_weeks, preds):
        adp_rows.append({"corp_customer": cc, "item": it, "week_start": wk, "forecast_weekly_adp": yhat})
adp_weekly = pd.DataFrame(adp_rows)

# Totals
tot_adp = adp_weekly["forecast_weekly_adp"].sum()
tot_sop = sep_weekly["forecast_weekly"].sum()
m1, m2 = st.columns(2)
m1.metric("Total ADP (from actuals) – Sep", f"{tot_adp:,.2f}")
m2.metric("Total S&OP baseline – Sep",       f"{tot_sop:,.2f}")

# Comparison table
compare = (sep_weekly.rename(columns={"forecast_weekly":"forecast_weekly_sop"})
           .merge(adp_weekly, on=["corp_customer","item","week_start"], how="outer"))
for col in ["forecast_weekly_sop","forecast_weekly_adp"]:
    compare[col] = compare[col].fillna(0.0)
compare["delta"]     = compare["forecast_weekly_adp"] - compare["forecast_weekly_sop"]
compare["delta_pct"] = np.where(compare["forecast_weekly_sop"]>0,
                                compare["delta"]/compare["forecast_weekly_sop"], np.nan)

st.subheader("Comparison: ADP vs S&OP (Weekly, September)")
st.dataframe(compare.sort_values(["corp_customer","item","week_start"]).head(30),
             use_container_width=True)

# =========================
# Export to Excel (Summary + Sep_Weekly_SOP + Sep_Weekly_ADP + Compare + Accuracy if available)
# =========================
from io import BytesIO
output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    # Summary
    summary_rows = {
        "Total_SOP_Sep": sep_weekly["forecast_weekly"].sum(),
        "Total_ADP_Sep": adp_weekly["forecast_weekly_adp"].sum(),
    }
    if overall_wape is not None:
        summary_rows["WAPE_Sep"]  = overall_wape
    if overall_smape is not None:
        summary_rows["sMAPE_Sep"] = overall_smape
    pd.DataFrame([summary_rows]).to_excel(writer, sheet_name="Summary", index=False)

    # S&OP baseline weekly
    sep_weekly.rename(columns={"forecast_weekly":"forecast_weekly_sop"}) \
              .sort_values(["corp_customer","item","week_start"]) \
              .to_excel(writer, sheet_name="Sep_Weekly_SOP", index=False)

    # Actuals-driven weekly
    adp_weekly.sort_values(["corp_customer","item","week_start"]) \
              .to_excel(writer, sheet_name="Sep_Weekly_ADP", index=False)

    # Comparison
    compare.sort_values(["corp_customer","item","week_start"]) \
           .to_excel(writer, sheet_name="Compare_SOP_vs_ADP", index=False)

    # Accuracy (optional)
    if accuracy_table is not None and len(accuracy_table) > 0:
        accuracy_table.to_excel(writer, sheet_name="Accuracy_By_Series", index=False)

data = output.getvalue()
st.download_button(
    label="⬇️ Download Excel",
    data=data,
    file_name="September_Forecast_Test.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
