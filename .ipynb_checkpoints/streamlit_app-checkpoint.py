# streamlit_app.py — September Test MVP (A+B)
import re
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="September Weekly Forecast – Test", layout="centered")

st.title("September Weekly Forecast – Test")
st.caption("Upload your Excel. The app reads Base & Frecuency, extracts Sep–Dec S&OP, and prepares frequency profiles. No placeholders.")

# ---------------------------
# File upload
# ---------------------------
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Please upload the Excel file to continue.")
    st.stop()

# Try opening workbook
try:
    xl = pd.ExcelFile(uploaded)
except Exception as e:
    st.error(f"Could not read workbook: {e}")
    st.stop()

st.success("File uploaded.")
with st.expander("Detected sheets", expanded=False):
    st.write(xl.sheet_names)

# ---------------------------
# Helpers for Part B (parsing)
# ---------------------------
def find_header_row(xl_file, sheet, expected_keys, probe_rows=30, search_top=10):
    """Find a likely header row by scoring first rows against expected column names."""
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
    # case-insensitive fallback
    clower = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in clower:
            return df.columns[clower.index(cand.lower())]
    return None

# ---------------------------
# Part B — Parse Base & Frecuency, build S&OP monthly table (Sep–Dec), build frequency weights
# ---------------------------
st.header("Data Parsing")

# Load sheets
try:
    base_df, base_hdr, base_score = load_with_auto_header(uploaded, "Base", BASE_KEYS)
    freq_df, freq_hdr, freq_score = load_with_auto_header(uploaded, "Frecuency", FREQ_KEYS)
except Exception as e:
    st.error(f"Could not parse required sheets: {e}")
    st.stop()

st.write(f"**Base** header row: {base_hdr}  ·  **Frecuency** header row: {freq_hdr}")

# Detect columns in Base
DATE_COL  = find_col(base_df, BASE_KEYS["date"])
ITEM_COL  = find_col(base_df, BASE_KEYS["item"])
QTY_COL   = find_col(base_df, BASE_KEYS["qty"])
MONTH_COL = find_col(base_df, BASE_KEYS["month"])
CORP_COL  = find_col(base_df, BASE_KEYS["corp"])

# Detect columns in Frecuency
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
dates = pd.to_datetime(base_df[DATE_COL], errors="coerce")
actuals = base_df.loc[dates.notna(), [DATE_COL, ITEM_COL, QTY_COL, CORP_COL]].copy()
actuals[DATE_COL] = dates[dates.notna()]
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
    if s in MONTH_NAME_MAP:
        return MONTH_NAME_MAP[s]
    if s[:3] in MONTH_NAME_MAP:
        return MONTH_NAME_MAP[s[:3]]
    m = re.search(r'(\d{1,2})$', s)  # captures 9 from 'P-9'
    return int(m.group(1)) if m and 1 <= int(m.group(1)) <= 12 else np.nan

base_df["_MonthNum"] = base_df[MONTH_COL].apply(parse_month_token)

# S&OP rows = no real date + Month in {9..12}
sop_raw = base_df.loc[dates.isna() & base_df["_MonthNum"].isin([9,10,11,12]),
                      [CORP_COL, ITEM_COL, "_MonthNum", QTY_COL]].copy()
sop_raw[QTY_COL] = pd.to_numeric(sop_raw[QTY_COL], errors="coerce")

sop_monthly = (sop_raw.dropna(subset=[QTY_COL])
               .groupby([CORP_COL, ITEM_COL, "_MonthNum"], as_index=False)[QTY_COL]
               .sum()
               .rename(columns={CORP_COL: "corp_customer",
                                ITEM_COL: "item",
                                "_MonthNum": "Month",
                                QTY_COL: "SOP_Monthly"}))

st.success(f"S&OP monthly (Sep–Dec) rows: {len(sop_monthly)}")
with st.expander("Preview S&OP monthly", expanded=False):
    st.dataframe(sop_monthly.head(12), use_container_width=True)

# Build frequency weights from Frecuency
weekday_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_cols = [c for c in freq_df.columns if any(w in c.lower() for w in weekday_names)]
dot1 = [c for c in weekday_cols if ".1" in c]
use_cols = dot1 if dot1 else weekday_cols  # prefer the .1 set if present

freq_use = freq_df[[FREQ_CORP_COL, FREQ_SEQ_COL] + use_cols].copy()

# normalize weekday shares row-wise to 1.0 (if present)
if use_cols:
    row_sums = freq_use[use_cols].replace({np.nan: 0}).sum(axis=1)
    for c in use_cols:
        freq_use[c] = np.where(row_sums > 0, freq_use[c].fillna(0) / row_sums, 0.0)

seq_to_weekday = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
def parse_sequence(seq):
    allowed = set()
    if pd.isna(seq): return allowed
    for ch in str(seq):
        if ch.isdigit():
            d = int(ch)
            if d in seq_to_weekday:
                allowed.add(seq_to_weekday[d])
    return allowed

freq_use["AllowedDays"] = freq_use[FREQ_SEQ_COL].apply(parse_sequence)

def build_weights(row):
    allowed = row["AllowedDays"]
    if use_cols:
        weights = {col.split(".")[0].split()[0].capitalize(): float(row[col]) if pd.notna(row[col]) else 0.0
                   for col in use_cols}
        if allowed:
            for wd in weekday_names:
                if wd not in allowed:
                    weights[wd] = 0.0
        total = sum(weights.get(wd, 0.0) for wd in weekday_names)
        if total > 0:
            return {wd: weights.get(wd, 0.0) / total for wd in weekday_names}
        elif allowed:
            eq = 1.0 / len(allowed)
            return {wd: (eq if wd in allowed else 0.0) for wd in weekday_names}
        else:
            return {wd: 0.0 for wd in weekday_names}
    else:
        if allowed:
            eq = 1.0 / len(allowed)
            return {wd: (eq if wd in allowed else 0.0) for wd in weekday_names}
        return {wd: 0.0 for wd in weekday_names}

freq_use["Weights"] = freq_use.apply(build_weights, axis=1)
freq_weights_lookup = dict(zip(freq_use[FREQ_CORP_COL], freq_use["Weights"]))

st.success(f"Frequency profiles loaded: {len(freq_weights_lookup)}")

st.info("Parsing complete. Next you can add September allocation (S&OP → daily via weekday weights → weekly).")

# =========================
# Part C — September allocation (+ QA, + accuracy if Sep actuals exist) and Export
# =========================
st.header("September Weekly Forecast")

# 1) Build September 2025 calendar (Monday-based weeks)
sep_start, sep_end = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-30")
cal = pd.DataFrame({"date": pd.date_range(sep_start, sep_end, freq="D")})
cal["weekday_name"] = cal["date"].dt.day_name()                       # Monday..Sunday
cal["week_start"]   = cal["date"] - pd.to_timedelta(cal["date"].dt.weekday, unit="D")
weekday_counts = cal["weekday_name"].value_counts().to_dict()

weekday_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def get_weights_for_corp(corp):
    w = freq_weights_lookup.get(corp)
    if w is None:
        # fallback: equal split Mon–Fri
        eq = 1.0/5.0
        w = {d: (eq if d in ["Monday","Tuesday","Wednesday","Thursday","Friday"] else 0.0)
             for d in weekday_names}
    # ensure all keys + normalize
    w = {d: float(w.get(d, 0.0)) for d in weekday_names}
    total = sum(w.values())
    if total > 0:
        w = {d: v/total for d, v in w.items()}
    return w

# 2) Filter S&OP to September and standardize column names for allocation
if "corp_customer" not in sop_monthly.columns:
    sop_monthly = sop_monthly.rename(columns={"Customer Group":"corp_customer",
                                              "Itemcode":"item",
                                              "SOP_Monthly":"SOP_Monthly"})
sop_sep = sop_monthly[sop_monthly["Month"] == 9].copy()

# 3) Allocate monthly → daily → weekly
alloc_rows = []
for _, r in sop_sep.iterrows():
    corp = r["corp_customer"]
    item = r["item"]
    monthly_total = float(r["SOP_Monthly"]) if pd.notna(r["SOP_Monthly"]) else 0.0
    if monthly_total <= 0:
        continue

    w = get_weights_for_corp(corp)  # weekday -> share (sum=1)
    per_wd_month = {wd: monthly_total * w.get(wd, 0.0) for wd in weekday_names}
    per_wd_day   = {wd: (per_wd_month[wd] / weekday_counts.get(wd, 1)) for wd in weekday_names}

    for _, c in cal.iterrows():
        wd = c["weekday_name"]
        qty = per_wd_day.get(wd, 0.0)
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

sep_weekly = (alloc_daily_sep
              .groupby(["corp_customer","item","week_start"], as_index=False)["qty_allocated"]
              .sum()
              .rename(columns={"qty_allocated":"forecast_weekly"}))

# 4) QA — totals must match S&OP monthly
alloc_sum = sep_weekly.groupby(["corp_customer","item"], as_index=False)["forecast_weekly"].sum()
cmp = sop_sep.merge(alloc_sum, on=["corp_customer","item"], how="left")
cmp["diff"] = cmp["SOP_Monthly"] - cmp["forecast_weekly"]
cmp["abs_diff"] = cmp["diff"].abs()

total_sop   = cmp["SOP_Monthly"].sum()
total_alloc = cmp["forecast_weekly"].sum()
total_abs   = cmp["abs_diff"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total S&OP September", f"{total_sop:,.2f}")
col2.metric("Total Allocated September", f"{total_alloc:,.2f}")
col3.metric("Total |Diff|", f"{total_abs:,.6f}")

mismatches = cmp[cmp["abs_diff"] > 1e-6]
if len(mismatches) > 0:
    st.error(f"Mismatched pairs: {len(mismatches)} (see sample below)")
    st.dataframe(mismatches.head(10), use_container_width=True)
else:
    st.success("Allocation integrity ✅ (S&OP total equals allocated total).")

st.subheader("September Weekly Forecast (preview)")
st.dataframe(sep_weekly.sort_values(["corp_customer","item","week_start"]).head(20),
             use_container_width=True)

# 5) Accuracy (auto) — if Base contains September actuals with real dates, compute weekly accuracy
dates_all = pd.to_datetime(base_df[DATE_COL], errors="coerce")
has_sep_actuals = (dates_all.dropna().between(sep_start, sep_end)).any()

accuracy_table = None
overall_wape = None
overall_smape = None

def WAPE(y, yhat):
    denom = np.sum(np.abs(y))
    return np.nan if denom == 0 else np.sum(np.abs(y - yhat)) / denom

def sMAPE(y, yhat):
    denom = (np.abs(y) + np.abs(yhat))
    mask = denom > 0
    return np.nan if mask.sum() == 0 else np.mean(2.0 * np.abs(y - yhat)[mask] / denom[mask])

st.subheader("Accuracy (September)")
if has_sep_actuals:
    actuals_sep = base_df.loc[dates_all.between(sep_start, sep_end), [DATE_COL, ITEM_COL, QTY_COL, CORP_COL]].copy()
    actuals_sep[DATE_COL] = pd.to_datetime(actuals_sep[DATE_COL], errors="coerce")
    actuals_sep[QTY_COL]  = pd.to_numeric(actuals_sep[QTY_COL], errors="coerce").fillna(0)
    actuals_sep["week_start"] = actuals_sep[DATE_COL] - pd.to_timedelta(actuals_sep[DATE_COL].dt.weekday, unit="D")
    weekly_actuals = (actuals_sep
                      .groupby([CORP_COL, ITEM_COL, "week_start"], as_index=False)[QTY_COL]
                      .sum()
                      .rename(columns={CORP_COL:"corp_customer", ITEM_COL:"item", QTY_COL:"actual"}))

    joined = sep_weekly.merge(weekly_actuals, on=["corp_customer","item","week_start"], how="inner")
    if len(joined) == 0:
        st.warning("September actuals detected, but no overlapping (corp,item,week) with forecast.")
    else:
        overall_wape  = WAPE(joined["actual"].values, joined["forecast_weekly"].values)
        overall_smape = sMAPE(joined["actual"].values, joined["forecast_weekly"].values)

        c1, c2 = st.columns(2)
        c1.metric("WAPE (Sep)",  f"{overall_wape:.3f}")
        c2.metric("sMAPE (Sep)", f"{overall_smape:.3f}")

        accuracy_table = (joined
                          .groupby(["corp_customer","item"], as_index=False)
                          .apply(lambda g: pd.Series({
                              "WAPE":  WAPE(g["actual"].values, g["forecast_weekly"].values),
                              "sMAPE": sMAPE(g["actual"].values, g["forecast_weekly"].values),
                              "Vol":   g["actual"].sum()
                          }))
                          .sort_values("Vol", ascending=False)
                          .reset_index(drop=True))
        with st.expander("Accuracy by (Corporate, Item)", expanded=False):
            st.dataframe(accuracy_table.head(30), use_container_width=True)
else:
    st.info("No September actuals detected in the Base sheet (dates 2025-09-01 to 2025-09-30). "
            "Upload a file with September daily actuals to compute accuracy.")

# 5b) Optional mini chart — weekly A vs F for a selected (corp, item)
st.subheader("Mini Chart (Weekly: Forecast vs Actual)")

# Build the selectable list from the forecast table
pairs = sep_weekly[["corp_customer","item"]].drop_duplicates().sort_values(["corp_customer","item"])
if len(pairs) == 0:
    st.info("No September forecast pairs to visualize.")
else:
    # Select corporate first, then item filtered by corporate
    corp_list = pairs["corp_customer"].unique().tolist()
    sel_corp = st.selectbox("Select Corporate Customer", corp_list)

    item_list = pairs.loc[pairs["corp_customer"] == sel_corp, "item"].unique().tolist()
    sel_item = st.selectbox("Select Item", item_list)

    # Filter forecast table
    f_sel = sep_weekly[(sep_weekly["corp_customer"] == sel_corp) &
                       (sep_weekly["item"] == sel_item)].copy()

    # Prepare a plotting frame with week_start as index
    plot_df = f_sel[["week_start","forecast_weekly"]].set_index("week_start").sort_index()
    plot_df.rename(columns={"forecast_weekly":"Forecast"}, inplace=True)

    # If we computed weekly actuals for September, merge them in
    if "weekly_actuals" not in locals():
        # compute only if has_sep_actuals and we didn't already build weekly_actuals above
        if has_sep_actuals:
            actuals_sep = base_df.loc[dates_all.between(sep_start, sep_end),
                                      [DATE_COL, ITEM_COL, QTY_COL, CORP_COL]].copy()
            actuals_sep[DATE_COL] = pd.to_datetime(actuals_sep[DATE_COL], errors="coerce")
            actuals_sep[QTY_COL]  = pd.to_numeric(actuals_sep[QTY_COL], errors="coerce").fillna(0)
            actuals_sep["week_start"] = actuals_sep[DATE_COL] - pd.to_timedelta(actuals_sep[DATE_COL].dt.weekday, unit="D")
            weekly_actuals = (actuals_sep
                              .groupby([CORP_COL, ITEM_COL, "week_start"], as_index=False)[QTY_COL]
                              .sum()
                              .rename(columns={CORP_COL:"corp_customer", ITEM_COL:"item", QTY_COL:"Actual"}))
        else:
            weekly_actuals = pd.DataFrame(columns=["corp_customer","item","week_start","Actual"])

    if len(weekly_actuals):
        a_sel = weekly_actuals[(weekly_actuals["corp_customer"] == sel_corp) &
                               (weekly_actuals["item"] == sel_item)].copy()
        a_sel = a_sel[["week_start","Actual"]].set_index("week_start").sort_index()
        plot_df = plot_df.join(a_sel, how="left")

    # Show the chart (lines). If there are NaNs for Actual, Streamlit just won’t plot that series.
    st.line_chart(plot_df)

    # Also show the numeric table for the selected pair
    st.dataframe(plot_df.reset_index().rename(columns={"week_start":"Week Start"}), use_container_width=True)


# 6) Export to Excel (Summary + Sep_Weekly + optional Accuracy)
from io import BytesIO
output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    # Summary
    summary_rows = {
        "Total_SOP_Sep": total_sop,
        "Total_Allocated_Sep": total_alloc,
        "TotalAbsDiff": total_abs,
    }
    if overall_wape is not None:
        summary_rows["WAPE_Sep"]  = overall_wape
    if overall_smape is not None:
        summary_rows["sMAPE_Sep"] = overall_smape
    pd.DataFrame([summary_rows]).to_excel(writer, sheet_name="Summary", index=False)

    # Sep weekly forecast
    sep_weekly.sort_values(["corp_customer","item","week_start"]).to_excel(
        writer, sheet_name="Sep_Weekly", index=False
    )

    # Accuracy table if we computed it
    if accuracy_table is not None and len(accuracy_table) > 0:
        accuracy_table.to_excel(writer, sheet_name="Accuracy_By_Series", index=False)

data = output.getvalue()
st.download_button(
    label="⬇️ Download Excel",
    data=data,
    file_name="September_Forecast_Test.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
