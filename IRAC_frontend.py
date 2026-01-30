import io
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
from datetime import datetime, timedelta

# --------------- CONFIG / CONSTANTS --------------- #

APP_TITLE = "IRAC - Inventory Risk & Availability Control"
RELEASE_VERSION = "v 0.50"
RELEASE_DATE = "Released Feb 2026"

DEFAULT_COMPANIES = [
    {"company_id": "COMPANY_A", "name": "Global Tech Industries"},
    {"company_id": "COMPANY_B", "name": "Prime Logistics Corp"},
]

DEFAULT_COMPANY_CONFIG = {
    "time_bucket": "month",
    "history_months": 24,
    "forecast_months": 18,
    "planning_horizon_months": 6,
    "service_levels": {"default": 0.95},
    "risk_thresholds": {
        "shortage_days": 2,    # Very critical
        "attention_days": 10,  # Warning zone
        "excess_days": 90,
    },
    "aggregation": {
        "default_level": "location",
    },
}

# --------------- UTILS --------------- #

def month_range(end_date: datetime, months_back: int) -> pd.DatetimeIndex:
    end = pd.Timestamp(end_date).normalize() + pd.offsets.MonthEnd(0)
    dates = pd.date_range(end - pd.DateOffset(months=months_back - 1), end, freq="M")
    return dates

def gen_seasonal_demand_series(periods, base_level, seasonal_amplitude=0.3, noise_std=0.15, random_state=None):
    rng = np.random.default_rng(random_state)
    t = np.arange(periods)
    seasonal = 1.0 + seasonal_amplitude * np.sin(2 * np.pi * t / 12.0)
    noise = rng.normal(loc=0.0, scale=noise_std, size=periods)
    series = base_level * seasonal * (1.0 + noise)
    series = np.clip(series, a_min=0, a_max=None)
    return series.round().astype(int)

def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None: return pd.DataFrame()
    return pd.read_csv(uploaded_file)

def compute_avg_daily_demand(df_demand: pd.DataFrame) -> pd.DataFrame:
    if df_demand is None or df_demand.empty: return pd.DataFrame()
    df = df_demand.copy()
    df["date"] = pd.to_datetime(df["date"])
    agg = (
        df.groupby(["material_id", "location_id"])
        .agg(
            total_demand=("qty_demand", "sum"),
            days_span=("date", lambda x: (x.max() - x.min()).days + 1),
        )
        .reset_index()
    )
    agg["avg_daily_demand"] = agg["total_demand"] / agg["days_span"].replace(0, 1)
    return agg

def classify_risk(df_inventory, avg_daily, config):
    if df_inventory is None or df_inventory.empty: return pd.DataFrame()
    df = df_inventory.copy()
    df = df.merge(
        avg_daily[["material_id", "location_id", "avg_daily_demand"]],
        on=["material_id", "location_id"],
        how="left",
    )
    df["avg_daily_demand"].fillna(0.0, inplace=True)
    
    # Coverage Calculation
    df["coverage_days"] = np.where(
        df["avg_daily_demand"] > 0,
        df["qty_on_hand"] / df["avg_daily_demand"],
        np.inf,
    )
    
    # Ideal Stock (Simplified: 30 days)
    df["ideal_stock_qty"] = df["avg_daily_demand"] * 30 
    
    thr = config["risk_thresholds"]
    def risk_label(row):
        cov = row["coverage_days"]
        if cov <= thr["shortage_days"]: return "RED"
        elif cov <= thr["attention_days"]: return "YELLOW"
        else: return "GREEN"
    
    df["risk_status"] = df.apply(risk_label, axis=1)
    return df

# --------------- DEMO DATA GENERATION --------------- #

def generate_demo_data(company_id, config=None, seed=None):
    if config is None: config = DEFAULT_COMPANY_CONFIG
    
    # Fully Random RNG
    rng = np.random.default_rng(seed)

    # Random Dimensions
    n_materials = rng.integers(40, 120)
    n_locations = rng.integers(3, 8)

    # Materials
    materials = []
    for i in range(n_materials):
        mid = f"MAT_{i:03d}"
        abc = rng.choice(["A", "B", "C"], p=[0.15, 0.55, 0.3])
        materials.append({
            "company_id": company_id,
            "material_id": mid,
            "material_desc": f"Part {mid} - {rng.choice(['Pro', 'Eco', 'Lite', 'Max'])}",
            "uom": "EA",
            "material_group": f"GRP_{rng.integers(0, 5)}",
            "abc_class": abc,
            "unit_price": round(rng.uniform(5, 800), 2)
        })

    # Locations
    locations = []
    for j in range(n_locations):
        lid = f"LOC_{j:02d}"
        locations.append({
            "company_id": company_id,
            "location_id": lid,
            "location_name": f"WH {lid} ({rng.choice(['Hub', 'Spoke'])})",
            "location_type": rng.choice(["PLANT", "DC"]),
            "region": f"REGION_{j % 3}",
        })

    df_materials = pd.DataFrame(materials)
    df_locations = pd.DataFrame(locations)

    today = datetime.today().date()
    hist_dates = month_range(today, config.get("history_months", 24))
    fcst_dates = month_range(today + pd.DateOffset(months=18), 18)

    demand_rows, forecast_rows = [], []
    
    # Demand & Forecast
    for _, m in df_materials.iterrows():
        for _, l in df_locations.iterrows():
            base = rng.uniform(10, 800)
            
            d_series = gen_seasonal_demand_series(len(hist_dates), base, 0.3, 0.2, rng.integers(1, 1e9))
            for d, q in zip(hist_dates, d_series):
                demand_rows.append({"company_id": company_id, "material_id": m["material_id"], "location_id": l["location_id"], "date": d.date(), "qty_demand": q})
            
            f_series = gen_seasonal_demand_series(len(fcst_dates), base * rng.uniform(0.9, 1.2), 0.3, 0.1, rng.integers(1, 1e9))
            for d, q in zip(fcst_dates, f_series):
                forecast_rows.append({"company_id": company_id, "material_id": m["material_id"], "location_id": l["location_id"], "date": d.date(), "qty_forecast": q})

    df_demand = pd.DataFrame(demand_rows)
    df_forecast = pd.DataFrame(forecast_rows)

    # Inventory Snapshot
    inv_rows = []
    snapshot_date = today
    avg_dem = df_demand.groupby(["material_id", "location_id"])["qty_demand"].mean().reset_index()

    for _, row in avg_dem.iterrows():
        mid, lid = row["material_id"], row["location_id"]
        amd = row["qty_demand"]
        
        # Risk Logic: 15% Red, 20% Yellow, 65% Green (with some outliers)
        risk_roll = rng.random()
        
        if risk_roll < 0.15: # RED
            cov_days = rng.uniform(0, 2)
        elif risk_roll < 0.35: # YELLOW
            cov_days = rng.uniform(2.1, 10)
        else: # GREEN & OUTLIERS
            if rng.random() < 0.10:
                cov_days = rng.uniform(120, 365) # Excess
            else:
                cov_days = rng.uniform(10.1, 60) # Healthy
            
        qty_on_hand = int(amd/30.0 * cov_days)
        inv_rows.append({
            "company_id": company_id,
            "snapshot_date": snapshot_date,
            "material_id": mid,
            "location_id": lid,
            "qty_on_hand": qty_on_hand
        })

    df_inventory = pd.DataFrame(inv_rows)
    
    # Supply Orders
    po_rows = []
    po_cnt = 1
    for _, inv in df_inventory.iterrows():
        has_po = False
        if inv["qty_on_hand"] < 50: has_po = rng.random() < 0.8
        else: has_po = rng.random() < 0.3
        
        if has_po:
            n = rng.integers(1, 3)
            for _ in range(n):
                qty = rng.integers(50, 500)
                days_due = rng.integers(-5, 60) 
                po_rows.append({
                    "company_id": company_id,
                    "order_id": f"PO_{po_cnt:05d}",
                    "material_id": inv["material_id"],
                    "location_id": inv["location_id"],
                    "qty_inbound": qty,
                    "due_date": snapshot_date + timedelta(days=int(days_due)),
                    "order_type": "PO"
                })
                po_cnt += 1
                
    df_open_supply = pd.DataFrame(po_rows)

    return {
        "materials": df_materials, "locations": df_locations, "demand": df_demand,
        "forecast": df_forecast, "inventory": df_inventory, "leadtime": pd.DataFrame(),
        "open_supply": df_open_supply
    }

# --------------- UI HELPERS --------------- #

def _fmt(x):
    try:
        val = int(round(float(x)))
        return f"{val:,}".replace(",", ".")
    except: return str(x)

def _get_card_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800&display=swap');
        body { font-family: "Source Sans Pro", sans-serif; margin: 0; color: #31333F; }
        .card {
            border-radius: 8px; padding: 12px 16px; margin: 6px 0;
            border: 1px solid rgba(0,0,0,0.05); display: flex; align-items: center; gap: 15px;
            font-size: 14px;
        }
        .pill {
            display:inline-block; padding:4px 10px; border-radius:12px; font-weight:700; font-size:12px;
        }
        .lbl { font-size:11px; color:#7D98A3; font-weight:700; margin-bottom:4px; }
        .val { font-weight:700; color:#213644; font-size:15px; }
    </style>
    """

def render_risk_cards(df_view):
    if df_view.empty:
        st.info("No items match filters.")
        return

    cards_html = [_get_card_css()]
    max_display = 100
    display_df = df_view.sort_values(["coverage_days", "avg_daily_demand"], ascending=[True, False]).head(max_display)
    
    for _, row in display_df.iterrows():
        bg = "#F3FFF6"
        if row["risk_status"] == "YELLOW": bg = "#FFFBEA"
        if row["risk_status"] == "RED": bg = "#FFF4F4"

        html = f"""
        <div class="card" style="background: {bg};">
            <div style="flex:0 0 200px;">
                <div style="font-weight:700; color:#111;">{row.get('location_name', row['location_id'])}</div>
                <div style="color:#666; font-size:12px;">{row.get('material_desc', row['material_id'])}</div>
            </div>
            <div style="flex:1; display:flex; gap:20px;">
                <div><div class="lbl">AVG DEMAND</div><div class="val">{_fmt(row['avg_daily_demand'])}/day</div></div>
                <div><div class="lbl">ON HAND</div><div class="val">{_fmt(row['qty_on_hand'])}</div></div>
                <div><div class="lbl">COVERAGE</div>
                     <span class="pill" style="background:{'#ff4b4b' if row['risk_status']=='RED' else '#ffa421' if row['risk_status']=='YELLOW' else '#21c354'}; color:white;">
                     {_fmt(row['coverage_days'])} days</span>
                </div>
            </div>
            <div style="flex:0 0 80px; text-align:right;">
                <div class="lbl">RISK</div>
                <div style="font-weight:800;">{row['risk_status']}</div>
            </div>
        </div>
        """
        cards_html.append(textwrap.dedent(html))
        
    if len(df_view) > max_display:
        st.warning(f"Showing top {max_display} most critical items out of {len(df_view)} total.")

    components.html("\n".join(cards_html), height=min(600, len(display_df)*85), scrolling=True)

def render_supply_cards(df_supply):
    if df_supply.empty:
        st.info("No open supply orders.")
        return
        
    cards_html = [_get_card_css()]
    for _, row in df_supply.sort_values("due_date").iterrows():
        is_late = pd.to_datetime(row['due_date']) < datetime.now()
        bg = "#FFF4F4" if is_late else "#F9F9F9"
        
        html = f"""
        <div class="card" style="background:{bg};">
            <div style="flex:0 0 150px; font-weight:700;">{row['order_id']}</div>
            <div style="flex:1;">
                <div class="lbl">DUE DATE</div>
                <div class="val" style="color:{'#D93025' if is_late else '#333'}">
                    {pd.to_datetime(row['due_date']).strftime('%Y-%m-%d')}
                    {' (LATE)' if is_late else ''}
                </div>
            </div>
            <div style="flex:1;">
                <div class="lbl">QUANTITY</div>
                <div class="val">{_fmt(row['qty_inbound'])}</div>
            </div>
            <div style="flex:1;">
                 <div class="lbl">TYPE</div>
                 <div class="pill" style="background:#E8EAED; color:#333;">{row.get('order_type','PO')}</div>
            </div>
        </div>
        """
        cards_html.append(textwrap.dedent(html))
        
    components.html("\n".join(cards_html), height=min(400, len(df_supply)*85), scrolling=True)

def render_inventory_card(row_inv):
    if row_inv.empty: return
    if isinstance(row_inv, pd.DataFrame): row_inv = row_inv.iloc[0]
    css = _get_card_css()
    html = f"""
    {css}
    <div class="card" style="background:#F0F4F8; border-left: 4px solid #0B67A4;">
        <div style="flex:1;">
            <div class="lbl">SNAPSHOT DATE</div>
            <div class="val">{pd.to_datetime(row_inv['snapshot_date']).strftime('%Y-%m-%d')}</div>
        </div>
        <div style="flex:1;">
            <div class="lbl">QTY ON HAND</div>
            <div class="val" style="font-size:20px;">{_fmt(row_inv['qty_on_hand'])}</div>
        </div>
        <div style="flex:1;">
            <div class="lbl">LOCATION</div>
            <div class="val">{row_inv['location_id']}</div>
        </div>
    </div>
    """
    components.html(html, height=100)

# --------------- MAIN APP --------------- #

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    
    # --- HEADER ---
    c1, c2 = st.columns([8, 3])
    with c1:
        st.markdown(f"""<div style="display:flex; align-items:center; gap:12px;">
                <div style="font-size:44px; line-height:1;">📦</div>
                <div><h1 style="margin:0; font-size:30px; color:#0F2933;">{APP_TITLE}</h1></div>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div style="display:flex; gap:10px; justify-content:flex-end; align-items:center; height:100%; padding-top:10px;">
                <span style="background-color: #E6F2FF; color: #0056b3; padding: 6px 12px; border-radius: 16px; font-weight: 600; font-family: 'Source Sans Pro', sans-serif; font-size: 14px;">{RELEASE_VERSION}</span>
                <span style="background-color: #F3E5F5; color: #6A1B9A; padding: 6px 12px; border-radius: 16px; font-weight: 600; font-family: 'Source Sans Pro', sans-serif; font-size: 14px;">{RELEASE_DATE}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # --- SIDEBAR: Data Dictionary (Pure Markdown Fix) ---
    with st.sidebar.expander("📚 Data Dictionary / Template Schema", expanded=False):
        # Using pure Markdown to avoid HTML rendering issues
        st.markdown("""
        **1. demand_history.csv**
        * `material_id` (str)
        * `location_id` (str)
        * `date` (YYYY-MM-DD)
        * `qty_demand` (float)

        **2. forecast.csv**
        * `material_id` (str)
        * `location_id` (str)
        * `date` (YYYY-MM-DD)
        * `qty_forecast` (float)

        **3. inventory_snapshot.csv**
        * `material_id` (str)
        * `location_id` (str)
        * `snapshot_date` (YYYY-MM-DD)
        * `qty_on_hand` (float)

        **4. open_supply.csv**
        * `order_id` (str)
        * `material_id`, `location_id`
        * `qty_inbound` (float)
        * `due_date` (YYYY-MM-DD)
        
        *Optional: materials.csv, locations.csv*
        """)

    # --- SIDEBAR: Logic Explainer ---
    with st.sidebar.expander("ℹ️ How Calculations Work", expanded=False):
        st.markdown("""
        **Avg Daily Demand:** Total history / days in window.
        
        **Coverage (Days):** `Qty On Hand / Avg Daily Demand`
        
        **Risk Status:**
        * 🔴 **RED:** Coverage ≤ Critical (e.g. 2 days)
        * 🟡 **YELLOW:** Coverage ≤ Attention (e.g. 10 days)
        * 🟢 **GREEN:** Healthy Coverage
        
        **Projected Stock:** `Current + Inbound - Forecast`
        """)

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("Configuration")
    sel_company = st.sidebar.selectbox("Company", [c["company_id"] for c in DEFAULT_COMPANIES])
    
    config = DEFAULT_COMPANY_CONFIG.copy()
    
    st.sidebar.subheader("Planning Horizon")
    config["planning_horizon_months"] = st.sidebar.slider("Months to Plan", 1, 24, 6)

    st.sidebar.subheader("Risk Thresholds (Days)")
    config["risk_thresholds"]["shortage_days"] = st.sidebar.number_input("Critical (RED)", 0, 20, 2)
    config["risk_thresholds"]["attention_days"] = st.sidebar.number_input("Attention (YELLOW)", 1, 60, 10)
    config["risk_thresholds"]["excess_days"] = st.sidebar.number_input("Excess Stock", 30, 365, 90)

    # --- DATA LOADING SECTION ---
    with st.expander("🗂️ Data Setup (Demo or Upload)", expanded=False):
        st.info("Click below to generate fully random datasets (variable items, locations, and risk).")
        if st.button("✨ Generate & Load Demo Data", use_container_width=True):
            with st.spinner("Generating fully random dataset..."):
                data = generate_demo_data(sel_company, config, seed=None)
                st.session_state["data"] = data
                st.success(f"Generated {len(data['materials'])} Materials across {len(data['locations'])} Locations.")
        
        st.markdown("---")
        st.markdown("**OR Upload Manual CSV Files**")
        
        c1, c2 = st.columns(2)
        with c1:
            u_dem = st.file_uploader("Demand", type=["csv"], key="u1")
            u_fcs = st.file_uploader("Forecast", type=["csv"], key="u2")
            u_inv = st.file_uploader("Inventory", type=["csv"], key="u3")
        with c2:
            u_sup = st.file_uploader("Supply", type=["csv"], key="u4")
            u_mat = st.file_uploader("Materials", type=["csv"], key="u5")
            u_loc = st.file_uploader("Locations", type=["csv"], key="u6")

        if u_dem and u_inv:
            if st.button("Load Uploaded Files", use_container_width=True):
                m_data = {
                    "demand": parse_uploaded_csv(u_dem),
                    "forecast": parse_uploaded_csv(u_fcs) if u_fcs else pd.DataFrame(),
                    "inventory": parse_uploaded_csv(u_inv),
                    "open_supply": parse_uploaded_csv(u_sup) if u_sup else pd.DataFrame(),
                    "materials": parse_uploaded_csv(u_mat) if u_mat else pd.DataFrame(),
                    "locations": parse_uploaded_csv(u_loc) if u_loc else pd.DataFrame(),
                    "leadtime": pd.DataFrame()
                }
                st.session_state["data"] = m_data
                st.success("Uploaded Data Loaded!")

    # --- APP LOGIC ---
    if "data" not in st.session_state:
        st.info("Please expand the 'Data Setup' box above to Load Demo Data or Upload your files.")
        return

    data = st.session_state["data"]
    
    # Pre-processing
    df_inv = data["inventory"].copy()
    if not df_inv.empty: df_inv["snapshot_date"] = pd.to_datetime(df_inv["snapshot_date"])
    
    df_dem = data["demand"].copy()
    if not df_dem.empty: df_dem["date"] = pd.to_datetime(df_dem["date"])
    
    avg_daily = compute_avg_daily_demand(df_dem)
    df_risk = classify_risk(df_inv, avg_daily, config)
    
    if not data["materials"].empty:
        df_risk = df_risk.merge(data["materials"][["material_id", "material_desc", "abc_class", "unit_price"]], on="material_id", how="left")
        df_risk["total_value"] = df_risk["qty_on_hand"] * df_risk.get("unit_price", 10.0)
        
    if not data["locations"].empty:
        df_risk = df_risk.merge(data["locations"][["location_id", "location_name", "region"]], on="location_id", how="left")

    # --- METRICS SECTION (Improved) ---
    st.markdown("### 📊 Portfolio Health")
    
    # Calculate Counts
    n_tot = len(df_risk)
    n_red = len(df_risk[df_risk["risk_status"]=="RED"])
    n_yel = len(df_risk[df_risk["risk_status"]=="YELLOW"])
    n_grn = len(df_risk[df_risk["risk_status"]=="GREEN"])
    
    # Calculate Percentages
    pct_red = (n_red / n_tot * 100) if n_tot > 0 else 0
    pct_yel = (n_yel / n_tot * 100) if n_tot > 0 else 0
    pct_grn = (n_grn / n_tot * 100) if n_tot > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    
    m1.metric("Total SKU-Locations", n_tot)
    
    # Displaying % in 'delta' but removing the arrow color logic by using delta_color='off' 
    # or just normal strings to avoid confusion.
    m2.metric("Critical (RED)", n_red, delta=f"{pct_red:.1f}% of Total", delta_color="inverse")
    m3.metric("Warning (YELLOW)", n_yel, delta=f"{pct_yel:.1f}% of Total", delta_color="inverse")
    m4.metric("Healthy (GREEN)", n_grn, delta=f"{pct_grn:.1f}% of Total", delta_color="normal")

    # SECTION 2: RISK ANALYSIS
    st.markdown("---")
    st.header("2. Risk Analysis")
    
    f1, f2, f3 = st.columns(3)
    ft_risk = f1.multiselect("Filter Status", ["RED", "YELLOW", "GREEN"], default=["RED", "YELLOW"])
    abc_opts = sorted(df_risk["abc_class"].dropna().unique()) if "abc_class" in df_risk else []
    ft_abc = f2.multiselect("Filter ABC", abc_opts)
    reg_opts = sorted(df_risk["region"].dropna().unique()) if "region" in df_risk else []
    ft_loc = f3.multiselect("Filter Region", reg_opts)

    df_view = df_risk.copy()
    if ft_risk: df_view = df_view[df_view["risk_status"].isin(ft_risk)]
    if ft_abc: df_view = df_view[df_view["abc_class"].isin(ft_abc)]
    if ft_loc: df_view = df_view[df_view["region"].isin(ft_loc)]

    c_risk_table, c_risk_chart = st.columns([3, 2])
    
    with c_risk_table:
        st.subheader("Priority Action List")
        render_risk_cards(df_view)

    with c_risk_chart:
        st.subheader("Inventory Insights")
        
        if not df_view.empty:
            # 1. NEW CHART: Inventory Value at Risk (Stacked Bar)
            if "total_value" in df_view.columns:
                chart_val = alt.Chart(df_view).mark_bar().encode(
                    x=alt.X('region', title='Region'),
                    y=alt.Y('sum(total_value)', title='Inventory Value ($)'),
                    color=alt.Color('risk_status', scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'], range=['#D93025', '#F9AB00', '#1E8E3E'])),
                    tooltip=['region', 'risk_status', 'sum(total_value)']
                ).properties(height=220, title="Inventory Value at Risk by Region")
                st.altair_chart(chart_val, use_container_width=True)

            # 2. NEW CHART: Coverage Distribution (Histogram)
            # Create bins for coverage to show distribution shape
            chart_hist = alt.Chart(df_view).mark_bar().encode(
                x=alt.X('coverage_days', bin=alt.Bin(maxbins=20), title='Coverage Days (Binned)'),
                y=alt.Y('count()', title='Number of Items'),
                color=alt.Color('risk_status', legend=None, scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'], range=['#D93025', '#F9AB00', '#1E8E3E'])),
                tooltip=['count()']
            ).properties(height=220, title="Coverage Distribution (Stock Shape)")
            st.altair_chart(chart_hist, use_container_width=True)

        else:
            st.info("No data for chart.")

    # SECTION 3: DRILLDOWN
    st.markdown("---")
    st.header("3. Detailed Planning View")
    
    if df_risk.empty:
        st.warning("No data available.")
        return

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        s_mat = st.selectbox("Select Material", sorted(df_risk["material_id"].unique()))
    with col_sel2:
        valid_locs = df_risk[df_risk["material_id"] == s_mat]["location_id"].unique()
        s_loc = st.selectbox("Select Location", sorted(valid_locs))

    dd_hist = df_dem[(df_dem["material_id"]==s_mat) & (df_dem["location_id"]==s_loc)].copy()
    
    if not data["forecast"].empty:
        dd_fcst = data["forecast"][(data["forecast"]["material_id"]==s_mat) & (data["forecast"]["location_id"]==s_loc)].copy()
    else: dd_fcst = pd.DataFrame()
        
    dd_inv = df_inv[(df_inv["material_id"]==s_mat) & (df_inv["location_id"]==s_loc)].copy()
    
    if not data["open_supply"].empty:
        dd_supp = data["open_supply"][(data["open_supply"]["material_id"]==s_mat) & (data["open_supply"]["location_id"]==s_loc)].copy()
    else: dd_supp = pd.DataFrame()

    tab1, tab2 = st.tabs(["📉 Demand & Supply Projection", "📋 Transaction Details"])

    with tab1:
        base = alt.Chart(dd_hist).mark_line(color='gray').encode(x='date', y='qty_demand', tooltip=['date', 'qty_demand'])
        if not dd_fcst.empty:
            dd_fcst["date"] = pd.to_datetime(dd_fcst["date"])
            line_f = alt.Chart(dd_fcst).mark_line(strokeDash=[5,5], color='blue').encode(x='date', y='qty_forecast', tooltip=['date', 'qty_forecast'])
            st.altair_chart((base + line_f).interactive(), use_container_width=True)
        else:
            st.altair_chart(base.interactive(), use_container_width=True)
        
        st.subheader("Projected Inventory Balance (90 Days)")
        if not dd_inv.empty:
            start_stock = dd_inv.iloc[0]["qty_on_hand"]
            avg_daily_fcst = dd_fcst["qty_forecast"].mean() / 30.0 if not dd_fcst.empty else 0
            
            today = pd.Timestamp.today().normalize()
            dates = pd.date_range(today, periods=90, freq='D')
            proj_df = pd.DataFrame({"date": dates})
            proj_df["outflow"] = avg_daily_fcst
            proj_df["inflow"] = 0
            
            if not dd_supp.empty:
                supp_agg = dd_supp.groupby("due_date")["qty_inbound"].sum().reset_index()
                supp_agg["due_date"] = pd.to_datetime(supp_agg["due_date"])
                proj_df = proj_df.merge(supp_agg, left_on="date", right_on="due_date", how="left").fillna(0)
                proj_df["inflow"] = proj_df["qty_inbound"]
            
            proj_df["net_change"] = proj_df["inflow"] - proj_df["outflow"]
            proj_df["projected_stock"] = start_stock + proj_df["net_change"].cumsum()
            
            ss_level = avg_daily_fcst * 10
            
            base = alt.Chart(proj_df).encode(x='date')
            line = base.mark_line(color='#0B67A4').encode(y=alt.Y('projected_stock', title='Projected Stock'))
            rule0 = base.mark_rule(color='red', strokeWidth=2).encode(y=alt.datum(0))
            ruless = base.mark_rule(color='orange', strokeDash=[3,3]).encode(y=alt.datum(ss_level))
            
            st.altair_chart((line + rule0 + ruless).interactive(), use_container_width=True)

    with tab2:
        c_inv, c_supp = st.columns(2)
        with c_inv:
            st.subheader("Current Inventory")
            render_inventory_card(dd_inv)
        with c_supp:
            st.subheader("Inbound Supply Orders")
            render_supply_cards(dd_supp)

if __name__ == "__main__":
    main()
