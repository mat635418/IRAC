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
RELEASE_VERSION = "v 0.40"
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

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)

def compute_avg_daily_demand(df_demand: pd.DataFrame) -> pd.DataFrame:
    if df_demand is None or df_demand.empty:
        return pd.DataFrame()
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
    if df_inventory is None or df_inventory.empty:
        return pd.DataFrame()
    df = df_inventory.copy()
    df = df.merge(
        avg_daily[["material_id", "location_id", "avg_daily_demand"]],
        on=["material_id", "location_id"],
        how="left",
    )
    df["avg_daily_demand"].fillna(0.0, inplace=True)
    df["coverage_days"] = np.where(
        df["avg_daily_demand"] > 0,
        df["qty_on_hand"] / df["avg_daily_demand"],
        np.inf,
    )
    thr = config["risk_thresholds"]
    def risk_label(row):
        cov = row["coverage_days"]
        if cov <= thr["shortage_days"]: return "RED"
        elif cov <= thr["attention_days"]: return "YELLOW"
        else: return "GREEN"
    df["risk_status"] = df.apply(risk_label, axis=1)
    return df

# --------------- DEMO DATA GENERATION --------------- #

def generate_demo_data(company_id, config=None, n_materials=80, n_locations=8, seed=42):
    """
    Scaled up generation with biased risk distribution.
    """
    if config is None: config = DEFAULT_COMPANY_CONFIG
    rng = np.random.default_rng(seed)

    # Materials
    materials = []
    for i in range(n_materials):
        mid = f"MAT_{i:03d}"
        abc = rng.choice(["A", "B", "C"], p=[0.2, 0.5, 0.3])
        materials.append({
            "company_id": company_id,
            "material_id": mid,
            "material_desc": f"Component {mid} - {rng.choice(['Standard', 'Premium', 'Basic'])}",
            "uom": "EA",
            "material_group": f"GRP_{i % 5}",
            "abc_class": abc,
            "unit_price": round(rng.uniform(10, 500), 2)
        })

    # Locations
    locations = []
    for j in range(n_locations):
        lid = f"LOC_{j:02d}"
        locations.append({
            "company_id": company_id,
            "location_id": lid,
            "location_name": f"Warehouse {lid} ({rng.choice(['North', 'South', 'East', 'West'])})",
            "location_type": rng.choice(["PLANT", "DC"]),
            "region": f"REGION_{j % 3}",
        })

    df_materials = pd.DataFrame(materials)
    df_locations = pd.DataFrame(locations)

    today = datetime.today().date()
    hist_dates = month_range(today, config.get("history_months", 24))
    fcst_dates = month_range(today + pd.DateOffset(months=18), 18)

    demand_rows, forecast_rows = [], []
    
    # Generate Demand & Forecast
    for _, m in df_materials.iterrows():
        for _, l in df_locations.iterrows():
            base = rng.uniform(20, 1000)
            
            # Demand
            d_series = gen_seasonal_demand_series(len(hist_dates), base, 0.3, 0.2, rng.integers(1,1e5))
            for d, q in zip(hist_dates, d_series):
                demand_rows.append({"company_id": company_id, "material_id": m["material_id"], "location_id": l["location_id"], "date": d.date(), "qty_demand": q})
            
            # Forecast
            f_series = gen_seasonal_demand_series(len(fcst_dates), base * rng.uniform(0.9, 1.2), 0.3, 0.1, rng.integers(1,1e5))
            for d, q in zip(fcst_dates, f_series):
                forecast_rows.append({"company_id": company_id, "material_id": m["material_id"], "location_id": l["location_id"], "date": d.date(), "qty_forecast": q})

    df_demand = pd.DataFrame(demand_rows)
    df_forecast = pd.DataFrame(forecast_rows)

    # Inventory Snapshot - BIAS FOR RISK
    inv_rows = []
    snapshot_date = today
    avg_dem = df_demand.groupby(["material_id", "location_id"])["qty_demand"].mean().reset_index()

    for _, row in avg_dem.iterrows():
        mid, lid = row["material_id"], row["location_id"]
        amd = row["qty_demand"]
        
        # Risk Bias Logic:
        # 30% Critical (Red) -> 0 to 2 days coverage
        # 30% Warning (Yellow) -> 3 to 10 days coverage
        # 40% Healthy (Green) -> > 10 days coverage
        risk_roll = rng.random()
        
        if risk_roll < 0.30: # RED
            cov_days = rng.uniform(0, 2)
        elif risk_roll < 0.60: # YELLOW
            cov_days = rng.uniform(2.1, 10)
        else: # GREEN
            cov_days = rng.uniform(10.1, 60)
            
        qty_on_hand = int(amd/30.0 * cov_days) # approx daily conversion
        
        inv_rows.append({
            "company_id": company_id,
            "snapshot_date": snapshot_date,
            "material_id": mid,
            "location_id": lid,
            "qty_on_hand": qty_on_hand
        })

    df_inventory = pd.DataFrame(inv_rows)
    
    # Lead Time & Supply
    # (Simplified for brevity, just generating Supply Orders)
    po_rows = []
    po_cnt = 1
    for _, inv in df_inventory.iterrows():
        # High probability of open orders for low stock items
        has_po = False
        if inv["qty_on_hand"] < 100: has_po = rng.random() < 0.9
        else: has_po = rng.random() < 0.4
        
        if has_po:
            n = rng.integers(1, 4)
            for _ in range(n):
                qty = rng.integers(50, 500)
                # Due dates: some late (negative), some soon
                days_due = rng.integers(-5, 45) 
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
    """Format numeric values with thousands separator."""
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
    
    # Limit display to avoid browser crash on huge demo data
    max_display = 100
    display_df = df_view.sort_values(["coverage_days", "avg_daily_demand"], ascending=[True, False]).head(max_display)
    
    for _, row in display_df.iterrows():
        bg = "#F3FFF6" # Green
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
        # Late?
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
    # Assuming row_inv is a Series or single row DF
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
    
    # HEADER
    c1, c2 = st.columns([3, 1])
    c1.title(f"📦 {APP_TITLE}")
    c2.markdown(f"**{RELEASE_VERSION}** | {RELEASE_DATE}")
    st.markdown("---")

    # SIDEBAR
    st.sidebar.header("Configuration")
    sel_company = st.sidebar.selectbox("Company", [c["company_id"] for c in DEFAULT_COMPANIES])
    config = DEFAULT_COMPANY_CONFIG.copy()
    
    # DATA LOADING
    st.header("1. Data Loading")
    col_btn, col_txt = st.columns([1, 4])
    if col_btn.button("Generate & Load Demo Data"):
        with st.spinner("Generating large dataset..."):
            data = generate_demo_data(sel_company, config)
            st.session_state["data"] = data
            st.success("Data Loaded!")
    
    if "data" not in st.session_state:
        st.info("Please click the button above to load data.")
        return

    data = st.session_state["data"]
    
    # PRE-PROCESSING
    df_inv = data["inventory"].copy()
    df_inv["snapshot_date"] = pd.to_datetime(df_inv["snapshot_date"])
    df_dem = data["demand"].copy()
    df_dem["date"] = pd.to_datetime(df_dem["date"])
    
    avg_daily = compute_avg_daily_demand(df_dem)
    df_risk = classify_risk(df_inv, avg_daily, config)
    
    # Merge Metadata
    df_risk = df_risk.merge(data["materials"][["material_id", "material_desc", "abc_class"]], on="material_id", how="left")
    df_risk = df_risk.merge(data["locations"][["location_id", "location_name", "region"]], on="location_id", how="left")

    # METRICS
    st.markdown("### 📊 High-Level KPIs")
    m1, m2, m3, m4 = st.columns(4)
    n_red = len(df_risk[df_risk["risk_status"]=="RED"])
    n_yel = len(df_risk[df_risk["risk_status"]=="YELLOW"])
    n_tot = len(df_risk)
    
    m1.metric("Total SKU-Locations", n_tot)
    m2.metric("Critical Shortages (RED)", n_red, delta=f"{n_red/n_tot:.1%}", delta_color="inverse")
    m3.metric("Low Coverage (YELLOW)", n_yel, delta=f"{n_yel/n_tot:.1%}", delta_color="inverse")
    m4.metric("Avg Coverage", f"{df_risk['coverage_days'].replace([np.inf, -np.inf], np.nan).median():.1f} days")

    # SECTION 2: RISK
    st.markdown("---")
    st.header("2. Risk Analysis")
    
    f1, f2, f3 = st.columns(3)
    ft_risk = f1.multiselect("Filter Status", ["RED", "YELLOW", "GREEN"], default=["RED", "YELLOW"])
    ft_abc = f2.multiselect("Filter ABC", ["A", "B", "C"])
    ft_loc = f3.multiselect("Filter Region", sorted(df_risk["region"].dropna().unique()))

    df_view = df_risk.copy()
    if ft_risk: df_view = df_view[df_view["risk_status"].isin(ft_risk)]
    if ft_abc: df_view = df_view[df_view["abc_class"].isin(ft_abc)]
    if ft_loc: df_view = df_view[df_view["region"].isin(ft_loc)]

    c_risk_table, c_risk_chart = st.columns([3, 2])
    
    with c_risk_table:
        st.subheader("Priority Action List")
        render_risk_cards(df_view)

    with c_risk_chart:
        st.subheader("Risk Matrix (Demand vs Coverage)")
        # Scatter Plot: X=AvgDailyDemand, Y=Coverage, Color=Risk
        # We clamp coverage to 60 days for visualization so charts don't squash
        df_chart = df_view.copy()
        df_chart["vis_coverage"] = df_chart["coverage_days"].clip(upper=60)
        
        chart = alt.Chart(df_chart).mark_circle(size=80).encode(
            x=alt.X('avg_daily_demand', title='Avg Daily Demand'),
            y=alt.Y('vis_coverage', title='Coverage (Days) [Capped at 60]'),
            color=alt.Color('risk_status', scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'], range=['#D93025', '#F9AB00', '#1E8E3E'])),
            tooltip=['material_id', 'location_id', 'coverage_days', 'avg_daily_demand']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    # SECTION 3: DRILLDOWN
    st.markdown("---")
    st.header("3. Detailed Planning View")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        s_mat = st.selectbox("Select Material", sorted(df_risk["material_id"].unique()))
    with col_sel2:
        # Filter locations valid for this material
        valid_locs = df_risk[df_risk["material_id"] == s_mat]["location_id"].unique()
        s_loc = st.selectbox("Select Location", sorted(valid_locs))

    # PREPARE DRILLDOWN DATA
    dd_hist = df_dem[(df_dem["material_id"]==s_mat) & (df_dem["location_id"]==s_loc)].copy()
    dd_fcst = data["forecast"][(data["forecast"]["material_id"]==s_mat) & (data["forecast"]["location_id"]==s_loc)].copy()
    dd_inv = df_inv[(df_inv["material_id"]==s_mat) & (df_inv["location_id"]==s_loc)].copy()
    dd_supp = data["open_supply"][(data["open_supply"]["material_id"]==s_mat) & (data["open_supply"]["location_id"]==s_loc)].copy()

    # LAYOUT: Top = Charts, Bottom = Card Tables
    tab1, tab2 = st.tabs(["📉 Demand & Supply Projection", "📋 Transaction Details"])

    with tab1:
        # 1. Demand vs Forecast Chart (Altair)
        c_hist = alt.Chart(dd_hist).mark_line(color='gray').encode(
            x='date', y='qty_demand', tooltip=['date', 'qty_demand']
        )
        c_fcst = alt.Chart(dd_fcst).mark_line(strokeDash=[5,5], color='blue').encode(
            x='date', y='qty_forecast', tooltip=['date', 'qty_forecast']
        )
        
        st.subheader("Demand History vs Forecast")
        st.altair_chart((c_hist + c_fcst).interactive(), use_container_width=True)
        
        # 2. Projected Inventory Balance (NEW VISUAL)
        st.subheader("Projected Inventory Balance (90 Days)")
        
        # Simple daily projection logic
        if not dd_inv.empty:
            start_stock = dd_inv.iloc[0]["qty_on_hand"]
            avg_daily_fcst = dd_fcst["qty_forecast"].mean() / 30.0 if not dd_fcst.empty else 0
            
            # Create 90 day range
            today = pd.Timestamp.today().normalize()
            dates = pd.date_range(today, periods=90, freq='D')
            proj_df = pd.DataFrame({"date": dates})
            
            # 1. Outflow (Forecast)
            proj_df["outflow"] = avg_daily_fcst
            
            # 2. Inflow (Supply)
            proj_df["inflow"] = 0
            if not dd_supp.empty:
                supp_agg = dd_supp.groupby("due_date")["qty_inbound"].sum().reset_index()
                supp_agg["due_date"] = pd.to_datetime(supp_agg["due_date"])
                proj_df = proj_df.merge(supp_agg, left_on="date", right_on="due_date", how="left").fillna(0)
                proj_df["inflow"] = proj_df["qty_inbound"]
            
            # 3. CumSum
            proj_df["net_change"] = proj_df["inflow"] - proj_df["outflow"]
            proj_df["projected_stock"] = start_stock + proj_df["net_change"].cumsum()
            
            # 4. Chart with Threshold Zones
            # Define safety stock line (e.g., 10 days of demand)
            ss_level = avg_daily_fcst * 10
            
            base = alt.Chart(proj_df).encode(x='date')
            line = base.mark_line(color='#0B67A4').encode(y=alt.Y('projected_stock', title='Projected Stock'))
            
            # Red zone area (below 0)
            rule0 = base.mark_rule(color='red', strokeWidth=2).encode(y=alt.datum(0))
            # SS Rule
            ruless = base.mark_rule(color='orange', strokeDash=[3,3]).encode(y=alt.datum(ss_level))
            
            st.altair_chart((line + rule0 + ruless).interactive(), use_container_width=True)
            st.caption("Blue: Projected Stock | Orange Dashed: Estimated Safety Stock (10 days) | Red: Stockout")

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
