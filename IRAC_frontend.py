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
RELEASE_VERSION = "v 0.65"
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
    "service_levels": {"default": 0.95}, # 95% Service Level
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

def compute_demand_stats(df_demand: pd.DataFrame) -> pd.DataFrame:
    """Computes Avg Daily Demand AND Standard Deviation (for SS calc)."""
    if df_demand is None or df_demand.empty: return pd.DataFrame()
    df = df_demand.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Total Aggregates
    agg = (
        df.groupby(["material_id", "location_id"])
        .agg(
            total_demand=("qty_demand", "sum"),
            days_span=("date", lambda x: (x.max() - x.min()).days + 1),
            std_dev_monthly=("qty_demand", "std") # Std Dev of the monthly buckets
        )
        .reset_index()
    )
    
    agg["avg_daily_demand"] = agg["total_demand"] / agg["days_span"].replace(0, 1)
    
    # Approx daily std dev from monthly (assuming 30 days)
    # math: sigma_daily = sigma_monthly / sqrt(30)
    agg["std_dev_daily"] = agg["std_dev_monthly"].fillna(0) / np.sqrt(30)
    
    return agg

def get_z_score(service_level):
    """Returns Z-score for common service levels."""
    lookup = {0.50: 0.00, 0.80: 0.84, 0.90: 1.28, 0.95: 1.645, 0.98: 2.055, 0.99: 2.33, 0.999: 3.09}
    return lookup.get(service_level, 1.645)

def calculate_inventory_parameters(df_risk, config):
    """
    Method 5 Logic + Conversion to Days for Projection
    """
    sl = config["service_levels"]["default"]
    z_score = get_z_score(sl)
    
    if "lead_time_days" not in df_risk.columns:
        df_risk["lead_time_days"] = 10
        
    order_cycle_days = 30 
    
    # 1. Safety Stock Qty (Method 5)
    # SS = Z * std_dev_daily * sqrt(lead_time)
    df_risk["ss_qty"] = z_score * df_risk["std_dev_daily"] * np.sqrt(df_risk["lead_time_days"])
    
    # 2. Convert SS to "Days Coverage" (for dynamic projection)
    # If demand evolves, we assume the "Days of Safety" remains the stable policy
    # SS_days = SS_qty / Avg_Daily_History
    df_risk["ss_days"] = np.where(df_risk["avg_daily_demand"] > 0, df_risk["ss_qty"] / df_risk["avg_daily_demand"], 0)
    
    # 3. ROP and Max in Days
    # ROP_days = LeadTime + SS_days
    df_risk["rop_days"] = df_risk["lead_time_days"] + df_risk["ss_days"]
    
    # Max_days = ROP_days + OrderCycle
    df_risk["max_days"] = df_risk["rop_days"] + order_cycle_days

    # Calculate static quantities for current view
    df_risk["ss_qty"] = df_risk["ss_qty"].round().astype(int)
    df_risk["rop_qty"] = (df_risk["avg_daily_demand"] * df_risk["rop_days"]).round().astype(int)
    df_risk["max_qty"] = (df_risk["avg_daily_demand"] * df_risk["max_days"]).round().astype(int)
    
    return df_risk

def classify_risk(df_inventory, df_stats, config):
    if df_inventory is None or df_inventory.empty: return pd.DataFrame()
    df = df_inventory.copy()
    
    # Merge Statistics
    df = df.merge(
        df_stats[["material_id", "location_id", "avg_daily_demand", "std_dev_daily"]],
        on=["material_id", "location_id"],
        how="left",
    )
    df["avg_daily_demand"].fillna(0.0, inplace=True)
    df["std_dev_daily"].fillna(0.0, inplace=True)
    
    # Calculate Method 5 Parameters
    df = calculate_inventory_parameters(df, config)
    
    # Coverage Calculation
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

def generate_demo_data(company_id, config=None, seed=None):
    """
    Generate comprehensive demo data for IRAC analysis.
    
    This function creates realistic inventory scenarios with:
    - 500-1000 materials (inclusive) to provide rich dataset for analysis
    - 8-15 locations (inclusive) to simulate complex distribution networks
    - Three location types: PLANT (manufacturing), DC (distribution center), RDC (regional distribution center)
    - Diverse risk scenarios including overstock situations
    
    Args:
        company_id: Identifier for the company
        config: Configuration dictionary (uses DEFAULT_COMPANY_CONFIG if None)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing generated datasets: materials, locations, demand,
        forecast, inventory, open_supply
    """
    if config is None: 
        config = DEFAULT_COMPANY_CONFIG
    
    rng = np.random.default_rng(seed)

    # Generate 500-1000 materials (inclusive) for comprehensive analysis
    # Note: rng.integers(500, 1001) generates values from 500 to 1000 inclusive
    n_materials = rng.integers(500, 1001)
    
    # Generate 8-15 locations (inclusive) to simulate diverse distribution networks
    # Note: rng.integers(8, 16) generates values from 8 to 15 inclusive
    n_locations = rng.integers(8, 16)

    # Generate materials with ABC classification
    # ABC distribution: 15% A-class (high value), 55% B-class, 30% C-class (low value)
    materials = []
    for i in range(n_materials):
        mid = f"MAT_{i:04d}"  # 4 digits to support up to 9999 materials
        abc = rng.choice(["A", "B", "C"], p=[0.15, 0.55, 0.3])
        lt = rng.integers(5, 45)  # Lead time: 5-45 days
        
        materials.append({
            "company_id": company_id,
            "material_id": mid,
            "material_desc": f"Part {mid} - {rng.choice(['Pro', 'Eco', 'Lite', 'Max', 'Plus', 'Elite', 'Prime', 'Standard'])}",
            "uom": "EA",
            "material_group": f"GRP_{rng.integers(0, 10)}",  # 10 material groups for better categorization
            "abc_class": abc,
            "unit_price": round(rng.uniform(5, 800), 2),
            "lead_time_days": lt
        })

    # Generate locations (distribution centers and plants)
    # Diverse location types across multiple regions
    locations = []
    for j in range(n_locations):
        lid = f"LOC_{j:02d}"
        locations.append({
            "company_id": company_id,
            "location_id": lid,
            "location_name": f"WH {lid} ({rng.choice(['Hub', 'Spoke', 'Regional', 'Central'])})",
            "location_type": rng.choice(["PLANT", "DC", "RDC"]),  # Added RDC (Regional Distribution Center)
            "region": f"REGION_{j % 5}",  # 5 regions for better geographic distribution
        })

    df_materials = pd.DataFrame(materials)
    df_locations = pd.DataFrame(locations)

    today = datetime.today().date()
    hist_dates = month_range(today, config.get("history_months", 24))
    fcst_dates = month_range(today + pd.DateOffset(months=18), 18)

    demand_rows, forecast_rows = [], []
    
    for _, m in df_materials.iterrows():
        for _, l in df_locations.iterrows():
            base = rng.uniform(10, 800)
            
            d_series = gen_seasonal_demand_series(len(hist_dates), base, 0.3, 0.25, rng.integers(1, 1e9))
            for d, q in zip(hist_dates, d_series):
                demand_rows.append({"company_id": company_id, "material_id": m["material_id"], "location_id": l["location_id"], "date": d.date(), "qty_demand": q})
            
            f_series = gen_seasonal_demand_series(len(fcst_dates), base * rng.uniform(0.9, 1.2), 0.3, 0.1, rng.integers(1, 1e9))
            for d, q in zip(fcst_dates, f_series):
                forecast_rows.append({"company_id": company_id, "material_id": m["material_id"], "location_id": l["location_id"], "date": d.date(), "qty_forecast": q})

    df_demand = pd.DataFrame(demand_rows)
    df_forecast = pd.DataFrame(forecast_rows)

    # Generate inventory snapshots with diverse coverage scenarios
    # This creates realistic distribution including:
    # - Critical shortages (RED: ~7%)
    # - Warnings (YELLOW: ~13%) 
    # - Healthy stock (GREEN: ~60%)
    # - Overstock situations (WAY ABOVE MAX: ~20% - this is key for rebalancing)
    inv_rows = []
    snapshot_date = today
    avg_dem = df_demand.groupby(["material_id", "location_id"])["qty_demand"].mean().reset_index()

    for _, row in avg_dem.iterrows():
        mid, lid = row["material_id"], row["location_id"]
        amd = row["qty_demand"]

        # Enhanced replenishment logic to create more overstock scenarios:
        # - 7% RED (critical shortage)
        # - 13% YELLOW (warning) 
        # - 60% GREEN (healthy range: 10-60 days)
        # - 20% EXTREME OVERSTOCK (100-300 days) - way above max for rebalancing opportunities
        risk_roll = rng.random()
        
        if risk_roll < 0.07:
            # RED: very low coverage, urgent replenishment needed
            cov_days = rng.uniform(0.1, 2)
        elif risk_roll < 0.20:
            # YELLOW: moderate coverage, attention needed
            cov_days = rng.uniform(2.5, 10)
        elif risk_roll < 0.80:
            # GREEN: healthy stock levels
            green_roll = rng.random()
            if green_roll < 0.4:
                cov_days = rng.uniform(10.5, 20)
            elif green_roll < 0.8:
                cov_days = rng.uniform(20, 60)
            else:
                cov_days = rng.uniform(60, 100)
        else:
            # EXTREME OVERSTOCK: way above max, candidates for rebalancing
            # These are the key items that could be transferred to shortage locations
            overstock_roll = rng.random()
            if overstock_roll < 0.5:
                cov_days = rng.uniform(100, 180)  # High overstock
            else:
                cov_days = rng.uniform(180, 300)  # Extreme overstock

        # Add realistic variance to simulate day-by-day fluctuations
        fluct = rng.uniform(-0.1, 0.1) * cov_days
        cov_days = max(0, cov_days + fluct)

        # Calculate quantity on hand based on coverage days
        qty_on_hand = int(np.round(amd / 30.0 * cov_days + rng.normal(0, amd * 0.12)))
        qty_on_hand = max(qty_on_hand, 0)
        
        inv_rows.append({
            "company_id": company_id,
            "snapshot_date": snapshot_date,
            "material_id": mid,
            "location_id": lid,
            "qty_on_hand": qty_on_hand
        })

    df_inventory = pd.DataFrame(inv_rows)
    
    # Generate open purchase orders (future receipts)
    # Items with low stock are more likely to have POs
    po_rows = []
    po_cnt = 1
    
    for _, inv in df_inventory.iterrows():
        # Probability of having a PO depends on current stock level
        has_po = False
        if inv["qty_on_hand"] < 50:
            has_po = rng.random() < 0.8  # 80% chance for low stock items
        else:
            has_po = rng.random() < 0.3  # 30% chance for other items
        
        if has_po:
            n = rng.integers(1, 3)  # 1-2 PO lines per item
            for _ in range(n):
                qty = rng.integers(50, 500)
                days_due = rng.integers(-5, 60)  # Some POs may be overdue (negative days)
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
        "materials": df_materials, 
        "locations": df_locations, 
        "demand": df_demand,
        "forecast": df_forecast, 
        "inventory": df_inventory, 
        "leadtime": pd.DataFrame(),
        "open_supply": df_open_supply
    }


# --------------- REBALANCING ANALYSIS --------------- #

def identify_rebalancing_opportunities(df_risk):
    """
    Identify materials that are overstocked in some locations but short in others.
    
    This function finds rebalancing opportunities where:
    - Material has excess stock (>100 days) in at least one location (SOURCE)
    - Same material has shortage (RED/YELLOW) in at least one other location (DESTINATION)
    
    Args:
        df_risk: DataFrame with risk classification for all material-location combinations
        
    Returns:
        DataFrame with rebalancing proposals including source/destination locations and quantities
    """
    if df_risk.empty:
        return pd.DataFrame()
    
    # Group by material to find rebalancing candidates
    rebalancing_proposals = []
    
    for material_id in df_risk["material_id"].unique():
        mat_df = df_risk[df_risk["material_id"] == material_id].copy()
        
        # Find overstock locations (potential sources) - coverage > 100 days
        overstock = mat_df[mat_df["coverage_days"] > 100].copy()
        
        # Find shortage locations (potential destinations) - RED or YELLOW status
        shortage = mat_df[mat_df["risk_status"].isin(["RED", "YELLOW"])].copy()
        
        # If we have both overstock and shortage for this material, create proposals
        if not overstock.empty and not shortage.empty:
            for _, src in overstock.iterrows():
                for _, dest in shortage.iterrows():
                    # Calculate transferable quantity
                    # From source: excess above max level
                    excess_qty = max(0, src["qty_on_hand"] - src["max_qty"])
                    
                    # To destination: gap to reach max level (considering inbound)
                    # Note: qty_inbound should be available from earlier merge in main logic
                    dest_gap = max(0, dest["max_qty"] - dest["qty_on_hand"] - dest.get("qty_inbound", 0))
                    
                    # Proposed transfer is the minimum of excess and gap
                    transfer_qty = min(excess_qty, dest_gap)
                    
                    if transfer_qty > 0:
                        # Calculate coverage improvement for destination
                        # Use small epsilon to avoid division by zero
                        dest_avg_demand = dest["avg_daily_demand"] if dest["avg_daily_demand"] > 0.01 else 0.01
                        coverage_improvement = transfer_qty / dest_avg_demand
                        
                        # Calculate total value of transfer
                        transfer_value = transfer_qty * src.get("unit_price", 0)
                        
                        rebalancing_proposals.append({
                            "material_id": material_id,
                            "material_desc": src.get("material_desc", material_id),
                            "abc_class": src.get("abc_class", ""),
                            "source_location": src["location_id"],
                            "source_location_name": src.get("location_name", src["location_id"]),
                            "source_coverage_days": src["coverage_days"],
                            "source_qty_on_hand": src["qty_on_hand"],
                            "dest_location": dest["location_id"],
                            "dest_location_name": dest.get("location_name", dest["location_id"]),
                            "dest_coverage_days": dest["coverage_days"],
                            "dest_risk_status": dest["risk_status"],
                            "dest_qty_on_hand": dest["qty_on_hand"],
                            "transfer_qty": int(transfer_qty),
                            "coverage_improvement_days": round(coverage_improvement, 1),
                            "transfer_value": round(transfer_value, 2),
                            "unit_price": src.get("unit_price", 0)
                        })
    
    if not rebalancing_proposals:
        return pd.DataFrame()
    
    df_rebalancing = pd.DataFrame(rebalancing_proposals)
    # Sort by highest value transfers first
    df_rebalancing = df_rebalancing.sort_values("transfer_value", ascending=False)
    
    return df_rebalancing


def aggregate_material_metrics(df_risk, data):
    """
    Aggregate all metrics by material for comprehensive material view.
    
    Args:
        df_risk: Risk-classified inventory data
        data: Full dataset dictionary
        
    Returns:
        DataFrame with material-level aggregations
    """
    if df_risk.empty:
        return pd.DataFrame()
    
    # Aggregate metrics by material
    material_agg = df_risk.groupby("material_id").agg({
        "qty_on_hand": "sum",
        "avg_daily_demand": "sum",
        "total_value": "sum",
        "coverage_days": "mean",
        "abc_class": "first",
        "material_desc": "first",
        "unit_price": "first"
    }).reset_index()
    
    # Count locations and risk distribution per material
    risk_counts = df_risk.groupby(["material_id", "risk_status"]).size().unstack(fill_value=0).reset_index()
    material_agg = material_agg.merge(risk_counts, on="material_id", how="left")
    
    # Fill missing risk columns
    for col in ["RED", "YELLOW", "GREEN"]:
        if col not in material_agg.columns:
            material_agg[col] = 0
    
    material_agg["total_locations"] = material_agg["RED"] + material_agg["YELLOW"] + material_agg["GREEN"]
    
    # Calculate summary metrics
    material_agg["avg_coverage_days"] = material_agg["coverage_days"].round(1)
    material_agg["total_qty_on_hand"] = material_agg["qty_on_hand"].astype(int)
    material_agg["total_value_formatted"] = material_agg["total_value"].round(0).astype(int)
    
    return material_agg


def aggregate_location_metrics(df_risk, data):
    """
    Aggregate all metrics by location for comprehensive location view.
    
    Args:
        df_risk: Risk-classified inventory data
        data: Full dataset dictionary
        
    Returns:
        DataFrame with location-level aggregations
    """
    if df_risk.empty:
        return pd.DataFrame()
    
    # Aggregate metrics by location
    location_agg = df_risk.groupby("location_id").agg({
        "qty_on_hand": "sum",
        "avg_daily_demand": "sum",
        "total_value": "sum",
        "coverage_days": "mean",
        "location_name": "first",
        "region": "first"
    }).reset_index()
    
    # Count materials and risk distribution per location
    risk_counts = df_risk.groupby(["location_id", "risk_status"]).size().unstack(fill_value=0).reset_index()
    location_agg = location_agg.merge(risk_counts, on="location_id", how="left")
    
    # Fill missing risk columns
    for col in ["RED", "YELLOW", "GREEN"]:
        if col not in location_agg.columns:
            location_agg[col] = 0
    
    location_agg["total_materials"] = location_agg["RED"] + location_agg["YELLOW"] + location_agg["GREEN"]
    
    # Calculate summary metrics
    location_agg["avg_coverage_days"] = location_agg["coverage_days"].round(1)
    location_agg["total_qty_on_hand"] = location_agg["qty_on_hand"].astype(int)
    location_agg["total_value_formatted"] = location_agg["total_value"].round(0).astype(int)
    
    return location_agg


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
        
        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .metric-box {
            padding: 15px; 
            border-radius: 8px; 
            border: 1px solid #eee; 
            text-align: left;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .metric-lbl { font-size: 12px; color: #666; font-weight: 600; margin-bottom: 5px; }
        .metric-val { font-size: 28px; font-weight: 700; color: #333; line-height: 1.1; }
        .metric-sub { font-size: 12px; margin-top: 4px; font-weight: 600; }
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
                <div><div class="lbl">SAFETY STOCK</div><div class="val" style="color:#666">{_fmt(row.get('ss_qty',0))}</div></div>
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

    # --- SIDEBAR: Logo ---
    st.sidebar.image("IRAC_logo.jpg", use_container_width=False, width=250)

    # --- SIDEBAR: Logic Explainer ---
    with st.sidebar.expander("🧮 Method 5 Formula Logic", expanded=False):
        st.latex(r"SS = Z \times \sigma_d \times \sqrt{LT}")
        st.latex(r"ROP = (AvgDaily \times LT) + SS")
        st.latex(r"Max = ROP + CycleStock")
        st.markdown("""
        **Variables:**
        * $Z$: Service Level Factor (e.g. 1.645 for 95%)
        * $\sigma_d$: Std Dev of Daily Demand
        * $LT$: Lead Time (Days)
        """)

    # --- SIDEBAR: Data Dictionary ---
    with st.sidebar.expander("📚 Data Dictionary / Template Schema", expanded=False):
        st.markdown("""
<div style="font-size:13px;">
  <b>1. demand_history.csv</b><br>
  - material_id, location_id (str)<br>
  - date (YYYY-MM-DD)<br>
  - qty_demand (float)<br><br>

  <b>2. forecast.csv</b><br>
  - material_id, location_id (str)<br>
  - date (YYYY-MM-DD)<br>
  - qty_forecast (float)<br><br>

  <b>3. inventory_snapshot.csv</b><br>
  - material_id, location_id (str)<br>
  - snapshot_date (YYYY-MM-DD)<br>
  - qty_on_hand (float)<br><br>
</div>
        """, unsafe_allow_html=True)

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("Configuration")
    sel_company = st.sidebar.selectbox("Company", [c["company_id"] for c in DEFAULT_COMPANIES])
    
    config = DEFAULT_COMPANY_CONFIG.copy()
    
    st.sidebar.subheader("Planning Horizon")
    config["planning_horizon_months"] = st.sidebar.slider("Months to Plan", 1, 24, 6)

    st.sidebar.subheader("Risk Thresholds (Days)")
    config["risk_thresholds"]["shortage_days"] = st.sidebar.number_input("Critical (RED)", 0, 20, 2)
    config["risk_thresholds"]["attention_days"] = st.sidebar.number_input("Attention (YELLOW)", 1, 60, 10)
    
    # --- DATA LOADING SECTION ---
    with st.expander("🗂️ Data Setup (Demo or Upload)", expanded=False):
        st.info("Click below to generate fully random datasets (variable items, locations, and risk).")
        if st.button("✨ Generate & Load Demo Data", use_container_width=True):
            with st.spinner("Generating fully random dataset..."):
                data = generate_demo_data(sel_company, config, seed=None)
                st.session_state["data"] = data
                st.success(f"Generated {len(data['materials'])} Materials.")
        
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
    
    # CALCULATE METRICS
    df_stats = compute_demand_stats(df_dem)
    
    if not data["materials"].empty:
        df_stats = df_stats.merge(data["materials"][["material_id", "lead_time_days"]], on="material_id", how="left")
    
    df_risk = classify_risk(df_inv, df_stats, config)
    
    if not data["materials"].empty:
        df_risk = df_risk.merge(data["materials"][["material_id", "material_desc", "abc_class", "unit_price"]], on="material_id", how="left")
        df_risk["total_value"] = df_risk["qty_on_hand"] * df_risk.get("unit_price", 10.0)
        
    if not data["locations"].empty:
        df_risk = df_risk.merge(data["locations"][["location_id", "location_name", "region"]], on="location_id", how="left")
    
    # Pre-calculate MRP logic
    if not data["open_supply"].empty:
        supply_agg = data["open_supply"].groupby(["material_id", "location_id"])["qty_inbound"].sum().reset_index()
        df_risk = df_risk.merge(supply_agg, on=["material_id", "location_id"], how="left")
        df_risk["qty_inbound"] = df_risk["qty_inbound"].fillna(0)
    else:
        df_risk["qty_inbound"] = 0
        
    df_risk["net_stock"] = df_risk["qty_on_hand"] + df_risk["qty_inbound"]
    df_risk["transfer_proposal"] = np.maximum(0, df_risk["max_qty"] - df_risk["net_stock"])
    
    mrp_df = df_risk[
        (df_risk["risk_status"].isin(["RED", "YELLOW"])) & 
        (df_risk["transfer_proposal"] > 0)
    ].copy()

    # --- METRICS SECTION ---
    
    n_tot = len(df_risk)
    if n_tot > 0:
        counts = df_risk["risk_status"].value_counts()
        n_red = counts.get("RED", 0)
        n_yel = counts.get("YELLOW", 0)
        n_grn = counts.get("GREEN", 0)
        pct_red, pct_yel, pct_grn = (n_red/n_tot)*100, (n_yel/n_tot)*100, (n_grn/n_tot)*100
    else:
        n_red=n_yel=n_grn=0; pct_red=pct_yel=pct_grn=0.0

    st.markdown("### 📊 Portfolio Health")
    col_kpi_left, col_kpi_right = st.columns(2)
    
    with col_kpi_left:
        st.markdown(_get_card_css(), unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-lbl">Total SKU-Locations</div>
                <div class="metric-val">{n_tot}</div>
                <div class="metric-sub" style="color:#666;">Total Active</div>
            </div>
            <div class="metric-box">
                <div class="metric-lbl">Healthy (GREEN)</div>
                <div class="metric-val">{n_grn}</div>
                <div class="metric-sub" style="color:#1E8E3E;">{pct_grn:.1f}% of Total</div>
            </div>
            <div class="metric-box">
                <div class="metric-lbl">Warning (YELLOW)</div>
                <div class="metric-val">{n_yel}</div>
                <div class="metric-sub" style="color:#F9AB00;">{pct_yel:.1f}% of Total</div>
            </div>
            <div class="metric-box">
                <div class="metric-lbl">Critical (RED)</div>
                <div class="metric-val">{n_red}</div>
                <div class="metric-sub" style="color:#D93025;">{pct_red:.1f}% of Total</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi_right:
        if n_tot > 0:
            source = df_risk["risk_status"].value_counts().reset_index()
            source.columns = ["risk_status", "count"]
            
            base = alt.Chart(source).encode(theta=alt.Theta("count", stack=True))
            
            pie = base.mark_arc(outerRadius=120, innerRadius=70).encode(
                color=alt.Color("risk_status", legend=None, scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'], range=['#D93025', '#F9AB00', '#1E8E3E'])),
                order=alt.Order("risk_status", sort="descending"),
                tooltip=["risk_status", "count"]
            )
            
            text = base.mark_text(radius=145, size=24, fontWeight='bold').encode(
                text=alt.Text("count"),
                order=alt.Order("risk_status", sort="descending"),
                color=alt.value("black")
            )
            
            st.altair_chart(pie + text, use_container_width=True)
        else:
            st.write("No data")

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

    # New Section for Obsolescence (High Coverage > 100 days)
    st.markdown("### 🐢 Potential Obsolescence (Coverage > 100 Days)")
    df_excess = df_view[df_view["coverage_days"] > 100].copy()
    
    if not df_excess.empty:
        # Metrics for excess
        total_excess_val = df_excess["total_value"].sum()
        count_excess = len(df_excess)
        
        c_ex1, c_ex2 = st.columns([1, 2])
        with c_ex1:
            st.metric("Total Value at Risk", f"${total_excess_val:,.0f}")
            st.metric("Count of Items", count_excess)
        
        with c_ex2:
            st.markdown("**Top 10 Worst Offenders**")
            st.dataframe(
                df_excess[["material_id", "location_id", "coverage_days", "qty_on_hand", "total_value"]]
                .sort_values("coverage_days", ascending=False)
                .head(10),
                column_config={
                    "coverage_days": st.column_config.NumberColumn("Coverage Days", format="%d"),
                    "qty_on_hand": st.column_config.NumberColumn("On Hand", format="%d"),
                    "total_value": st.column_config.NumberColumn("Total Value", format="$%d"),
                },
                use_container_width=True
            )
    else:
        st.success("No items with extreme overstock (> 100 days) found in current filter.")

    st.markdown("---")
    
    # Split: Risk Cards (60%) vs Scatter Matrix (40%)
    c_risk_table, c_risk_chart = st.columns([0.6, 0.4])
    
    with c_risk_table:
        st.subheader("Priority Action List")
        render_risk_cards(df_view)

    with c_risk_chart:
        st.subheader("Inventory Insights")
        if not df_view.empty:
            df_chart = df_view.copy()
            df_chart["vis_coverage"] = df_chart["coverage_days"].clip(upper=60)
            
            scatter = alt.Chart(df_chart).mark_circle(size=80).encode(
                x=alt.X('avg_daily_demand', title='Avg Daily Demand'),
                y=alt.Y('vis_coverage', title='Coverage (Days) [Capped 60]'),
                color=alt.Color('risk_status', scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'], range=['#D93025', '#F9AB00', '#1E8E3E'])),
                tooltip=['material_id', 'coverage_days']
            ).interactive().properties(height=350, title="Risk Matrix (Demand vs Coverage)")
            
            st.altair_chart(scatter, use_container_width=True)
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
        item_params = df_risk[(df_risk["material_id"]==s_mat) & (df_risk["location_id"]==s_loc)].iloc[0]
        ss_days = item_params["ss_days"]
        rop_days = item_params["rop_days"]
        max_days = item_params["max_days"]
        
        base = alt.Chart(dd_hist).mark_line(color='gray').encode(x='date', y='qty_demand', tooltip=['date', 'qty_demand'])
        if not dd_fcst.empty:
            dd_fcst["date"] = pd.to_datetime(dd_fcst["date"])
            line_f = alt.Chart(dd_fcst).mark_line(strokeDash=[5,5], color='blue').encode(x='date', y='qty_forecast', tooltip=['date', 'qty_forecast'])
            st.altair_chart((base + line_f).interactive(), use_container_width=True)
        else:
            st.altair_chart(base.interactive(), use_container_width=True)
        
        st.subheader("Projected Inventory Corridor (Dynamic Policy)")
        if not dd_inv.empty:
            start_stock = dd_inv.iloc[0]["qty_on_hand"]
            today = pd.Timestamp.today().normalize()
            dates = pd.date_range(today, periods=90, freq='D')
            proj_df = pd.DataFrame({"date": dates})
            
            if not dd_fcst.empty:
                dd_fcst_daily = dd_fcst.set_index("date").resample("D").interpolate(method="linear").reset_index()
                proj_df = proj_df.merge(dd_fcst_daily[["date", "qty_forecast"]], on="date", how="left").fillna(method="bfill").fillna(0)
            else:
                proj_df["qty_forecast"] = 0
            
            proj_df["daily_demand"] = proj_df["qty_forecast"] / 30.0 
            
            proj_df["dynamic_ss"] = proj_df["daily_demand"] * ss_days
            proj_df["dynamic_rop"] = proj_df["daily_demand"] * rop_days
            proj_df["dynamic_max"] = proj_df["daily_demand"] * max_days

            proj_df["outflow"] = proj_df["daily_demand"]
            proj_df["inflow"] = 0
            
            if not dd_supp.empty:
                supp_agg = dd_supp.groupby("due_date")["qty_inbound"].sum().reset_index()
                supp_agg["due_date"] = pd.to_datetime(supp_agg["due_date"])
                proj_df = proj_df.merge(supp_agg, left_on="date", right_on="due_date", how="left").fillna(0)
                proj_df["inflow"] = proj_df["qty_inbound"]
            
            proj_df["net_change"] = proj_df["inflow"] - proj_df["outflow"]
            proj_df["projected_stock"] = start_stock + proj_df["net_change"].cumsum()
            
            base = alt.Chart(proj_df).encode(x='date')
            
            line_stock = base.mark_line(color='#0B67A4', strokeWidth=3).encode(y=alt.Y('projected_stock', title='Stock Level'))
            line_ss = base.mark_line(color='red', strokeDash=[4,4]).encode(y='dynamic_ss')
            line_rop = base.mark_line(color='orange', strokeDash=[4,4]).encode(y='dynamic_rop')
            line_max = base.mark_line(color='green', strokeDash=[4,4]).encode(y='dynamic_max')
            
            st.altair_chart((line_stock + line_ss + line_rop + line_max).interactive(), use_container_width=True)
            st.caption("Red: Safety Stock | Orange: ROP | Green: Max Stock (Dynamic based on forecasted demand)")

    with tab2:
        c_inv, c_supp = st.columns(2)
        with c_inv:
            st.subheader("Current Inventory")
            render_inventory_card(dd_inv)
        with c_supp:
            st.subheader("Inbound Supply Orders")
            render_supply_cards(dd_supp)

    # --- SECTION 4: MRP ---
    st.markdown("---")
    st.header("4. MRP & Replenishment Proposal")
    
    if not mrp_df.empty:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("##### 🚀 Proposed Stock Transfers (To restore corridor health)")
            # Replaced style.background_gradient with st.column_config.ProgressColumn
            st.dataframe(
                mrp_df[["material_id", "location_id", "risk_status", "qty_on_hand", "qty_inbound", "max_qty", "transfer_proposal"]]
                .sort_values("transfer_proposal", ascending=False),
                column_config={
                    "qty_on_hand": st.column_config.NumberColumn("Current Stock", format="%d"),
                    "qty_inbound": st.column_config.NumberColumn("Inbound", format="%d"),
                    "max_qty": st.column_config.NumberColumn("Max Target", format="%d"),
                    "transfer_proposal": st.column_config.ProgressColumn(
                        "Suggested Order",
                        format="%d",
                        min_value=0,
                        max_value=int(mrp_df["transfer_proposal"].max()),
                    ),
                },
                use_container_width=True,
                hide_index=True
            )
        with c2:
            st.markdown("##### Proposal by Location")
            bar_chart = alt.Chart(mrp_df).mark_bar().encode(
                x=alt.X('location_id', title=None),
                y=alt.Y('sum(transfer_proposal)', title='Total Qty'),
                color='location_id'
            )
            st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.success("No urgent replenishment needed! All Red/Yellow items have sufficient inbound supply.")

    # --- SECTION 5: REBALANCING OPPORTUNITIES ---
    st.markdown("---")
    st.header("5. 🔄 Rebalancing Opportunities")
    st.markdown("""
    **Smart Inventory Rebalancing**: This analysis identifies materials that are overstocked in some locations 
    but critically short in others. By transferring excess inventory, you can reduce waste and improve service levels simultaneously.
    """)
    
    # Identify rebalancing opportunities
    df_rebalancing = identify_rebalancing_opportunities(df_risk)
    
    if not df_rebalancing.empty:
        # Summary metrics
        col_rb1, col_rb2, col_rb3, col_rb4 = st.columns(4)
        
        with col_rb1:
            st.metric(
                "Rebalancing Opportunities", 
                len(df_rebalancing),
                help="Number of potential transfers identified"
            )
        
        with col_rb2:
            total_transfer_value = df_rebalancing["transfer_value"].sum()
            st.metric(
                "Total Transfer Value", 
                f"${total_transfer_value:,.0f}",
                help="Total value of inventory that could be rebalanced"
            )
        
        with col_rb3:
            unique_materials = df_rebalancing["material_id"].nunique()
            st.metric(
                "Materials Affected", 
                unique_materials,
                help="Number of unique materials with rebalancing opportunities"
            )
        
        with col_rb4:
            avg_improvement = df_rebalancing["coverage_improvement_days"].mean()
            st.metric(
                "Avg Coverage Gain", 
                f"{avg_improvement:.1f} days",
                help="Average coverage improvement at destination locations"
            )
        
        st.markdown("---")
        
        # Filters for rebalancing view
        f_rb1, f_rb2 = st.columns(2)
        
        with f_rb1:
            abc_filter = st.multiselect(
                "Filter by ABC Class",
                options=sorted(df_rebalancing["abc_class"].unique()),
                help="Focus on specific material classes"
            )
        
        with f_rb2:
            risk_filter = st.multiselect(
                "Filter by Destination Risk",
                options=["RED", "YELLOW"],
                default=["RED", "YELLOW"],
                help="Show transfers to locations with specific risk status"
            )
        
        # Apply filters
        df_rebal_filtered = df_rebalancing.copy()
        if abc_filter:
            df_rebal_filtered = df_rebal_filtered[df_rebal_filtered["abc_class"].isin(abc_filter)]
        if risk_filter:
            df_rebal_filtered = df_rebal_filtered[df_rebal_filtered["dest_risk_status"].isin(risk_filter)]
        
        # Display rebalancing proposals
        st.markdown("### 📊 Detailed Rebalancing Proposals")
        
        if not df_rebal_filtered.empty:
            # Show top proposals
            st.dataframe(
                df_rebal_filtered[[
                    "material_id", "material_desc", "abc_class",
                    "source_location_name", "source_coverage_days",
                    "dest_location_name", "dest_risk_status", "dest_coverage_days",
                    "transfer_qty", "coverage_improvement_days", "transfer_value"
                ]].head(100),
                column_config={
                    "material_id": st.column_config.TextColumn("Material ID", width="small"),
                    "material_desc": st.column_config.TextColumn("Description", width="medium"),
                    "abc_class": st.column_config.TextColumn("ABC", width="small"),
                    "source_location_name": st.column_config.TextColumn("From Location", width="medium"),
                    "source_coverage_days": st.column_config.NumberColumn("Source Days", format="%.1f"),
                    "dest_location_name": st.column_config.TextColumn("To Location", width="medium"),
                    "dest_risk_status": st.column_config.TextColumn("Dest Risk", width="small"),
                    "dest_coverage_days": st.column_config.NumberColumn("Dest Days", format="%.1f"),
                    "transfer_qty": st.column_config.NumberColumn("Transfer Qty", format="%d"),
                    "coverage_improvement_days": st.column_config.NumberColumn("Days Gained", format="%.1f"),
                    "transfer_value": st.column_config.NumberColumn("Value", format="$%.0f"),
                },
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Visualization: Top materials by transfer value
            st.markdown("### 📈 Top Rebalancing Opportunities by Value")
            
            col_vis1, col_vis2 = st.columns(2)
            
            with col_vis1:
                # Bar chart of top materials
                top_materials = df_rebal_filtered.groupby("material_id").agg({
                    "transfer_value": "sum",
                    "transfer_qty": "sum"
                }).reset_index().sort_values("transfer_value", ascending=False).head(15)
                
                chart_mat = alt.Chart(top_materials).mark_bar().encode(
                    x=alt.X('material_id', sort='-y', title='Material'),
                    y=alt.Y('transfer_value', title='Total Transfer Value ($)'),
                    color=alt.value('#0B67A4'),
                    tooltip=['material_id', 'transfer_value', 'transfer_qty']
                ).properties(
                    title="Top 15 Materials by Transfer Value",
                    height=300
                )
                st.altair_chart(chart_mat, use_container_width=True)
            
            with col_vis2:
                # Distribution by ABC class
                abc_summary = df_rebal_filtered.groupby("abc_class").agg({
                    "transfer_value": "sum"
                }).reset_index()
                
                chart_abc = alt.Chart(abc_summary).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta('transfer_value', title='Total Value'),
                    color=alt.Color('abc_class', 
                                    scale=alt.Scale(domain=['A', 'B', 'C'], 
                                                   range=['#D93025', '#F9AB00', '#1E8E3E'])),
                    tooltip=['abc_class', 'transfer_value']
                ).properties(
                    title="Transfer Value by ABC Class",
                    height=300
                )
                st.altair_chart(chart_abc, use_container_width=True)
        else:
            st.info("No rebalancing opportunities match the selected filters.")
    else:
        st.success("✅ No rebalancing opportunities found. Inventory distribution is optimal across locations!")

    # --- SECTION 6: BY MATERIAL VIEW ---
    st.markdown("---")
    st.header("6. 📦 By Material Analysis")
    st.markdown("""
    **Material-Centric View**: Comprehensive analysis of each material across all locations.
    Understand total inventory position, risk distribution, and location-specific dynamics.
    """)
    
    # Generate material-level aggregations
    df_material_agg = aggregate_material_metrics(df_risk, data)
    
    if not df_material_agg.empty:
        # Material selection
        selected_material = st.selectbox(
            "Select Material for Detailed Analysis",
            options=sorted(df_material_agg["material_id"].unique()),
            help="Choose a material to see comprehensive metrics"
        )
        
        # Filter for selected material
        mat_data = df_material_agg[df_material_agg["material_id"] == selected_material].iloc[0]
        mat_locations = df_risk[df_risk["material_id"] == selected_material].copy()
        
        # Display material header
        st.markdown(f"### 🔍 Material: **{selected_material}** - {mat_data['material_desc']}")
        
        # Key metrics for this material
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        with col_m1:
            st.metric("ABC Class", mat_data["abc_class"])
        
        with col_m2:
            st.metric("Total Locations", int(mat_data["total_locations"]))
        
        with col_m3:
            st.metric("Total On Hand", f"{int(mat_data['total_qty_on_hand']):,}")
        
        with col_m4:
            st.metric("Avg Coverage", f"{mat_data['avg_coverage_days']:.1f} days")
        
        with col_m5:
            st.metric("Total Value", f"${int(mat_data['total_value_formatted']):,}")
        
        st.markdown("---")
        
        # Create tabs for different analyses
        tab_mat1, tab_mat2, tab_mat3 = st.tabs([
            "📊 Location Distribution", 
            "📈 Coverage Analysis", 
            "⚠️ Risk Summary"
        ])
        
        with tab_mat1:
            st.markdown("#### Inventory Distribution Across Locations")
            
            # Table view
            st.dataframe(
                mat_locations[[
                    "location_id", "location_name", "region", "risk_status",
                    "qty_on_hand", "coverage_days", "avg_daily_demand"
                ]].sort_values("qty_on_hand", ascending=False),
                column_config={
                    "location_id": "Location ID",
                    "location_name": "Location",
                    "region": "Region",
                    "risk_status": st.column_config.TextColumn("Risk"),
                    "qty_on_hand": st.column_config.NumberColumn("On Hand", format="%d"),
                    "coverage_days": st.column_config.NumberColumn("Coverage (Days)", format="%.1f"),
                    "avg_daily_demand": st.column_config.NumberColumn("Avg Daily Demand", format="%.1f")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Bar chart of quantities by location
            st.markdown("##### Quantity Distribution")
            chart_loc_qty = alt.Chart(mat_locations).mark_bar().encode(
                x=alt.X('location_id', sort='-y', title='Location'),
                y=alt.Y('qty_on_hand', title='Quantity On Hand'),
                color=alt.Color('risk_status', 
                               scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'], 
                                             range=['#D93025', '#F9AB00', '#1E8E3E'])),
                tooltip=['location_id', 'qty_on_hand', 'coverage_days', 'risk_status']
            ).properties(height=300)
            st.altair_chart(chart_loc_qty, use_container_width=True)
        
        with tab_mat2:
            st.markdown("#### Coverage Days Analysis")
            
            # Coverage distribution chart
            col_cov1, col_cov2 = st.columns(2)
            
            with col_cov1:
                # Scatter plot of coverage vs demand
                mat_loc_chart = mat_locations.copy()
                mat_loc_chart["vis_coverage"] = mat_loc_chart["coverage_days"].clip(upper=150)
                
                scatter_mat = alt.Chart(mat_loc_chart).mark_circle(size=150).encode(
                    x=alt.X('avg_daily_demand', title='Avg Daily Demand'),
                    y=alt.Y('vis_coverage', title='Coverage Days (capped at 150)'),
                    color=alt.Color('risk_status',
                                   scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'],
                                                 range=['#D93025', '#F9AB00', '#1E8E3E'])),
                    tooltip=['location_id', 'coverage_days', 'avg_daily_demand', 'risk_status']
                ).properties(
                    title="Coverage vs Demand by Location",
                    height=300
                )
                st.altair_chart(scatter_mat, use_container_width=True)
            
            with col_cov2:
                # Coverage distribution histogram
                hist_coverage = alt.Chart(mat_locations).mark_bar().encode(
                    x=alt.X('coverage_days', bin=alt.Bin(maxbins=20), title='Coverage Days'),
                    y=alt.Y('count()', title='Number of Locations'),
                    color=alt.value('#0B67A4'),
                    tooltip=['count()']
                ).properties(
                    title="Coverage Distribution",
                    height=300
                )
                st.altair_chart(hist_coverage, use_container_width=True)
        
        with tab_mat3:
            st.markdown("#### Risk Status Summary")
            
            # Risk distribution metrics
            risk_counts = mat_locations["risk_status"].value_counts()
            
            col_risk1, col_risk2, col_risk3 = st.columns(3)
            
            with col_risk1:
                red_count = risk_counts.get("RED", 0)
                st.metric(
                    "🔴 Critical Locations", 
                    red_count,
                    help="Locations with critical shortage"
                )
            
            with col_risk2:
                yellow_count = risk_counts.get("YELLOW", 0)
                st.metric(
                    "🟡 Warning Locations", 
                    yellow_count,
                    help="Locations needing attention"
                )
            
            with col_risk3:
                green_count = risk_counts.get("GREEN", 0)
                st.metric(
                    "🟢 Healthy Locations", 
                    green_count,
                    help="Locations with adequate stock"
                )
            
            # Pie chart of risk distribution
            risk_df = pd.DataFrame({
                'risk_status': list(risk_counts.index),
                'count': list(risk_counts.values)
            })
            
            pie_risk = alt.Chart(risk_df).mark_arc(outerRadius=120).encode(
                theta=alt.Theta('count', stack=True),
                color=alt.Color('risk_status',
                               scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'],
                                             range=['#D93025', '#F9AB00', '#1E8E3E']),
                               legend=alt.Legend(title="Risk Status")),
                tooltip=['risk_status', 'count']
            ).properties(
                title="Risk Distribution Across Locations",
                height=300
            )
            st.altair_chart(pie_risk, use_container_width=True)
    else:
        st.warning("No material data available for analysis.")

    # --- SECTION 7: BY LOCATION VIEW ---
    st.markdown("---")
    st.header("7. 🏭 By Location Analysis")
    st.markdown("""
    **Location-Centric View**: Deep dive into warehouse or plant performance.
    Monitor inventory health, identify bottlenecks, and optimize location-specific operations.
    """)
    
    # Generate location-level aggregations
    df_location_agg = aggregate_location_metrics(df_risk, data)
    
    if not df_location_agg.empty:
        # Location selection
        selected_location = st.selectbox(
            "Select Location for Detailed Analysis",
            options=sorted(df_location_agg["location_id"].unique()),
            help="Choose a location to see comprehensive metrics"
        )
        
        # Filter for selected location
        loc_data = df_location_agg[df_location_agg["location_id"] == selected_location].iloc[0]
        loc_materials = df_risk[df_risk["location_id"] == selected_location].copy()
        
        # Display location header
        st.markdown(f"### 🔍 Location: **{selected_location}** - {loc_data['location_name']}")
        
        # Key metrics for this location
        col_l1, col_l2, col_l3, col_l4, col_l5 = st.columns(5)
        
        with col_l1:
            st.metric("Region", loc_data["region"])
        
        with col_l2:
            st.metric("Total Materials", int(loc_data["total_materials"]))
        
        with col_l3:
            st.metric("Total On Hand", f"{int(loc_data['total_qty_on_hand']):,}")
        
        with col_l4:
            st.metric("Avg Coverage", f"{loc_data['avg_coverage_days']:.1f} days")
        
        with col_l5:
            st.metric("Total Value", f"${int(loc_data['total_value_formatted']):,}")
        
        st.markdown("---")
        
        # Create tabs for different analyses
        tab_loc1, tab_loc2, tab_loc3 = st.tabs([
            "📊 Material Distribution", 
            "📈 ABC Analysis", 
            "⚠️ Risk Summary"
        ])
        
        with tab_loc1:
            st.markdown("#### Inventory Distribution by Material")
            
            # Table view - top materials
            st.markdown("##### Top Materials by Value")
            st.dataframe(
                loc_materials[[
                    "material_id", "material_desc", "abc_class", "risk_status",
                    "qty_on_hand", "coverage_days", "total_value"
                ]].sort_values("total_value", ascending=False).head(50),
                column_config={
                    "material_id": "Material ID",
                    "material_desc": "Description",
                    "abc_class": "ABC",
                    "risk_status": st.column_config.TextColumn("Risk"),
                    "qty_on_hand": st.column_config.NumberColumn("On Hand", format="%d"),
                    "coverage_days": st.column_config.NumberColumn("Coverage (Days)", format="%.1f"),
                    "total_value": st.column_config.NumberColumn("Value", format="$%.0f")
                },
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Total value by risk status
            st.markdown("##### Inventory Value by Risk Status")
            value_by_risk = loc_materials.groupby("risk_status")["total_value"].sum().reset_index()
            
            chart_value_risk = alt.Chart(value_by_risk).mark_bar().encode(
                x=alt.X('risk_status', title='Risk Status', sort=['RED', 'YELLOW', 'GREEN']),
                y=alt.Y('total_value', title='Total Value ($)'),
                color=alt.Color('risk_status',
                               scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'],
                                             range=['#D93025', '#F9AB00', '#1E8E3E'])),
                tooltip=['risk_status', 'total_value']
            ).properties(height=300)
            st.altair_chart(chart_value_risk, use_container_width=True)
        
        with tab_loc2:
            st.markdown("#### ABC Classification Analysis")
            
            col_abc1, col_abc2 = st.columns(2)
            
            with col_abc1:
                # ABC distribution
                abc_summary = loc_materials.groupby("abc_class").agg({
                    "material_id": "count",
                    "total_value": "sum"
                }).reset_index()
                abc_summary.columns = ["abc_class", "count", "total_value"]
                
                st.dataframe(
                    abc_summary,
                    column_config={
                        "abc_class": "ABC Class",
                        "count": st.column_config.NumberColumn("Material Count", format="%d"),
                        "total_value": st.column_config.NumberColumn("Total Value", format="$%.0f")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Pie chart for ABC value distribution
                pie_abc = alt.Chart(abc_summary).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta('total_value', title='Value'),
                    color=alt.Color('abc_class',
                                   scale=alt.Scale(domain=['A', 'B', 'C'],
                                                 range=['#D93025', '#F9AB00', '#1E8E3E'])),
                    tooltip=['abc_class', 'total_value', 'count']
                ).properties(
                    title="Value Distribution by ABC",
                    height=250
                )
                st.altair_chart(pie_abc, use_container_width=True)
            
            with col_abc2:
                # Risk distribution within each ABC class
                abc_risk = loc_materials.groupby(["abc_class", "risk_status"]).size().reset_index(name='count')
                
                chart_abc_risk = alt.Chart(abc_risk).mark_bar().encode(
                    x=alt.X('abc_class', title='ABC Class'),
                    y=alt.Y('count', title='Number of Materials'),
                    color=alt.Color('risk_status',
                                   scale=alt.Scale(domain=['RED', 'YELLOW', 'GREEN'],
                                                 range=['#D93025', '#F9AB00', '#1E8E3E']),
                                   legend=alt.Legend(title="Risk Status")),
                    tooltip=['abc_class', 'risk_status', 'count']
                ).properties(
                    title="Risk Distribution by ABC Class",
                    height=300
                )
                st.altair_chart(chart_abc_risk, use_container_width=True)
        
        with tab_loc3:
            st.markdown("#### Risk Status Summary")
            
            # Risk distribution metrics
            risk_counts_loc = loc_materials["risk_status"].value_counts()
            
            col_risk_l1, col_risk_l2, col_risk_l3 = st.columns(3)
            
            with col_risk_l1:
                red_count_loc = risk_counts_loc.get("RED", 0)
                st.metric(
                    "🔴 Critical Materials", 
                    red_count_loc,
                    help="Materials with critical shortage"
                )
            
            with col_risk_l2:
                yellow_count_loc = risk_counts_loc.get("YELLOW", 0)
                st.metric(
                    "🟡 Warning Materials", 
                    yellow_count_loc,
                    help="Materials needing attention"
                )
            
            with col_risk_l3:
                green_count_loc = risk_counts_loc.get("GREEN", 0)
                st.metric(
                    "🟢 Healthy Materials", 
                    green_count_loc,
                    help="Materials with adequate stock"
                )
            
            # Show critical items that need attention
            st.markdown("##### 🚨 Priority Items (RED Status)")
            critical_items = loc_materials[loc_materials["risk_status"] == "RED"].sort_values(
                "total_value", ascending=False
            )
            
            if not critical_items.empty:
                st.dataframe(
                    critical_items[[
                        "material_id", "material_desc", "abc_class",
                        "qty_on_hand", "coverage_days", "avg_daily_demand", "total_value"
                    ]].head(20),
                    column_config={
                        "material_id": "Material ID",
                        "material_desc": "Description",
                        "abc_class": "ABC",
                        "qty_on_hand": st.column_config.NumberColumn("On Hand", format="%d"),
                        "coverage_days": st.column_config.NumberColumn("Coverage", format="%.1f"),
                        "avg_daily_demand": st.column_config.NumberColumn("Daily Demand", format="%.1f"),
                        "total_value": st.column_config.NumberColumn("Value", format="$%.0f")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("✅ No critical items at this location!")
    else:
        st.warning("No location data available for analysis.")

if __name__ == "__main__":
    main()
