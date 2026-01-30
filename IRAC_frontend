import io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# --------------- CONFIG / CONSTANTS --------------- #

APP_TITLE = "IRAC - Inventory Risk & Availability Control (Frontend MVP)"

DEFAULT_COMPANIES = [
    {"company_id": "COMPANY_A", "name": "Company A"},
    {"company_id": "COMPANY_B", "name": "Company B"},
]

# Default config used for generated demo data and as a fallback
DEFAULT_COMPANY_CONFIG = {
    "time_bucket": "month",           # "week" or "month"
    "history_months": 24,
    "forecast_months": 18,
    "planning_horizon_months": 6,
    "service_levels": {
        "A": 0.98,
        "B": 0.95,
        "C": 0.90,
        "default": 0.95,
    },
    # Risk thresholds defined in coverage days
    "risk_thresholds": {
        "shortage_days": 0,   # stockout or negative coverage
        "attention_days": 7,  # coverage < 7 days -> YELLOW
        "excess_days": 90,    # coverage > 90 days -> BLUE
    },
    "aggregation": {
        "default_level": "location",
        "allow_rollup_to_region": True,
    },
}

# --------------- UTILS --------------- #


def month_range(end_date: datetime, months_back: int) -> pd.DatetimeIndex:
    """Generate month-end dates going backwards."""
    # Use MonthEnd periods backward from end_date
    end = pd.Timestamp(end_date).normalize() + pd.offsets.MonthEnd(0)
    dates = pd.date_range(end - pd.DateOffset(months=months_back - 1),
                          end,
                          freq="M")
    return dates


def gen_seasonal_demand_series(
    periods: int,
    base_level: float,
    seasonal_amplitude: float = 0.3,
    noise_std: float = 0.15,
    random_state=None,
) -> np.ndarray:
    """
    Create a demand time series with seasonality and noise.
    periods: number of time points (months or weeks)
    base_level: average demand level
    seasonal_amplitude: relative amplitude of seasonality (0-1)
    noise_std: relative noise level
    """
    rng = np.random.default_rng(random_state)
    t = np.arange(periods)

    # Simple yearly seasonality (assuming monthly or weekly):
    # 2 * pi * t / 12 for monthly seasonality
    # If using weekly, it's still fine (just approximate annual cycle).
    seasonal = 1.0 + seasonal_amplitude * np.sin(2 * np.pi * t / 12.0)

    noise = rng.normal(loc=0.0, scale=noise_std, size=periods)
    series = base_level * seasonal * (1.0 + noise)
    series = np.clip(series, a_min=0, a_max=None)
    return series.round().astype(int)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to a CSV in bytes (for in-memory upload simulation)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


def compute_avg_daily_demand(df_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average daily demand per material/location
    from demand history (which is at arbitrary date frequency).
    Assumes df has columns: material_id, location_id, date, qty_demand.
    """
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


def classify_risk(
    df_inventory: pd.DataFrame,
    avg_daily: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Merge inventory snapshot with avg daily demand and classify risk based on coverage.
    df_inventory: snapshot with columns [material_id, location_id, snapshot_date, qty_on_hand]
    avg_daily: per item-location avg_daily_demand
    config: company config with risk thresholds
    """
    if df_inventory is None or df_inventory.empty:
        return pd.DataFrame()

    df = df_inventory.copy()
    df = df.merge(
        avg_daily[["material_id", "location_id", "avg_daily_demand"]],
        on=["material_id", "location_id"],
        how="left",
    )

    df["avg_daily_demand"].fillna(0.0, inplace=True)

    # Coverage in days
    # If avg_daily_demand == 0, treat coverage as very large (or 0).
    df["coverage_days"] = np.where(
        df["avg_daily_demand"] > 0,
        df["qty_on_hand"] / df["avg_daily_demand"],
        np.inf,
    )

    thr = config["risk_thresholds"]
    def risk_label(row):
        cov = row["coverage_days"]
        if cov <= thr["shortage_days"]:
            return "RED - shortage"
        elif cov <= thr["attention_days"]:
            return "YELLOW - attention"
        elif cov >= thr["excess_days"]:
            return "BLUE - excess"
        else:
            return "GREEN - healthy"

    df["risk_status"] = df.apply(risk_label, axis=1)
    return df


# --------------- DEMO DATA GENERATION --------------- #


def generate_demo_data(
    company_id: str,
    config: dict = None,
    n_materials: int = 20,
    n_locations: int = 5,
    seed: int = 42,
):
    """
    Generate realistic fake data for:
    - material master
    - location master
    - demand history (24 months)
    - forecast (18 months)
    - inventory snapshot
    - lead time history
    - open supply orders
    """

    if config is None:
        config = DEFAULT_COMPANY_CONFIG

    rng = np.random.default_rng(seed)

    # Dimensions
    materials = []
    for i in range(n_materials):
        mid = f"MAT_{i:03d}"
        abc = rng.choice(["A", "B", "C"], p=[0.3, 0.4, 0.3])
        materials.append(
            {
                "company_id": company_id,
                "material_id": mid,
                "material_desc": f"Material {i:03d}",
                "uom": "EA",
                "material_group": f"GROUP_{i % 4}",
                "abc_class": abc,
                "criticality": "Strategic" if abc == "A" else "Standard",
            }
        )

    locations = []
    for j in range(n_locations):
        lid = f"LOC_{j:02d}"
        ltype = rng.choice(["PLANT", "DC"])
        locations.append(
            {
                "company_id": company_id,
                "location_id": lid,
                "location_name": f"Location {lid}",
                "location_type": ltype,
                "region": f"REGION_{j % 2}",
            }
        )

    df_materials = pd.DataFrame(materials)
    df_locations = pd.DataFrame(locations)

    # Time
    today = datetime.today().date()
    history_months = config.get("history_months", 24)
    forecast_months = config.get("forecast_months", 18)

    hist_dates = month_range(today, history_months)
    fcst_dates = month_range(today + pd.DateOffset(months=forecast_months), forecast_months)

    # Demand History & Forecast
    demand_rows = []
    forecast_rows = []
    for _, m in df_materials.iterrows():
        for _, l in df_locations.iterrows():
            base = rng.uniform(50, 800)  # base monthly demand

            # Demand history
            demand_series = gen_seasonal_demand_series(
                periods=len(hist_dates),
                base_level=base,
                seasonal_amplitude=rng.uniform(0.1, 0.4),
                noise_std=rng.uniform(0.05, 0.2),
                random_state=rng.integers(1, 10_000),
            )
            for date, qty in zip(hist_dates, demand_series):
                demand_rows.append(
                    {
                        "company_id": company_id,
                        "material_id": m["material_id"],
                        "location_id": l["location_id"],
                        "date": date.date(),
                        "qty_demand": qty,
                    }
                )

            # Forecast - slightly trend up or down + noise
            trend_factor = rng.uniform(0.9, 1.1)
            fcst_series = gen_seasonal_demand_series(
                periods=len(fcst_dates),
                base_level=base * trend_factor,
                seasonal_amplitude=rng.uniform(0.1, 0.4),
                noise_std=rng.uniform(0.05, 0.2),
                random_state=rng.integers(1, 10_000),
            )
            for date, qty in zip(fcst_dates, fcst_series):
                forecast_rows.append(
                    {
                        "company_id": company_id,
                        "material_id": m["material_id"],
                        "location_id": l["location_id"],
                        "date": date.date(),
                        "qty_forecast": qty,
                    }
                )

    df_demand = pd.DataFrame(demand_rows)
    df_forecast = pd.DataFrame(forecast_rows)

    # Inventory Snapshot (roughly 1-3 months of coverage)
    inv_rows = []
    snapshot_date = today
    avg_demand = (
        df_demand.groupby(["material_id", "location_id"])["qty_demand"]
        .mean()
        .reset_index()
        .rename(columns={"qty_demand": "avg_monthly_demand"})
    )

    for _, row in avg_demand.iterrows():
        mid = row["material_id"]
        lid = row["location_id"]
        amd = row["avg_monthly_demand"]
        coverage_months = rng.uniform(0.1, 3.0)
        qty_on_hand = max(0, int(amd * coverage_months + rng.normal(0, amd * 0.1)))
        inv_rows.append(
            {
                "company_id": company_id,
                "snapshot_date": snapshot_date,
                "material_id": mid,
                "location_id": lid,
                "qty_on_hand": qty_on_hand,
            }
        )

    df_inventory = pd.DataFrame(inv_rows)

    # Lead Time History (per item-location, monthly aggregated)
    lt_rows = []
    for _, m in df_materials.iterrows():
        for _, l in df_locations.iterrows():
            planned_lt = rng.integers(5, 45)  # planned lead time in days
            lt_variability = rng.uniform(0.1, 0.4)  # as fraction of planned
            for i in range(history_months):
                month_end = hist_dates[i].date()
                # Simulate 1-3 orders per month
                n_orders = rng.integers(1, 4)
                for _ in range(n_orders):
                    planned = planned_lt
                    actual = int(
                        max(
                            1,
                            rng.normal(planned, planned * lt_variability),
                        )
                    )
                    order_date = month_end - timedelta(days=actual + rng.integers(0, 10))
                    receipt_date = order_date + timedelta(days=actual)
                    lt_rows.append(
                        {
                            "company_id": company_id,
                            "material_id": m["material_id"],
                            "location_id": l["location_id"],
                            "planned_lt_days": planned,
                            "actual_lt_days": actual,
                            "order_date": order_date,
                            "receipt_date": receipt_date,
                        }
                    )

    df_leadtime = pd.DataFrame(lt_rows)

    # Open Supply Orders (some future POs)
    po_rows = []
    po_id_counter = 1
    for _, inv in df_inventory.iterrows():
        mid = inv["material_id"]
        lid = inv["location_id"]
        amd = avg_demand.query(
            "material_id == @mid and location_id == @lid"
        )["avg_monthly_demand"].values
        amd = amd[0] if len(amd) else rng.uniform(50, 500)

        # probability of having open supply
        if rng.random() < 0.6:
            n_pos = rng.integers(1, 3)
            for _ in range(n_pos):
                qty = int(max(0, rng.normal(amd, amd * 0.3)))
                due_in_days = rng.integers(5, 90)
                due_date = snapshot_date + timedelta(days=int(due_in_days))
                po_rows.append(
                    {
                        "company_id": company_id,
                        "order_id": f"PO_{po_id_counter:06d}",
                        "material_id": mid,
                        "location_id": lid,
                        "qty_inbound": qty,
                        "due_date": due_date,
                        "order_type": "PO",
                    }
                )
                po_id_counter += 1

    df_open_supply = pd.DataFrame(po_rows)

    return {
        "materials": df_materials,
        "locations": df_locations,
        "demand": df_demand,
        "forecast": df_forecast,
        "inventory": df_inventory,
        "leadtime": df_leadtime,
        "open_supply": df_open_supply,
    }


# --------------- STREAMLIT APP --------------- #


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # --- SIDEBAR: Company & Config ---
    st.sidebar.header("Company & Configuration")

    company_options = [c["company_id"] for c in DEFAULT_COMPANIES]
    selected_company = st.sidebar.selectbox("Select company", company_options)

    # For now we use a single default config per company.
    # Later: load from DB / file.
    config = DEFAULT_COMPANY_CONFIG.copy()

    # Allow user to tweak a few config parameters
    st.sidebar.subheader("Planning Settings")
    config["time_bucket"] = st.sidebar.selectbox(
        "Time bucket",
        options=["week", "month"],
        index=1 if config["time_bucket"] == "month" else 0,
    )
    config["planning_horizon_months"] = st.sidebar.slider(
        "Planning horizon (months)",
        min_value=1,
        max_value=24,
        value=config["planning_horizon_months"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Risk Thresholds (Coverage in Days)")
    rt = config["risk_thresholds"]
    rt["shortage_days"] = st.sidebar.number_input(
        "Shortage threshold (≤ days)",
        min_value=0,
        max_value=60,
        value=rt["shortage_days"],
        step=1,
    )
    rt["attention_days"] = st.sidebar.number_input(
        "Attention if coverage ≤ days",
        min_value=1,
        max_value=60,
        value=rt["attention_days"],
        step=1,
    )
    rt["excess_days"] = st.sidebar.number_input(
        "Excess if coverage ≥ days",
        min_value=30,
        max_value=365,
        value=rt["excess_days"],
        step=5,
    )

    config["risk_thresholds"] = rt

    # --- MAIN: Data Upload & Demo Data --- #
    st.header("1. Load Baseline Data")

    st.markdown(
        """
IRAC requires a small standard dataset:
- Historical demand
- Forecast
- Inventory snapshot
- Lead time history
- Open supply orders (optional but recommended)
- Material & location masters (optional, for nicer labels)

You can either:

- **Upload your own CSV files**, or  
- **Click "Load Demo Data"** to auto-generate realistic fake data and test the app.
"""
    )

    col_demo, col_info = st.columns([1, 3])

    with col_demo:
        if st.button("Load Demo Data (Generate Fake Company Dataset)"):
            demo = generate_demo_data(company_id=selected_company, config=config)

            # Cache in session_state as if they came from uploads
            st.session_state["demo_data"] = demo
            st.success("Demo data generated and loaded into the app.")

    with col_info:
        st.info(
            "Tip: Start by clicking **Load Demo Data**. "
            "You can then inspect the tables and risk classification before integrating real data."
        )

    st.markdown("---")

    # Uploaders
    st.subheader("Upload Baseline Files (CSV)")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_demand = st.file_uploader(
            "Historical Demand (demand_history.csv)",
            type=["csv"],
            key="upload_demand",
        )
        uploaded_forecast = st.file_uploader(
            "Forecast (forecast.csv)",
            type=["csv"],
            key="upload_forecast",
        )
        uploaded_inventory = st.file_uploader(
            "Inventory Snapshot (inventory_snapshot.csv)",
            type=["csv"],
            key="upload_inventory",
        )

    with col2:
        uploaded_leadtime = st.file_uploader(
            "Lead Time History (leadtime_history.csv)",
            type=["csv"],
            key="upload_leadtime",
        )
        uploaded_open_supply = st.file_uploader(
            "Open Supply Orders (open_supply.csv)",
            type=["csv"],
            key="upload_open_supply",
        )
        uploaded_materials = st.file_uploader(
            "Material Master (materials.csv) [optional]",
            type=["csv"],
            key="upload_materials",
        )
        uploaded_locations = st.file_uploader(
            "Location Master (locations.csv) [optional]",
            type=["csv"],
            key="upload_locations",
        )

    # Decide whether to use demo data or uploaded data
    use_demo = "demo_data" in st.session_state and st.session_state["demo_data"] is not None

    if use_demo:
        demo = st.session_state["demo_data"]
        df_materials = demo["materials"]
        df_locations = demo["locations"]
        df_demand = demo["demand"]
        df_forecast = demo["forecast"]
        df_inventory = demo["inventory"]
        df_leadtime = demo["leadtime"]
        df_open_supply = demo["open_supply"]
    else:
        df_materials = parse_uploaded_csv(uploaded_materials) if uploaded_materials else pd.DataFrame()
        df_locations = parse_uploaded_csv(uploaded_locations) if uploaded_locations else pd.DataFrame()
        df_demand = parse_uploaded_csv(uploaded_demand) if uploaded_demand else pd.DataFrame()
        df_forecast = parse_uploaded_csv(uploaded_forecast) if uploaded_forecast else pd.DataFrame()
        df_inventory = parse_uploaded_csv(uploaded_inventory) if uploaded_inventory else pd.DataFrame()
        df_leadtime = parse_uploaded_csv(uploaded_leadtime) if uploaded_leadtime else pd.DataFrame()
        df_open_supply = parse_uploaded_csv(uploaded_open_supply) if uploaded_open_supply else pd.DataFrame()

    # Validate minimal required data
    required_loaded = not df_demand.empty and not df_inventory.empty

    if not required_loaded:
        st.warning(
            "Minimal required files: **Historical Demand** and **Inventory Snapshot**.\n\n"
            "Either upload them, or click **Load Demo Data**."
        )
        # If no minimal data, we still show some info tables if present, but skip computations
    else:
        st.success("Minimal required data is loaded. You can now explore risk and coverage.")

    # --------------- PREVIEW DATA --------------- #
    with st.expander("Preview Loaded DataFrames"):
        st.write("**Materials**")
        st.dataframe(df_materials.head(20))
        st.write("**Locations**")
        st.dataframe(df_locations.head(20))
        st.write("**Demand History**")
        st.dataframe(df_demand.head(20))
        st.write("**Forecast**")
        st.dataframe(df_forecast.head(20))
        st.write("**Inventory Snapshot**")
        st.dataframe(df_inventory.head(20))
        st.write("**Lead Time History**")
        st.dataframe(df_leadtime.head(20))
        st.write("**Open Supply Orders**")
        st.dataframe(df_open_supply.head(20))

    # --------------- ANALYSIS: COVERAGE & RISK --------------- #
    st.markdown("---")
    st.header("2. Inventory Coverage & Risk Classification")

    if not required_loaded:
        st.info("Once minimal data is loaded, this section will compute coverage and risk status.")
        return

    # Ensure basic schema
    for name, df in [
        ("Demand", df_demand),
        ("Inventory", df_inventory),
    ]:
        missing_cols = []
        if name == "Demand":
            for col in ["material_id", "location_id", "date", "qty_demand"]:
                if col not in df.columns:
                    missing_cols.append(col)
        elif name == "Inventory":
            for col in ["material_id", "location_id", "snapshot_date", "qty_on_hand"]:
                if col not in df.columns:
                    missing_cols.append(col)

        if missing_cols:
            st.error(
                f"{name} data is missing expected columns: {missing_cols}. "
                f"Please check your uploaded CSV or adjust the demo generator."
            )
            return

    # Type coercion
    df_demand["date"] = pd.to_datetime(df_demand["date"])
    df_inventory["snapshot_date"] = pd.to_datetime(df_inventory["snapshot_date"])

    avg_daily_df = compute_avg_daily_demand(df_demand)
    df_risk = classify_risk(df_inventory, avg_daily_df, config=config)

    st.subheader("Risk Summary (per Item-Location)")

    # Add optional joins for nicer display
    if not df_materials.empty:
        df_risk = df_risk.merge(
            df_materials[["material_id", "material_desc", "abc_class"]],
            on="material_id",
            how="left",
        )

    if not df_locations.empty:
        df_risk = df_risk.merge(
            df_locations[["location_id", "location_name", "region"]],
            on="location_id",
            how="left",
        )

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        abc_filter = st.multiselect(
            "Filter by ABC class",
            options=sorted(df_risk["abc_class"].dropna().unique().tolist())
            if "abc_class" in df_risk.columns
            else [],
            default=None,
        )
    with col_f2:
        region_filter = st.multiselect(
            "Filter by Region",
            options=sorted(df_risk["region"].dropna().unique().tolist())
            if "region" in df_risk.columns
            else [],
            default=None,
        )
    with col_f3:
        risk_filter = st.multiselect(
            "Filter by Risk Status",
            options=sorted(df_risk["risk_status"].dropna().unique().tolist()),
            default=None,
        )

    df_view = df_risk.copy()
    if abc_filter:
        df_view = df_view[df_view["abc_class"].isin(abc_filter)]
    if region_filter:
        df_view = df_view[df_view["region"].isin(region_filter)]
    if risk_filter:
        df_view = df_view[df_view["risk_status"].isin(risk_filter)]

    st.dataframe(
        df_view.sort_values(["risk_status", "coverage_days"], ascending=[True, True])
    )

    # Quick aggregated view
    st.subheader("Risk Distribution")

    risk_counts = (
        df_risk["risk_status"].value_counts().rename_axis("risk_status").reset_index(name="count")
    )
    st.bar_chart(
        data=risk_counts.set_index("risk_status"),
        use_container_width=True,
    )

    # --------------- ITEM-LEVEL DRILLDOWN --------------- #
    st.markdown("---")
    st.header("3. Item-Level Drilldown (Demand, Inventory, Supply)")

    # Drilldown selectors
    unique_items = sorted(df_risk["material_id"].unique().tolist())
    unique_locs = sorted(df_risk["location_id"].unique().tolist())

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        sel_item = st.selectbox("Select Material", unique_items)
    with col_i2:
        sel_loc = st.selectbox("Select Location", unique_locs)

    # Filter for selected item/location
    d_hist = df_demand.query(
        "material_id == @sel_item and location_id == @sel_loc"
    ).copy()
    d_fcst = df_forecast.query(
        "material_id == @sel_item and location_id == @sel_loc"
    ).copy() if not df_forecast.empty else pd.DataFrame()
    d_inv = df_inventory.query(
        "material_id == @sel_item and location_id == @sel_loc"
    ).copy()
    d_os = df_open_supply.query(
        "material_id == @sel_item and location_id == @sel_loc"
    ).copy() if not df_open_supply.empty else pd.DataFrame()

    # Time-series chart: demand history + forecast
    st.subheader("Demand History & Forecast")

    ts_hist = d_hist[["date", "qty_demand"]].rename(columns={"qty_demand": "demand"})
    ts_hist.set_index("date", inplace=True)

    if not d_fcst.empty:
        d_fcst["date"] = pd.to_datetime(d_fcst["date"])
        ts_fcst = d_fcst[["date", "qty_forecast"]].rename(
            columns={"qty_forecast": "forecast"}
        )
        ts_fcst.set_index("date", inplace=True)
        ts_all = ts_hist.join(ts_fcst, how="outer")
    else:
        ts_all = ts_hist

    st.line_chart(ts_all, use_container_width=True)

    # Show inventory and open supply
    st.subheader("Inventory & Open Supply")

    if not d_inv.empty:
        st.write("**Inventory Snapshot**")
        st.dataframe(d_inv)
    else:
        st.write("No inventory snapshot for this item-location.")

    if not d_os.empty:
        st.write("**Open Supply Orders**")
        st.dataframe(d_os)
    else:
        st.write("No open supply orders for this item-location.")


if __name__ == "__main__":
    main()
