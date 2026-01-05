import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

# --- CONFIGURATION & CONSTANTS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "images"

# File Paths
DB1B_MARKET_PATH = DATA_DIR / "T_DB1B_MARKET.csv"
T100_PATH = DATA_DIR / "T_T100D_SEGMENT_US_CARRIER_ONLY.csv"

# Global Constants
KEYS = ["year", "origin_airport_id", "dest_airport_id"]
MILES_TO_KM = 1.60934
SEAT_CO2_KG_PER_KM = 0.09  # Fallback value

# Ensure directories exist
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# --- DATA PREPARATION FUNCTIONS ---

def _mode(series: pd.Series) -> object:
    """Helper: Returns the mode of a series, or NaN if empty."""
    mode = series.mode(dropna=True)
    return mode.iat[0] if not mode.empty else np.nan

def prep_db1b(market_path: Path) -> pd.DataFrame:
    """
    Loads and aggregates DB1B Market data (Pricing).
    Calculates passenger-weighted average fares per route.
    """
    cols = {
        "YEAR": "year",
        "ORIGIN_AIRPORT_ID": "origin_airport_id",
        "DEST_AIRPORT_ID": "dest_airport_id",
        "PASSENGERS": "passengers",
        "MARKET_FARE": "market_fare",
        "MARKET_DISTANCE": "distance",
    }
    
    # Load only necessary columns
    df = pd.read_csv(market_path, usecols=cols.keys()).rename(columns=cols)

    # Calculate weighted fare
    df["fare_weighted"] = df["market_fare"] * df["passengers"]

    # Group by Route (Year, Origin, Dest)
    grouped = (
        df.groupby(KEYS, as_index=False)
        .agg(
            passengers=("passengers", "sum"),
            avg_distance=("distance", "mean"),
            avg_fare=("fare_weighted", "sum"),
        )
    )

    # Finalize weighted average
    grouped["avg_fare"] = grouped["avg_fare"] / grouped["passengers"]
    return grouped


def prep_t100(path: Path) -> pd.DataFrame:
    """
    Loads and aggregates T100 Segment data (Operations/Capacity).
    Aggregates to quarterly level to match DB1B granularity.
    """
    cols = {
        "YEAR": "year",
        "MONTH": "month",
        "ORIGIN_AIRPORT_ID": "origin_airport_id",
        "DEST_AIRPORT_ID": "dest_airport_id",
        "UNIQUE_CARRIER": "carrier",
        "AIRCRAFT_TYPE": "aircraft_type",
        "SEATS": "seats",
        "PASSENGERS": "reported_passengers",
        "PAYLOAD": "payload",
        "AIR_TIME": "air_time",
    }
    df = pd.read_csv(path, usecols=cols.keys()).rename(columns=cols)
    
    # Aggregate logic
    grouped = (
        df.groupby(KEYS, as_index=False)
        .agg(
            seats=("seats", "sum"),
            reported_passengers=("reported_passengers", "sum"),
            payload=("payload", "sum"),
            air_time=("air_time", "sum"),
            carrier=("carrier", _mode),        # Most frequent carrier
            aircraft_type=("aircraft_type", _mode), # Most frequent aircraft
        )
    )
    return grouped


def get_aircraft_co2_emissions(path: Path) -> dict:
    """
    Calculates estimated CO2 per seat-km based on payload and airtime.
    Returns a dictionary mapping Aircraft Type ID -> CO2 Value.
    """
    if not path.exists():
        print(f"Warning: {path} not found. Returning empty emissions dict.")
        return {}

    df = pd.read_csv(path, usecols=["AIRCRAFT_TYPE", "SEATS", "PAYLOAD", "AIR_TIME"])
    
    # Filter invalid rows
    df = df[(df["AIR_TIME"] > 0) & (df["SEATS"] > 0) & (df["PAYLOAD"] > 0)].copy()
    
    # Emissions Formula (Approximate)
    df["co2_per_seat_hour"] = (df["PAYLOAD"] * 3.15) / (df["SEATS"] * (df["AIR_TIME"] / 60))
    df["co2_per_seat_km"] = df["co2_per_seat_hour"] / 800  # Assumes avg speed 800km/h
    
    emissions = df.groupby("AIRCRAFT_TYPE")["co2_per_seat_km"].mean().to_dict()
    return emissions

# Pre-load emissions map (Module Level Execution)
# Note: This runs immediately when you import the file.
AIRCRAFT_CO2_EMISSIONS = get_aircraft_co2_emissions(T100_PATH)


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering: Adds CO2 metrics, OHE for aircraft, and Carrier flags.
    """
    df = df.copy()
    
    # Physical Conversions
    df["distance_km"] = df["avg_distance"] * MILES_TO_KM
    df["fare"] = df["avg_fare"]
    
    # Carbon Metrics
    df["co2_per_seat_km"] = df["aircraft_type"].map(AIRCRAFT_CO2_EMISSIONS).fillna(SEAT_CO2_KG_PER_KM)
    df["co2_per_km"] = df["seats"] * df["co2_per_seat_km"]
    
    # Per Pax Metrics (Avoid div by zero)
    pax = df["reported_passengers"].where(df["reported_passengers"] > 0)
    df["co2_per_pax_km"] = (df["co2_per_km"] / pax).replace([np.inf, -np.inf], np.nan)

    # One-Hot Encoding: Top 10 Aircraft Types
    top_n_aircraft = 10
    top_aircraft_types = df["aircraft_type"].value_counts().head(top_n_aircraft).index.tolist()
    for ac_type in top_aircraft_types:
        df[f"aircraft_type_{ac_type}"] = (df["aircraft_type"] == ac_type).astype(int)

    # Airline Flags
    major_airlines = {"AA", "DL", "UA", "WN", "AS", "B6"}
    df["is_major_airline"] = df["carrier"].isin(major_airlines).astype(int)

    return df


def merge_data(save_excel: bool = True) -> pd.DataFrame:
    """
    Main Pipeline: Loads DB1B & T100, merges them, enriches features, and saves result.
    """
    print("Loading DB1B...")
    db1b = prep_db1b(DB1B_MARKET_PATH)
    
    print("Loading T100...")
    t100 = prep_t100(T100_PATH)
    
    print("Merging datasets...")
    merged = enrich_features(db1b.merge(t100, on=KEYS, how="inner"))

    print(f"Merged rows: {len(merged):,}")
    
    if merged.empty:
        print("CRITICAL WARNING: Merge resulted in 0 rows. Check Years/Airport IDs.")
        return merged
        
    if save_excel:
        output_path = DATA_DIR / "T_DB1B_COUPON_clean.xlsx"
        merged.to_excel(output_path, index=False)
        print(f"Saved merged data to {output_path}")
        
    return merged


# --- VISUALIZATION FUNCTIONS ---

def save_and_show(fig: plt.Figure, name: str) -> None:
    """Saves plot to images/ directory and displays it."""
    fig.tight_layout()
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=200)
    print(f"Saved plot -> {path}")
    plt.show()
    plt.close(fig)

def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Prints descriptive statistics for key columns."""
    cols = [c for c in ["fare", "distance_km", "co2_per_km", "passengers"] if c in df.columns]
    summary = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    print("Metric summary:\n", summary)
    return summary

def bar_missingness(df: pd.DataFrame) -> None:
    """Plots a stacked bar chart of missing vs present values."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values detected.")
        return

    total = len(df)
    present = total - missing
    missing_pct = (missing / total) * 100

    plot_df = pd.DataFrame({'Present': present, 'Missing': missing})
    plot_df = plot_df.sort_values('Missing', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind='bar', stacked=True, ax=ax,
                 color=['#2ecc71', '#e74c3c'], width=0.75)
    
    ax.set_title("Missing Values Analysis")
    ax.set_ylabel("Count")
    
    save_and_show(fig, "missingness")

def hist_plot(df: pd.DataFrame, column: str, log: bool = False) -> None:
    """Plots histogram with optional log scale."""
    if column not in df.columns: return
    data = df[column].dropna()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log1p(data) if log else data, bins=40, color="#1f77b4", alpha=0.8)
    ax.set_title(f"{'Log ' if log else ''}Histogram of {column}")
    save_and_show(fig, f"hist_{'log_' if log else ''}{column}")

def kde_plot(df: pd.DataFrame, column: str, log: bool = False) -> None:
    """Plots Kernel Density Estimate."""
    if column not in df.columns: return
    data = df[column].dropna()
    
    if log:
        data = data[data > 0]
        if data.empty: return
        data = np.log(data)
    
    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), 400)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_grid, kde(x_grid), color="#2ca02c")
    ax.set_title(f"Density of {column}")
    save_and_show(fig, f"kde_{'log_' if log else ''}{column}")

def bar_top(df: pd.DataFrame, column: str, top_n: int, title: str, slug: str) -> None:
    """Plots horizontal bar chart for categorical counts."""
    if column not in df.columns: return
    counts = df[column].value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.sort_values().plot(kind="barh", ax=ax, color="#ff7f0e")
    ax.set_title(title)
    save_and_show(fig, slug)

def scatter_plot(df: pd.DataFrame, x: str, y: str, slug: str) -> None:
    """Plots basic scatter plot."""
    if x not in df.columns or y not in df.columns: return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[x], df[y], alpha=0.3, s=10, color="#9467bd")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    save_and_show(fig, f"scatter_{slug}")