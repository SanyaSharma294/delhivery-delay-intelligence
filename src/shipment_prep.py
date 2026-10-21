"""Shared prep functions and constants for the Delhivery Delay Intelligence Pipeline."""

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw: pd.DataFrame, medians: dict = None) -> tuple:
    """Clean raw shipments DataFrame and return (clean_df, medians)."""
    df = raw.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    if "service_type" in df.columns:
        df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    if "payment_mode" in df.columns:
        df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    if "declared_value" in df.columns:
        cleaned_val = (
            df["declared_value"]
            .astype(str)
            .str.replace("\u20b9", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["declared_value_inr"] = pd.to_numeric(cleaned_val, errors="coerce")
        df = df.drop(columns=["declared_value"])
    if "dispatch_time" in df.columns:
        df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    if "parcel_weight_kg" in df.columns:
        df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }

    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])

    return df, medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Left merge hubs onto clean shipments and construct pipeline features."""
    df = clean.copy()

    # Left merge hubs table
    df = df.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id",
        how="left",
    )
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"].astype(float))

    return df
