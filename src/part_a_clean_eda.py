"""Part A: Data Foundation, Verification Checks, and Delay Evidence."""

import os
import numpy as np
import pandas as pd
from shipment_prep import FEATURES, NOISE_COLUMNS, build_features, clean_shipments


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    raw_rows = len(raw)

    dupes_mask = raw.duplicated()
    duplicate_rows_removed = int(dupes_mask.sum())
    raw_deduped = raw.drop_duplicates().copy()
    rows_after_dedup = len(raw_deduped)
    missing_distance_km_before = int(raw_deduped["distance_km"].isna().sum())
    missing_parcel_weight_before = int(raw_deduped["parcel_weight_kg"].isna().sum())
    sentinel_weight_rows = int((raw_deduped["parcel_weight_kg"] > 100).sum())

    cleaned_val_raw = (
        raw_deduped["declared_value"]
        .astype(str)
        .str.replace("\u20b9", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    missing_declared_value_before = int(
        pd.to_numeric(cleaned_val_raw, errors="coerce").isna().sum()
    )
    missing_payment_mode_before = int(raw_deduped["payment_mode"].isna().sum())

    service_type_spellings_before = int(raw_deduped["service_type"].nunique())
    clean_df, medians = clean_shipments(raw)
    service_type_spellings_after = int(clean_df["service_type"].nunique())
    model_df = build_features(clean_df, hubs)
    rows_after_hub_merge = len(model_df)
    model_df["is_delayed"] = (
        model_df["delivery_hours"] > model_df["promised_hours"]
    ).astype(int)
    model_df.to_csv("data/modelling_table.csv", index=False)
    nulls_in_features_and_target = int(
        model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
    )
    delayed_count = int(model_df["is_delayed"].sum())
    delay_rate = round(float(model_df["is_delayed"].mean()), 4)
    q1 = float(model_df["delivery_hours"].quantile(0.25))
    q3 = float(model_df["delivery_hours"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(
        (
            (model_df["delivery_hours"] < lower_bound)
            | (model_df["delivery_hours"] > upper_bound)
        ).sum()
    )

    dec_val_mean = round(float(model_df["declared_value_inr"].mean()), 2)
    dec_val_median = round(float(model_df["declared_value_inr"].median()), 2)
    log_val_mean = round(float(model_df["log_declared_value"].mean()), 4)
    log_val_median = round(float(model_df["log_declared_value"].median()), 4)

    checks_records = [
        {"metric": "raw_rows", "value": raw_rows},
        {"metric": "duplicate_rows_removed", "value": duplicate_rows_removed},
        {"metric": "rows_after_dedup", "value": rows_after_dedup},
        {"metric": "missing_distance_km_before", "value": missing_distance_km_before},
        {"metric": "missing_parcel_weight_before", "value": missing_parcel_weight_before},
        {"metric": "sentinel_weight_rows", "value": sentinel_weight_rows},
        {"metric": "missing_declared_value_before", "value": missing_declared_value_before},
        {"metric": "missing_payment_mode_before", "value": missing_payment_mode_before},
        {"metric": "service_type_spellings_before", "value": service_type_spellings_before},
        {"metric": "service_type_spellings_after", "value": service_type_spellings_after},
        {"metric": "median_distance_km", "value": round(medians["distance_km"], 2)},
        {"metric": "median_parcel_weight_kg", "value": round(medians["parcel_weight_kg"], 2)},
        {"metric": "median_declared_value_inr", "value": round(medians["declared_value_inr"], 2)},
        {"metric": "rows_after_hub_merge", "value": rows_after_hub_merge},
        {"metric": "nulls_in_features_and_target", "value": nulls_in_features_and_target},
        {"metric": "delayed_count", "value": delayed_count},
        {"metric": "delay_rate", "value": delay_rate},
        {"metric": "hours_q1", "value": round(q1, 2)},
        {"metric": "hours_q3", "value": round(q3, 2)},
        {"metric": "hours_lower_bound", "value": round(lower_bound, 2)},
        {"metric": "hours_upper_bound", "value": round(upper_bound, 2)},
        {"metric": "hours_outlier_count", "value": outlier_count},
        {"metric": "declared_value_mean_inr", "value": dec_val_mean},
        {"metric": "declared_value_median_inr", "value": dec_val_median},
        {"metric": "log_declared_value_mean", "value": log_val_mean},
        {"metric": "log_declared_value_median", "value": log_val_median},
    ]

    checks_df = pd.DataFrame(checks_records)
    checks_df.to_csv("outputs/part_a_checks.csv", index=False)
    print("outputs/part_a_checks.csv generated successfully.")

    model_df["distance_band"] = pd.cut(
        model_df["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
    )

    evidence_rows = []
    dimensions = ["service_type", "hub_tier", "distance_band"]

    for dim in dimensions:
        grouped = model_df.groupby(dim, observed=True)
        for grp_val, grp_data in grouped:
            shipments = len(grp_data)
            delayed = int(grp_data["is_delayed"].sum())
            d_rate = round(float(delayed / shipments), 4)
            mean_h = round(float(grp_data["delivery_hours"].mean()), 4)
            evidence_rows.append({
                "dimension": dim,
                "group": str(grp_val),
                "shipments": shipments,
                "delayed": delayed,
                "delay_rate": d_rate,
                "mean_hours": mean_h,
            })

    evidence_df = pd.DataFrame(evidence_rows)
    evidence_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    print("outputs/part_a_delay_by_group.csv generated successfully.")
    numeric_cols = model_df.select_dtypes(include=[np.number]).columns
    corr_matrix = model_df[numeric_cols].corr().round(4)
    corr_matrix.to_csv("outputs/part_a_correlations.csv", index=True)
    print("outputs/part_a_correlations.csv generated successfully.")


if __name__ == "__main__":
    main()