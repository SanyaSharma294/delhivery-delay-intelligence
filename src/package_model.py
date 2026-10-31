"""Part D Task 5: Package Model, Threshold, Feature Order, Impute Values, and Verification."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from shipment_prep import FEATURES, clean_shipments

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    n_estimators = int(cfg["n_estimators"])
    max_d_raw = cfg["max_depth"]
    max_depth = None if str(max_d_raw).strip().lower() in ["unlimited", "none", "nan"] else int(max_d_raw)
    min_samples_leaf = int(cfg["min_samples_leaf"])
    threshold = float(cfg["threshold"])
    selector_k = int(cfg["selector_k"])
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES].copy()
    y = df["is_delayed"].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    pipeline = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
        ("model", rf),
    ])
    pipeline.fit(X_train, y_train)
    in_memory_probs = pipeline.predict_proba(X_test)[:, 1]
    joblib.dump(pipeline, "models/delay_model.pkl")

    pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)

    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)
    impute_df = pd.DataFrame([
        {"column": col, "median": round(float(val), 4)}
        for col, val in medians.items()
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)
    print("Artifacts saved into models/ successfully.")
    reloaded_pipeline = joblib.load("models/delay_model.pkl")
    reloaded_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]
    diff = float(np.max(np.abs(in_memory_probs - reloaded_probs)))
    print(f"Reload test max absolute diff: {diff}")
    reload_check = pd.DataFrame([
        {"metric": "test_rows_scored", "value": len(X_test)},
        {"metric": "reload_max_abs_probability_diff", "value": round(diff, 10)},
        {"metric": "threshold", "value": threshold},
    ])
    reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)
    print("outputs/part_d_reload_check.csv written.")


if __name__ == "__main__":
    main()