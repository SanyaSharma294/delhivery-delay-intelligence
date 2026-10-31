"""Part D Task 6: Operational Scoring Script on Unseen Shipments."""

import os
import joblib
import pandas as pd
from shipment_prep import clean_shipments, build_features

def main():
    os.makedirs("outputs", exist_ok=True)
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")
    pipeline = joblib.load("models/delay_model.pkl")
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute_records = pd.read_csv("models/impute_values.csv").set_index("column")["median"].to_dict()
    clean_df, _ = clean_shipments(new_raw, medians=impute_records)
    feat_df = build_features(clean_df, hubs)
    X_score = feat_df[feature_order].copy()
    probs = pipeline.predict_proba(X_score)[:, 1]
    feat_df["delay_probability"] = probs.round(4)
    feat_df["flagged"] = (feat_df["delay_probability"] >= threshold).astype(int)
    feat_df = feat_df.merge(
        hub_segments[["hub_id", "segment"]],
        on="hub_id",
        how="left"
    )
    def assign_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            return "Proactive customer call"
        return "No action"

    feat_df["action"] = feat_df.apply(assign_action, axis=1)
    out_cols = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    actions_df = feat_df[out_cols].copy()
    actions_df.to_csv("outputs/new_shipment_actions.csv", index=False)
    print("outputs/new_shipment_actions.csv written successfully.")


if __name__ == "__main__":
    main()