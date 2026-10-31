"""Part D: Hub Profile, K-Means Clustering, Separation Evidence, and Playbook Segments."""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    hub_profile = (
        df.groupby("hub_id", as_index=False)
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .sort_values(by="hub_id")
        .reset_index(drop=True)
    )
    hub_profile_out = hub_profile.copy()
    for col in [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]:
        hub_profile_out[col] = hub_profile_out[col].round(4)

    hub_profile_out.to_csv("outputs/hub_profile.csv", index=False)
    print("outputs/hub_profile.csv written.")
    profile_cols = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(hub_profile[profile_cols])
    dist_matrix = squareform(pdist(scaled_matrix, metric="euclidean"))
    n_hubs = len(hub_profile)
    separation_records = []
    models_dict = {}

    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        models_dict[k] = (km, labels)
        counts = pd.Series(labels).value_counts().sort_values(ascending=False)
        sizes_str = "-".join(map(str, counts.values))
        within_dists = []
        across_dists = []

        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = dist_matrix[i, j]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        ratio = (np.mean(across_dists) / np.mean(within_dists)) if within_dists else 0.0

        separation_records.append({
            "K": k,
            "across_over_within": round(float(ratio), 4),
            "cluster_sizes": sizes_str,
        })

    pd.DataFrame(separation_records).to_csv("outputs/part_d_separation.csv", index=False)
    print("outputs/part_d_separation.csv written.")
    km_3, labels_3 = models_dict[3]
    hub_profile["cluster"] = labels_3
    cluster_delay = hub_profile.groupby("cluster")["delay_rate"].mean()
    sorted_clusters = cluster_delay.sort_values().index.tolist()
    cluster_to_segment = {
        sorted_clusters[0]: "Standard",
        sorted_clusters[1]: "Watch",
        sorted_clusters[2]: "Escalate",
    }
    hub_profile["segment"] = hub_profile["cluster"].map(cluster_to_segment)
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = hub_profile.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id",
        how="left",
    )
    outputs_hub_segments = hub_segments[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ].sort_values(by="hub_id")
    outputs_hub_segments.to_csv("outputs/hub_segments.csv", index=False)
    print("outputs/hub_segments.csv written.")
    segment_prof = (
        hub_segments.groupby("segment", as_index=False)
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
    )
    seg_order = {"Standard": 0, "Watch": 1, "Escalate": 2}
    segment_prof["order"] = segment_prof["segment"].map(seg_order)
    segment_prof = segment_prof.sort_values(by="order").drop(columns=["order"])

    for col in [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]:
        segment_prof[col] = segment_prof[col].round(4)

    segment_prof.to_csv("outputs/segment_profile.csv", index=False)
    print("outputs/segment_profile.csv written.")
    expected_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_segments["expected_segment"] = hub_segments["hub_tier"].map(expected_segment)
    mismatches = hub_segments[hub_segments["segment"] != hub_segments["expected_segment"]]
    mismatch_df = mismatches[["hub_id", "hub_city", "hub_tier", "segment"]].sort_values(by="hub_id")
    mismatch_df.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
    print("outputs/tier_vs_segment_mismatch.csv written.")
    
if __name__ == "__main__":
    main()