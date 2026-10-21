
# Delhivery Last-Mile Delay Intelligence Pipeline

## Part A: Data Foundation & Delay Evidence Findings

1. **Service Type Delay Rates**: Express service has the highest delay rate across delivery modes at **48.97%** (119 out of 243 shipments delayed), followed by Standard at **20.56%** (132 out of 642 shipments), and Economy at **8.57%** (27 out of 315 shipments). Express delivery faces substantial breach risks due to tighter promised SLA windows (36 hours) during peak demand.
2. **Hub Tier Disparities**: Hub operational performance degrades sharply by tier. Hub Tier 1 records a **4.00%** delay rate (16 delayed out of 400), Tier 2 increases to **21.50%** (86 delayed out of 400), and Tier 3 surges to **44.00%** (176 delayed out of 400), showing that remote or regional hubs face severe operational bottlenecks.
3. **Distance Band Influence**: Delay rates escalate dramatically across dispatch distances:
   - **0–200 km**: 2.19% delay rate (6 of 274 shipments)
   - **201–400 km**: 13.68% delay rate (77 of 563 shipments)
   - **401–600 km**: 43.64% delay rate (120 of 275 shipments)
   - **601+ km**: 85.23% delay rate (75 of 88 shipments)
   Shipments exceeding 600 km breach promised delivery timelines over 85% of the time.
4. **Top Correlations with Delivery Hours**: Excluding `delivery_hours` itself, the top three features most strongly correlated with delivery duration by absolute magnitude are:
   - `promised_hours`: **+0.8370**
   - `distance_km`: **+0.4010**
   - `hub_tier`: **+0.3240**
5. **Declared Value Log-Transformation Justification**: Prior to transformation, raw `declared_value_inr` is heavily right-skewed with a mean of **INR 1,533.05** substantially exceeding the median of **INR 1,121.50**. Following natural log transformation, `log_declared_value` normalizes to a mean of **7.0280** and median of **7.0220**, preventing extreme parcel valuations from distorting downstream linear and distance-based estimators.


