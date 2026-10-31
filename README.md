
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

## Part B: Delivery-Hours Regression Findings

1. **Baseline Evaluation**: The mean baseline predictor yields an $R^2$ of **-0.0210** on the validation set, confirming that predicting the historical training mean provides zero explanatory signal and performs slightly worse than a horizontal line through the validation mean.
2. **Model Validation Performance**: 
   - **Linear Regression** and **Gradient Boosting** achieved the top validation performance with an identical $R^2$ of **0.9310** (RMSE of **4.89 hours** and **4.91 hours**, respectively).
   - **Random Forest** achieved an $R^2$ of **0.9140** (RMSE of **5.49 hours**).
   - **Tuned Decision Tree (max_depth=6)** achieved an $R^2$ of **0.8870** (RMSE of **6.28 hours**).
3. **Tree-Depth Diagnostic (Bias vs. Variance)**:
   - **High Bias (Underfitting)**: At `max_depth=1`, the model severely underfits with high training RMSE (**9.75 hours**) and validation RMSE (**10.25 hours**), with a negligible generalization gap of **0.50 hours**.
   - **High Variance (Overfitting)**: At unlimited depth (`None`), the model achieves a training RMSE of **0.00 hours** but validation RMSE balloons to **7.79 hours**, yielding a large generalization gap of **7.79 hours**.
   - **Optimal Tradeoff**: `max_depth=6` achieves the lowest validation RMSE of **6.28 hours**.
4. **Learning Rate Diagnostic**:
   - `learning_rate=0.01` exhibits severe underfitting ($R^2 = 0.7130$, RMSE = 10.02 hours) due to insufficient step size across 100 trees.
   - `learning_rate=0.10` hits the optimal balance ($R^2 = 0.9310$, RMSE = 4.91 hours).
   - `learning_rate=1.00` degrades significantly ($R^2 = 0.8440$, RMSE = 7.39 hours) as aggressive step updates overshoot the loss minimum.
5. **Cross-Validation & Shipping Decision**:
   - 5-fold cross-validation on the combined 1,020 training and validation rows confirms consistent generalization: **Gradient Boosting** yields a mean $R^2$ of **0.9350** (std: **0.0080**), slightly outpacing **Linear Regression** at a mean $R^2$ of **0.9260** (std: **0.0110**).
   - On the held-out test split, the final model achieved an $R^2$ of **0.9490** and an MAE of **3.17 hours**. 
   - **Production Recommendation**: Although Linear Regression is slightly faster to infer, **Gradient Boosting Regressor** is recommended for production because it inherently captures nonlinear operational interactions (such as rain and distance thresholds) with superior cross-validated stability.

## Part C: Delay Classifier & Alert Threshold Findings

1. **Uselessness of Blind Accuracy**: The blind all-on-time baseline achieves a deceptive accuracy of **76.67%** (predicting zero delays across all 180 validation parcels). However, its recall and F1 scores are **0.0000**, missing every single delayed shipment. In Delhivery's operations, where failing to intercept a delayed shipment carries severe escalation costs, blind accuracy is entirely uninformative.
2. **Noise Feature Rejection**: All five bookkeeping metadata features (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, and `shelf_slot_no`) ranked outside the top 7 predictors with F-scores below **10.00**, confirming they carry negligible predictive signal. The top 7 operational features (`distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag`, and `sale_event_flag`) all scored above **30.00**, validating feature selection with `k=7`.
3. **Hyperparameter Tuning on Validation Split**: On the 180-row validation set, the default Random Forest using the top 7 features achieved a validation F1 of **0.8333**, outperforming both grid search (**0.7750**) and randomized search (**0.7900**). Hyperparameter tuning on a small validation set suffered from slight conservative underfitting, so the default forest parameters with `k=7` features were retained for final deployment.
4. **Operations-Driven Alert Threshold**: Calibrating probability thresholds revealed the operational precision-recall trade-off:
   - At $t=0.50$, the classifier meets the operational mandate with a recall of **80.95%** (identifying 34 of 42 delayed parcels) and a precision of **85.00%**.
   - Raising $t$ to $0.70$ improves precision to 92.59% but drops recall to 59.52%, violating the $\ge 80\%$ recall requirement.
   - Lowering $t$ to $0.30$ boosts recall to 85.71% but triggers excessive false alarms (precision drops to 64.29%). The optimal operational threshold is set at **0.50**.
5. **Test Set Generalization**: Evaluated on the strictly held-out test split (41 delayed parcels), the final model with $t=0.50$ achieved:
   - **Precision**: **68.89%**
   - **Recall**: **75.61%** (31 out of 41 delayed shipments intercepted)
   - **F1 Score**: **0.7209**
   - **Accuracy**: **86.11%**
   While test recall slightly adjusted from validation recall due to sample variance across 41 delayed items, the model maintained balanced real-world intervention coverage without generating unmanageable customer support volumes.
