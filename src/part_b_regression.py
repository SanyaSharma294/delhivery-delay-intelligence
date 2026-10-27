"""Part B: Delivery-Hours Regression Pipeline."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from shipment_prep import FEATURES


def main():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES].copy()
    y = df["delivery_hours"].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    train_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_train_scaled, axis=0))))
    test_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_test_scaled, axis=0))))
    checks_df = pd.DataFrame([
        {"metric": "train_rows", "value": len(X_train)},
        {"metric": "val_rows", "value": len(X_val)},
        {"metric": "test_rows", "value": len(X_test)},
        {"metric": "train_scaled_max_abs_mean", "value": round(train_scaled_max_abs_mean, 6)},
        {"metric": "test_scaled_max_abs_mean", "value": round(test_scaled_max_abs_mean, 4)},
    ])
    checks_df.to_csv("outputs/part_b_checks.csv", index=False)
    print("outputs/part_b_checks.csv written.")
    depth_candidates = [1, 2, 3, 4, 6, 8, 12, None]
    tree_depth_records = []
    best_val_rmse = float("inf")
    best_depth = None

    for d in depth_candidates:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)

        train_pred = dt.predict(X_train)
        val_pred = dt.predict(X_val)

        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        label = "unlimited" if d is None else str(d)
        tree_depth_records.append({
            "max_depth": label,
            "train_rmse": round(train_rmse, 4),
            "val_rmse": round(val_rmse, 4),
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_depth = d

    pd.DataFrame(tree_depth_records).to_csv("outputs/part_b_tree_depth.csv", index=False)
    print(f"outputs/part_b_tree_depth.csv written. (Best depth: {best_depth})")
    mean_val_pred = np.full(shape=len(y_val), fill_value=y_train.mean())
    baseline_mae = mean_absolute_error(y_val, mean_val_pred)
    baseline_mse = mean_squared_error(y_val, mean_val_pred)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(y_val, mean_val_pred)

    models = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_records = [
        {
            "model": "mean_baseline",
            "split": "val",
            "MAE": round(baseline_mae, 4),
            "MSE": round(baseline_mse, 4),
            "RMSE": round(baseline_rmse, 4),
            "R2": round(baseline_r2, 4),
        }
    ]

    val_rmses = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)

        val_rmses[name] = rmse
        metrics_records.append({
            "model": name,
            "split": "val",
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        })
    lr_candidates = [0.01, 0.1, 1.0]
    lr_records = []

    for lr in lr_candidates:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        preds = gb.predict(X_val)

        v_rmse = np.sqrt(mean_squared_error(y_val, preds))
        v_r2 = r2_score(y_val, preds)

        lr_records.append({
            "learning_rate": lr,
            "val_rmse": round(v_rmse, 4),
            "val_r2": round(v_r2, 4),
        })

    pd.DataFrame(lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    print("outputs/part_b_gb_learning_rate.csv written.")
    ranked_candidates = sorted(val_rmses.items(), key=lambda item: item[1])
    top_two = [item[0] for item in ranked_candidates[:2]]
    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []

    for model_name in top_two:
        m = models[model_name]
        scores = cross_val_score(m, X_train_val, y_train_val, cv=cv, scoring="r2")
        cv_records.append({
            "model": model_name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })

    pd.DataFrame(cv_records).to_csv("outputs/part_b_cv.csv", index=False)
    print("outputs/part_b_cv.csv written.")
    final_model_name = ranked_candidates[0][0]
    final_model = models[final_model_name]
    test_preds = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_preds)

    metrics_records.append({
        "model": final_model_name,
        "split": "test",
        "MAE": round(test_mae, 4),
        "MSE": round(test_mse, 4),
        "RMSE": round(test_rmse, 4),
        "R2": round(test_r2, 4),
    })

    pd.DataFrame(metrics_records).to_csv("outputs/part_b_metrics.csv", index=False)
    print(f"outputs/part_b_metrics.csv written. (Winning model: {final_model_name})")


if __name__ == "__main__":
    main()