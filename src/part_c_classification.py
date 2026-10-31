"""Part C: Delay Classifier and Alert Threshold Pipeline."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from shipment_prep import FEATURES, NOISE_COLUMNS

def get_cm_metrics(y_true, y_pred, zero_division=0):
    """Return TN, FP, FN, TP, accuracy, precision, recall, f1 rounded appropriately."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = round(accuracy_score(y_true, y_pred), 4)
    prec = round(precision_score(y_true, y_pred, zero_division=zero_division), 4)
    rec = round(recall_score(y_true, y_pred, zero_division=zero_division), 4)
    f1 = round(f1_score(y_true, y_pred, zero_division=zero_division), 4)
    return int(tn), int(fp), int(fn), int(tp), acc, prec, rec, f1
def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES].copy()
    y = df["is_delayed"].copy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Delayed in train / val / test: {int(y_train.sum())} / {int(y_val.sum())} / {int(y_test.sum())}")
    models_records = []
    blind_preds = np.zeros(len(y_val), dtype=int)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, blind_preds, zero_division=0)
    models_records.append({
        "model": "blind_all_on_time",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, lr_preds)
    models_records.append({
        "model": "logistic_regression",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })
    depths = [2, 3, 4, 5, 6, 8, 10]
    best_dt_f1 = -1.0
    best_dt_depth = None
    best_dt_preds = None

    for d in depths:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_val)
        score_f1 = f1_score(y_val, dt_preds, zero_division=0)
        if score_f1 > best_dt_f1:
            best_dt_f1 = score_f1
            best_dt_depth = d
            best_dt_preds = dt_preds
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, best_dt_preds)
    models_records.append({
        "model": f"decision_tree_depth_{best_dt_depth}",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)
    rf_preds = rf_default.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, rf_preds)
    models_records.append({
        "model": "random_forest_default",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    pd.DataFrame(models_records).to_csv("outputs/part_c_models.csv", index=False)
    print("outputs/part_c_models.csv written.")
    rf_importances = rf_default.feature_importances_
    selector_all = SelectKBest(score_func=f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_

    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": np.round(f_scores, 4),
        "rf_importance": np.round(rf_importances, 4),
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
    })
    feature_scores_df["f_rank"] = feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    feature_scores_df = feature_scores_df.sort_values(by="f_rank").reset_index(drop=True)
    feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
    print("outputs/part_c_feature_scores.csv written.")
    k_values = [5, 7, 10, 12, 17]
    selectk_records = []
    for k in k_values:
        k_pipe = Pipeline([
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        k_pipe.fit(X_train, y_train)
        val_pred = k_pipe.predict(X_val)
        val_f1 = round(f1_score(y_val, val_pred, zero_division=0), 4)
        selectk_records.append({"k": k, "val_f1": val_f1})
    pd.DataFrame(selectk_records).to_csv("outputs/part_c_selectk.csv", index=False)
    print("outputs/part_c_selectk.csv written.")
    selector_7 = SelectKBest(score_func=f_classif, k=7)
    X_train_k7 = selector_7.fit_transform(X_train, y_train)
    X_val_k7 = selector_7.transform(X_val)
    X_test_k7 = selector_7.transform(X_test)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring=None, 
    )
    grid_search.fit(X_train_k7, y_train)
    grid_val_preds = grid_search.predict(X_val_k7)
    grid_val_f1 = round(f1_score(y_val, grid_val_preds, zero_division=0), 4)
    rand_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42,
        scoring=None,
    )
    rand_search.fit(X_train_k7, y_train)
    rand_val_preds = rand_search.predict(X_val_k7)
    rand_val_f1 = round(f1_score(y_val, rand_val_preds, zero_division=0), 4)
    untuned_forest_k7 = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42
    )
    untuned_forest_k7.fit(X_train_k7, y_train)
    untuned_val_preds = untuned_forest_k7.predict(X_val_k7)
    untuned_val_f1 = round(f1_score(y_val, untuned_val_preds, zero_division=0), 4)

    tuning_records = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid_search.best_params_),
            "best_cv_accuracy": round(float(grid_search.best_score_), 4),
            "val_f1": grid_val_f1,
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rand_search.best_params_),
            "best_cv_accuracy": round(float(rand_search.best_score_), 4),
            "val_f1": rand_val_f1,
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": untuned_val_f1,
        },
    ]

    tuning_df = pd.DataFrame(tuning_records)
    tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)
    print("outputs/part_c_tuning.csv written.")
    candidate_order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
    tuning_dict = {row["search"]: row for _, row in tuning_df.iterrows()}

    best_config_name = max(
        candidate_order,
        key=lambda name: (tuning_dict[name]["val_f1"], -candidate_order.index(name))
    )

    if best_config_name == "default_forest_same_7_features":
        chosen_rf_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    elif best_config_name == "grid_search":
        chosen_rf_params = grid_search.best_params_
    else:
        chosen_rf_params = rand_search.best_params_

    final_model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **chosen_rf_params))
    ])
    final_model.fit(X_train, y_train)
    val_probs = final_model.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_records = []

    chosen_threshold = 0.1
    best_t_found = False
    for t in sorted(thresholds, reverse=True):
        t_preds = (val_probs >= t).astype(int)
        r = recall_score(y_val, t_preds, zero_division=0)
        if r >= 0.80 and not best_t_found:
            chosen_threshold = t
            best_t_found = True

    for t in thresholds:
        t_preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, t_preds)
        thresh_records.append({
            "threshold": t,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        })

    pd.DataFrame(thresh_records).to_csv("outputs/part_c_thresholds.csv", index=False)
    print(f"outputs/part_c_thresholds.csv written. (Chosen threshold: {chosen_threshold})")
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_test, test_preds)

    test_record = [{
        "threshold": chosen_threshold,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    }]
    pd.DataFrame(test_record).to_csv("outputs/part_c_final_test.csv", index=False)
    print("outputs/part_c_final_test.csv written.")

    max_d_repr = "unlimited" if chosen_rf_params.get("max_depth") is None else chosen_rf_params["max_depth"]
    config_record = [{
        "selector_k": 7,
        "n_estimators": chosen_rf_params["n_estimators"],
        "max_depth": max_d_repr,
        "min_samples_leaf": chosen_rf_params["min_samples_leaf"],
        "threshold": chosen_threshold,
        "chosen_from": best_config_name,
    }]
    pd.DataFrame(config_record).to_csv("outputs/part_c_final_config.csv", index=False)
    print("outputs/part_c_final_config.csv written.")


if __name__ == "__main__":
    main()