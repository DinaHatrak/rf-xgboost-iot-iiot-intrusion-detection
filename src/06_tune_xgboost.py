import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================================================
# Paths
# =========================================================
PROJECT_DIR = Path(
    r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost"
)

SPLIT_DIR = PROJECT_DIR / "splits"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Macro PR-AUC
# =========================================================
def macro_pr_auc(y_true, prob_attack):

    y_true = np.asarray(y_true)
    prob_attack = np.asarray(prob_attack)

    ap_attack = average_precision_score(
        y_true,
        prob_attack
    )

    ap_normal = average_precision_score(
        1 - y_true,
        1 - prob_attack
    )

    macro_ap = (
        ap_attack + ap_normal
    ) / 2

    return ap_attack, ap_normal, macro_ap


# =========================================================
# Small XGBoost tuning grid
# =========================================================
XGB_CONFIGS = [

    {
        "config_id": "XGB01",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.10,
        "subsample": 1.0,
        "colsample_bytree": 1.0
    },

    {
        "config_id": "XGB02",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "config_id": "XGB03",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.10,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "config_id": "XGB04",
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "config_id": "XGB05",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.10,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "config_id": "XGB06",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.20,
        "subsample": 1.0,
        "colsample_bytree": 1.0
    },
]


# =========================================================
# Dataset runner
# =========================================================
def tune_xgb(
    dataset_name,
    train_file,
    val_file,
    target_col,
    metadata_cols,
    target_mapping=None
):

    print("\n" + "=" * 85)
    print(f"XGBOOST TUNING: {dataset_name}")
    print("=" * 85)

    # -----------------------------------------------------
    # Load train / validation ONLY
    # -----------------------------------------------------
    train = pd.read_csv(
        train_file,
        low_memory=False
    )

    val = pd.read_csv(
        val_file,
        low_memory=False
    )

    print("Train:", train.shape)
    print("Validation:", val.shape)

    # -----------------------------------------------------
    # Target mapping
    # -----------------------------------------------------
    if target_mapping is not None:

        train[target_col] = (
            train[target_col]
            .map(target_mapping)
        )

        val[target_col] = (
            val[target_col]
            .map(target_mapping)
        )

    y_train = train[target_col].astype(int)
    y_val = val[target_col].astype(int)

    # -----------------------------------------------------
    # Model features
    # -----------------------------------------------------
    excluded = (
        [target_col]
        + metadata_cols
    )

    feature_cols = [
        c for c in train.columns
        if c not in excluded
    ]

    X_train = train[
        feature_cols
    ].copy()

    X_val = val[
        feature_cols
    ].copy()

    print(
        "Model features:",
        len(feature_cols)
    )

    # -----------------------------------------------------
    # Types
    # -----------------------------------------------------
    numeric_cols = (
        X_train
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    categorical_cols = [
        c for c in feature_cols
        if c not in numeric_cols
    ]

    print(
        "Numeric:",
        len(numeric_cols)
    )

    print(
        "Categorical:",
        len(categorical_cols)
    )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_cols
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_cols
            )
        ],
        remainder="drop"
    )

    # -----------------------------------------------------
    # Fit preprocessing TRAIN ONLY
    # -----------------------------------------------------
    print("\nFitting preprocessing...")

    prep_start = time.perf_counter()

    X_train_t = (
        preprocessor
        .fit_transform(X_train)
    )

    X_val_t = (
        preprocessor
        .transform(X_val)
    )

    prep_time = (
        time.perf_counter()
        - prep_start
    )

    print(
        f"Preprocessing time: "
        f"{prep_time:.3f} sec"
    )

    print(
        "Transformed train shape:",
        X_train_t.shape
    )

    # =====================================================
    # Configurations
    # =====================================================
    rows = []

    for i, cfg in enumerate(
        XGB_CONFIGS,
        start=1
    ):

        print("\n" + "-" * 85)

        print(
            f"{dataset_name} | "
            f"{cfg['config_id']} | "
            f"{i}/{len(XGB_CONFIGS)}"
        )

        print("-" * 85)

        print(cfg)

        model = XGBClassifier(

            n_estimators=
                cfg["n_estimators"],

            max_depth=
                cfg["max_depth"],

            learning_rate=
                cfg["learning_rate"],

            subsample=
                cfg["subsample"],

            colsample_bytree=
                cfg["colsample_bytree"],

            objective="binary:logistic",

            eval_metric="logloss",

            tree_method="hist",

            random_state=42,

            n_jobs=-1
        )

        # -------------------------------------------------
        # Training
        # -------------------------------------------------
        start = time.perf_counter()

        model.fit(
            X_train_t,
            y_train
        )

        train_time = (
            time.perf_counter()
            - start
        )

        # -------------------------------------------------
        # Inference
        # -------------------------------------------------
        start = time.perf_counter()

        val_prob = (
            model
            .predict_proba(
                X_val_t
            )[:, 1]
        )

        val_pred = (
            val_prob >= 0.5
        ).astype(int)

        inference_time = (
            time.perf_counter()
            - start
        )

        inference_ms = (
            inference_time
            / len(X_val_t)
            * 1000
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------
        accuracy = accuracy_score(
            y_val,
            val_pred
        )

        macro_precision = (
            precision_score(
                y_val,
                val_pred,
                average="macro",
                zero_division=0
            )
        )

        macro_recall = (
            recall_score(
                y_val,
                val_pred,
                average="macro",
                zero_division=0
            )
        )

        macro_f1 = (
            f1_score(
                y_val,
                val_pred,
                average="macro",
                zero_division=0
            )
        )

        attack_precision = (
            precision_score(
                y_val,
                val_pred,
                pos_label=1,
                zero_division=0
            )
        )

        attack_recall = (
            recall_score(
                y_val,
                val_pred,
                pos_label=1,
                zero_division=0
            )
        )

        attack_f1 = (
            f1_score(
                y_val,
                val_pred,
                pos_label=1,
                zero_division=0
            )
        )

        roc_auc = (
            roc_auc_score(
                y_val,
                val_prob
            )
        )

        (
            ap_attack,
            ap_normal,
            macro_ap
        ) = macro_pr_auc(
            y_val,
            val_prob
        )

        cm = confusion_matrix(
            y_val,
            val_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        # -------------------------------------------------
        # Display
        # -------------------------------------------------
        print(
            f"Macro-F1       : "
            f"{macro_f1:.6f}"
        )

        print(
            f"Macro PR-AUC   : "
            f"{macro_ap:.6f}"
        )

        print(
            f"ROC-AUC        : "
            f"{roc_auc:.6f}"
        )

        print(
            f"Attack Recall  : "
            f"{attack_recall:.6f}"
        )

        print(
            f"Train time     : "
            f"{train_time:.3f} sec"
        )

        print(
            f"Inference/sample: "
            f"{inference_ms:.6f} ms"
        )

        print(
            f"CM: TN={tn}, FP={fp}, "
            f"FN={fn}, TP={tp}"
        )

        rows.append({

            "Dataset":
                dataset_name,

            "Config_ID":
                cfg["config_id"],

            "n_estimators":
                cfg["n_estimators"],

            "max_depth":
                cfg["max_depth"],

            "learning_rate":
                cfg["learning_rate"],

            "subsample":
                cfg["subsample"],

            "colsample_bytree":
                cfg["colsample_bytree"],

            "Accuracy":
                accuracy,

            "Macro_Precision":
                macro_precision,

            "Macro_Recall":
                macro_recall,

            "Macro_F1":
                macro_f1,

            "Attack_Precision":
                attack_precision,

            "Attack_Recall":
                attack_recall,

            "Attack_F1":
                attack_f1,

            "ROC_AUC":
                roc_auc,

            "PR_AUC_Attack":
                ap_attack,

            "PR_AUC_Normal":
                ap_normal,

            "Macro_PR_AUC":
                macro_ap,

            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,

            "Training_Time_sec":
                train_time,

            "Inference_Time_sec":
                inference_time,

            "Inference_per_sample_ms":
                inference_ms
        })

    # =====================================================
    # Ranking
    # PRIMARY = Macro-F1
    # TIE-BREAKER = Macro PR-AUC
    # =====================================================
    results = pd.DataFrame(
        rows
    )

    results = (
        results
        .sort_values(
            by=[
                "Macro_F1",
                "Macro_PR_AUC"
            ],
            ascending=[
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )

    best = results.iloc[0]

    print("\n" + "=" * 85)
    print(
        f"BEST XGBOOST CONFIGURATION: "
        f"{dataset_name}"
    )
    print("=" * 85)

    print(
        results[
            [
                "Config_ID",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "Macro_F1",
                "Macro_PR_AUC",
                "ROC_AUC",
                "Training_Time_sec"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSELECTED:")

    print(
        best[
            [
                "Config_ID",
                "Macro_F1",
                "Macro_PR_AUC",
                "ROC_AUC"
            ]
        ]
    )

    # -----------------------------------------------------
    # Save tuning table
    # -----------------------------------------------------
    output_csv = (
        RESULTS_DIR /
        f"06_xgb_tuning_{dataset_name.lower()}.csv"
    )

    results.to_csv(
        output_csv,
        index=False
    )

    # -----------------------------------------------------
    # Save best config
    # -----------------------------------------------------
    selected_id = (
        best["Config_ID"]
    )

    selected_config = next(
        cfg for cfg in XGB_CONFIGS
        if cfg["config_id"]
        == selected_id
    )

    output_json = (
        RESULTS_DIR /
        f"06_xgb_best_{dataset_name.lower()}.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selected_config,
            f,
            indent=4
        )

    print("\nSaved:")
    print(output_csv)
    print(output_json)

    return results


# =========================================================
# Edge-IIoTset
# =========================================================
edge_results = tune_xgb(

    dataset_name="Edge-IIoTset",

    train_file=(
        SPLIT_DIR /
        "edgeiiot_train.csv"
    ),

    val_file=(
        SPLIT_DIR /
        "edgeiiot_val.csv"
    ),

    target_col="Attack_label",

    metadata_cols=[
        "Attack_type"
    ],

    target_mapping=None
)


# =========================================================
# IoTID20
# =========================================================
iot_results = tune_xgb(

    dataset_name="IoTID20",

    train_file=(
        SPLIT_DIR /
        "iotid20_train.csv"
    ),

    val_file=(
        SPLIT_DIR /
        "iotid20_val.csv"
    ),

    target_col="Label",

    metadata_cols=[
        "Cat",
        "Sub_Cat"
    ],

    target_mapping={
        "Normal": 0,
        "Anomaly": 1
    }
)


# =========================================================
# Combined
# =========================================================
combined = pd.concat(
    [
        edge_results,
        iot_results
    ],
    ignore_index=True
)

combined_file = (
    RESULTS_DIR /
    "06_xgb_tuning_all.csv"
)

combined.to_csv(
    combined_file,
    index=False
)

print("\n" + "=" * 85)
print("XGBOOST TUNING COMPLETE")
print("=" * 85)

print("\nTEST SETS HAVE NOT BEEN USED.")

print("\nCombined results:")
print(combined_file)