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

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

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
# Small tuning grid
#
# Intentionally limited.
# This is SIMPLE hyperparameter tuning,
# not a large exhaustive search.
# =========================================================
RF_CONFIGS = [

    {
        "config_id": "RF01",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample"
    },

    {
        "config_id": "RF02",
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample"
    },

    {
        "config_id": "RF03",
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample"
    },

    {
        "config_id": "RF04",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample"
    },

    {
        "config_id": "RF05",
        "n_estimators": 400,
        "max_depth": 30,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample"
    },

    {
        "config_id": "RF06",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None
    },
]


# =========================================================
# Dataset runner
# =========================================================
def tune_rf(
    dataset_name,
    train_file,
    val_file,
    target_col,
    metadata_cols,
    target_mapping=None
):

    print("\n" + "=" * 85)
    print(f"RANDOM FOREST TUNING: {dataset_name}")
    print("=" * 85)

    # -----------------------------------------------------
    # Load Train / Validation ONLY
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
    # Target conversion
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

    y_train = (
        train[target_col]
        .astype(int)
    )

    y_val = (
        val[target_col]
        .astype(int)
    )

    # -----------------------------------------------------
    # Remove target + metadata
    # -----------------------------------------------------
    excluded = (
        [target_col] +
        metadata_cols
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
    # Feature types
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
    #
    # RF does NOT require scaling.
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
            ),
        ],
        remainder="drop"
    )

    # -----------------------------------------------------
    # Fit preprocessing ON TRAIN ONLY
    #
    # We transform once and reuse for all RF configs.
    # This saves substantial runtime.
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
        time.perf_counter() -
        prep_start
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
    # Test each configuration
    # =====================================================
    rows = []

    for i, cfg in enumerate(
        RF_CONFIGS,
        start=1
    ):

        print("\n" + "-" * 85)

        print(
            f"{dataset_name} | "
            f"{cfg['config_id']} | "
            f"{i}/{len(RF_CONFIGS)}"
        )

        print("-" * 85)

        print(cfg)

        # -------------------------------------------------
        # Model
        # -------------------------------------------------
        model = RandomForestClassifier(

            n_estimators=
                cfg["n_estimators"],

            max_depth=
                cfg["max_depth"],

            min_samples_leaf=
                cfg["min_samples_leaf"],

            max_features=
                cfg["max_features"],

            class_weight=
                cfg["class_weight"],

            random_state=42,

            n_jobs=-1
        )

        # -------------------------------------------------
        # Training time
        # -------------------------------------------------
        start = time.perf_counter()

        model.fit(
            X_train_t,
            y_train
        )

        train_time = (
            time.perf_counter() -
            start
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
            time.perf_counter() -
            start
        )

        inference_ms = (
            inference_time /
            len(X_val_t) *
            1000
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

        tn, fp, fn, tp = (
            cm.ravel()
        )

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

        # -------------------------------------------------
        # Store
        # -------------------------------------------------
        rows.append({

            "Dataset":
                dataset_name,

            "Config_ID":
                cfg["config_id"],

            "n_estimators":
                cfg["n_estimators"],

            "max_depth":
                str(cfg["max_depth"]),

            "min_samples_leaf":
                cfg["min_samples_leaf"],

            "max_features":
                cfg["max_features"],

            "class_weight":
                str(
                    cfg["class_weight"]
                ),

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
    # Results table
    # =====================================================
    results = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------
    # PRIMARY selection metric:
    #
    # 1. Macro-F1
    # 2. Macro PR-AUC as tie-breaker
    #
    # TEST IS NOT USED.
    # -----------------------------------------------------
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
        f"BEST RF CONFIGURATION: "
        f"{dataset_name}"
    )
    print("=" * 85)

    print(
        results[
            [
                "Config_ID",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "class_weight",
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
    # Save all candidate results
    # -----------------------------------------------------
    output_csv = (
        RESULTS_DIR /
        f"05_rf_tuning_{dataset_name.lower()}.csv"
    )

    results.to_csv(
        output_csv,
        index=False
    )

    # -----------------------------------------------------
    # Recover selected config
    # -----------------------------------------------------
    selected_id = (
        best["Config_ID"]
    )

    selected_config = next(
        cfg for cfg in RF_CONFIGS
        if cfg["config_id"] ==
        selected_id
    )

    output_json = (
        RESULTS_DIR /
        f"05_rf_best_{dataset_name.lower()}.json"
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
# EDGE-IIoTset
# =========================================================
edge_results = tune_rf(

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
iot_results = tune_rf(

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
# Combined results
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
    "05_rf_tuning_all.csv"
)

combined.to_csv(
    combined_file,
    index=False
)

print("\n" + "=" * 85)
print("RANDOM FOREST TUNING COMPLETE")
print("=" * 85)

print(
    "\nTEST SETS HAVE NOT BEEN USED."
)

print(
    "\nCombined results:"
)

print(combined_file)