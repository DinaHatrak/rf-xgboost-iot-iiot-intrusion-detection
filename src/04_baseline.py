import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

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
#
# We calculate AP for:
#   Attack = positive class
#   Normal = positive class
#
# Then average them.
# This is more informative for our imbalanced datasets.
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

    macro_ap = (ap_attack + ap_normal) / 2

    return ap_attack, ap_normal, macro_ap


# =========================================================
# Evaluation
# =========================================================
def evaluate_model(
    dataset_name,
    train_file,
    val_file,
    target_col,
    metadata_cols,
    target_mapping=None
):

    print("\n" + "=" * 80)
    print(f"BASELINE: {dataset_name}")
    print("=" * 80)

    # -----------------------------------------------------
    # Load TRAIN and VALIDATION only
    # TEST is intentionally untouched.
    # -----------------------------------------------------
    train = pd.read_csv(
        train_file,
        low_memory=False
    )

    val = pd.read_csv(
        val_file,
        low_memory=False
    )

    print("\nTrain shape:", train.shape)
    print("Validation shape:", val.shape)

    # -----------------------------------------------------
    # Target conversion
    # -----------------------------------------------------
    if target_mapping is not None:

        train[target_col] = train[target_col].map(
            target_mapping
        )

        val[target_col] = val[target_col].map(
            target_mapping
        )

    if train[target_col].isna().any():
        raise ValueError(
            f"Unexpected target value in {dataset_name} TRAIN."
        )

    if val[target_col].isna().any():
        raise ValueError(
            f"Unexpected target value in {dataset_name} VALIDATION."
        )

    y_train = train[target_col].astype(int)
    y_val = val[target_col].astype(int)

    # -----------------------------------------------------
    # Remove target + metadata
    # -----------------------------------------------------
    excluded = [target_col] + metadata_cols

    feature_cols = [
        c for c in train.columns
        if c not in excluded
    ]

    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()

    print("Model feature count:", len(feature_cols))

    # -----------------------------------------------------
    # Identify numeric / categorical columns
    # -----------------------------------------------------
    numeric_cols = (
        X_train
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    categorical_cols = [
        c for c in feature_cols
        if c not in numeric_cols
    ]

    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", len(categorical_cols))

    if categorical_cols:
        print("\nCategorical features:")
        for c in categorical_cols:
            print(
                f" - {c}: "
                f"{X_train[c].nunique(dropna=False)} train categories"
            )

    # -----------------------------------------------------
    # Numeric preprocessing
    #
    # IMPORTANT:
    # Median imputation and scaling are FIT ON TRAIN ONLY.
    # -----------------------------------------------------
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # -----------------------------------------------------
    # Categorical preprocessing
    #
    # OneHotEncoder is FIT ON TRAIN ONLY.
    # Unknown validation values are safely ignored.
    # -----------------------------------------------------
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10
                )
            ),
        ]
    )

    # -----------------------------------------------------
    # Combined preprocessing
    # -----------------------------------------------------
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
    # Logistic Regression baseline
    #
    # class_weight='balanced' is useful because both
    # datasets are strongly imbalanced.
    # -----------------------------------------------------
    model = LogisticRegression(
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )

    # =====================================================
    # TRAINING TIME
    # Includes preprocessing fit + model fit
    # =====================================================
    print("\nTraining Logistic Regression...")

    train_start = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train
    )

    train_time = (
        time.perf_counter() - train_start
    )

    print(
        f"Training time: {train_time:.3f} seconds"
    )

    # =====================================================
    # INFERENCE TIME
    # Includes preprocessing + probabilities/predictions
    # =====================================================
    inference_start = time.perf_counter()

    val_prob = pipeline.predict_proba(
        X_val
    )[:, 1]

    val_pred = (
        val_prob >= 0.5
    ).astype(int)

    inference_time = (
        time.perf_counter() - inference_start
    )

    inference_per_sample_ms = (
        inference_time /
        len(X_val) *
        1000
    )

    # =====================================================
    # Metrics
    # =====================================================
    accuracy = accuracy_score(
        y_val,
        val_pred
    )

    macro_precision = precision_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    attack_precision = precision_score(
        y_val,
        val_pred,
        pos_label=1,
        zero_division=0
    )

    attack_recall = recall_score(
        y_val,
        val_pred,
        pos_label=1,
        zero_division=0
    )

    attack_f1 = f1_score(
        y_val,
        val_pred,
        pos_label=1,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        val_prob
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

    # =====================================================
    # Display
    # =====================================================
    print("\n" + "-" * 70)
    print("VALIDATION RESULTS")
    print("-" * 70)

    print(f"Accuracy        : {accuracy:.6f}")
    print(f"Macro Precision : {macro_precision:.6f}")
    print(f"Macro Recall    : {macro_recall:.6f}")
    print(f"Macro F1        : {macro_f1:.6f}")

    print(f"\nAttack Precision: {attack_precision:.6f}")
    print(f"Attack Recall   : {attack_recall:.6f}")
    print(f"Attack F1       : {attack_f1:.6f}")

    print(f"\nROC-AUC         : {roc_auc:.6f}")

    print(f"PR-AUC Attack   : {ap_attack:.6f}")
    print(f"PR-AUC Normal   : {ap_normal:.6f}")
    print(f"Macro PR-AUC    : {macro_ap:.6f}")

    print("\nConfusion Matrix:")
    print(cm)

    print(
        f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}"
    )

    print(
        f"\nTraining time       : {train_time:.3f} sec"
    )

    print(
        f"Inference time      : {inference_time:.6f} sec"
    )

    print(
        "Inference/sample    : "
        f"{inference_per_sample_ms:.6f} ms"
    )

    # =====================================================
    # Result row
    # =====================================================
    result = {
        "Dataset": dataset_name,
        "Model": "Logistic Regression",
        "Evaluation_Set": "Validation",
        "Threshold": 0.5,

        "Accuracy": accuracy,

        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,

        "Attack_Precision": attack_precision,
        "Attack_Recall": attack_recall,
        "Attack_F1": attack_f1,

        "ROC_AUC": roc_auc,

        "PR_AUC_Attack": ap_attack,
        "PR_AUC_Normal": ap_normal,
        "Macro_PR_AUC": macro_ap,

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,

        "Training_Time_sec": train_time,
        "Inference_Time_sec": inference_time,
        "Inference_per_sample_ms":
            inference_per_sample_ms
    }

    return result


# =========================================================
# Run Edge-IIoTset
# =========================================================
edge_result = evaluate_model(

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
# Run IoTID20
# =========================================================
iot_result = evaluate_model(

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
# Save results
# =========================================================
results = pd.DataFrame(
    [
        edge_result,
        iot_result
    ]
)

output_file = (
    RESULTS_DIR /
    "04_baseline_validation_results.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("BASELINE COMPLETE")
print("=" * 80)

print("\nSummary:")
print(
    results[
        [
            "Dataset",
            "Model",
            "Macro_F1",
            "Macro_PR_AUC",
            "ROC_AUC",
            "Training_Time_sec",
            "Inference_per_sample_ms"
        ]
    ].to_string(index=False)
)

print("\nResults saved to:")
print(output_file)

print("\nIMPORTANT:")
print("TEST SETS HAVE NOT BEEN USED.")