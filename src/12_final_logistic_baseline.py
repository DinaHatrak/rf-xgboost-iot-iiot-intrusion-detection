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
from sklearn.exceptions import ConvergenceWarning

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

warnings.simplefilter("always", ConvergenceWarning)


# =========================================================
# PATHS
# =========================================================
PROJECT = Path(
    r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost"
)

SPLITS = PROJECT / "splits"
RESULTS = PROJECT / "results"

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================
SEEDS = [1, 2, 3]
THRESHOLD = 0.5

# Increased only to allow convergence.
# This is NOT hyperparameter tuning.
MAX_ITER = 5000


# =========================================================
# METRICS
# =========================================================
def macro_ap(
    y_true,
    prob_attack
):

    ap_attack = average_precision_score(
        y_true,
        prob_attack
    )

    ap_normal = average_precision_score(
        1 - y_true,
        1 - prob_attack
    )

    macro_average_precision = (
        ap_attack + ap_normal
    ) / 2

    return (
        ap_attack,
        ap_normal,
        macro_average_precision
    )


# =========================================================
# DATASET RUNNER
# =========================================================
def run_logistic_final(
    dataset_name,
    prefix,
    target_col,
    metadata_cols,
    target_mapping=None
):

    print("\n" + "=" * 95)

    print(
        f"FINAL LOGISTIC REGRESSION BASELINE: "
        f"{dataset_name}"
    )

    print("=" * 95)

    # -----------------------------------------------------
    # Load fixed partitions
    # -----------------------------------------------------
    train = pd.read_csv(
        SPLITS / f"{prefix}_train.csv",
        low_memory=False
    )

    val = pd.read_csv(
        SPLITS / f"{prefix}_val.csv",
        low_memory=False
    )

    test = pd.read_csv(
        SPLITS / f"{prefix}_test.csv",
        low_memory=False
    )

    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

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

        test[target_col] = (
            test[target_col]
            .map(target_mapping)
        )

    # -----------------------------------------------------
    # Train + Validation
    # -----------------------------------------------------
    final_train = pd.concat(
        [
            train,
            val
        ],
        ignore_index=True
    )

    print(
        "\nFinal training set:",
        final_train.shape
    )

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------
    excluded = (
        [target_col]
        +
        metadata_cols
    )

    feature_cols = [
        c for c in final_train.columns
        if c not in excluded
    ]

    X_train = (
        final_train[
            feature_cols
        ].copy()
    )

    X_test = (
        test[
            feature_cols
        ].copy()
    )

    y_train = (
        final_train[
            target_col
        ]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test[
            target_col
        ]
        .astype(int)
        .to_numpy()
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
        "Model features:",
        len(feature_cols)
    )

    print(
        "Numeric:",
        len(numeric_cols)
    )

    print(
        "Categorical:",
        len(categorical_cols)
    )

    # =====================================================
    # PREPROCESSING
    # =====================================================
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
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

    print(
        "\nFitting preprocessing "
        "on Train+Validation only..."
    )

    start = time.perf_counter()

    X_train_t = (
        preprocessor
        .fit_transform(
            X_train
        )
    )

    preprocessing_train_time = (
        time.perf_counter()
        -
        start
    )

    start = time.perf_counter()

    X_test_t = (
        preprocessor
        .transform(
            X_test
        )
    )

    preprocessing_test_time = (
        time.perf_counter()
        -
        start
    )

    print(
        "Transformed train:",
        X_train_t.shape
    )

    print(
        "Transformed test:",
        X_test_t.shape
    )

    print(
        f"Preprocessing train: "
        f"{preprocessing_train_time:.4f} s"
    )

    print(
        f"Preprocessing test: "
        f"{preprocessing_test_time:.4f} s"
    )

    rows = []

    # =====================================================
    # FINAL SEEDS
    # =====================================================
    for seed in SEEDS:

        print(
            "\n" + "-" * 80
        )

        print(
            f"{dataset_name} | "
            f"Logistic Regression | "
            f"Seed {seed}"
        )

        model = LogisticRegression(

            solver="saga",

            max_iter=MAX_ITER,

            class_weight="balanced",

            random_state=seed
        )

        # -------------------------------------------------
        # Training
        # -------------------------------------------------
        start = time.perf_counter()

        model.fit(
            X_train_t,
            y_train
        )

        training_time = (
            time.perf_counter()
            -
            start
        )

        # -------------------------------------------------
        # Convergence check
        # -------------------------------------------------
        n_iter_used = int(
            np.max(
                model.n_iter_
            )
        )

        converged = (
            n_iter_used
            <
            MAX_ITER
        )

        print(
            f"Iterations used : "
            f"{n_iter_used}"
        )

        print(
            f"Converged       : "
            f"{converged}"
        )

        if not converged:

            print(
                "WARNING: Logistic Regression "
                f"reached MAX_ITER={MAX_ITER}."
            )

        # -------------------------------------------------
        # Inference
        # -------------------------------------------------
        start = time.perf_counter()

        probability = (
            model
            .predict_proba(
                X_test_t
            )[:, 1]
        )

        inference_time = (
            time.perf_counter()
            -
            start
        )

        prediction = (
            probability >= THRESHOLD
        ).astype(int)

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------
        accuracy = accuracy_score(
            y_test,
            prediction
        )

        macro_precision = precision_score(
            y_test,
            prediction,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            y_test,
            prediction,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            prediction,
            average="macro",
            zero_division=0
        )

        attack_precision = precision_score(
            y_test,
            prediction,
            pos_label=1,
            zero_division=0
        )

        attack_recall = recall_score(
            y_test,
            prediction,
            pos_label=1,
            zero_division=0
        )

        attack_f1 = f1_score(
            y_test,
            prediction,
            pos_label=1,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probability
        )

        (
            ap_attack,
            ap_normal,
            macro_average_precision
        ) = macro_ap(
            y_test,
            probability
        )

        cm = confusion_matrix(
            y_test,
            prediction,
            labels=[0, 1]
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        inference_per_sample_ms = (
            inference_time
            /
            len(y_test)
            *
            1000
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------
        print(
            f"Macro-F1        : "
            f"{macro_f1:.6f}"
        )

        print(
            f"Macro-AP        : "
            f"{macro_average_precision:.6f}"
        )

        print(
            f"ROC-AUC         : "
            f"{roc_auc:.6f}"
        )

        print(
            f"Macro Precision : "
            f"{macro_precision:.6f}"
        )

        print(
            f"Macro Recall    : "
            f"{macro_recall:.6f}"
        )

        print(
            f"Attack Recall   : "
            f"{attack_recall:.6f}"
        )

        print(
            f"Training time   : "
            f"{training_time:.3f} sec"
        )

        print(
            f"Inference/sample: "
            f"{inference_per_sample_ms:.6f} ms"
        )

        print(
            f"CM: TN={tn}, "
            f"FP={fp}, "
            f"FN={fn}, "
            f"TP={tp}"
        )

        rows.append({

            "Dataset":
                dataset_name,

            "Model":
                "Logistic Regression",

            "Seed":
                seed,

            "Threshold":
                THRESHOLD,

            "Max_Iter":
                MAX_ITER,

            "N_Iter_Used":
                n_iter_used,

            "Converged":
                converged,

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

            "AP_Attack":
                ap_attack,

            "AP_Normal":
                ap_normal,

            "Macro_AP":
                macro_average_precision,

            "TN":
                int(tn),

            "FP":
                int(fp),

            "FN":
                int(fn),

            "TP":
                int(tp),

            "Model_Training_Time_sec":
                training_time,

            "Inference_per_sample_ms":
                inference_per_sample_ms,

            "Preprocessing_Train_sec":
                preprocessing_train_time,

            "Preprocessing_Test_sec":
                preprocessing_test_time
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# EDGE-IIoTset
# =========================================================
edge_results = run_logistic_final(

    dataset_name="Edge-IIoTset",

    prefix="edgeiiot",

    target_col="Attack_label",

    metadata_cols=[
        "Attack_type"
    ],

    target_mapping=None
)


# =========================================================
# IoTID20
# =========================================================
iot_results = run_logistic_final(

    dataset_name="IoTID20",

    prefix="iotid20",

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
# COMBINE
# =========================================================
all_results = pd.concat(
    [
        edge_results,
        iot_results
    ],
    ignore_index=True
)


# =========================================================
# SAVE PER-SEED
# =========================================================
per_seed_file = (
    RESULTS /
    "12_logistic_final_test_per_seed_convergence_checked.csv"
)

all_results.to_csv(
    per_seed_file,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================
summary = (

    all_results

    .groupby(
        [
            "Dataset",
            "Model"
        ]
    )

    .agg(

        Macro_F1_Mean=(
            "Macro_F1",
            "mean"
        ),

        Macro_F1_SD=(
            "Macro_F1",
            "std"
        ),

        Macro_AP_Mean=(
            "Macro_AP",
            "mean"
        ),

        Macro_AP_SD=(
            "Macro_AP",
            "std"
        ),

        ROC_AUC_Mean=(
            "ROC_AUC",
            "mean"
        ),

        ROC_AUC_SD=(
            "ROC_AUC",
            "std"
        ),

        Macro_Precision_Mean=(
            "Macro_Precision",
            "mean"
        ),

        Macro_Precision_SD=(
            "Macro_Precision",
            "std"
        ),

        Macro_Recall_Mean=(
            "Macro_Recall",
            "mean"
        ),

        Macro_Recall_SD=(
            "Macro_Recall",
            "std"
        ),

        Attack_Recall_Mean=(
            "Attack_Recall",
            "mean"
        ),

        Attack_Recall_SD=(
            "Attack_Recall",
            "std"
        ),

        Training_Time_Mean_sec=(
            "Model_Training_Time_sec",
            "mean"
        ),

        Training_Time_SD_sec=(
            "Model_Training_Time_sec",
            "std"
        ),

        Inference_ms_Mean=(
            "Inference_per_sample_ms",
            "mean"
        ),

        Inference_ms_SD=(
            "Inference_per_sample_ms",
            "std"
        ),

        N_Iter_Mean=(
            "N_Iter_Used",
            "mean"
        ),

        N_Iter_Max=(
            "N_Iter_Used",
            "max"
        ),

        All_Runs_Converged=(
            "Converged",
            "all"
        )
    )

    .reset_index()
)


summary_file = (
    RESULTS /
    "12_logistic_final_test_summary_convergence_checked.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# =========================================================
# FINAL DISPLAY
# =========================================================
print("\n" + "=" * 105)

print(
    "FINAL LOGISTIC REGRESSION BASELINE — "
    "CONVERGENCE CHECKED"
)

print("=" * 105)

print(
    summary
    .round(6)
    .to_string(
        index=False
    )
)

print("\nSaved:")
print(per_seed_file)
print(summary_file)


# =========================================================
# FINAL SAFETY MESSAGE
# =========================================================
if all(
    all_results[
        "Converged"
    ]
):

    print(
        "\nPASS: All Logistic Regression runs "
        "converged before reaching MAX_ITER."
    )

    print(
        "These are the LR numbers to use "
        "in the final manuscript."
    )

else:

    print(
        "\nWARNING: At least one Logistic Regression "
        "run reached MAX_ITER."
    )

    print(
        "Do not finalize the LR baseline yet."
    )