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

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================================================
# PATHS
# =========================================================
PROJECT_DIR = Path(
    r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost"
)

SPLIT_DIR = PROJECT_DIR / "splits"
RESULTS_DIR = PROJECT_DIR / "results"
PRED_DIR = RESULTS_DIR / "predictions"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FINAL SEEDS
# =========================================================
SEEDS = [1, 2, 3]

THRESHOLD = 0.5


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

    return (
        ap_attack,
        ap_normal,
        macro_ap
    )


# =========================================================
# Load JSON configuration
# =========================================================
def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Configuration not found:\n{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =========================================================
# DATASET RUNNER
# =========================================================
def run_final_experiment(

    dataset_name,
    prefix,
    target_col,
    metadata_cols,
    rf_config_file,
    xgb_config_file,
    target_mapping=None
):

    print("\n" + "=" * 90)
    print(f"FINAL TEST EXPERIMENT: {dataset_name}")
    print("=" * 90)

    # -----------------------------------------------------
    # Load fixed splits
    # -----------------------------------------------------
    train = pd.read_csv(
        SPLIT_DIR / f"{prefix}_train.csv",
        low_memory=False
    )

    val = pd.read_csv(
        SPLIT_DIR / f"{prefix}_val.csv",
        low_memory=False
    )

    test = pd.read_csv(
        SPLIT_DIR / f"{prefix}_test.csv",
        low_memory=False
    )

    print("\nOriginal fixed splits:")
    print("Train      :", train.shape)
    print("Validation :", val.shape)
    print("Test       :", test.shape)

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
    # FINAL training data:
    # Train + Validation
    #
    # Hyperparameters are already frozen.
    # Test remains completely separate.
    # -----------------------------------------------------
    train_final = pd.concat(
        [train, val],
        ignore_index=True
    )

    print("\nFinal training set:")
    print(train_final.shape)

    print("\nFixed final test set:")
    print(test.shape)

    # -----------------------------------------------------
    # Targets
    # -----------------------------------------------------
    y_train = (
        train_final[target_col]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test[target_col]
        .astype(int)
        .to_numpy()
    )

    # -----------------------------------------------------
    # Model features
    # -----------------------------------------------------
    excluded = (
        [target_col]
        + metadata_cols
    )

    feature_cols = [
        c for c in train_final.columns
        if c not in excluded
    ]

    X_train = train_final[
        feature_cols
    ].copy()

    X_test = test[
        feature_cols
    ].copy()

    print(
        "\nModel feature count:",
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
        "Numeric features:",
        len(numeric_cols)
    )

    print(
        "Categorical features:",
        len(categorical_cols)
    )

    # =====================================================
    # PREPROCESSING
    #
    # Fit ONLY using final training set.
    # Test is transformed only.
    # =====================================================
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

    print("\nFitting final preprocessing...")

    prep_train_start = time.perf_counter()

    X_train_t = (
        preprocessor
        .fit_transform(X_train)
    )

    prep_train_time = (
        time.perf_counter()
        - prep_train_start
    )

    prep_test_start = time.perf_counter()

    X_test_t = (
        preprocessor
        .transform(X_test)
    )

    prep_test_time = (
        time.perf_counter()
        - prep_test_start
    )

    print(
        f"Preprocessing fit/transform train: "
        f"{prep_train_time:.4f} sec"
    )

    print(
        f"Preprocessing test transform: "
        f"{prep_test_time:.4f} sec"
    )

    print(
        "Transformed train:",
        X_train_t.shape
    )

    print(
        "Transformed test :",
        X_test_t.shape
    )

    # =====================================================
    # LOAD FROZEN HYPERPARAMETERS
    # =====================================================
    rf_cfg = load_json(
        rf_config_file
    )

    xgb_cfg = load_json(
        xgb_config_file
    )

    print("\nFrozen RF configuration:")
    print(rf_cfg)

    print("\nFrozen XGBoost configuration:")
    print(xgb_cfg)

    all_rows = []

    # =====================================================
    # MODELS
    # =====================================================
    for model_name in [
        "Random Forest",
        "XGBoost"
    ]:

        print("\n" + "#" * 90)
        print(model_name)
        print("#" * 90)

        for seed in SEEDS:

            print(
                f"\n{dataset_name} | "
                f"{model_name} | "
                f"Seed {seed}"
            )

            # -------------------------------------------------
            # Build model
            # -------------------------------------------------
            if model_name == "Random Forest":

                model = RandomForestClassifier(

                    n_estimators=
                        rf_cfg["n_estimators"],

                    max_depth=
                        rf_cfg["max_depth"],

                    min_samples_leaf=
                        rf_cfg["min_samples_leaf"],

                    max_features=
                        rf_cfg["max_features"],

                    class_weight=
                        rf_cfg["class_weight"],

                    random_state=seed,

                    n_jobs=-1
                )

            else:

                model = XGBClassifier(

                    n_estimators=
                        xgb_cfg["n_estimators"],

                    max_depth=
                        xgb_cfg["max_depth"],

                    learning_rate=
                        xgb_cfg["learning_rate"],

                    subsample=
                        xgb_cfg["subsample"],

                    colsample_bytree=
                        xgb_cfg["colsample_bytree"],

                    objective=
                        "binary:logistic",

                    eval_metric=
                        "logloss",

                    tree_method=
                        "hist",

                    random_state=seed,

                    n_jobs=-1
                )

            # =================================================
            # MODEL TRAINING TIME
            # =================================================
            train_start = time.perf_counter()

            model.fit(
                X_train_t,
                y_train
            )

            model_train_time = (
                time.perf_counter()
                - train_start
            )

            # =================================================
            # MODEL INFERENCE TIME
            # =================================================
            infer_start = time.perf_counter()

            test_prob = (
                model
                .predict_proba(
                    X_test_t
                )[:, 1]
            )

            model_inference_time = (
                time.perf_counter()
                - infer_start
            )

            test_pred = (
                test_prob >= THRESHOLD
            ).astype(int)

            infer_per_sample_ms = (
                model_inference_time
                / len(X_test_t)
                * 1000
            )

            # =================================================
            # METRICS
            # =================================================
            accuracy = accuracy_score(
                y_test,
                test_pred
            )

            macro_precision = precision_score(
                y_test,
                test_pred,
                average="macro",
                zero_division=0
            )

            macro_recall = recall_score(
                y_test,
                test_pred,
                average="macro",
                zero_division=0
            )

            macro_f1 = f1_score(
                y_test,
                test_pred,
                average="macro",
                zero_division=0
            )

            attack_precision = precision_score(
                y_test,
                test_pred,
                pos_label=1,
                zero_division=0
            )

            attack_recall = recall_score(
                y_test,
                test_pred,
                pos_label=1,
                zero_division=0
            )

            attack_f1 = f1_score(
                y_test,
                test_pred,
                pos_label=1,
                zero_division=0
            )

            roc_auc = roc_auc_score(
                y_test,
                test_prob
            )

            (
                ap_attack,
                ap_normal,
                macro_ap
            ) = macro_pr_auc(
                y_test,
                test_prob
            )

            cm = confusion_matrix(
                y_test,
                test_pred,
                labels=[0, 1]
            )

            tn, fp, fn, tp = (
                cm.ravel()
            )

            # =================================================
            # COMPUTATIONAL COST
            # =================================================
            end_to_end_train_time = (
                prep_train_time
                + model_train_time
            )

            end_to_end_inference_time = (
                prep_test_time
                + model_inference_time
            )

            # =================================================
            # DISPLAY
            # =================================================
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
                f"Macro Precision: "
                f"{macro_precision:.6f}"
            )

            print(
                f"Macro Recall   : "
                f"{macro_recall:.6f}"
            )

            print(
                f"Attack Recall  : "
                f"{attack_recall:.6f}"
            )

            print(
                f"Model train    : "
                f"{model_train_time:.4f} sec"
            )

            print(
                f"Inference/sample: "
                f"{infer_per_sample_ms:.6f} ms"
            )

            print(
                f"CM: TN={tn}, FP={fp}, "
                f"FN={fn}, TP={tp}"
            )

            # =================================================
            # SAVE PREDICTIONS
            # =================================================
            prediction_data = {

                "row_id":
                    np.arange(
                        len(test)
                    ),

                "y_true":
                    y_test,

                "prob_attack":
                    test_prob,

                "y_pred":
                    test_pred
            }

            # Add metadata for later error analysis
            for meta in metadata_cols:

                if meta in test.columns:

                    prediction_data[
                        meta
                    ] = (
                        test[meta]
                        .to_numpy()
                    )

            predictions = pd.DataFrame(
                prediction_data
            )

            safe_model = (
                model_name
                .lower()
                .replace(" ", "_")
            )

            pred_file = (
                PRED_DIR /
                f"{prefix}_{safe_model}_seed{seed}.csv"
            )

            predictions.to_csv(
                pred_file,
                index=False
            )

            # =================================================
            # SAVE ROW
            # =================================================
            all_rows.append({

                "Dataset":
                    dataset_name,

                "Model":
                    model_name,

                "Seed":
                    seed,

                "Threshold":
                    THRESHOLD,

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

                "TN":
                    tn,

                "FP":
                    fp,

                "FN":
                    fn,

                "TP":
                    tp,

                "Preprocessing_Train_sec":
                    prep_train_time,

                "Preprocessing_Test_sec":
                    prep_test_time,

                "Model_Training_Time_sec":
                    model_train_time,

                "Model_Inference_Time_sec":
                    model_inference_time,

                "Inference_per_sample_ms":
                    infer_per_sample_ms,

                "End_to_End_Training_sec":
                    end_to_end_train_time,

                "End_to_End_Inference_sec":
                    end_to_end_inference_time
            })

    return pd.DataFrame(
        all_rows
    )


# =========================================================
# EDGE-IIoTset
# =========================================================
edge_results = run_final_experiment(

    dataset_name="Edge-IIoTset",

    prefix="edgeiiot",

    target_col="Attack_label",

    metadata_cols=[
        "Attack_type"
    ],

    rf_config_file=(
        RESULTS_DIR /
        "05_rf_best_edge-iiotset.json"
    ),

    xgb_config_file=(
        RESULTS_DIR /
        "06_xgb_best_edge-iiotset.json"
    ),

    target_mapping=None
)


# =========================================================
# IoTID20
# =========================================================
iot_results = run_final_experiment(

    dataset_name="IoTID20",

    prefix="iotid20",

    target_col="Label",

    metadata_cols=[
        "Cat",
        "Sub_Cat"
    ],

    rf_config_file=(
        RESULTS_DIR /
        "05_rf_best_iotid20.json"
    ),

    xgb_config_file=(
        RESULTS_DIR /
        "06_xgb_best_iotid20.json"
    ),

    target_mapping={
        "Normal": 0,
        "Anomaly": 1
    }
)


# =========================================================
# COMBINE ALL FINAL TEST RESULTS
# =========================================================
all_results = pd.concat(
    [
        edge_results,
        iot_results
    ],
    ignore_index=True
)

per_seed_file = (
    RESULTS_DIR /
    "07_final_test_per_seed.csv"
)

all_results.to_csv(
    per_seed_file,
    index=False
)


# =========================================================
# MEAN ± SD SUMMARY
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

        Macro_PR_AUC_Mean=(
            "Macro_PR_AUC",
            "mean"
        ),

        Macro_PR_AUC_SD=(
            "Macro_PR_AUC",
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
        )

    )

    .reset_index()
)


summary_file = (
    RESULTS_DIR /
    "07_final_test_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# =========================================================
# DISPLAY FINAL SUMMARY
# =========================================================
print("\n" + "=" * 100)
print("FINAL TEST RESULTS — 3 SEEDS")
print("=" * 100)

pd.set_option(
    "display.max_columns",
    None
)

print(
    summary.round(6).to_string(
        index=False
    )
)


# =========================================================
# SIMPLE WINNER REPORT
# =========================================================
print("\n" + "=" * 100)
print("WINNERS BY DATASET — MACRO-F1")
print("=" * 100)

for dataset in summary[
    "Dataset"
].unique():

    subset = (
        summary[
            summary["Dataset"]
            == dataset
        ]
        .sort_values(
            "Macro_F1_Mean",
            ascending=False
        )
    )

    winner = subset.iloc[0]

    print(
        f"\n{dataset}: "
        f"{winner['Model']} "
        f"(Macro-F1 = "
        f"{winner['Macro_F1_Mean']:.6f} "
        f"± "
        f"{winner['Macro_F1_SD']:.6f})"
    )


print("\nSaved per-seed results:")
print(per_seed_file)

print("\nSaved summary:")
print(summary_file)

print("\nPredictions saved in:")
print(PRED_DIR)

print("\nFINAL TEST EVALUATION COMPLETE.")