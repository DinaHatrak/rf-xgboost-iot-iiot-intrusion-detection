import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
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
PROJECT = Path(
    r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost"
)

SPLITS = PROJECT / "splits"
RESULTS = PROJECT / "results"
PRED = RESULTS / "predictions"
ANALYSIS = RESULTS / "analysis"

ANALYSIS.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================
SEEDS = [1, 2, 3]

THRESHOLD = 0.5

ABLATION_FEATURES = [
    "tcp.seq",
    "tcp.ack",
    "tcp.checksum",
    "tcp.ack_raw"
]

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 321


# =========================================================
# HELPERS
# =========================================================
def section(title):

    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def macro_pr_auc(
    y_true,
    prob_attack
):

    ap_attack = (
        average_precision_score(
            y_true,
            prob_attack
        )
    )

    ap_normal = (
        average_precision_score(
            1 - y_true,
            1 - prob_attack
        )
    )

    return (
        ap_attack,
        ap_normal,
        (
            ap_attack
            +
            ap_normal
        ) / 2
    )


# =========================================================
# PART 1
# XGBOOST SEED AUDIT
# =========================================================
def audit_seeds(
    dataset_name,
    prefix
):

    section(
        f"XGBOOST SEED AUDIT — {dataset_name}"
    )

    dfs = {}

    for seed in SEEDS:

        file = (
            PRED /
            f"{prefix}_xgboost_seed{seed}.csv"
        )

        dfs[seed] = pd.read_csv(
            file
        )

        print(
            f"Loaded Seed {seed}: "
            f"{len(dfs[seed])} rows"
        )

    # Verify same fixed test set
    reference_y = (
        dfs[1]["y_true"]
        .to_numpy()
    )

    for seed in [2, 3]:

        if not np.array_equal(
            reference_y,
            dfs[seed]["y_true"].to_numpy()
        ):

            raise ValueError(
                "Test-set order changed."
            )

    rows = []

    for a, b in [
        (1, 2),
        (1, 3),
        (2, 3)
    ]:

        prob_a = (
            dfs[a]["prob_attack"]
            .to_numpy()
        )

        prob_b = (
            dfs[b]["prob_attack"]
            .to_numpy()
        )

        pred_a = (
            dfs[a]["y_pred"]
            .to_numpy()
        )

        pred_b = (
            dfs[b]["y_pred"]
            .to_numpy()
        )

        abs_diff = np.abs(
            prob_a - prob_b
        )

        label_diff = int(
            np.sum(
                pred_a != pred_b
            )
        )

        probability_equal = (
            np.array_equal(
                prob_a,
                prob_b
            )
        )

        rows.append({

            "Dataset":
                dataset_name,

            "Seed_A":
                a,

            "Seed_B":
                b,

            "Different_Hard_Predictions":
                label_diff,

            "Probabilities_Exactly_Equal":
                probability_equal,

            "Mean_Absolute_Probability_Difference":
                abs_diff.mean(),

            "Max_Absolute_Probability_Difference":
                abs_diff.max()
        })

        print(
            f"\nSeed {a} vs Seed {b}"
        )

        print(
            "Different hard predictions:",
            label_diff
        )

        print(
            "Probabilities exactly equal:",
            probability_equal
        )

        print(
            "Mean |prob difference|:",
            f"{abs_diff.mean():.10f}"
        )

        print(
            "Max |prob difference|:",
            f"{abs_diff.max():.10f}"
        )

    result = pd.DataFrame(
        rows
    )

    output = (
        ANALYSIS /
        f"10_seed_audit_{prefix}.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    # Interpretation
    same_labels = (
        result[
            "Different_Hard_Predictions"
        ] == 0
    ).all()

    probability_changed = (
        ~result[
            "Probabilities_Exactly_Equal"
        ]
    ).any()

    print("\nINTERPRETATION:")

    if (
        same_labels
        and
        probability_changed
    ):

        print(
            "PASS: seeds changed probability outputs, "
            "but no sample crossed threshold 0.5."
        )

        print(
            "Therefore Macro-F1/Precision/Recall can "
            "legitimately have SD = 0."
        )

    elif (
        same_labels
        and
        not probability_changed
    ):

        print(
            "WARNING: probabilities and labels are "
            "exactly identical across seeds."
        )

    else:

        print(
            "PASS: seeds changed hard predictions."
        )

    print("\nSaved:")
    print(output)


# =========================================================
# PREPROCESSOR
# =========================================================
def build_preprocessor(
    X_train
):

    numeric_cols = (
        X_train
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    categorical_cols = [

        c for c
        in X_train.columns

        if c not in numeric_cols
    ]

    numeric_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipe = Pipeline(
        [
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

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipe,
                numeric_cols
            ),

            (
                "categorical",
                categorical_pipe,
                categorical_cols
            )
        ],

        remainder="drop"
    )


# =========================================================
# METRICS
# =========================================================
def calculate_metrics(
    y_true,
    probability
):

    prediction = (
        probability
        >= THRESHOLD
    ).astype(int)

    (
        ap_attack,
        ap_normal,
        macro_ap
    ) = macro_pr_auc(
        y_true,
        probability
    )

    cm = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    return {

        "Macro_Precision":
            precision_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0
            ),

        "Macro_Recall":
            recall_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0
            ),

        "Macro_F1":
            f1_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0
            ),

        "Attack_Recall":
            recall_score(
                y_true,
                prediction,
                pos_label=1,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                probability
            ),

        "PR_AUC_Attack":
            ap_attack,

        "PR_AUC_Normal":
            ap_normal,

        "Macro_PR_AUC":
            macro_ap,

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp)
    }


# =========================================================
# PART 2
# EDGE ABLATION
# =========================================================
def edge_ablation():

    section(
        "EDGE-IIoTset POST-HOC ABLATION"
    )

    train = pd.read_csv(
        SPLITS /
        "edgeiiot_train.csv",
        low_memory=False
    )

    val = pd.read_csv(
        SPLITS /
        "edgeiiot_val.csv",
        low_memory=False
    )

    test = pd.read_csv(
        SPLITS /
        "edgeiiot_test.csv",
        low_memory=False
    )

    train_final = pd.concat(
        [
            train,
            val
        ],
        ignore_index=True
    )

    target = "Attack_label"

    missing = [

        c for c
        in ABLATION_FEATURES

        if c
        not in train_final.columns
    ]

    if missing:

        raise ValueError(
            f"Missing features: {missing}"
        )

    excluded = [

        target,
        "Attack_type",

        *ABLATION_FEATURES
    ]

    feature_cols = [

        c for c
        in train_final.columns

        if c not in excluded
    ]

    print(
        "Removed:",
        ABLATION_FEATURES
    )

    print(
        "Original feature count: 38"
    )

    print(
        "Ablated feature count:",
        len(feature_cols)
    )

    X_train = (
        train_final[
            feature_cols
        ].copy()
    )

    X_test = (
        test[
            feature_cols
        ].copy()
    )

    y_train = (
        train_final[
            target
        ]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test[
            target
        ]
        .astype(int)
        .to_numpy()
    )

    # -----------------------------------------------------
    # Fit preprocessing on Train+Val only
    # -----------------------------------------------------
    preprocessor = (
        build_preprocessor(
            X_train
        )
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
        - start
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
        - start
    )

    print(
        "Transformed Train:",
        X_train_t.shape
    )

    print(
        "Transformed Test:",
        X_test_t.shape
    )

    # -----------------------------------------------------
    # Frozen hyperparameters — NO RETUNING
    # -----------------------------------------------------
    rf_cfg = load_json(

        RESULTS /
        "05_rf_best_edge-iiotset.json"
    )

    xgb_cfg = load_json(

        RESULTS /
        "06_xgb_best_edge-iiotset.json"
    )

    print(
        "\nFrozen RF:",
        rf_cfg
    )

    print(
        "Frozen XGBoost:",
        xgb_cfg
    )

    rows = []

    for model_name in [
        "Random Forest",
        "XGBoost"
    ]:

        print(
            "\n" + "-" * 80
        )

        print(model_name)

        for seed in SEEDS:

            if (
                model_name
                ==
                "Random Forest"
            ):

                model = (
                    RandomForestClassifier(

                        n_estimators=
                            rf_cfg[
                                "n_estimators"
                            ],

                        max_depth=
                            rf_cfg[
                                "max_depth"
                            ],

                        min_samples_leaf=
                            rf_cfg[
                                "min_samples_leaf"
                            ],

                        max_features=
                            rf_cfg[
                                "max_features"
                            ],

                        class_weight=
                            rf_cfg[
                                "class_weight"
                            ],

                        random_state=
                            seed,

                        n_jobs=-1
                    )
                )

                safe_name = (
                    "random_forest"
                )

            else:

                model = (
                    XGBClassifier(

                        n_estimators=
                            xgb_cfg[
                                "n_estimators"
                            ],

                        max_depth=
                            xgb_cfg[
                                "max_depth"
                            ],

                        learning_rate=
                            xgb_cfg[
                                "learning_rate"
                            ],

                        subsample=
                            xgb_cfg[
                                "subsample"
                            ],

                        colsample_bytree=
                            xgb_cfg[
                                "colsample_bytree"
                            ],

                        objective=
                            "binary:logistic",

                        eval_metric=
                            "logloss",

                        tree_method=
                            "hist",

                        random_state=
                            seed,

                        n_jobs=-1
                    )
                )

                safe_name = (
                    "xgboost"
                )

            # ------------------------------------------------
            # Training
            # ------------------------------------------------
            start = (
                time.perf_counter()
            )

            model.fit(
                X_train_t,
                y_train
            )

            training_time = (
                time.perf_counter()
                - start
            )

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------
            start = (
                time.perf_counter()
            )

            probability = (
                model
                .predict_proba(
                    X_test_t
                )[:, 1]
            )

            inference_time = (
                time.perf_counter()
                - start
            )

            metrics = (
                calculate_metrics(
                    y_test,
                    probability
                )
            )

            rows.append({

                "Dataset":
                    "Edge-IIoTset",

                "Model":
                    model_name,

                "Seed":
                    seed,

                "Removed_Features":
                    ";".join(
                        ABLATION_FEATURES
                    ),

                "Remaining_Model_Features":
                    len(
                        feature_cols
                    ),

                "Training_Time_sec":
                    training_time,

                "Inference_per_sample_ms":
                    (
                        inference_time
                        /
                        len(y_test)
                        *
                        1000
                    ),

                **metrics
            })

            prediction = (
                probability
                >= THRESHOLD
            ).astype(int)

            pred_file = (

                ANALYSIS /

                f"10_edge_ablation_"
                f"{safe_name}_"
                f"seed{seed}.csv"
            )

            pd.DataFrame({

                "row_id":
                    np.arange(
                        len(y_test)
                    ),

                "y_true":
                    y_test,

                "prob_attack":
                    probability,

                "y_pred":
                    prediction,

                "Attack_type":
                    test[
                        "Attack_type"
                    ].to_numpy()

            }).to_csv(
                pred_file,
                index=False
            )

            print(
                f"Seed {seed} | "
                f"Macro-F1="
                f"{metrics['Macro_F1']:.6f} | "
                f"Macro PR-AUC="
                f"{metrics['Macro_PR_AUC']:.6f} | "
                f"ROC-AUC="
                f"{metrics['ROC_AUC']:.6f} | "
                f"Macro Recall="
                f"{metrics['Macro_Recall']:.6f} | "
                f"FP={metrics['FP']} | "
                f"FN={metrics['FN']}"
            )

    # =====================================================
    # SAVE PER-SEED
    # =====================================================
    per_seed = pd.DataFrame(
        rows
    )

    per_seed_file = (

        ANALYSIS /
        "10_edge_ablation_per_seed.csv"
    )

    per_seed.to_csv(
        per_seed_file,
        index=False
    )

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = (

        per_seed

        .groupby(
            "Model"
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

            Macro_Recall_Mean=(
                "Macro_Recall",
                "mean"
            ),

            Attack_Recall_Mean=(
                "Attack_Recall",
                "mean"
            ),

            FP_Mean=(
                "FP",
                "mean"
            ),

            FN_Mean=(
                "FN",
                "mean"
            ),

            Training_Time_Mean_sec=(
                "Training_Time_sec",
                "mean"
            ),

            Inference_ms_Mean=(
                "Inference_per_sample_ms",
                "mean"
            )
        )

        .reset_index()
    )

    summary_file = (

        ANALYSIS /
        "10_edge_ablation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        "\nABLATION SUMMARY:"
    )

    print(
        summary
        .round(6)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # ORIGINAL VS ABLATION
    # =====================================================
    original = pd.read_csv(

        RESULTS /
        "07_final_test_summary.csv"
    )

    original = original[

        original["Dataset"]
        ==
        "Edge-IIoTset"

    ].copy()

    comparison = (

        original[
            [
                "Model",
                "Macro_F1_Mean",
                "Macro_PR_AUC_Mean",
                "ROC_AUC_Mean",
                "Macro_Recall_Mean",
                "Attack_Recall_Mean"
            ]
        ]

        .merge(

            summary[
                [
                    "Model",
                    "Macro_F1_Mean",
                    "Macro_PR_AUC_Mean",
                    "ROC_AUC_Mean",
                    "Macro_Recall_Mean",
                    "Attack_Recall_Mean"
                ]
            ],

            on="Model",

            suffixes=(
                "_Original",
                "_Ablation"
            )
        )
    )

    for metric in [

        "Macro_F1",
        "Macro_PR_AUC",
        "ROC_AUC",
        "Macro_Recall",
        "Attack_Recall"

    ]:

        comparison[
            f"{metric}_Delta"
        ] = (

            comparison[
                f"{metric}_Mean_Ablation"
            ]

            -

            comparison[
                f"{metric}_Mean_Original"
            ]
        )

    comparison_file = (

        ANALYSIS /
        "10_edge_ablation_vs_original.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    print(
        "\nORIGINAL VS ABLATION:"
    )

    print(

        comparison[
            [
                "Model",

                "Macro_F1_Mean_Original",

                "Macro_F1_Mean_Ablation",

                "Macro_F1_Delta",

                "Macro_Recall_Mean_Original",

                "Macro_Recall_Mean_Ablation"
            ]
        ]

        .round(6)

        .to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(per_seed_file)
    print(summary_file)
    print(comparison_file)


# =========================================================
# PART 3
# BOOTSTRAP ON THE ABLATION
# =========================================================
def bootstrap_ablation():

    section(
        "EDGE ABLATION — PAIRED BOOTSTRAP"
    )

    rf = pd.read_csv(

        ANALYSIS /
        "10_edge_ablation_"
        "random_forest_seed1.csv"
    )

    xgb = pd.read_csv(

        ANALYSIS /
        "10_edge_ablation_"
        "xgboost_seed1.csv"
    )

    y = (
        rf["y_true"]
        .to_numpy()
    )

    rf_pred = (
        rf["y_pred"]
        .to_numpy()
    )

    xgb_pred = (
        xgb["y_pred"]
        .to_numpy()
    )

    rf_f1 = f1_score(
        y,
        rf_pred,
        average="macro"
    )

    xgb_f1 = f1_score(
        y,
        xgb_pred,
        average="macro"
    )

    observed_difference = (
        xgb_f1
        -
        rf_f1
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    differences = []

    n = len(y)

    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):

        idx = rng.integers(
            0,
            n,
            size=n
        )

        rf_boot = f1_score(

            y[idx],
            rf_pred[idx],

            average="macro"
        )

        xgb_boot = f1_score(

            y[idx],
            xgb_pred[idx],

            average="macro"
        )

        differences.append(
            xgb_boot
            -
            rf_boot
        )

    differences = np.asarray(
        differences
    )

    lower = np.percentile(
        differences,
        2.5
    )

    upper = np.percentile(
        differences,
        97.5
    )

    probability_positive = (
        differences > 0
    ).mean()

    result = pd.DataFrame(
        [
            {
                "RF_Macro_F1":
                    rf_f1,

                "XGB_Macro_F1":
                    xgb_f1,

                "Difference_XGB_minus_RF":
                    observed_difference,

                "Bootstrap_CI_Lower":
                    lower,

                "Bootstrap_CI_Upper":
                    upper,

                "Bootstrap_Resamples_Favoring_XGB":
                    probability_positive,

                "Bootstrap_Iterations":
                    BOOTSTRAP_ITERATIONS
            }
        ]
    )

    output = (

        ANALYSIS /
        "10_edge_ablation_bootstrap.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    print(
        f"RF Macro-F1      : "
        f"{rf_f1:.6f}"
    )

    print(
        f"XGBoost Macro-F1 : "
        f"{xgb_f1:.6f}"
    )

    print(
        f"Difference       : "
        f"{observed_difference:.6f}"
    )

    print(
        "95% bootstrap CI : "
        f"[{lower:.6f}, "
        f"{upper:.6f}]"
    )

    print(
        "Resamples favoring XGB: "
        f"{probability_positive:.4f}"
    )

    print(
        "\nSaved:",
        output
    )


# =========================================================
# PART 4
# ENVIRONMENT DOCUMENTATION
# =========================================================
def save_environment():

    section(
        "ENVIRONMENT / TIMING DOCUMENTATION"
    )

    environment = f"""
Python: {platform.python_version()}
Operating system: {platform.platform()}
Processor: {platform.processor()}
Logical CPUs: {os.cpu_count()}

NumPy: {np.__version__}
Pandas: {pd.__version__}
scikit-learn: {sklearn.__version__}
XGBoost: {xgboost.__version__}

Random Forest n_jobs = -1
XGBoost n_jobs = -1

Final test batch size = 22,500 samples.
Inference was measured using batch predict_proba.
Preprocessing time was measured separately from model fit/predict time.

The three-seed SD measures training-randomness variability
on the SAME fixed Train/Validation/Test split.

It does not measure split-to-split variability.

The paired bootstrap measures uncertainty associated with
resampling the fixed test set.

It does not measure retraining variability.
"""

    output = (

        ANALYSIS /
        "10_environment_and_timing.txt"
    )

    output.write_text(
        environment,
        encoding="utf-8"
    )

    print(environment)

    print(
        "Saved:",
        output
    )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # -----------------------------------------
    # 1. Verify seed behavior
    # -----------------------------------------
    audit_seeds(
        "Edge-IIoTset",
        "edgeiiot"
    )

    audit_seeds(
        "IoTID20",
        "iotid20"
    )

    # -----------------------------------------
    # 2. Targeted Edge sensitivity analysis
    # -----------------------------------------
    edge_ablation()

    # -----------------------------------------
    # 3. Statistical comparison after ablation
    # -----------------------------------------
    bootstrap_ablation()

    # -----------------------------------------
    # 4. Save reproducibility information
    # -----------------------------------------
    save_environment()

    section(
        "ALL METHODOLOGICAL CHECKS COMPLETE"
    )

    print(
        "No hyperparameter tuning was "
        "performed here."
    )

    print(
        "The ablation is a POST-HOC "
        "SENSITIVITY ANALYSIS."
    )