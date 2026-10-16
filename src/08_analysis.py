import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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

ANALYSIS_DIR = RESULTS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================
INTERPRETATION_SEED = 1

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 123


# =========================================================
# LOAD JSON
# =========================================================
def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
def feature_importance_analysis(

    dataset_name,
    prefix,
    target_col,
    metadata_cols,
    target_mapping,
    rf_config_file,
    xgb_config_file
):

    print("\n" + "=" * 90)
    print(f"FEATURE IMPORTANCE: {dataset_name}")
    print("=" * 90)

    train = pd.read_csv(
        SPLIT_DIR / f"{prefix}_train.csv",
        low_memory=False
    )

    val = pd.read_csv(
        SPLIT_DIR / f"{prefix}_val.csv",
        low_memory=False
    )

    # Test is NOT needed for feature importance
    data = pd.concat(
        [train, val],
        ignore_index=True
    )

    if target_mapping is not None:

        data[target_col] = (
            data[target_col]
            .map(target_mapping)
        )

    y = (
        data[target_col]
        .astype(int)
        .to_numpy()
    )

    excluded = (
        [target_col]
        + metadata_cols
    )

    feature_cols = [
        c for c in data.columns
        if c not in excluded
    ]

    X = data[
        feature_cols
    ].copy()

    numeric_cols = (
        X
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
        remainder="drop",
        verbose_feature_names_out=True
    )

    print("Fitting preprocessing...")

    X_t = preprocessor.fit_transform(X)

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    print(
        "Transformed feature count:",
        len(feature_names)
    )

    rf_cfg = load_json(
        rf_config_file
    )

    xgb_cfg = load_json(
        xgb_config_file
    )

    models = {

        "Random_Forest":
            RandomForestClassifier(

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

                random_state=
                    INTERPRETATION_SEED,

                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(

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

                random_state=
                    INTERPRETATION_SEED,

                n_jobs=-1
            )
    }

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name} "
            f"for feature importance..."
        )

        model.fit(
            X_t,
            y
        )

        importance = (
            model.feature_importances_
        )

        importance_df = pd.DataFrame({

            "Feature":
                feature_names,

            "Importance":
                importance
        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        output_file = (
            ANALYSIS_DIR /
            f"08_feature_importance_"
            f"{prefix}_{model_name.lower()}.csv"
        )

        importance_df.to_csv(
            output_file,
            index=False
        )

        print("\nTop 15 features:")

        print(
            importance_df
            .head(15)
            .to_string(index=False)
        )

        print("\nSaved:")
        print(output_file)


# =========================================================
# ERROR ANALYSIS
# =========================================================
def error_analysis(

    dataset_name,
    prefix,
    metadata_col
):

    print("\n" + "=" * 90)
    print(f"ERROR ANALYSIS: {dataset_name}")
    print("=" * 90)

    for model_file, model_name in [

        (
            "random_forest",
            "Random Forest"
        ),

        (
            "xgboost",
            "XGBoost"
        )

    ]:

        pred_file = (
            PRED_DIR /
            f"{prefix}_{model_file}_seed1.csv"
        )

        df = pd.read_csv(
            pred_file
        )

        # -------------------------------------------------
        # Error type
        # -------------------------------------------------
        conditions = [

            (
                (df["y_true"] == 0)
                &
                (df["y_pred"] == 0)
            ),

            (
                (df["y_true"] == 0)
                &
                (df["y_pred"] == 1)
            ),

            (
                (df["y_true"] == 1)
                &
                (df["y_pred"] == 0)
            ),

            (
                (df["y_true"] == 1)
                &
                (df["y_pred"] == 1)
            )
        ]

        choices = [
            "TN",
            "FP",
            "FN",
            "TP"
        ]

        df["Error_Type"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        print(
            f"\n{model_name}:"
        )

        print(
            df["Error_Type"]
            .value_counts()
        )

        # -------------------------------------------------
        # Attack-type recall
        # -------------------------------------------------
        if metadata_col in df.columns:

            attacks = df[
                df["y_true"] == 1
            ].copy()

            breakdown = (

                attacks

                .groupby(
                    metadata_col
                )

                .agg(

                    Samples=(
                        "y_true",
                        "size"
                    ),

                    Detected=(
                        "y_pred",
                        "sum"
                    )

                )

                .reset_index()
            )

            breakdown[
                "Missed"
            ] = (
                breakdown["Samples"]
                -
                breakdown["Detected"]
            )

            breakdown[
                "Recall"
            ] = (
                breakdown["Detected"]
                /
                breakdown["Samples"]
            )

            breakdown = (
                breakdown
                .sort_values(
                    "Recall",
                    ascending=True
                )
            )

            print(
                "\nAttack-category recall:"
            )

            print(
                breakdown
                .to_string(
                    index=False
                )
            )

            breakdown_file = (
                ANALYSIS_DIR /
                f"08_error_breakdown_"
                f"{prefix}_{model_file}.csv"
            )

            breakdown.to_csv(
                breakdown_file,
                index=False
            )

            print("\nSaved:")
            print(breakdown_file)

        # -------------------------------------------------
        # Save actual errors
        # -------------------------------------------------
        errors = df[
            df["Error_Type"]
            .isin(
                ["FP", "FN"]
            )
        ].copy()

        error_file = (
            ANALYSIS_DIR /
            f"08_errors_"
            f"{prefix}_{model_file}.csv"
        )

        errors.to_csv(
            error_file,
            index=False
        )


# =========================================================
# PAIRED BOOTSTRAP
#
# Directly estimates confidence interval for:
#
# Macro-F1(XGBoost) - Macro-F1(Random Forest)
# =========================================================
def bootstrap_comparison(
    dataset_name,
    prefix
):

    print("\n" + "=" * 90)
    print(
        f"PAIRED BOOTSTRAP: {dataset_name}"
    )
    print("=" * 90)

    rf = pd.read_csv(

        PRED_DIR /
        f"{prefix}_random_forest_seed1.csv"

    )

    xgb = pd.read_csv(

        PRED_DIR /
        f"{prefix}_xgboost_seed1.csv"

    )

    if not np.array_equal(
        rf["y_true"].to_numpy(),
        xgb["y_true"].to_numpy()
    ):

        raise ValueError(
            "RF and XGBoost test labels differ."
        )

    y_true = (
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

    observed_rf = f1_score(
        y_true,
        rf_pred,
        average="macro"
    )

    observed_xgb = f1_score(
        y_true,
        xgb_pred,
        average="macro"
    )

    observed_diff = (
        observed_xgb
        - observed_rf
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = len(y_true)

    differences = np.empty(
        BOOTSTRAP_ITERATIONS
    )

    print(
        f"Running "
        f"{BOOTSTRAP_ITERATIONS} "
        f"bootstrap iterations..."
    )

    for i in range(
        BOOTSTRAP_ITERATIONS
    ):

        idx = rng.integers(
            0,
            n,
            size=n
        )

        y_boot = y_true[idx]

        rf_boot = rf_pred[idx]

        xgb_boot = xgb_pred[idx]

        rf_f1 = f1_score(
            y_boot,
            rf_boot,
            average="macro"
        )

        xgb_f1 = f1_score(
            y_boot,
            xgb_boot,
            average="macro"
        )

        differences[i] = (
            xgb_f1
            - rf_f1
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

    print(
        f"\nRF Macro-F1       : "
        f"{observed_rf:.6f}"
    )

    print(
        f"XGBoost Macro-F1  : "
        f"{observed_xgb:.6f}"
    )

    print(
        f"Observed difference: "
        f"{observed_diff:.6f}"
    )

    print(
        "95% bootstrap CI  : "
        f"[{lower:.6f}, "
        f"{upper:.6f}]"
    )

    print(
        "Bootstrap P(diff > 0): "
        f"{probability_positive:.4f}"
    )

    return {

        "Dataset":
            dataset_name,

        "RF_Macro_F1":
            observed_rf,

        "XGB_Macro_F1":
            observed_xgb,

        "Difference_XGB_minus_RF":
            observed_diff,

        "Bootstrap_CI_Lower":
            lower,

        "Bootstrap_CI_Upper":
            upper,

        "Bootstrap_Probability_Diff_Positive":
            probability_positive,

        "Bootstrap_Iterations":
            BOOTSTRAP_ITERATIONS
    }


# =========================================================
# FEATURE IMPORTANCE — EDGE
# =========================================================
feature_importance_analysis(

    dataset_name="Edge-IIoTset",

    prefix="edgeiiot",

    target_col="Attack_label",

    metadata_cols=[
        "Attack_type"
    ],

    target_mapping=None,

    rf_config_file=(
        RESULTS_DIR /
        "05_rf_best_edge-iiotset.json"
    ),

    xgb_config_file=(
        RESULTS_DIR /
        "06_xgb_best_edge-iiotset.json"
    )
)


# =========================================================
# FEATURE IMPORTANCE — IoTID20
# =========================================================
feature_importance_analysis(

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
    },

    rf_config_file=(
        RESULTS_DIR /
        "05_rf_best_iotid20.json"
    ),

    xgb_config_file=(
        RESULTS_DIR /
        "06_xgb_best_iotid20.json"
    )
)


# =========================================================
# ERROR ANALYSIS
# =========================================================
error_analysis(

    dataset_name="Edge-IIoTset",

    prefix="edgeiiot",

    metadata_col="Attack_type"
)


error_analysis(

    dataset_name="IoTID20",

    prefix="iotid20",

    metadata_col="Sub_Cat"
)


# =========================================================
# BOOTSTRAP COMPARISON
# =========================================================
bootstrap_rows = []

bootstrap_rows.append(

    bootstrap_comparison(
        "Edge-IIoTset",
        "edgeiiot"
    )

)

bootstrap_rows.append(

    bootstrap_comparison(
        "IoTID20",
        "iotid20"
    )

)

bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_file = (
    ANALYSIS_DIR /
    "08_bootstrap_comparison.csv"
)

bootstrap_df.to_csv(
    bootstrap_file,
    index=False
)


# =========================================================
# FINAL
# =========================================================
print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

print("\nBootstrap summary:")

print(
    bootstrap_df
    .round(6)
    .to_string(index=False)
)

print("\nSaved analysis directory:")
print(ANALYSIS_DIR)