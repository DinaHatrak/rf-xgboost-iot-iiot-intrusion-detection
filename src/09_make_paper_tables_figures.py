import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================
PROJECT_DIR = Path(
    r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost"
)

RESULTS_DIR = PROJECT_DIR / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
PROCESSED_DIR = PROJECT_DIR / "processed"
SPLIT_DIR = PROJECT_DIR / "splits"

PAPER_DIR = PROJECT_DIR / "paper_outputs"
TABLE_DIR = PAPER_DIR / "tables"
FIGURE_DIR = PAPER_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# DISPLAY SETTINGS
# =========================================================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# =========================================================
# HELPER
# =========================================================
def save_figure(fig, filename):

    png_file = FIGURE_DIR / f"{filename}.png"
    pdf_file = FIGURE_DIR / f"{filename}.pdf"

    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", png_file)
    print("Saved:", pdf_file)


def mean_sd_string(mean, sd, decimals=4):

    return (
        f"{mean:.{decimals}f} "
        f"± {sd:.{decimals}f}"
    )


# =========================================================
# LOAD FINAL RESULTS
# =========================================================
summary = pd.read_csv(
    RESULTS_DIR /
    "07_final_test_summary.csv"
)

per_seed = pd.read_csv(
    RESULTS_DIR /
    "07_final_test_per_seed.csv"
)

bootstrap = pd.read_csv(
    ANALYSIS_DIR /
    "08_bootstrap_comparison.csv"
)


# =========================================================
# TABLE 1
# DATASET CHARACTERISTICS
# =========================================================
print("\n" + "=" * 90)
print("TABLE 1 — DATASET CHARACTERISTICS")
print("=" * 90)

dataset_rows = []

datasets = [

    {
        "Dataset": "Edge-IIoTset",
        "processed_file":
            PROCESSED_DIR /
            "edgeiiot_clean.csv",
        "sample_file":
            SPLIT_DIR /
            "edgeiiot_sample_150k.csv",
        "target":
            "Attack_label",
        "positive":
            1,
        "feature_count":
            38
    },

    {
        "Dataset": "IoTID20",
        "processed_file":
            PROCESSED_DIR /
            "iotid20_clean.csv",
        "sample_file":
            SPLIT_DIR /
            "iotid20_sample_150k.csv",
        "target":
            "Label",
        "positive":
            "Anomaly",
        "feature_count":
            69
    }
]


for item in datasets:

    processed = pd.read_csv(
        item["processed_file"],
        low_memory=False
    )

    sampled = pd.read_csv(
        item["sample_file"],
        low_memory=False
    )

    target = item["target"]
    positive = item["positive"]

    positive_count = (
        sampled[target] == positive
    ).sum()

    negative_count = (
        len(sampled) -
        positive_count
    )

    dataset_rows.append({

        "Dataset":
            item["Dataset"],

        "Cleaned_Records":
            len(processed),

        "Experimental_Sample":
            len(sampled),

        "Model_Features":
            item["feature_count"],

        "Attack_Records":
            int(positive_count),

        "Normal_Records":
            int(negative_count),

        "Attack_Percent":
            (
                positive_count /
                len(sampled)
                * 100
            ),

        "Normal_Percent":
            (
                negative_count /
                len(sampled)
                * 100
            ),

        "Train":
            105000,

        "Validation":
            22500,

        "Test":
            22500
    })


table1 = pd.DataFrame(
    dataset_rows
)

table1.to_csv(
    TABLE_DIR /
    "Table1_Dataset_Characteristics.csv",
    index=False
)

print(
    table1.round(3).to_string(
        index=False
    )
)


# =========================================================
# TABLE 2
# SELECTED HYPERPARAMETERS
# =========================================================
print("\n" + "=" * 90)
print("TABLE 2 — SELECTED HYPERPARAMETERS")
print("=" * 90)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


selected_configs = [

    (
        "Edge-IIoTset",
        "Random Forest",
        RESULTS_DIR /
        "05_rf_best_edge-iiotset.json"
    ),

    (
        "Edge-IIoTset",
        "XGBoost",
        RESULTS_DIR /
        "06_xgb_best_edge-iiotset.json"
    ),

    (
        "IoTID20",
        "Random Forest",
        RESULTS_DIR /
        "05_rf_best_iotid20.json"
    ),

    (
        "IoTID20",
        "XGBoost",
        RESULTS_DIR /
        "06_xgb_best_iotid20.json"
    )
]


hyper_rows = []


for dataset, model, path in selected_configs:

    cfg = load_json(path)

    if model == "Random Forest":

        parameters = (

            f"n_estimators={cfg['n_estimators']}; "
            f"max_depth={cfg['max_depth']}; "
            f"min_samples_leaf={cfg['min_samples_leaf']}; "
            f"max_features={cfg['max_features']}; "
            f"class_weight={cfg['class_weight']}"
        )

    else:

        parameters = (

            f"n_estimators={cfg['n_estimators']}; "
            f"max_depth={cfg['max_depth']}; "
            f"learning_rate={cfg['learning_rate']}; "
            f"subsample={cfg['subsample']}; "
            f"colsample_bytree={cfg['colsample_bytree']}"
        )

    hyper_rows.append({

        "Dataset":
            dataset,

        "Model":
            model,

        "Selected_Parameters":
            parameters
    })


table2 = pd.DataFrame(
    hyper_rows
)

table2.to_csv(
    TABLE_DIR /
    "Table2_Selected_Hyperparameters.csv",
    index=False
)

print(
    table2.to_string(
        index=False
    )
)


# =========================================================
# TABLE 3
# FINAL PERFORMANCE — MEAN ± SD
# =========================================================
print("\n" + "=" * 90)
print("TABLE 3 — FINAL TEST PERFORMANCE")
print("=" * 90)

table3_rows = []


for _, row in summary.iterrows():

    table3_rows.append({

        "Dataset":
            row["Dataset"],

        "Model":
            row["Model"],

        "Macro_F1":
            mean_sd_string(
                row["Macro_F1_Mean"],
                row["Macro_F1_SD"]
            ),

        "Macro_PR_AUC":
            mean_sd_string(
                row["Macro_PR_AUC_Mean"],
                row["Macro_PR_AUC_SD"]
            ),

        "ROC_AUC":
            mean_sd_string(
                row["ROC_AUC_Mean"],
                row["ROC_AUC_SD"]
            ),

        "Macro_Precision":
            mean_sd_string(
                row["Macro_Precision_Mean"],
                row["Macro_Precision_SD"]
            ),

        "Macro_Recall":
            mean_sd_string(
                row["Macro_Recall_Mean"],
                row["Macro_Recall_SD"]
            ),

        "Attack_Recall":
            mean_sd_string(
                row["Attack_Recall_Mean"],
                row["Attack_Recall_SD"]
            )
    })


table3 = pd.DataFrame(
    table3_rows
)

table3.to_csv(
    TABLE_DIR /
    "Table3_Final_Test_Performance.csv",
    index=False
)

print(
    table3.to_string(
        index=False
    )
)


# =========================================================
# TABLE 4
# COMPUTATIONAL COST
# =========================================================
print("\n" + "=" * 90)
print("TABLE 4 — COMPUTATIONAL COST")
print("=" * 90)

table4_rows = []


for _, row in summary.iterrows():

    table4_rows.append({

        "Dataset":
            row["Dataset"],

        "Model":
            row["Model"],

        "Training_Time_sec":
            mean_sd_string(
                row["Training_Time_Mean_sec"],
                row["Training_Time_SD_sec"],
                decimals=3
            ),

        "Inference_per_sample_ms":
            mean_sd_string(
                row["Inference_ms_Mean"],
                row["Inference_ms_SD"],
                decimals=6
            )
    })


table4 = pd.DataFrame(
    table4_rows
)

table4.to_csv(
    TABLE_DIR /
    "Table4_Computational_Cost.csv",
    index=False
)

print(
    table4.to_string(
        index=False
    )
)


# =========================================================
# TABLE 5
# BOOTSTRAP COMPARISON
# =========================================================
print("\n" + "=" * 90)
print("TABLE 5 — PAIRED BOOTSTRAP COMPARISON")
print("=" * 90)

table5 = bootstrap[

    [
        "Dataset",
        "RF_Macro_F1",
        "XGB_Macro_F1",
        "Difference_XGB_minus_RF",
        "Bootstrap_CI_Lower",
        "Bootstrap_CI_Upper",
        "Bootstrap_Iterations"
    ]

].copy()


table5["95%_CI"] = (

    "["

    + table5[
        "Bootstrap_CI_Lower"
    ].map(
        lambda x:
        f"{x:.6f}"
    )

    + ", "

    + table5[
        "Bootstrap_CI_Upper"
    ].map(
        lambda x:
        f"{x:.6f}"
    )

    + "]"
)


table5 = table5[

    [
        "Dataset",
        "RF_Macro_F1",
        "XGB_Macro_F1",
        "Difference_XGB_minus_RF",
        "95%_CI",
        "Bootstrap_Iterations"
    ]

]


table5.to_csv(
    TABLE_DIR /
    "Table5_Bootstrap_Comparison.csv",
    index=False
)

print(
    table5.round(6).to_string(
        index=False
    )
)


# =========================================================
# TABLE 6
# TOP-10 FEATURE IMPORTANCE
# =========================================================
print("\n" + "=" * 90)
print("TABLE 6 — TOP FEATURES")
print("=" * 90)

feature_files = [

    (
        "Edge-IIoTset",
        "Random Forest",
        ANALYSIS_DIR /
        "08_feature_importance_edgeiiot_random_forest.csv"
    ),

    (
        "Edge-IIoTset",
        "XGBoost",
        ANALYSIS_DIR /
        "08_feature_importance_edgeiiot_xgboost.csv"
    ),

    (
        "IoTID20",
        "Random Forest",
        ANALYSIS_DIR /
        "08_feature_importance_iotid20_random_forest.csv"
    ),

    (
        "IoTID20",
        "XGBoost",
        ANALYSIS_DIR /
        "08_feature_importance_iotid20_xgboost.csv"
    )
]


feature_rows = []


for dataset, model, file in feature_files:

    fi = pd.read_csv(file).head(10)

    for rank, row in enumerate(
        fi.itertuples(),
        start=1
    ):

        clean_feature = (
            row.Feature
            .replace(
                "numeric__",
                ""
            )
            .replace(
                "categorical__",
                ""
            )
        )

        feature_rows.append({

            "Dataset":
                dataset,

            "Model":
                model,

            "Rank":
                rank,

            "Feature":
                clean_feature,

            "Importance":
                row.Importance
        })


table6 = pd.DataFrame(
    feature_rows
)

table6.to_csv(
    TABLE_DIR /
    "Table6_Top10_Feature_Importance.csv",
    index=False
)

print(
    table6.head(20)
    .round(6)
    .to_string(
        index=False
    )
)


# =========================================================
# FIGURE 1
# EXPERIMENTAL WORKFLOW
# =========================================================
fig, ax = plt.subplots(
    figsize=(14, 3.7)
)

ax.axis("off")

workflow = [

    "Raw\nDatasets",

    "Leakage-aware\nCleaning",

    "Stratified\nSampling",

    "70 / 15 / 15\nSplit",

    "Simple\nTuning",

    "3-Seed\nFinal Test",

    "Performance,\nCost & XAI"

]

x_positions = np.linspace(
    0.06,
    0.94,
    len(workflow)
)


for i, (x, text) in enumerate(
    zip(
        x_positions,
        workflow
    )
):

    ax.text(

        x,
        0.5,
        text,

        ha="center",
        va="center",

        fontsize=10,

        bbox=dict(
            boxstyle="round,pad=0.55",
            fill=False,
            linewidth=1.3
        )
    )

    if i < len(workflow) - 1:

        ax.annotate(

            "",

            xy=(
                x_positions[i + 1] - 0.045,
                0.5
            ),

            xytext=(
                x + 0.045,
                0.5
            ),

            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.2
            )
        )


ax.set_title(
    "Experimental workflow",
    fontsize=13
)

save_figure(
    fig,
    "Figure1_Experimental_Workflow"
)


# =========================================================
# FIGURE 2
# MACRO-F1 COMPARISON
# =========================================================
plot_data = summary.copy()

datasets_order = [
    "Edge-IIoTset",
    "IoTID20"
]

models_order = [
    "Random Forest",
    "XGBoost"
]

x = np.arange(
    len(datasets_order)
)

width = 0.34

fig, ax = plt.subplots(
    figsize=(8, 5)
)


for i, model in enumerate(
    models_order
):

    subset = (

        plot_data[
            plot_data["Model"]
            == model
        ]

        .set_index(
            "Dataset"
        )

        .loc[
            datasets_order
        ]
    )

    positions = (
        x
        +
        (
            i - 0.5
        ) * width
    )

    bars = ax.bar(

        positions,

        subset[
            "Macro_F1_Mean"
        ],

        width,

        yerr=
            subset[
                "Macro_F1_SD"
            ],

        capsize=4,

        label=model
    )

    for bar, value in zip(
        bars,
        subset[
            "Macro_F1_Mean"
        ]
    ):

        ax.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            value + 0.002,

            f"{value:.4f}",

            ha="center",
            va="bottom",

            fontsize=9
        )


ax.set_ylabel(
    "Macro-F1"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    datasets_order
)

ax.set_ylim(
    0.90,
    1.01
)

ax.set_title(
    "Final Macro-F1 comparison on the fixed test sets"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "Figure2_Macro_F1_Comparison"
)


# =========================================================
# FIGURE 3A
# TRAINING TIME
# =========================================================
fig, ax = plt.subplots(
    figsize=(8, 5)
)


for i, model in enumerate(
    models_order
):

    subset = (

        summary[
            summary["Model"]
            == model
        ]

        .set_index(
            "Dataset"
        )

        .loc[
            datasets_order
        ]
    )

    positions = (
        x
        +
        (
            i - 0.5
        ) * width
    )

    bars = ax.bar(

        positions,

        subset[
            "Training_Time_Mean_sec"
        ],

        width,

        yerr=
            subset[
                "Training_Time_SD_sec"
            ],

        capsize=4,

        label=model
    )

    for bar, value in zip(
        bars,
        subset[
            "Training_Time_Mean_sec"
        ]
    ):

        ax.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            value + 0.08,

            f"{value:.2f}s",

            ha="center",
            va="bottom",

            fontsize=9
        )


ax.set_ylabel(
    "Training time (seconds)"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    datasets_order
)

ax.set_title(
    "Model training cost"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "Figure3A_Training_Time"
)


# =========================================================
# FIGURE 3B
# INFERENCE TIME
# =========================================================
fig, ax = plt.subplots(
    figsize=(8, 5)
)


for i, model in enumerate(
    models_order
):

    subset = (

        summary[
            summary["Model"]
            == model
        ]

        .set_index(
            "Dataset"
        )

        .loc[
            datasets_order
        ]
    )

    positions = (
        x
        +
        (
            i - 0.5
        ) * width
    )

    bars = ax.bar(

        positions,

        subset[
            "Inference_ms_Mean"
        ],

        width,

        yerr=
            subset[
                "Inference_ms_SD"
            ],

        capsize=4,

        label=model
    )

    for bar, value in zip(
        bars,
        subset[
            "Inference_ms_Mean"
        ]
    ):

        ax.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            value + 0.00005,

            f"{value:.6f}",

            ha="center",
            va="bottom",

            fontsize=8
        )


ax.set_ylabel(
    "Inference time per sample (ms)"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    datasets_order
)

ax.set_title(
    "Inference cost"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "Figure3B_Inference_Time"
)


# =========================================================
# FIGURES 4
# TOP-10 FEATURE IMPORTANCE
# One figure for each dataset/model pair
# =========================================================
for dataset, model, file in feature_files:

    fi = (
        pd.read_csv(file)
        .head(10)
        .copy()
    )

    fi["Feature"] = (

        fi["Feature"]

        .str.replace(
            "numeric__",
            "",
            regex=False
        )

        .str.replace(
            "categorical__",
            "",
            regex=False
        )
    )

    fi = fi.sort_values(
        "Importance",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.barh(
        fi["Feature"],
        fi["Importance"]
    )

    ax.set_xlabel(
        "Feature importance"
    )

    ax.set_title(
        f"Top-10 features — "
        f"{dataset} — {model}"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    fig.tight_layout()

    safe_dataset = (
        dataset
        .lower()
        .replace("-", "")
    )

    safe_model = (
        model
        .lower()
        .replace(" ", "_")
    )

    save_figure(
        fig,
        f"Figure4_{safe_dataset}_{safe_model}_Top10"
    )


# =========================================================
# BONUS TABLE
# MODEL IMPROVEMENT
# =========================================================
comparison_rows = []


for dataset in datasets_order:

    dataset_summary = (
        summary[
            summary["Dataset"]
            == dataset
        ]
        .set_index("Model")
    )

    rf = dataset_summary.loc[
        "Random Forest"
    ]

    xgb = dataset_summary.loc[
        "XGBoost"
    ]

    comparison_rows.append({

        "Dataset":
            dataset,

        "Macro_F1_RF":
            rf["Macro_F1_Mean"],

        "Macro_F1_XGBoost":
            xgb["Macro_F1_Mean"],

        "Absolute_Improvement":
            (
                xgb["Macro_F1_Mean"]
                -
                rf["Macro_F1_Mean"]
            ),

        "Relative_Improvement_Percent":
            (
                (
                    xgb["Macro_F1_Mean"]
                    -
                    rf["Macro_F1_Mean"]
                )
                /
                rf["Macro_F1_Mean"]
                *
                100
            ),

        "Training_Speedup_X":
            (
                rf[
                    "Training_Time_Mean_sec"
                ]
                /
                xgb[
                    "Training_Time_Mean_sec"
                ]
            ),

        "Inference_Speedup_X":
            (
                rf[
                    "Inference_ms_Mean"
                ]
                /
                xgb[
                    "Inference_ms_Mean"
                ]
            )
    })


comparison = pd.DataFrame(
    comparison_rows
)

comparison.to_csv(
    TABLE_DIR /
    "Supplementary_Model_Comparison.csv",
    index=False
)


print("\n" + "=" * 100)
print("PAPER OUTPUT GENERATION COMPLETE")
print("=" * 100)

print("\nTables:")
print(TABLE_DIR)

print("\nFigures:")
print(FIGURE_DIR)

print("\nModel comparison:")
print(
    comparison
    .round(4)
    .to_string(
        index=False
    )
)