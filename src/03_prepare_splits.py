import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# =========================================================
# Paths
# =========================================================
PROJECT_DIR = Path(
    r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost"
)

PROCESSED_DIR = PROJECT_DIR / "processed"
SPLIT_DIR = PROJECT_DIR / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Global settings
# =========================================================
SAMPLE_SIZE = 150_000
SAMPLE_SEED = 42
SPLIT_SEED = 42

TEST_SIZE = 0.15
VAL_SIZE_TOTAL = 0.15

# After removing 15% test:
# validation fraction inside remaining 85%
VAL_SIZE_REMAINING = VAL_SIZE_TOTAL / (1 - TEST_SIZE)


# =========================================================
# Helper
# =========================================================
def prepare_dataset(
    name,
    input_file,
    target_col,
    metadata_cols
):

    print("\n" + "=" * 75)
    print(f"PREPARING: {name}")
    print("=" * 75)

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print("Cleaned shape:", df.shape)

    # -----------------------------------------------------
    # Validate target
    # -----------------------------------------------------
    if target_col not in df.columns:
        raise ValueError(
            f"Target '{target_col}' not found in {name}"
        )

    print("\nOriginal cleaned distribution:")
    print(df[target_col].value_counts())

    print("\nPercentages:")
    print(
        df[target_col]
        .value_counts(normalize=True)
        .mul(100)
        .round(3)
    )

    # -----------------------------------------------------
    # Stratified sample
    # -----------------------------------------------------
    if len(df) > SAMPLE_SIZE:

        sampled_parts = []

        class_counts = df[target_col].value_counts()

        for label, count in class_counts.items():

            class_df = df[
                df[target_col] == label
            ]

            fraction = count / len(df)

            n_class = round(
                SAMPLE_SIZE * fraction
            )

            sampled_class = class_df.sample(
                n=min(n_class, len(class_df)),
                random_state=SAMPLE_SEED
            )

            sampled_parts.append(
                sampled_class
            )

        sampled = pd.concat(
            sampled_parts,
            ignore_index=True
        )

        # Correct rare rounding difference
        if len(sampled) > SAMPLE_SIZE:

            sampled = sampled.sample(
                n=SAMPLE_SIZE,
                random_state=SAMPLE_SEED
            )

        elif len(sampled) < SAMPLE_SIZE:

            missing_n = SAMPLE_SIZE - len(sampled)

            remaining = df.drop(
                index=sampled.index,
                errors="ignore"
            )

            extra = remaining.sample(
                n=missing_n,
                random_state=SAMPLE_SEED
            )

            sampled = pd.concat(
                [sampled, extra],
                ignore_index=True
            )

    else:

        sampled = df.copy()

    sampled = sampled.sample(
        frac=1,
        random_state=SAMPLE_SEED
    ).reset_index(drop=True)

    print("\nSampled shape:", sampled.shape)

    print("\nSample distribution:")
    print(sampled[target_col].value_counts())

    print("\nSample percentages:")
    print(
        sampled[target_col]
        .value_counts(normalize=True)
        .mul(100)
        .round(3)
    )

    # -----------------------------------------------------
    # 70 / 15 / 15 split
    #
    # Step 1:
    # 85% train+validation
    # 15% test
    # -----------------------------------------------------
    train_val, test = train_test_split(
        sampled,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=sampled[target_col]
    )

    # -----------------------------------------------------
    # Step 2:
    # split remaining 85% into
    # 70% train + 15% validation
    # -----------------------------------------------------
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE_REMAINING,
        random_state=SPLIT_SEED,
        stratify=train_val[target_col]
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------
    print("\n" + "-" * 50)
    print("FINAL SPLITS")
    print("-" * 50)

    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    for split_name, split_df in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test)
    ]:

        print(f"\n{split_name} distribution:")

        counts = split_df[
            target_col
        ].value_counts()

        percentages = (
            split_df[target_col]
            .value_counts(normalize=True)
            .mul(100)
            .round(3)
        )

        print(counts)
        print(percentages)

    # -----------------------------------------------------
    # Define model columns
    #
    # Metadata retained in CSV for later error analysis,
    # but NOT used by models.
    # -----------------------------------------------------
    model_features = [
        c for c in sampled.columns
        if c not in (
            [target_col] + metadata_cols
        )
    ]

    print(
        "\nModel feature count:",
        len(model_features)
    )

    print(
        "Metadata excluded from model:",
        metadata_cols
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    prefix = name.lower()

    sample_file = (
        SPLIT_DIR /
        f"{prefix}_sample_150k.csv"
    )

    train_file = (
        SPLIT_DIR /
        f"{prefix}_train.csv"
    )

    val_file = (
        SPLIT_DIR /
        f"{prefix}_val.csv"
    )

    test_file = (
        SPLIT_DIR /
        f"{prefix}_test.csv"
    )

    sampled.to_csv(
        sample_file,
        index=False
    )

    train.to_csv(
        train_file,
        index=False
    )

    val.to_csv(
        val_file,
        index=False
    )

    test.to_csv(
        test_file,
        index=False
    )

    print("\nSaved:")
    print(sample_file)
    print(train_file)
    print(val_file)
    print(test_file)


# =========================================================
# Edge-IIoTset
# =========================================================
prepare_dataset(
    name="edgeiiot",
    input_file=(
        PROCESSED_DIR /
        "edgeiiot_clean.csv"
    ),
    target_col="Attack_label",
    metadata_cols=[
        "Attack_type"
    ]
)


# =========================================================
# IoTID20
# =========================================================
prepare_dataset(
    name="iotid20",
    input_file=(
        PROCESSED_DIR /
        "iotid20_clean.csv"
    ),
    target_col="Label",
    metadata_cols=[
        "Cat",
        "Sub_Cat"
    ]
)


print("\n" + "=" * 75)
print("ALL DATASETS PREPARED SUCCESSFULLY")
print("=" * 75)