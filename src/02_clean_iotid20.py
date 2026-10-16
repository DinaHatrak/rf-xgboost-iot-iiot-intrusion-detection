import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# Project paths
# =========================================================
PROJECT_DIR = Path(
    r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost"
)

OUTPUT_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "iotid20_clean.csv"


# =========================================================
# Find IoTID20 automatically
# =========================================================
matches = list(
    PROJECT_DIR.rglob("IoT Network Intrusion Dataset.csv")
)

if len(matches) == 0:
    raise FileNotFoundError(
        "Could not find 'IoT Network Intrusion Dataset.csv'"
    )

if len(matches) > 1:
    print("WARNING: Multiple files found:")
    for m in matches:
        print(" -", m)
    print("\nUsing the first one.")

INPUT_FILE = matches[0]


# =========================================================
# Load
# =========================================================
print("=" * 70)
print("Loading IoTID20...")
print("=" * 70)

print("Input file:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE, low_memory=False)

df.columns = df.columns.str.strip()

print("\nOriginal shape:", df.shape)


# =========================================================
# Labels
# =========================================================
TARGET = "Label"
CAT = "Cat"
SUB_CAT = "Sub_Cat"

if TARGET not in df.columns:
    raise ValueError(f"{TARGET} not found.")

print("\nOriginal binary distribution:")
print(df[TARGET].value_counts(dropna=False))

print("\nOriginal binary percentages:")
print(
    df[TARGET]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(3)
)

if CAT in df.columns:
    print("\nAttack category distribution:")
    print(df[CAT].value_counts(dropna=False))

if SUB_CAT in df.columns:
    print("\nAttack sub-category distribution:")
    print(df[SUB_CAT].value_counts(dropna=False))


# =========================================================
# Remove obvious identifiers
# =========================================================
DROP_COLUMNS = [
    "Flow_ID",
    "Src_IP",
    "Dst_IP",
    "Timestamp",
]

existing_drop = [
    c for c in DROP_COLUMNS
    if c in df.columns
]

print("\nRemoving identifier columns:")

for c in existing_drop:
    print(" -", c)

df = df.drop(columns=existing_drop)


# =========================================================
# Normalize string values
# =========================================================
print("\nNormalizing categorical strings...")

for col in df.select_dtypes(include=["object"]).columns:

    # Preserve semantic label text but remove whitespace
    df[col] = df[col].astype(str).str.strip()

    df[col] = df[col].replace({
        "0.0": "0",
        "0.00": "0",
        "0.000": "0",
    })


# =========================================================
# Infinite values -> NaN
# Do NOT impute yet.
# Imputation must be fitted on TRAIN only later.
# =========================================================
numeric_cols = df.select_dtypes(
    include=[np.number]
).columns

inf_count = np.isinf(
    df[numeric_cols]
).sum()

inf_count = inf_count[inf_count > 0]

print("\nColumns containing +/-Inf:")

if len(inf_count) == 0:
    print("None")
else:
    print(inf_count)

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

print(
    "\nMissing values after Inf -> NaN:",
    int(df.isna().sum().sum())
)


# =========================================================
# Model-space duplicate removal
#
# Cat and Sub_Cat will NOT enter the binary classifier.
# Therefore they should not prevent detection of duplicate X,y rows.
# =========================================================
dedup_excluded = [CAT, SUB_CAT]

dedup_subset = [
    c for c in df.columns
    if c not in dedup_excluded
]

before = len(df)

df = df.drop_duplicates(
    subset=dedup_subset
).reset_index(drop=True)

removed = before - len(df)

print("\nDuplicates removed:", removed)
print("Shape after deduplication:", df.shape)


# =========================================================
# Constant feature removal
# Cat/Sub_Cat are audit metadata, not model features
# =========================================================
NON_FEATURES = [
    TARGET,
    CAT,
    SUB_CAT
]

feature_cols = [
    c for c in df.columns
    if c not in NON_FEATURES
]

constant_cols = [
    c for c in feature_cols
    if df[c].nunique(dropna=False) <= 1
]

print("\nConstant feature columns:", len(constant_cols))

for c in constant_cols:
    print(" -", c)

df = df.drop(columns=constant_cols)


# =========================================================
# Final audit
# =========================================================
print("\n" + "=" * 70)
print("FINAL IoTID20 AUDIT")
print("=" * 70)

print("\nFinal shape:")
print(df.shape)

print("\nFinal binary distribution:")
print(df[TARGET].value_counts())

print("\nFinal binary percentages:")
print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

print("\nData types:")
print(df.dtypes.value_counts())

print(
    "\nRemaining missing values:",
    int(df.isna().sum().sum())
)

remaining_features = [
    c for c in df.columns
    if c not in NON_FEATURES
]

print(
    "\nRemaining feature count:",
    len(remaining_features)
)


# =========================================================
# Missing-value report
# =========================================================
missing_by_col = (
    df.isna()
      .sum()
      .sort_values(ascending=False)
)

missing_by_col = missing_by_col[
    missing_by_col > 0
]

print("\nColumns with missing values:")

if len(missing_by_col) == 0:
    print("None")
else:
    print(missing_by_col)


# =========================================================
# Save
# =========================================================
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved cleaned dataset to:")
print(OUTPUT_FILE)

print("\nDONE.")