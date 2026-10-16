import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# Paths
# =========================================================
INPUT_FILE = Path(
    r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost\Edge-IIoTset dataset\Edge-IIoTset dataset\Selected dataset for ML and DL\ML-EdgeIIoT-dataset.csv"
)
OUTPUT_DIR = Path(r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost\processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


OUTPUT_FILE = OUTPUT_DIR / "edgeiiot_clean.csv"


# =========================================================
# Load
# =========================================================
print("=" * 70)
print("Loading Edge-IIoTset...")
print("=" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)


# =========================================================
# Check target
# =========================================================
TARGET = "Attack_label"
ATTACK_TYPE = "Attack_type"

if TARGET not in df.columns:
    raise ValueError(f"{TARGET} not found in dataset.")

print("\nOriginal label distribution:")
print(df[TARGET].value_counts(dropna=False))

print("\nOriginal label percentages:")
print(
    df[TARGET]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(3)
)


# =========================================================
# Drop leakage / identifier / payload fields
# =========================================================
DROP_COLUMNS = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "icmp.transmit_timestamp",
    "http.request.uri.query",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "mqtt.msg",
]

existing_drop = [c for c in DROP_COLUMNS if c in df.columns]

print("\nDropping identifier/payload columns:")
for c in existing_drop:
    print(" -", c)

df = df.drop(columns=existing_drop)


# =========================================================
# Attack_type is useful for auditing only,
# but must NOT enter binary classifier
# =========================================================
if ATTACK_TYPE in df.columns:
    print("\nAttack types:")
    print(df[ATTACK_TYPE].value_counts(dropna=False))


# =========================================================
# Normalize serialization artifacts
# e.g. categorical "0.0" vs "0"
# =========================================================
print("\nNormalizing categorical serialization artifacts...")

for col in df.select_dtypes(include=["object"]).columns:

    if col == ATTACK_TYPE:
        continue

    df[col] = df[col].astype(str).str.strip()

    df[col] = df[col].replace({
        "0.0": "0",
        "0.00": "0",
        "0.000": "0",
    })


# =========================================================
# Convert +/- inf to NaN
# =========================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)

missing_total = int(df.isna().sum().sum())

print("\nMissing values after normalization:", missing_total)


# =========================================================
# Remove duplicates AFTER identifier removal
# =========================================================
before = len(df)

dedup_subset = [c for c in df.columns if c != ATTACK_TYPE]
df = df.drop_duplicates(subset=dedup_subset).reset_index(drop=True)

removed = before - len(df)

print("\nDuplicates removed:", removed)
print("Shape after deduplication:", df.shape)


# =========================================================
# Remove constant feature columns
# Do NOT evaluate target or Attack_type
# =========================================================
feature_cols = [
    c for c in df.columns
    if c not in [TARGET, ATTACK_TYPE]
]

constant_cols = [
    c for c in feature_cols
    if df[c].nunique(dropna=False) <= 1
]

print("\nConstant columns:", len(constant_cols))

for c in constant_cols:
    print(" -", c)

df = df.drop(columns=constant_cols)


# =========================================================
# Final diagnostics
# =========================================================
print("\n" + "=" * 70)
print("FINAL EDGE-IIOTSET AUDIT")
print("=" * 70)

print("Final shape:", df.shape)

print("\nBinary label distribution:")
print(df[TARGET].value_counts())

print("\nBinary percentages:")
print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

print("\nData types:")
print(df.dtypes.value_counts())

print("\nMissing values:", int(df.isna().sum().sum()))

print("\nNumber of remaining feature columns:",
      len([c for c in df.columns if c not in [TARGET, ATTACK_TYPE]]))


# =========================================================
# Save cleaned dataset
# =========================================================
df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved cleaned dataset to:")
print(OUTPUT_FILE)

print("\nDONE.")