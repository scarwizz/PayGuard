"""
Data ingestion and memory-safe schema downcasting for PaySim transactions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: List[str] = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

DOWNCAST_DTYPES: Dict[str, str] = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "nameOrig": "object",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "object",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

VALID_TRANSACTION_TYPES: List[str] = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


def validate_schema(df: pd.DataFrame) -> None:
    """Validates that the DataFrame conforms to the expected 11-column PaySim schema."""
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed: Missing columns {missing_cols}")

    present_types = set(df["type"].dropna().unique())
    unknown_types = present_types - set(VALID_TRANSACTION_TYPES)
    if unknown_types:
        raise ValueError(f"Schema validation warning: Unknown transaction types {unknown_types}")


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """Calculates deep memory footprint in megabytes."""
    return float(df.memory_usage(deep=True).sum() / (1024 * 1024))


def downcast_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts numerical and categorical columns to memory-optimized representations."""
    validate_schema(df)
    initial_mem = get_memory_usage_mb(df)
    df = df.astype(DOWNCAST_DTYPES)
    final_mem = get_memory_usage_mb(df)
    logger.info("Downcasted DataFrame: %.2f MB -> %.2f MB", initial_mem, final_mem)
    return df


def ingest_paysim_csv(
    raw_csv_path: Union[str, Path],
    output_parquet_path: Union[str, Path] = "data/processed/paysim_typed.parquet",
    chunksize: Optional[int] = 1_000_000,
) -> pd.DataFrame:
    """Reads raw CSV in memory-safe chunks, downcasts dtypes, and persists to Parquet."""
    csv_file = Path(raw_csv_path)
    parquet_file = Path(output_parquet_path)
    parquet_file.parent.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {csv_file}")

    file_size_mb = csv_file.stat().st_size / (1024 * 1024)
    if chunksize is not None and file_size_mb > 200:
        chunks: List[pd.DataFrame] = []
        for chunk in pd.read_csv(csv_file, chunksize=chunksize, dtype=DOWNCAST_DTYPES):
            validate_schema(chunk)
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(csv_file, dtype=DOWNCAST_DTYPES)
        validate_schema(df)

    df.to_parquet(parquet_file, engine="pyarrow", compression="snappy", index=False)
    logger.info("Persisted %d rows to %s (%.2f MB)", len(df), parquet_file, parquet_file.stat().st_size / (1024 * 1024))
    return df


def load_processed_data(
    parquet_path: Union[str, Path] = "data/processed/paysim_typed.parquet"
) -> pd.DataFrame:
    """Loads the typed PaySim Parquet dataset from disk."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    return pd.read_parquet(path)
