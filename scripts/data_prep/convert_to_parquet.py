"""Convert raw Tianchi behavior logs to partitioned parquet files.

This script reads the official competition TXT files in batches, normalizes
missing values, enriches time-based features, and writes partitioned parquet
datasets to `data/processed/`.

Usage example:
    python scripts/data_prep/convert_to_parquet.py \
        --input-files rawdata/tianchi_fresh_comp_train_user_online_partA.txt \
        --input-files rawdata/tianchi_fresh_comp_train_user_online_partB.txt \
        --output-dir data/processed/raw_behavior \
        --rows-per-chunk 5000000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Convert raw Tianchi behavior logs to parquet",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="List of raw TXT files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/raw_behavior"),
        help="Directory to store parquet outputs.",
    )
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=5_000_000,
        help="Rows to process per batch to control memory usage.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/data_prep.log"),
        help="Log file path.",
    )
    return parser.parse_args()


def configure_logging(log_path: Path) -> None:
    """Configure logging to file and console."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def read_chunks(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    """Yield chunks from raw txt file."""

    column_names = [
        "user_id",
        "item_id",
        "behavior_type",
        "user_geohash",
        "item_category",
        "time",
    ]
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=column_names,
        chunksize=chunk_size,
        dtype={
            "user_id": "int64",
            "item_id": "int64",
            "behavior_type": "int8",
            "user_geohash": "string",
            "item_category": "int64",
            "time": "string",
        },
    )
    yield from reader


def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing values and derive time features."""

    df = df.copy()
    df["user_geohash"].fillna("_NA_", inplace=True)
    df["item_category"].fillna(-1, inplace=True)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H")
    df["event_day"] = df["time"].dt.strftime("%Y%m%d")
    df["event_hour"] = df["time"].dt.hour.astype("int8")
    return df


def write_parquet(df: pd.DataFrame, output_dir: Path) -> None:
    """Write dataframe to partitioned parquet by event_day."""

    for day, day_df in df.groupby("event_day"):
        day_dir = output_dir / f"event_day={day}"
        day_dir.mkdir(parents=True, exist_ok=True)
        file_path = day_dir / f"chunk_{day_df.index.min()}_{day_df.index.max()}.parquet"
        day_df.to_parquet(file_path, index=False)
        LOGGER.info("Saved parquet: %s rows=%d", file_path, len(day_df))


def process_file(path: Path, chunk_size: int, output_dir: Path) -> None:
    """Process a single raw file into parquet chunks."""

    LOGGER.info("Start processing %s", path)
    for idx, chunk in enumerate(read_chunks(path, chunk_size), start=1):
        chunk = normalize_chunk(chunk)
        write_parquet(chunk, output_dir)
        LOGGER.info("Completed chunk %d for %s", idx, path.name)
    LOGGER.info("Finished file %s", path)


def main() -> None:
    """Script entry point."""

    args = parse_args()
    configure_logging(args.log_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in args.input_files:
        process_file(Path(file_path), args.rows_per_chunk, args.output_dir)


if __name__ == "__main__":
    main()

