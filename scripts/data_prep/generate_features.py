"""Aggregate behavior features with DuckDB for high-throughput preprocessing.

本脚本是 `generate_features.py` 的 DuckDB 版本，直接在 SQL 层对
`convert_to_parquet.py` 产出的分区 Parquet 数据执行按日聚合与滚动窗口
统计，可显著提高大规模数据下的处理速度。

使用示例：

.. code-block:: bash

    python scripts/data_prep/generate_features_duckdb.py \
        --start-day 20141128 --end-day 20141218 --window 7 \
        --output-dir data/processed/features_duckdb

输出与原脚本保持一致：分别生成用户、商品的历史行为计数表，列后缀
`_{window}d` 表示滚动窗口长度。为了与 TorchEasyRec 配置对齐，所有
行为列均为 `int64` 类型。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

try:
    import duckdb
except ModuleNotFoundError as exc:  # pragma: no cover - fail fast for missing dep
    raise SystemExit(
        "DuckDB 未安装，请先执行 `pip install duckdb pyarrow` 再运行该脚本。",
    ) from exc

import pandas as pd

try:  # pragma: no cover - optional dependency for better UX
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency for better UX
    tqdm = None


LOGGER = logging.getLogger(__name__)
DATE_FMT = "%Y%m%d"
BEHAVIOR_COLUMNS = ["click", "fav", "cart", "buy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate aggregated features with DuckDB")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/raw_behavior"),
        help="Directory containing event_day partitions produced by convert_to_parquet.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/features-14d"),
        help="Directory to store aggregated feature parquet files.",
    )
    parser.add_argument(
        "--start-day",
        type=str,
        # required=True,
        default="20141125",
        help="First day (inclusive) to generate features, format YYYYMMDD.",
    )
    parser.add_argument(
        "--end-day",
        type=str,
        # required=True,
        default="20141217",
        help="Last day (inclusive) to generate features, format YYYYMMDD.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=14,
        help="Rolling window length in days (history strictly before target day).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/data_prep.log"),
        help="Log file path.",
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=None,
        help="DuckDB database文件路径或目录，缺省为内存模式。若指定目录将自动创建 duckdb.db。",
    )
    parser.add_argument(
        "--duckdb-temp-directory",
        type=Path,
        default=None,
        help="DuckDB 临时目录，若未指定则使用默认临时路径。建议指定到大容量磁盘。",
    )
    parser.add_argument(
        "--duckdb-max-temp-size",
        type=str,
        default=None,
        help="DuckDB PRAGMA max_temp_directory_size 的值，例如 '0'、'120GB'。",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        type=str,
        default=None,
        help="DuckDB PRAGMA memory_limit 的值，例如 '60GB'。",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="DuckDB 并行线程数，默认使用 CPU 核心数。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm/DuckDB progress output.",
    )
    return parser.parse_args()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def discover_days(input_dir: Path) -> list[str]:
    """Return sorted YYYYMMDD list from `event_day=*` partitions."""

    days: list[str] = []
    for path in input_dir.glob("event_day=*"):
        if not path.is_dir():
            continue
        _, _, suffix = path.name.partition("=")
        if suffix.isdigit() and len(suffix) == 8:
            days.append(suffix)
    days.sort()
    return days


def ensure_valid_days(
    available_days: Iterable[str],
    start_day: str,
    end_day: str,
    window: int,
) -> list[str]:
    """Filter target days that have at least `window` prior days available."""

    sorted_days = list(available_days)
    ordered = [day for day in sorted_days if start_day <= day <= end_day]
    if not ordered:
        raise SystemExit("指定区间内没有可用的 event_day 分区，请检查输入目录。")

    day_to_index = {day: idx for idx, day in enumerate(sorted_days)}
    valid: list[str] = []
    for day in ordered:
        idx = day_to_index[day]
        if idx < window:
            LOGGER.warning("Skip day %s due to insufficient history (%d < %d)", day, idx, window)
            continue
        valid.append(day)

    if not valid:
        raise SystemExit("所有目标日期都缺少足够历史，请调整 --start-day 或缩短 --window。")
    return valid


def compute_history_start(earliest_target: str, available_days: list[str], window: int) -> str:
    earliest_date = datetime.strptime(earliest_target, DATE_FMT).date()
    history_start_date = earliest_date - timedelta(days=window)
    available_min = datetime.strptime(available_days[0], DATE_FMT).date()
    return max(history_start_date, available_min).strftime(DATE_FMT)


def build_entity_query(
    *,
    entity_column: str,
    entity_alias: str,
    parquet_glob: str,
    history_start: str,
    end_day: str,
    valid_days: list[str],
    window: int,
) -> str:
    if not valid_days:
        raise ValueError("valid_days must not be empty")

    values_clause = ", ".join(f"('{day}')" for day in valid_days)

    return f"""
WITH target(day_str, target_dt) AS (
    SELECT day_str, strptime(day_str, '%Y%m%d')::DATE AS target_dt
    FROM (VALUES {values_clause}) AS v(day_str)
), logs AS (
    SELECT
        strptime(CAST(event_day AS VARCHAR), '%Y%m%d')::DATE AS event_dt,
        {entity_column} AS entity_id,
        behavior_type
    FROM read_parquet('{parquet_glob}')
    WHERE event_day BETWEEN '{history_start}' AND '{end_day}'
), window_log AS (
    SELECT
        t.day_str,
        l.entity_id,
        l.behavior_type,
        COUNT(*)::BIGINT AS cnt
    FROM target t
    JOIN logs l
      ON l.event_dt >= t.target_dt - INTERVAL '{window}' DAY
     AND l.event_dt < t.target_dt
    GROUP BY 1, 2, 3
), entity_pool AS (
    SELECT DISTINCT day_str, entity_id FROM window_log
)
SELECT
    e.day_str AS event_day,
    e.entity_id AS {entity_alias},
    COALESCE(SUM(CASE WHEN w.behavior_type = 1 THEN w.cnt ELSE 0 END), 0)::BIGINT AS click_{window}d,
    COALESCE(SUM(CASE WHEN w.behavior_type = 2 THEN w.cnt ELSE 0 END), 0)::BIGINT AS fav_{window}d,
    COALESCE(SUM(CASE WHEN w.behavior_type = 3 THEN w.cnt ELSE 0 END), 0)::BIGINT AS cart_{window}d,
    COALESCE(SUM(CASE WHEN w.behavior_type = 4 THEN w.cnt ELSE 0 END), 0)::BIGINT AS buy_{window}d
FROM entity_pool e
LEFT JOIN window_log w
  ON e.day_str = w.day_str AND e.entity_id = w.entity_id
GROUP BY 1, 2
ORDER BY 1, 2;
"""


def fetch_entity_features(
    conn: duckdb.DuckDBPyConnection,
    *,
    entity_column: str,
    entity_alias: str,
    parquet_glob: str,
    history_start: str,
    end_day: str,
    valid_days: list[str],
    window: int,
) -> pd.DataFrame:
    query = build_entity_query(
        entity_column=entity_column,
        entity_alias=entity_alias,
        parquet_glob=parquet_glob,
        history_start=history_start,
        end_day=end_day,
        valid_days=valid_days,
        window=window,
    )
    LOGGER.info("Executing DuckDB query for %s", entity_column)
    LOGGER.debug("DuckDB SQL for %s:\n%s", entity_column, query)
    return conn.execute(query).fetch_df()


def split_and_save(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    if df.empty:
        LOGGER.warning("No rows returned for %s, skip writing parquet.", prefix)
        return

    df = df.copy()
    df.sort_values(["event_day", df.columns[1]], inplace=True)
    grouped = df.groupby("event_day", sort=True)
    event_days = list(grouped.groups.keys())

    iterator = event_days
    if SHOW_PROGRESS and tqdm is not None:
        iterator = tqdm(event_days, desc=f"Writing {prefix}", unit="day")
    elif SHOW_PROGRESS:
        LOGGER.info("Writing %d day partitions for %s", len(event_days), prefix)

    for event_day in iterator:
        group = grouped.get_group(event_day)
        file_path = output_dir / f"{prefix}_{event_day}.parquet"
        payload = group.drop(columns=["event_day"])
        payload.to_parquet(file_path, index=False)
        LOGGER.info("Saved %s rows=%d", file_path, len(payload))


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)

    global SHOW_PROGRESS
    SHOW_PROGRESS = not args.no_progress

    args.output_dir.mkdir(parents=True, exist_ok=True)

    available_days = discover_days(args.input_dir)
    if not available_days:
        raise SystemExit("输入目录下未找到 event_day=* 分区，请先执行 convert_to_parquet.py。")

    valid_days = ensure_valid_days(available_days, args.start_day, args.end_day, args.window)
    if not valid_days:
        raise SystemExit("未找到满足窗口要求的日期，任务终止。")

    history_start = compute_history_start(valid_days[0], available_days, args.window)
    LOGGER.info(
        "Generating features for days %s to %s (window=%d, history from %s)",
        valid_days[0],
        valid_days[-1],
        args.window,
        history_start,
    )

    parquet_glob = str((args.input_dir / "event_day=*/chunk_*.parquet").resolve()).replace("'", "''")

    if args.duckdb_path:
        if args.duckdb_path.is_dir():
            db_path = args.duckdb_path / "duckdb.db"
        else:
            db_path = args.duckdb_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(database=str(db_path))
    else:
        conn = duckdb.connect(database=":memory:")

    threads = args.threads or (os.cpu_count() or 1)
    conn.execute(f"PRAGMA threads={threads}")
    conn.execute("PRAGMA preserve_insertion_order=false")

    if args.duckdb_temp_directory:
        temp_dir = args.duckdb_temp_directory.resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)
        conn.execute(f"PRAGMA temp_directory='{str(temp_dir).replace("'", "''")}'")
        LOGGER.info("DuckDB temp_directory set to %s", temp_dir)

    if args.duckdb_max_temp_size:
        conn.execute(f"PRAGMA max_temp_directory_size='{args.duckdb_max_temp_size}'")
        LOGGER.info(
            "DuckDB max_temp_directory_size set to %s",
            args.duckdb_max_temp_size,
        )

    if args.duckdb_memory_limit:
        conn.execute(f"PRAGMA memory_limit='{args.duckdb_memory_limit}'")
        LOGGER.info("DuckDB memory_limit set to %s", args.duckdb_memory_limit)

    if SHOW_PROGRESS:
        conn.execute("PRAGMA progress_bar_time=1.0")

    tasks = [
        ("user_id", "user_id", "user_features"),
        ("item_id", "item_id", "item_features"),
    ]

    results: dict[str, pd.DataFrame] = {}
    iterator = tasks
    if SHOW_PROGRESS and tqdm is not None:
        iterator = tqdm(tasks, desc="SQL queries", unit="entity")
    elif SHOW_PROGRESS:
        LOGGER.info("Executing %d SQL queries", len(tasks))

    for entity_column, entity_alias, prefix in iterator:
        df = fetch_entity_features(
            conn,
            entity_column=entity_column,
            entity_alias=entity_alias,
            parquet_glob=parquet_glob,
            history_start=history_start,
            end_day=args.end_day,
            valid_days=valid_days,
            window=args.window,
        )
        results[prefix] = df

    # 强制列类型为 int64，避免下游配置出现类型不一致
    for frame in results.values():
        for col in frame.columns:
            if col.endswith(f"_{args.window}d"):
                frame[col] = frame[col].fillna(0).astype("int64")

    split_and_save(results["user_features"], args.output_dir, "user_features")
    split_and_save(results["item_features"], args.output_dir, "item_features")

    LOGGER.info("DuckDB feature generation completed: days=%d window=%d", len(valid_days), args.window)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - top-level guard for CLI usage
        LOGGER.exception("Failed to generate features with DuckDB: %s", error)
        sys.exit(1)

