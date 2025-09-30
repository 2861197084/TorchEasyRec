"""Build training/evaluation/inference datasets for TorchEasyRec.

该脚本基于 `convert_to_parquet.py` 与 `generate_features_duckdb.py`
生成的分区数据，按日期拼接用户/商品特征并打上标签，输出
TorchEasyRec 所需的 Parquet 数据集。

特性：
- 使用 DuckDB 直接读取分区 Parquet，避免将 10 亿级日志一次性载入内存
- 支持配置训练集时间范围、验证日、预测日
- 默认将 `behavior_type == 4` 视为购买标签，可通过参数调整
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import duckdb

LOGGER = logging.getLogger(__name__)
DATE_FMT = "%Y%m%d"
DEFAULT_TRAIN_START = "20141125"
DEFAULT_TRAIN_END = "20141217"
DEFAULT_EVAL_DAY = "20141217"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for TorchEasyRec")
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/processed/features-7d"),
        help="目录：`generate_features.py` 输出的 user/item 特征集合。",
    )
    parser.add_argument(
        "--behavior-dir",
        type=Path,
        default=Path("data/processed/raw_behavior"),
        help="目录：`convert_to_parquet.py` 输出的行为日志分区。",
    )
    parser.add_argument(
        "--item-subset-path",
        type=Path,
        default='data/processed/raw_item/items.parquet',
        help="商品子集 Parquet/CSV 路径；缺省时自动读取 data/processed/raw_item/items.parquet（若存在）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="输出目录，脚本会写入 train/eval/predict 三个 parquet。",
    )
    parser.add_argument(
        "--train-start-day",
        type=str,
        default=DEFAULT_TRAIN_START,
        help="训练集开始日期 (含)，格式 YYYYMMDD。",
    )
    parser.add_argument(
        "--train-end-day",
        type=str,
        default=DEFAULT_TRAIN_END,
        help="训练集结束日期 (含)，格式 YYYYMMDD。",
    )
    parser.add_argument(
        "--eval-day",
        type=str,
        default=DEFAULT_EVAL_DAY,
        help="验证集日期，格式 YYYYMMDD。",
    )
    parser.add_argument(
        "--predict-day",
        type=str,
        default="20141218",
        help="推理集日期，默认为 20141218。",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="20141218",
        help="输出文件名前缀，例如 20141218 => 20141218_train.parquet。",
    )
    parser.add_argument(
        "--positive-behavior-type",
        type=int,
        default=4,
        help="被视为正样本的行为类型，默认为购买(4)。",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 8,
        help="DuckDB 并行线程数。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/data_prep.log"),
        help="日志文件路径。",
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


def discover_feature_days(features_dir: Path) -> list[str]:
    days = sorted(
        p.name.split("_")[-1].split(".parquet")[0]
        for p in features_dir.glob("user_features_*.parquet")
    )
    if not days:
        raise SystemExit("在 features_dir 下未找到 user_features_*.parquet，请先运行特征生成脚本。")
    return days


def ensure_paths(days: Iterable[str], features_dir: Path, behavior_dir: Path) -> None:
    missing_features: list[str] = []
    missing_behavior: list[str] = []
    for day in days:
        if not (features_dir / f"user_features_{day}.parquet").exists():
            missing_features.append(f"user_features_{day}.parquet")
        if not (features_dir / f"item_features_{day}.parquet").exists():
            missing_features.append(f"item_features_{day}.parquet")
        if not (behavior_dir / f"event_day={day}").exists():
            missing_behavior.append(f"event_day={day}")
    if missing_features:
        raise SystemExit(f"缺少特征文件: {missing_features}")
    if missing_behavior:
        raise SystemExit(f"缺少行为日志分区: {missing_behavior}")


def escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def build_feature_union(days: Sequence[str], features_dir: Path, entity: str) -> str:
    selects = []
    column_prefix = "user" if entity == "user" else "item"
    for day in days:
        feature_path = escape(features_dir / f"{entity}_features_{day}.parquet")
        selects.append(
            f"""
            SELECT
                '{day}' AS event_day,
                {entity}_id,
                click_7d AS {column_prefix}_click_7d,
                fav_7d AS {column_prefix}_fav_7d,
                cart_7d AS {column_prefix}_cart_7d,
                buy_7d AS {column_prefix}_buy_7d
            FROM read_parquet('{feature_path}')
            """
        )
    return "\nUNION ALL\n".join(selects)


def build_dataset_query(
    days: Sequence[str],
    behavior_dir: Path,
    features_dir: Path,
    positive_behavior: int | None,
    label_days: Sequence[str] | None,
    *,
    item_subset_scan: str | None = None,
) -> str:
    if not days:
        raise ValueError("days must not be empty")
    behavior_glob = escape(behavior_dir / "event_day=*/chunk_*.parquet")
    user_union = build_feature_union(days, features_dir, "user")
    item_union = build_feature_union(days, features_dir, "item")
    day_list = ",".join(f"'{day}'" for day in days)
    label_select = "NULL AS is_buy"
    final_label_select = "NULL AS is_buy"
    next_buy_cte = ""
    next_buy_join = ""
    subset_cte = ""
    subset_join = ""
    subset_filter = ""
    if item_subset_scan is not None:
        subset_cte = f"item_subset AS (\n    SELECT DISTINCT item_id FROM {item_subset_scan}\n)"
        subset_join = "\nJOIN item_subset s ON s.item_id = r.item_id"
        subset_filter = "\n      AND item_id IN (SELECT item_id FROM item_subset)"
    if positive_behavior is not None and label_days is not None:
        label_day_list = ",".join(f"'{day}'" for day in label_days)
        next_buy_filter = subset_filter if item_subset_scan is not None else ""
        next_buy_cte = (
            f"""next_buy AS (\n    SELECT\n        user_id,\n        item_id,\n        strftime('%Y%m%d', strptime(CAST(event_day AS VARCHAR), '%Y%m%d') - INTERVAL 1 DAY) AS prev_day,\n        1 AS is_buy\n    FROM read_parquet('{behavior_glob}')\n    WHERE event_day IN ({label_day_list})\n      AND behavior_type = {positive_behavior}{next_buy_filter}\n    GROUP BY user_id, item_id, prev_day\n)"""
        )
        final_label_select = "COALESCE(nb.is_buy, 0) AS is_buy"
        next_buy_join = "LEFT JOIN next_buy nb ON nb.prev_day = r.event_day AND nb.user_id = r.user_id AND nb.item_id = r.item_id"
    elif positive_behavior is not None and label_days is None:
        final_label_select = f"CASE WHEN behavior_type = {positive_behavior} THEN 1 ELSE 0 END AS is_buy"

    cte_clauses = []
    if subset_cte:
        cte_clauses.append(subset_cte)
    raw_clause = f"raw AS (\n    SELECT\n        user_id,\n        item_id,\n        behavior_type,\n        user_geohash,\n        item_category,\n        CAST(event_day AS VARCHAR) AS event_day,\n        event_hour,\n        time\n    FROM read_parquet('{behavior_glob}')\n    WHERE event_day IN ({day_list}){subset_filter}\n)"
    cte_clauses.append(raw_clause)
    if next_buy_cte:
        # next_buy_cte already contains leading comma/newline; remove them for consistency
        next_buy_clause = next_buy_cte.strip()[1:].strip() if next_buy_cte.startswith(",") else next_buy_cte.strip()
        cte_clauses.append(next_buy_clause)
    cte_clauses.append(f"user_feat AS (\n    {user_union}\n)".strip())
    cte_clauses.append(f"item_feat AS (\n    {item_union}\n)".strip())

    cte_sql = ",\n".join(cte_clauses)

    return f"""
WITH {cte_sql}
SELECT
    r.user_id,
    r.item_id,
    CAST(r.behavior_type AS BIGINT) AS behavior_type,
    COALESCE(r.user_geohash, '_NA_') AS user_geohash,
    r.item_category,
    r.event_day,
    CAST(datediff('day', DATE '2014-11-18', strptime(r.event_day, '%Y%m%d')) AS BIGINT) AS event_day_index,
    CAST(r.event_hour AS BIGINT) AS event_hour,
    uf.user_click_7d,
    uf.user_fav_7d,
    uf.user_cart_7d,
    uf.user_buy_7d,
    it.item_click_7d,
    it.item_fav_7d,
    it.item_cart_7d,
    it.item_buy_7d,
    {final_label_select}
FROM raw r{subset_join}
LEFT JOIN user_feat uf
    ON uf.event_day = r.event_day AND uf.user_id = r.user_id
LEFT JOIN item_feat it
    ON it.event_day = r.event_day AND it.item_id = r.item_id
{next_buy_join}
"""


def export_dataset(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing dataset to %s", output_path)
    conn.execute(
        f"COPY ({query}) TO '{escape(output_path)}' (FORMAT 'parquet', COMPRESSION ZSTD)"
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)

    all_days = discover_feature_days(args.features_dir)
    LOGGER.info("Detected feature days: %s ... %s (%d days)", all_days[0], all_days[-1], len(all_days))

    train_days = [day for day in all_days if args.train_start_day <= day <= args.train_end_day]
    if args.eval_day not in all_days:
        raise SystemExit(f"eval_day {args.eval_day} 不在特征目录中")
    if args.predict_day not in all_days:
        raise SystemExit(f"predict_day {args.predict_day} 不在特征目录中")

    ensure_paths(set(train_days + [args.eval_day, args.predict_day]), args.features_dir, args.behavior_dir)

    conn = duckdb.connect()
    conn.execute(f"PRAGMA threads={args.threads}")
    conn.execute("PRAGMA temp_directory='data/duckdb_tmp'")
    conn.execute("PRAGMA max_temp_directory_size='120GB'")
    conn.execute("PRAGMA preserve_insertion_order=false")

    eval_date = datetime.strptime(args.eval_day, DATE_FMT)
    eval_label_day = (eval_date + timedelta(days=1)).strftime(DATE_FMT)
    if eval_label_day not in all_days:
        LOGGER.warning("eval_label_day %s 不在特征目录中，将无法生成验证标签", eval_label_day)

    train_label_end = (datetime.strptime(args.train_end_day, DATE_FMT) + timedelta(days=1)).strftime(DATE_FMT)
    train_label_days = [day for day in all_days if args.train_start_day <= day <= train_label_end]

    item_subset_scan: str | None = None
    subset_path = args.item_subset_path
    if subset_path is None:
        default_subset = Path("data/processed/raw_item/items.parquet")
        if default_subset.exists():
            subset_path = default_subset
    if subset_path is not None:
        real_path = subset_path.resolve()
        escaped = str(real_path).replace("'", "''")
        if real_path.suffix.lower() == ".parquet":
            item_subset_scan = f"read_parquet('{escaped}')"
        else:
            item_subset_scan = f"read_csv_auto('{escaped}')"
        LOGGER.info("商品子集过滤启用：%s", real_path)
    else:
        LOGGER.warning("未提供商品子集文件，train/eval/predict 将包含全集商品。")

    train_query = build_dataset_query(
        train_days,
        args.behavior_dir,
        args.features_dir,
        args.positive_behavior_type,
        train_label_days,
        item_subset_scan=item_subset_scan,
    )
    eval_query = build_dataset_query(
        [args.eval_day],
        args.behavior_dir,
        args.features_dir,
        args.positive_behavior_type,
        [eval_label_day],
        item_subset_scan=item_subset_scan,
    )
    predict_query = build_dataset_query(
        [args.predict_day],
        args.behavior_dir,
        args.features_dir,
        None,
        None,
        item_subset_scan=item_subset_scan,
    )

    train_output = args.output_dir / f"{args.output_prefix}_train.parquet"
    eval_output = args.output_dir / f"{args.output_prefix}_eval.parquet"
    predict_output = args.output_dir / f"{args.output_prefix}_predict.parquet"

    export_dataset(conn, train_query, train_output)
    export_dataset(conn, eval_query, eval_output)
    export_dataset(conn, predict_query, predict_output)

    LOGGER.info(
        "Dataset preparation completed: train=%s eval=%s predict=%s",
        train_output,
        eval_output,
        predict_output,
    )


if __name__ == "__main__":
    main()
