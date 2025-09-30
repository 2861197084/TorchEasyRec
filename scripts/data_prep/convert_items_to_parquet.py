#!/usr/bin/env python3
"""Convert item subset (txt/csv) to Parquet for downstream DuckDB pipelines.

输出: data/processed/raw_item/items.parquet

支持:
- 自动推断分隔符/表头 (DuckDB read_csv_auto)
- 列: 至少需要 item_id; 可选 item_geohash, item_category
- 去重: 按 item_id 去重
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert item subset text to Parquet")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("tianchi_fresh_comp_train_item_online.txt"),
        help="源文件路径（txt/csv/parquet）。若为 parquet 将直接复制到目标位置。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/raw_item"),
        help="输出目录，文件名固定为 items.parquet",
    )
    parser.add_argument("--verbose", action="store_true", help="打印 debug 日志")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def convert_to_parquet(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".parquet":
        # 仅保留按 item_id 去重后的单列
        conn = duckdb.connect()
        src = escape(input_path)
        dst = escape(output_path)
        sql = f"""
        COPY (
          SELECT DISTINCT CAST(item_id AS BIGINT) AS item_id
          FROM read_parquet('{src}')
        ) TO '{dst}' (FORMAT PARQUET);
        """
        conn.execute(sql)
        conn.close()
        return

    conn = duckdb.connect()
    src = escape(input_path)
    dst = escape(output_path)

    # 自动读取 csv/txt，并尽可能保留 geohash/category（若存在）
    conn.execute(f"CREATE TEMP VIEW src AS SELECT * FROM read_csv_auto('{src}');")
    cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='src' ORDER BY ordinal_position"
    ).fetchall()]

    has_item_id = "item_id" in cols
    has_col0 = len(cols) > 0 and cols[0] == "column0"
    has_geohash = "item_geohash" in cols
    has_category = "item_category" in cols

    if not has_item_id and not has_col0:
        raise SystemExit("找不到 item_id 列，且无法从第一列推断，请检查源文件格式。")

    # 使用基础表达式（不带别名），在 SELECT 中统一别名与类型
    item_id_base_expr = "item_id" if has_item_id else "column0"
    # 输出仅包含单列 item_id，避免因 geohash/category 引入重复

    sql = f"""
    COPY (
      SELECT DISTINCT CAST({item_id_base_expr} AS BIGINT) AS item_id
      FROM src
    ) TO '{dst}' (FORMAT PARQUET);
    """
    conn.execute(sql)
    conn.close()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    output_path = args.output_dir / "items.parquet"
    logging.info("Converting %s -> %s", args.input_path, output_path)
    convert_to_parquet(args.input_path, output_path)
    logging.info("Done.")


if __name__ == "__main__":
    main()


