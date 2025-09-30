#!/usr/bin/env python3
"""根据命令行参数筛选预测结果并输出提交 txt 文件。"""

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
import duckdb


def configure_logging(verbose: bool) -> None:
    """配置日志级别。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def load_prediction(path: Path) -> Any:
    """已弃用：保留占位以兼容历史调用。当前流程使用 DuckDB 直接读取。"""
    logging.info("读取预测 parquet(compat): %s", path)
    return None
def escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def ensure_item_subset_parquet(path: Optional[Path]) -> Optional[Path]:
    """若提供 txt/csv 则先转换为 Parquet，返回 Parquet 路径。"""
    if path is None:
        return None
    if path.suffix.lower() == ".parquet":
        return path
    output = Path("data/processed/item_subset.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect()
    src = escape(path)
    dst = escape(output)
    # 自动推断 csv/txt 格式，并仅保留 item_id 列
    # 若源文件没有列名，则将第一列视为 item_id
    sql = f"""
    CREATE TEMP VIEW src AS SELECT * FROM read_csv_auto('{src}');
    -- 规范列名
    CREATE TEMP VIEW norm AS
    SELECT
        CASE WHEN 'item_id' IN (SELECT column_name FROM information_schema.columns WHERE table_name='src')
             THEN item_id
             ELSE CAST(column0 AS BIGINT)
        END AS item_id
    FROM src;
    COPY (SELECT DISTINCT CAST(item_id AS BIGINT) AS item_id FROM norm) TO '{dst}' (FORMAT PARQUET);
    """
    conn.execute(sql)
    conn.close()
    return output


def duckdb_write_submission(
    pred_path: Path,
    item_subset_parquet: Optional[Path],
    threshold: Optional[float],
    topk: Optional[int],
    min_prob: Optional[float],
    max_entries_per_user: Optional[int],
    output_path: Path,
    sep: str,
) -> None:
    conn = duckdb.connect()
    conn.execute("PRAGMA preserve_insertion_order=false")

    pred = escape(pred_path)
    out = escape(output_path)

    filters: list[str] = []
    if threshold is not None and threshold > 0:
        filters.append(f"probs >= {float(threshold)}")
    if min_prob is not None and min_prob > 0:
        filters.append(f"probs >= {float(min_prob)}")
    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    per_user_limit = 0
    if topk is not None and topk > 0:
        per_user_limit = topk
    if max_entries_per_user is not None and max_entries_per_user > 0:
        per_user_limit = min(per_user_limit or max_entries_per_user, max_entries_per_user)

    subset_join = ""
    if item_subset_parquet is not None:
        subset = escape(item_subset_parquet)
        subset_join = f"JOIN (SELECT DISTINCT CAST(item_id AS BIGINT) AS item_id FROM read_parquet('{subset}')) s USING(item_id)"

    rank_clause = ""
    if per_user_limit > 0:
        rank_clause = f"QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY probs DESC) <= {per_user_limit}"

    # 仅选择必要列并写出
    sql = f"""
    COPY (
      SELECT user_id, item_id
      FROM (
        SELECT user_id, item_id, CAST(probs AS DOUBLE) AS probs
        FROM read_parquet('{pred}')
        {where_clause}
      ) p
      {subset_join}
      {rank_clause}
      ORDER BY user_id, probs DESC
    ) TO '{out}' (HEADER false, DELIMITER '{sep}')
    """
    conn.execute(sql)
    conn.close()


# 旧的基于 Pandas 的过滤/写出逻辑已移除，统一改为 DuckDB 实现。


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成提交 txt 的清洗脚本")
    parser.add_argument(
        "--pred-path",
        type=Path,
        # required=True,
        default=Path("outputs/stage2_deepfm_v8/predict/part-0.parquet"),
        help="预测结果 parquet 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/submission.txt"),
        help="输出的提交 txt 路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.00,
        help="概率阈值（<=0 表示关闭），默认 0.02（与离线最优对齐）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="每个用户保留的 TopK（<=0 表示关闭），默认 10",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="额外的最小概率过滤值，默认 0.0（不生效）",
    )
    parser.add_argument(
        "--max-entries-per-user",
        type=int,
        default=0,
        help="每个用户最多保留的条数（<=0 表示不限制），默认 200",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\t",
        help="输出分隔符，默认为制表符",
    )
    parser.add_argument("--verbose", action="store_true", help="打印 debug 日志")
    parser.add_argument(
        "--item-subset-path",
        type=Path,
        default=None,
        help="商品子集 P 文件路径（可选，Parquet/CSV，需含 item_id 列）",
    )
    return parser.parse_args()


def main() -> None:
    """主函数。"""
    args = parse_args()
    configure_logging(args.verbose)

    subset_parquet = ensure_item_subset_parquet(args.item_subset_path)
    duckdb_write_submission(
        pred_path=args.pred_path,
        item_subset_parquet=subset_parquet,
        threshold=args.threshold,
        topk=args.topk,
        min_prob=args.min_prob,
        max_entries_per_user=args.max_entries_per_user,
        output_path=args.output,
        sep=args.separator,
    )

    logging.info("完成清洗，输出提交文件。")


if __name__ == "__main__":
    main()
