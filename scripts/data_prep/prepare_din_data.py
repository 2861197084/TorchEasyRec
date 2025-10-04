#!/usr/bin/env python3
"""生成 DIN 精排模型所需的数据集。

重构版说明：
1. 依赖 `prepare_mind_data.py` 输出的分日历史与统计结果，避免对旧版
   `generate_features.py` 的强耦合。
2. 按天处理召回候选，拼接用户历史序列、行为计数以及召回得分，构造
   训练 / 验证 / 预测数据。
3. 训练与验证阶段使用 `mind_train.parquet`、`mind_eval.parquet` 提供的正样本
   作为标签，预测阶段仅输出特征列。

输入依赖：
- `recall_dir/recall_YYYYMMDD.parquet`：召回候选文件，需包含
  `user_id, item_id, item_category, recall_score, recall_source`。
- `mind_history_dir/mind_YYYYMMDD.parquet`：用户历史序列。
- `user_stats_dir/user_stats_YYYYMMDD.parquet`：用户行为计数。
- `item_stats_dir/item_stats_YYYYMMDD.parquet`：物品行为计数。
- `mind_train.parquet`、`mind_eval.parquet`：正样本列表（含 event_day）。

输出：写入 `data/processed/rank/` 目录：
- `din_train.parquet`
- `din_eval.parquet`
- `din_predict.parquet`
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import duckdb

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    class _TqdmFallback:
        def __init__(self, total: Optional[int] = None, desc: Optional[str] = None) -> None:
            self.total = total
            self.desc = desc or "progress"
            self.count = 0

        def __enter__(self) -> "_TqdmFallback":
            print(f"[{self.desc}] start", flush=True)
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            print(f"[{self.desc}] done", flush=True)

        def update(self, n: int = 1) -> None:
            self.count += n
            if self.total:
                print(f"[{self.desc}] {self.count}/{self.total}", flush=True)
            else:
                print(f"[{self.desc}] {self.count}", flush=True)

        def close(self) -> None:
            pass

    def tqdm(*args, **kwargs):  # type: ignore
        return _TqdmFallback(total=kwargs.get("total"), desc=kwargs.get("desc"))


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 DIN 精排数据集")
    parser.add_argument(
        "--recall-dir",
        type=Path,
        default=Path("outputs/stage2_mind/recall"),
        help="召回候选目录，要求存在 recall_YYYYMMDD.parquet",
    )
    parser.add_argument(
        "--mind-train",
        type=Path,
        default=Path("data/processed/recall/mind_train.parquet"),
        help="MIND 训练样本（正样本列表）。",
    )
    parser.add_argument(
        "--mind-eval",
        type=Path,
        default=Path("data/processed/recall/mind_eval.parquet"),
        help="MIND 验证样本（正样本列表）。",
    )
    parser.add_argument(
        "--mind-history-dir",
        type=Path,
        default=Path("data/processed/recall/mind_user_history"),
        help="按日用户历史目录。",
    )
    parser.add_argument(
        "--user-stats-dir",
        type=Path,
        default=Path("data/processed/recall/mind_user_stats"),
        help="按日用户行为计数目录。",
    )
    parser.add_argument(
        "--item-stats-dir",
        type=Path,
        default=Path("data/processed/recall/mind_item_stats"),
        help="按日物品行为计数目录。",
    )
    parser.add_argument(
        "--train-start-day",
        type=str,
        default="20141125",
        help="训练集事件日开始（含）。",
    )
    parser.add_argument(
        "--train-end-day",
        type=str,
        default="20141217",
        help="训练集事件日结束（含）。",
    )
    parser.add_argument(
        "--eval-day",
        type=str,
        default="20141218",
        help="验证集事件日。",
    )
    parser.add_argument(
        "--predict-day",
        type=str,
        default="20141219",
        help="预测样本事件日。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/rank"),
        help="输出目录。",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="DuckDB 并行线程数。",
    )
    parser.add_argument(
        "--duckdb-temp-directory",
        type=Path,
        default=Path("data/duckdb_tmp"),
        help="DuckDB 临时目录。",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        type=str,
        default="32GB",
        help="DuckDB 内存上限 (例如 32GB)。",
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


def escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(f"{description} 不存在: {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iterate_days(start: str, end: str) -> Iterable[str]:
    current = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    while current <= end_dt:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def build_union_query(parts: list[Path]) -> str:
    selects = [f"SELECT * FROM read_parquet('{escape(p)}')" for p in parts]
    if not selects:
        return "SELECT * FROM read_parquet(NULL) WHERE 1=0"  # 空结果
    return " UNION ALL ".join(selects)


def write_parquet(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    output: Path,
    progress: Optional[str] = None,
) -> None:
    ensure_dir(output.parent)
    LOGGER.info("写出 %s", output)
    if progress:
        with tqdm(desc=progress, total=1, leave=False) as bar:
            conn.execute(
                f"COPY ({query}) TO '{escape(output)}' (FORMAT 'parquet', COMPRESSION ZSTD)"
            )
            bar.update(1)
    else:
        conn.execute(
            f"COPY ({query}) TO '{escape(output)}' (FORMAT 'parquet', COMPRESSION ZSTD)"
        )


def build_day_query(
    *,
    day: str,
    recall_path: Path,
    history_path: Path,
    user_stats_path: Path,
    item_stats_path: Path,
    positives_path: Optional[Path],
) -> str:
    ensure_file(recall_path, "召回文件")
    ensure_file(history_path, "用户历史文件")
    ensure_file(user_stats_path, "用户统计文件")
    ensure_file(item_stats_path, "物品统计文件")

    positives_cte = ""
    label_select = "0 AS label"
    if positives_path is not None:
        ensure_file(positives_path, "正样本文件")
        positives_cte = (
            f", pos AS (\n    SELECT DISTINCT user_id, item_id\n    FROM read_parquet('{escape(positives_path)}')\n    WHERE event_day = '{day}'\n)"
        )
        label_select = "CASE WHEN pos.user_id IS NOT NULL THEN 1 ELSE 0 END AS label"

    return f"""
WITH
cand AS (
    SELECT * FROM read_parquet('{escape(recall_path)}')
),
hist AS (
    SELECT * FROM read_parquet('{escape(history_path)}')
),
user_feat AS (
    SELECT * FROM read_parquet('{escape(user_stats_path)}')
),
item_feat AS (
    SELECT * FROM read_parquet('{escape(item_stats_path)}')
){positives_cte}
SELECT
    cand.user_id,
    cand.item_id,
    cand.item_category,
    cand.recall_score,
    cand.recall_source,
    hist.hist_item_seq,
    hist.hist_cate_seq,
    COALESCE(user_feat.click_count, 0) AS user_click_count,
    COALESCE(user_feat.fav_count, 0) AS user_fav_count,
    COALESCE(user_feat.cart_count, 0) AS user_cart_count,
    COALESCE(user_feat.buy_count, 0) AS user_buy_count,
    COALESCE(item_feat.click_count, 0) AS item_click_count,
    COALESCE(item_feat.fav_count, 0) AS item_fav_count,
    COALESCE(item_feat.cart_count, 0) AS item_cart_count,
    COALESCE(item_feat.buy_count, 0) AS item_buy_count,
    {label_select}
FROM cand
LEFT JOIN hist USING(user_id)
LEFT JOIN user_feat USING(user_id)
LEFT JOIN item_feat USING(item_id)
"""


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)

    ensure_file(args.mind_train, "MIND 训练文件")
    ensure_file(args.mind_eval, "MIND 验证文件")
    if not args.recall_dir.exists():
        raise SystemExit(f"召回目录不存在: {args.recall_dir}")

    temp_dir = args.duckdb_temp_directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect()
    conn.execute(f"PRAGMA threads={args.threads}")
    conn.execute(f"PRAGMA temp_directory='{escape(temp_dir)}'")
    conn.execute(f"PRAGMA memory_limit='{args.duckdb_memory_limit}'")
    conn.execute("PRAGMA preserve_insertion_order=false")

    ensure_dir(args.output_dir)

    train_parts: list[Path] = []
    for day in tqdm(list(iterate_days(args.train_start_day, args.train_end_day)), desc="din_train_days"):
        query = build_day_query(
            day=day,
            recall_path=args.recall_dir / f"recall_{day}.parquet",
            history_path=args.mind_history_dir / f"mind_{day}.parquet",
            user_stats_path=args.user_stats_dir / f"user_stats_{day}.parquet",
            item_stats_path=args.item_stats_dir / f"item_stats_{day}.parquet",
            positives_path=args.mind_train,
        )
        part_path = args.output_dir / f"tmp_din_train_{day}.parquet"
        write_parquet(conn, query, part_path, progress=f"train_{day}")
        train_parts.append(part_path)

    train_output = args.output_dir / "din_train.parquet"
    write_parquet(conn, build_union_query(train_parts), train_output, progress="train_merge")

    eval_query = build_day_query(
        day=args.eval_day,
        recall_path=args.recall_dir / f"recall_{args.eval_day}.parquet",
        history_path=args.mind_history_dir / f"mind_{args.eval_day}.parquet",
        user_stats_path=args.user_stats_dir / f"user_stats_{args.eval_day}.parquet",
        item_stats_path=args.item_stats_dir / f"item_stats_{args.eval_day}.parquet",
        positives_path=args.mind_eval,
    )
    eval_output = args.output_dir / "din_eval.parquet"
    write_parquet(conn, eval_query, eval_output, progress="din_eval")

    predict_query = build_day_query(
        day=args.predict_day,
        recall_path=args.recall_dir / f"recall_{args.predict_day}.parquet",
        history_path=args.mind_history_dir / f"mind_{args.predict_day}.parquet",
        user_stats_path=args.user_stats_dir / f"user_stats_{args.predict_day}.parquet",
        item_stats_path=args.item_stats_dir / f"item_stats_{args.predict_day}.parquet",
        positives_path=None,
    )
    predict_output = args.output_dir / "din_predict.parquet"
    write_parquet(conn, predict_query, predict_output, progress="din_predict")

    for part in train_parts:
        part.unlink(missing_ok=True)

    conn.close()
    LOGGER.info("DIN 数据准备完成")


if __name__ == "__main__":
    main()

