#!/usr/bin/env python3
"""生成 MIND 召回模型所需的数据集（Polars 版）。

改写要点：
1. 使用 Polars 惰性扫描原始日志，按日分块构造历史序列与滚动统计，避免
   DuckDB 大窗口导致的内存/磁盘爆炸。
2. 生成的目录结构保持不变：`mind_user_history/`、`mind_user_stats/`、
   `mind_item_stats/`、`mind_train.parquet`、`mind_eval.parquet`、
   `mind_predict_users.parquet`。
3. 默认按商品子集 P 过滤数据，支持窗口长度、序列长度等参数配置。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
DEFAULT_GEOHASH = "_NA_"  # 如果日志缺失 geohash，则使用默认占位符


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


@dataclass
class Paths:
    behavior_glob: Path
    item_subset: Optional[pl.Series]
    output_dir: Path

    @property
    def history_dir(self) -> Path:
        return self.output_dir / "mind_user_history"

    @property
    def user_stats_dir(self) -> Path:
        return self.output_dir / "mind_user_stats"

    @property
    def item_stats_dir(self) -> Path:
        return self.output_dir / "mind_item_stats"

    def ensure_dirs(self) -> None:
        for path in (self.output_dir, self.history_dir, self.user_stats_dir, self.item_stats_dir):
            path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 MIND 召回数据集（Polars 版）")
    parser.add_argument("--behavior-dir", type=Path, default=Path("data/processed/raw_behavior"))
    parser.add_argument("--item-subset-path", type=Path, default=Path("data/processed/raw_item/items.parquet"))
    parser.add_argument("--history-window-days", type=int, default=7)
    parser.add_argument("--stats-window-days", type=int, default=7)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--max-history-events", type=int, default=120)
    parser.add_argument("--label-behavior-type", type=int, default=4)
    parser.add_argument("--history-start-day", type=str, default="20141118")
    parser.add_argument("--train-start-day", type=str, default="20141125")
    parser.add_argument("--train-end-day", type=str, default="20141217")
    parser.add_argument("--eval-day", type=str, default="20141218")
    parser.add_argument("--predict-day", type=str, default="20141219")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/recall"))
    parser.add_argument("--log-file", type=Path, default=Path("logs/data_prep.log"))
    parser.add_argument("--chunk-size", type=int, default=5_000_000, help="单批处理的最大行数")
    parser.add_argument("--threads", type=int, default=8, help="Polars 线程数")
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


def iterate_days(start: str, end: str) -> Iterable[str]:
    current = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    while current <= end_dt:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def load_subset(path: Optional[Path]) -> Optional[pl.Series]:
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"商品子集文件不存在: {path}")
    df = pl.read_parquet(path) if path.suffix.lower() == ".parquet" else pl.read_csv(path)
    if "item_id" not in df.columns:
        raise SystemExit("商品子集文件必须包含 item_id 列")
    return df["item_id"].cast(pl.Int64)


def scan_behavior(paths: Paths, start_day: str, end_day: str) -> pl.LazyFrame:
    base = pl.scan_parquet(str(paths.behavior_glob), use_statistics=True)
    lf = (
        base
        .with_columns([
            pl.col("user_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
            pl.col("item_category").cast(pl.Int64),
            pl.col("user_geohash").cast(pl.Utf8).fill_null(DEFAULT_GEOHASH),
            pl.col("event_day").cast(pl.Utf8),
            pl.col("time"),
            pl.col("behavior_type").cast(pl.Int64),
        ])
        .filter((pl.col("event_day") >= start_day) & (pl.col("event_day") <= end_day))
    )
    if paths.item_subset is not None:
        subset_lf = paths.item_subset.unique().to_frame(name="item_id").lazy()
        lf = lf.join(subset_lf, on="item_id", how="inner")
    return lf


def build_history_for_day(
    paths: Paths,
    *,
    day: str,
    window_days: int,
    sequence_length: int,
    max_history_events: int,
    chunk_size: int,
) -> None:
    window_start = (datetime.strptime(day, "%Y%m%d") - timedelta(days=window_days - 1)).strftime("%Y%m%d")
    lf = scan_behavior(paths, window_start, day)
    df = lf.collect().sort(["user_id", "time", "item_id"])

    def build_history(group: pl.DataFrame) -> pl.DataFrame:
        items = group["item_id"].to_list()
        cates = group["item_category"].to_list()
        hist_items = items[-max_history_events:]
        hist_cates = cates[-max_history_events:]
        geohash = group["user_geohash"].to_list()
        geohash_value = geohash[-1] if geohash else DEFAULT_GEOHASH
        return pl.DataFrame(
            {
                "user_id": group["user_id"][0],
                "hist_item_seq": [";".join(map(str, hist_items[-sequence_length:]))],
                "hist_cate_seq": [";".join(map(str, hist_cates[-sequence_length:]))],
                "hist_len": [len(hist_items)],
                "user_geohash": [geohash_value or DEFAULT_GEOHASH],
                "event_day": [day],
            }
        )

    history = pl.concat(
        [build_history(group) for group in df.partition_by("user_id", maintain_order=True)]
        if df.height > 0 else [],
        how="vertical",
    )

    output_path = paths.history_dir / f"mind_{day}.parquet"
    history.write_parquet(output_path, compression="zstd")


def build_stats_for_day(
    paths: Paths,
    *,
    day: str,
    window_days: int,
    chunk_size: int,
) -> tuple[Path, Path]:
    window_start = (datetime.strptime(day, "%Y%m%d") - timedelta(days=window_days - 1)).strftime("%Y%m%d")
    lf = scan_behavior(paths, window_start, day)
    df = lf.collect()

    user_stats = (
        df.group_by("user_id", maintain_order=True)
        .agg([
            (pl.col("behavior_type") == 1).cast(pl.Int64).sum().alias("user_click_count"),
            (pl.col("behavior_type") == 2).cast(pl.Int64).sum().alias("user_fav_count"),
            (pl.col("behavior_type") == 3).cast(pl.Int64).sum().alias("user_cart_count"),
            (pl.col("behavior_type") == 4).cast(pl.Int64).sum().alias("user_buy_count"),
            pl.col("user_geohash").drop_nulls().last().fill_null(DEFAULT_GEOHASH).alias("user_geohash"),
        ])
        .with_columns(pl.lit(day).alias("event_day"))
    )

    item_stats = (
        df.group_by("item_id", maintain_order=True)
        .agg([
            (pl.col("behavior_type") == 1).cast(pl.Int64).sum().alias("item_click_count"),
            (pl.col("behavior_type") == 2).cast(pl.Int64).sum().alias("item_fav_count"),
            (pl.col("behavior_type") == 3).cast(pl.Int64).sum().alias("item_cart_count"),
            (pl.col("behavior_type") == 4).cast(pl.Int64).sum().alias("item_buy_count"),
            pl.col("item_category").first().alias("item_category"),
        ])
        .with_columns(pl.lit(day).alias("event_day"))
    )

    user_path = paths.user_stats_dir / f"user_stats_{day}.parquet"
    item_path = paths.item_stats_dir / f"item_stats_{day}.parquet"
    user_stats.write_parquet(user_path, compression="zstd")
    item_stats.write_parquet(item_path, compression="zstd")
    return user_path, item_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    pl.Config.set_tbl_cols(16)
    pl.Config.set_tbl_rows(20)
    try:
        pl.set_thread_pool_size(args.threads)
    except AttributeError:
        LOGGER.warning("当前 Polars 版本不支持动态设置线程池大小")

    if not args.behavior_dir.exists():
        raise SystemExit(f"行为日志目录不存在：{args.behavior_dir}")
    behavior_glob = args.behavior_dir / "event_day=*/chunk_*.parquet"

    subset_series = load_subset(args.item_subset_path)
    paths = Paths(behavior_glob=behavior_glob, item_subset=subset_series, output_dir=args.output_dir)
    paths.ensure_dirs()

    LOGGER.info("按日生成历史与统计 ...")
    for day in tqdm(list(iterate_days(args.train_start_day, args.predict_day)), desc="mind_days"):
        build_history_for_day(
            paths,
            day=day,
            window_days=args.history_window_days,
            sequence_length=args.sequence_length,
            max_history_events=args.max_history_events,
            chunk_size=args.chunk_size,
        )
        build_stats_for_day(
            paths,
            day=day,
            window_days=args.stats_window_days,
            chunk_size=args.chunk_size,
        )

    def build_dataset(
        *,
        day: str,
        paths: Paths,
        label_behavior: int,
    ) -> pl.DataFrame:
        window_lf = scan_behavior(paths, day, day)
        positives = (
            window_lf
            .filter(pl.col("behavior_type") == label_behavior)
            .select(["user_id", "item_id"])
            .unique()
            .collect()
        )

        history = pl.read_parquet(paths.history_dir / f"mind_{day}.parquet")
        user_stats = pl.read_parquet(paths.user_stats_dir / f"user_stats_{day}.parquet")
        item_stats = pl.read_parquet(paths.item_stats_dir / f"item_stats_{day}.parquet")

        candidates = (
            window_lf
            .select(["user_id", "item_id", "item_category"])
            .unique()
            .collect()
            .with_columns(pl.lit(day).alias("event_day"))
        )

        positives = positives.with_columns(pl.struct(["user_id", "item_id"]).alias("key"))
        candidates = candidates.with_columns(pl.struct(["user_id", "item_id"]).alias("key"))

        df = (
            candidates
            .join(history.drop("event_day"), on="user_id", how="left")
            .join(user_stats.drop("event_day"), on="user_id", how="left")
            .join(item_stats.drop("event_day"), on="item_id", how="left")
            .join(
                positives.with_columns(pl.lit(1).alias("label")),
                on=["user_id", "item_id"],
                how="left",
            )
            .drop("key")
            .with_columns([
                pl.col("hist_item_seq").fill_null(""),
                pl.col("hist_cate_seq").fill_null(""),
                pl.col("hist_len").fill_null(0),
                pl.col("user_click_count").fill_null(0),
                pl.col("user_fav_count").fill_null(0),
                pl.col("user_cart_count").fill_null(0),
                pl.col("user_buy_count").fill_null(0),
                pl.col("item_click_count").fill_null(0),
                pl.col("item_fav_count").fill_null(0),
                pl.col("item_cart_count").fill_null(0),
                pl.col("item_buy_count").fill_null(0),
                pl.col("label").fill_null(0).cast(pl.Int32),
            ])
        )
        return df

    LOGGER.info("生成训练样本 ...")
    train_frames = []
    for day in tqdm(list(iterate_days(args.train_start_day, args.train_end_day)), desc="mind_train"):
        df = build_dataset(day=day, paths=paths, label_behavior=args.label_behavior_type)
        train_frames.append(df)
    train_df = pl.concat(train_frames, how="vertical") if train_frames else pl.DataFrame()
    train_df.write_parquet(paths.output_dir / "mind_train.parquet", compression="zstd")

    LOGGER.info("生成验证样本 ...")
    eval_df = build_dataset(day=args.eval_day, paths=paths, label_behavior=args.label_behavior_type)
    eval_df.write_parquet(paths.output_dir / "mind_eval.parquet", compression="zstd")

    LOGGER.info("生成推理样本 ...")
    predict_history = pl.read_parquet(paths.history_dir / f"mind_{args.predict_day}.parquet")
    predict_history.write_parquet(paths.output_dir / "mind_predict_users.parquet", compression="zstd")

    LOGGER.info("生成物品特征 ...")
    predict_items = pl.read_parquet(paths.item_stats_dir / f"item_stats_{args.predict_day}.parquet")
    predict_items.write_parquet(paths.output_dir / "mind_item_features.parquet", compression="zstd")


if __name__ == "__main__":
    main()

