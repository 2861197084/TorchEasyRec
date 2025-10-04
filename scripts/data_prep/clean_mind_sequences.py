#!/usr/bin/env python3
"""清洗现有 MIND 数据，将历史序列列改成分号分隔字符串。

使用场景：已经运行过早期版本的 ``prepare_mind_data.py``，生成的
``hist_item_seq``/``hist_cate_seq`` 列为列表类型，TorchEasyRec 无法直接读取。
本脚本会就地覆盖指定目录下的 Parquet 文件，保持其他字段不变。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import polars as pl
from polars.datatypes import List
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import types as pa_types


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗 MIND 序列列")
    parser.add_argument(
        "--mind-dir",
        type=Path,
        default=Path("data/processed/recall"),
        help="MIND 数据输出目录",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="mind_user_history/mind_*.parquet",
        help="需要替换的历史序列文件通配符",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="mind_train.parquet",
        help="训练样本文件相对路径",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="mind_eval.parquet",
        help="验证样本文件相对路径",
    )
    parser.add_argument(
        "--predict-file",
        type=str,
        default="mind_predict_users.parquet",
        help="预测样本文件相对路径",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        help="写回 Parquet 时使用的压缩算法",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Polars 线程数",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def list_target_files(mind_dir: Path, glob_pattern: str, extra_files: Iterable[str]) -> list[Path]:
    files: list[Path] = sorted(mind_dir.glob(glob_pattern))
    for rel_path in extra_files:
        path = mind_dir / rel_path
        if path.exists():
            files.append(path)
        else:
            LOGGER.warning("文件不存在，跳过: %s", path)
    return files


def _build_conversions(schema: dict[str, pl.DataType]) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for col in ("hist_item_seq", "hist_cate_seq"):
        dtype = schema.get(col)
        if dtype is None:
            continue
        if dtype == List:
            exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(""))
                .otherwise(
                    pl.col(col)
                    .cast(pl.List(pl.Utf8))
                    .list.join(";")
                )
                .alias(col)
            )
        elif dtype == pl.Utf8:
            exprs.append(pl.col(col).fill_null("").alias(col))
        else:
            LOGGER.debug("列 %s 类型 %s 无需处理", col, dtype)
    return exprs


def convert_file(path: Path, *, compression: str) -> None:
    LOGGER.info("处理文件: %s", path)
    lf = pl.scan_parquet(str(path))
    schema = lf.schema
    exprs = _build_conversions(schema)
    if not exprs:
        LOGGER.info("列类型已符合要求，跳过: %s", path)
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    lf = lf.with_columns(exprs)
    lf.sink_parquet(tmp_path, compression=compression)

    table = pq.read_table(tmp_path)
    updated = False
    arrays = []
    fields = []
    for field in table.schema:
        column = table.column(field.name)
        if field.name in {"hist_item_seq", "hist_cate_seq"} and pa_types.is_large_string(field.type):
            column = column.cast(pa.string())
            field = pa.field(field.name, pa.string(), nullable=field.nullable)
            updated = True
        arrays.append(column)
        fields.append(field)

    if updated:
        table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
        pq.write_table(table, tmp_path, compression=compression)

    tmp_path.replace(path)
    _maybe_cast_label(path, compression)


def _maybe_cast_label(path: Path, compression: str) -> None:
    table = pq.read_table(path)
    if "label" not in table.schema.names:
        return
    field = table.schema.field("label")
    if field.type == pa.int32():
        return
    column = table.column("label").cast(pa.int32())
    new_table = table.set_column(
        table.schema.get_field_index("label"),
        pa.field("label", pa.int32(), nullable=field.nullable),
        column,
    )
    pq.write_table(new_table, path, compression=compression)


def main() -> None:
    args = parse_args()
    configure_logging()
    try:
        pl.threading.threadpool_size(args.threads)
    except AttributeError:  # 兼容旧版 Polars
        LOGGER.warning("当前 Polars 版本不支持动态设置线程数")

    target_files = list_target_files(
        args.mind_dir,
        args.glob,
        [args.train_file, args.eval_file, args.predict_file],
    )

    if not target_files:
        LOGGER.error("未找到任何需要清洗的文件")
        raise SystemExit(1)

    for path in target_files:
        convert_file(path, compression=args.compression)

    LOGGER.info("处理完成，共更新 %d 个文件", len(target_files))


if __name__ == "__main__":
    main()

