#!/usr/bin/env python3
"""根据命令行参数筛选预测结果并输出提交 txt 文件。"""

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq


def configure_logging(verbose: bool) -> None:
    """配置日志级别。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def load_prediction(path: Path) -> Any:
    """读取预测结果 parquet。"""
    logging.info("读取预测 parquet: %s", path)
    table = pq.read_table(path)
    return table.to_pandas()


def filter_by_threshold(prediction, threshold: float) -> Any:
    """按概率阈值筛选预测。"""
    return prediction[prediction["probs"] >= threshold]


def filter_by_topk(prediction, topk: int) -> Any:
    """对每个用户保留 TopK。"""
    sorted_df = prediction.sort_values(["user_id", "probs"], ascending=[True, False])
    return (
        sorted_df.groupby("user_id", group_keys=False)
        .head(topk)
        .loc[:, ["user_id", "item_id", "probs"]]
    )


def apply_filters(
    prediction,
    threshold: Optional[float],
    topk: Optional[int],
    min_prob: Optional[float],
    max_entries_per_user: Optional[int],
):
    """根据参数应用筛选逻辑。"""
    df = prediction

    if threshold is not None and threshold > 0:
        logging.info("按阈值 %.4f 筛选", threshold)
        df = filter_by_threshold(df, threshold)
    else:
        logging.info("阈值筛选未启用")

    if min_prob is not None and min_prob > 0:
        logging.info("按最小概率 %.4f 过滤", min_prob)
        df = df[df["probs"] >= min_prob]

    if topk is not None and topk > 0:
        logging.info("按 Top%d 筛选", topk)
        df = filter_by_topk(df, topk)
    else:
        logging.info("TopK 筛选未启用")

    if max_entries_per_user is not None and max_entries_per_user > 0:
        logging.info("限制每个用户最多 %d 条", max_entries_per_user)
        df = (
            df.sort_values(["user_id", "probs"], ascending=[True, False])
            .groupby("user_id", group_keys=False)
            .head(max_entries_per_user)
        )

    return df


def write_submission(df, output_path: Path, sep: str) -> None:
    """写出提交文件。"""
    logging.info("写出提交文件: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, ["user_id", "item_id"]].to_csv(
        output_path,
        sep=sep,
        index=False,
        header=False,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成提交 txt 的清洗脚本")
    parser.add_argument(
        "--pred-path",
        type=Path,
        # required=True,
        default=Path("outputs/stage2_deepfm_v6/predict/part-0.parquet"),
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
        default=0.018,
        help="概率阈值（<=0 表示关闭），默认 0.02",
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
    return parser.parse_args()


def main() -> None:
    """主函数。"""
    args = parse_args()
    configure_logging(args.verbose)

    prediction = load_prediction(args.pred_path)
    filtered = apply_filters(
        prediction,
        threshold=args.threshold,
        topk=args.topk,
        min_prob=args.min_prob,
        max_entries_per_user=args.max_entries_per_user,
    )

    logging.info("筛选后样本数: %d", len(filtered))
    write_submission(filtered, args.output, args.separator)

    logging.info("完成清洗，输出提交文件。")


if __name__ == "__main__":
    main()
