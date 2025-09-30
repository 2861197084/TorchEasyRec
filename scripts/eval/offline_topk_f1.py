#!/usr/bin/env python3
"""离线 TopK/F1 调参脚本。

2025-09-29 初版：基于 `part-0.parquet` 与验证集离线评估不同 TopK / 概率阈值。

该脚本读取预测结果 parquet 文件和带标签的验证集 parquet 文件，通过调整 TopK 或概率阈
值评估 F1/Precision/Recall，辅助选择提交策略。

用法示例：
    python scripts/eval/offline_topk_f1.py \
        --pred-path outputs/stage2_deepfm_v5/predict/part-0.parquet \
        --label-path data/processed/20141218_next_eval.parquet \
        --topk-list 5 10 20 50 \
        --threshold-list 0.01 0.005

脚本输出每个 TopK/阈值组合的指标，并保存最佳结果。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_parquet_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """读取指定列并返回 Pandas DataFrame。"""
    logging.info("读取数据：%s", path)
    table = pq.read_table(path, columns=list(columns))
    df = table.to_pandas()
    logging.info("数据规模：%d 行", len(df))
    return df


def build_ground_truth(label_df: pd.DataFrame) -> dict[int, set[int]]:
    """构建每个用户的购买集合。"""
    grouped = label_df[label_df["is_buy"] == 1].groupby("user_id")["item_id"]
    ground_truth = {uid: set(items.values) for uid, items in grouped}
    logging.info("有正样本用户数：%d", len(ground_truth))
    return ground_truth


def select_topk(pred_df: pd.DataFrame, topk: int) -> dict[int, list[int]]:
    """为每个用户选择 TopK 候选项。"""
    sorted_df = pred_df.sort_values(["user_id", "probs"], ascending=[True, False])
    topk_df = sorted_df.groupby("user_id").head(topk)
    result = topk_df.groupby("user_id")["item_id"].apply(list).to_dict()
    logging.debug("Top%d 用户数：%d", topk, len(result))
    return result


def select_threshold(pred_df: pd.DataFrame, threshold: float) -> dict[int, list[int]]:
    """按概率阈值选择候选项。"""
    filtered = pred_df[pred_df["probs"] >= threshold]
    result = filtered.sort_values(["user_id", "probs"], ascending=[True, False])
    result = result.groupby("user_id")["item_id"].apply(list).to_dict()
    logging.debug("阈值 %.4f 用户数：%d", threshold, len(result))
    return result


def evaluate_predictions(
    predictions: dict[int, Iterable[int]],
    ground_truth: dict[int, set[int]],
) -> tuple[float, float, float]:
    """计算 Precision、Recall、F1。"""
    tp = 0
    fp = 0
    fn = 0
    for user_id, items in predictions.items():
        pred_set = set(items)
        truth = ground_truth.get(user_id, set())
        tp += len(pred_set & truth)
        fp += len(pred_set - truth)
    for truth in ground_truth.values():
        fn += len(truth)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线 TopK/F1 评估脚本")
    parser.add_argument(
        "--pred-path", type=Path, default=Path("outputs/stage2_deepfm_v7/predict/part-0.parquet"), help="预测结果 parquet"
    )
    parser.add_argument("--label-path", type=Path, default=Path("data/processed/20141218_next_eval.parquet"), help="验证集 parquet")
    parser.add_argument(
        "--topk-list",
        type=int,
        nargs="*",
        default=[1, 3, 5, 10, 20],
        help="需要评估的 TopK 值列表",
    )
    parser.add_argument(
        "--threshold-list",
        type=float,
        nargs="*",
        default=[0.005, 0.01, 0.02, 0.05],
        help="需要评估的概率阈值列表",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/offline_eval_results.csv"),
        help="指标保存路径",
    )
    parser.add_argument("--verbose", action="store_true", help="打印 debug 日志")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    pred_df = read_parquet_columns(args.pred_path, ["user_id", "item_id", "probs"])
    label_df = read_parquet_columns(args.label_path, ["user_id", "item_id", "is_buy"])

    ground_truth = build_ground_truth(label_df)

    records: list[dict[str, float | int | str]] = []

    for topk in args.topk_list:
        logging.info("评估 Top%d", topk)
        predictions = select_topk(pred_df, topk)
        precision, recall, f1 = evaluate_predictions(predictions, ground_truth)
        logging.info(
            "Top%d -> Precision %.4f | Recall %.4f | F1 %.4f",
            topk,
            precision,
            recall,
            f1,
        )
        records.append(
            {
                "strategy": f"top{topk}",
                "topk": topk,
                "threshold": np.nan,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    for threshold in args.threshold_list:
        logging.info("评估阈值 >= %.4f", threshold)
        predictions = select_threshold(pred_df, threshold)
        precision, recall, f1 = evaluate_predictions(predictions, ground_truth)
        logging.info(
            "阈值 %.4f -> Precision %.4f | Recall %.4f | F1 %.4f",
            threshold,
            precision,
            recall,
            f1,
        )
        records.append(
            {
                "strategy": f"threshold_{threshold:.4f}",
                "topk": np.nan,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    result_df = pd.DataFrame(records)
    result_df.sort_values("f1", ascending=False, inplace=True)
    result_df.to_csv(args.output, index=False)
    logging.info("结果写入：%s", args.output)
    logging.info("最佳策略：%s", result_df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
