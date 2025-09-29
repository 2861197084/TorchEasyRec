## 项目目标
- 利用 TorchEasyRec 框架完成天池移动电商推荐赛（参见 `Task.md`）的离线训练、评估与预测流程。
- 依据比赛要求输出 2014-12-19 用户对商品子集的购买预测结果，并以 F1 指标为核心优化目标。

## 阶段规划
1. **数据理解与治理**
   - 统计 `rawdata` 下用户行为 A/B 分片与商品子集文件的规模、字段完整性。
   - 设计日期与用户双重切分策略，生成训练集、验证集与预测日标签。
2. **特征工程与数据预处理**
   - 编写 `convert_to_parquet.py`、`generate_features.py`、`prepare_dataset.py` 三阶段脚本，将原始 TXT 转成分区 Parquet、聚合特征并生成训练/评估/预测样本（详见 `scripts/data_prep/README.md`）。
   - 构建用户/商品统计、地理偏好、序列行为等特征，确保字段定义与 TorchEasyRec `IdFeature`、`RawFeature`、`SequenceFeature` 规范一致 [^1]。
3. **模型方案与配置**
   - 初始排序模型选用 Wide&Deep，后续评估 DIN、DeepFM 等；召回阶段酌情引入 DSSM/MIND。
   - 所有配置文件放入 `configs/`，遵循 `阶段_模型_版本.config` 命名，引用官方配置字段格式 [^1]。
4. **训练、评估与记录**
   - 使用 TorchEasyRec CLI 执行训练、评估、导出模型；指标记录到 `docs/experiments.csv`。
   - 额外编写脚本计算比赛制 F1 值，形成统一验证流程。
5. **推理与提交**
   - 通过 `predict` 流程生成候选并做后处理（去重、TopN）。
   - 提交文件统一命名 `outputs/submission_模型_版本.tsv`。

## 关键产物
- 预处理脚本与文档：`scripts/data_prep/`
- 配置集合：`configs/staging/`（实验）、`configs/production/`（最终）
- 模型与预测：`models/`、`outputs/`

## 阶段性配置方案（逐步补充）

- **2025-09-30｜Stage2 DeepFM v6（新调参方案）**
  - 训练命令（新增模型目录 `models/stage2_deepfm_v6`）：

    ```bash
    torchrun --master_addr=localhost --master_port=29511 \
      --nnodes=1 --nproc-per-node=1 --node_rank=0 \
      -m tzrec.train_eval \
      --pipeline_config_path /root/autodl-tmp/TorchEasyRec/configs/staging/stage2_deepfm_v1.config \
      --model_dir /root/autodl-tmp/TorchEasyRec/models/stage2_deepfm_v6
    ```

  - TensorBoard 监控（GPU 实例本地预览）：

    ```bash
    tensorboard --logdir /root/autodl-tmp/TorchEasyRec/models/stage2_deepfm_v6 --port 6006
    ```

  - 模型导出：

    ```bash
    torchrun --master_addr=localhost --master_port=29511 \
      --nnodes=1 --nproc-per-node=1 --node_rank=0 \
      -m tzrec.export \
      --pipeline_config_path /root/autodl-tmp/TorchEasyRec/models/stage2_deepfm_v6/pipeline.config \
      --export_dir /root/autodl-tmp/TorchEasyRec/models/stage2_deepfm_v6/export
    ```

  - 预测产出：

    ```bash
    torchrun --master_addr=localhost --master_port=29511 \
      --nnodes=1 --nproc-per-node=1 --node_rank=0 \
      -m tzrec.predict \
      --scripted_model_path /root/autodl-tmp/TorchEasyRec/models/stage2_deepfm_v6/export \
      --predict_input_path /root/autodl-tmp/TorchEasyRec/data/processed/20141218_next_predict.parquet \
      --predict_output_path /root/autodl-tmp/TorchEasyRec/outputs/stage2_deepfm_v6/predict \
      --reserved_columns user_id,item_id
    ```

  - 离线阈值扫描：

    ```bash
    python /root/autodl-tmp/TorchEasyRec/scripts/eval/offline_topk_f1.py \
      --pred-path /root/autodl-tmp/TorchEasyRec/outputs/stage2_deepfm_v6/predict/part-0.parquet \
      --label-path /root/autodl-tmp/TorchEasyRec/data/processed/20141218_next_eval.parquet \
      --topk-list 5 10 20 50 \
      --threshold-list 0.015 0.02 0.025 \
      --output /root/autodl-tmp/TorchEasyRec/outputs/stage2_deepfm_v6/offline_eval_results.csv
    ```

  - 提交文件清洗脚本（命令行参数方式）：

    ```bash
    python /root/autodl-tmp/TorchEasyRec/scripts/eval/generate_submission.py \
      --pred-path /root/autodl-tmp/TorchEasyRec/outputs/stage2_deepfm_v6/predict/part-0.parquet \
      --output /root/autodl-tmp/TorchEasyRec/outputs/stage2_deepfm_v6/submission_stage2_deepfm_v6.txt \
      --threshold 0.02 \
      --topk 200 \
      --max-entries-per-user 200
    ```

    - 默认 `threshold=0.02`、`topk=200`、`max_entries_per_user=200`、`separator=\t`，可按需覆盖。
    - 输出文件不含表头，满足比赛提交格式要求。


## 风险与应对
- **数据规模巨大**：采用分批转换与多进程处理，必要时在转换脚本中引入分区字段。
- **特征缺失**：对空的地理信息等字段提供兜底值，确保 hash bucket 覆盖。
- **指标差异**：保持 TorchEasyRec 内置 AUC 与外部 F1 双指标监控。

[^1]: TorchEasyRec 官方文档《数据与特征》《模型》《使用指南》章节提供配置字段说明与示例 [https://torcheasyrec.readthedocs.io/zh-cn/latest/index.html](https://torcheasyrec.readthedocs.io/zh-cn/latest/index.html)。

