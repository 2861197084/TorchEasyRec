## 项目目标
- 利用 TorchEasyRec 框架完成天池移动电商推荐赛（参见 `Task.md`）的离线训练、评估与预测流程。
- 依据比赛要求输出 2014-12-19 用户对商品子集的购买预测结果，并以 F1 指标为核心优化目标。

## 阶段规划
1. **数据理解与治理**
   - 统计 `rawdata` 下用户行为 A/B 分片与商品子集文件的规模、字段完整性。
   - 设计日期与用户双重切分策略，生成训练集、验证集与预测日标签。
2. **特征工程与数据预处理**
   - 编写 `convert_to_parquet.py`、`convert_items_to_parquet.py`（新增）、`generate_features.py`、`prepare_dataset.py` 四阶段脚本，将原始 TXT 转成分区 Parquet、聚合特征并生成训练/评估/预测样本（详见 `scripts/data_prep/README.md`）。
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

- **2025-09-30｜Stage2 DeepFM **
  - 训练命令（新增模型目录 ：

    ```bash
    torchrun --master_addr=localhost --master_port=29511 \
      -m tzrec.train_eval \
      --pipeline_config_path /root/autodl-tmp/TorchEasyRec/configs/staging/stage2_mind_v1.config
    ```

  - TensorBoard 监控（GPU 实例本地预览）：

    ```bash
    tensorboard --logdir /root/autodl-tmp/TorchEasyRec/models/tage2_mind_v1 --port 6006
    ```

  - 模型导出：

    ```bash
    torchrun --master_addr=localhost --master_port=29511 \
      -m tzrec.export \
      --pipeline_config_path /root/autodl-tmp/TorchEasyRec/models/stage2_deepfm_v10/pipeline.config \
      –checkpoint_path /root/autodl-tmp/TorchEasyRec/models/stage2_deepfm_v10/model.ckpt-14015 \
      --export_dir /root/autodl-tmp/TorchEasyRec/models/stage2_deepfm_v10/export
    ```

  - 预测产出：

    ```bash
    torchrun --master_addr=localhost --master_port=29511 \
      --nnodes=1 --nproc-per-node=1 --node_rank=0 \
      -m tzrec.predict \
      --scripted_model_path /root/autodl-tmp/TorchEasyRec/models/stage2_deepfm_v10/export \
      --predict_input_path /root/autodl-tmp/TorchEasyRec/data/processed/20141218_v2_predict.parquet \
      --predict_output_path /root/autodl-tmp/TorchEasyRec/outputs/stage2_deepfm_v10/predict \
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
      --item-subset-path /root/autodl-tmp/TorchEasyRec/data/processed/raw_item/items.parquet \
      --output /root/autodl-tmp/TorchEasyRec/outputs/stage2_deepfm_v6/submission_stage2_deepfm_v6.txt \
      --threshold 0.02 \
      --topk 200 \
      --max-entries-per-user 200
    ```
## 商品子集 P 口径与数据流（重要）
1. 将官方 `tianchi_fresh_comp_train_item_online.txt` 转换为 Parquet：
   ```bash
   python scripts/data_prep/convert_items_to_parquet.py \
     --input-path /root/autodl-tmp/TorchEasyRec/tianchi_fresh_comp_train_item_online.txt
   ```
   输出：`data/processed/raw_item/items.parquet`

2. 生成窗口特征时在 DuckDB 内按 P 过滤：
   - `generate_features.py` 若检测到 `data/processed/raw_item/items.parquet`，会自动在窗口聚合前 `JOIN` P；也可通过 `--item-parquet` 指定路径。

3. 准备数据集时按 P 过滤候选与标签：
   - `prepare_dataset.py` 若未传 `--item-subset-path` 但检测到标准路径，会默认使用它；标签构造（次日购买）也限定在 P 内。

4. 提交文件写出前再次按 P 过滤：
   - `scripts/eval/generate_submission.py` 支持 `--item-subset-path`；若不传且已有标准路径，可直接使用该 Parquet，默认阈值 0.02。


    - 默认 `threshold=0.02`、`topk=200`、`max_entries_per_user=200`、`separator=\t`，可按需覆盖。
    - 输出文件不含表头，满足比赛提交格式要求。


## 风险与应对
- **数据规模巨大**：采用分批转换与多进程处理，必要时在转换脚本中引入分区字段。
- **特征缺失**：对空的地理信息等字段提供兜底值，确保 hash bucket 覆盖。
- **指标差异**：保持 TorchEasyRec 内置 AUC 与外部 F1 双指标监控。

[^1]: TorchEasyRec 官方文档《数据与特征》《模型》《使用指南》章节提供配置字段说明与示例 [https://torcheasyrec.readthedocs.io/zh-cn/latest/index.html](https://torcheasyrec.readthedocs.io/zh-cn/latest/index.html)。

