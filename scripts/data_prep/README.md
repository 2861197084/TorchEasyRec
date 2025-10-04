## 数据预处理说明

### 输入数据
- 用户行为明细：`rawdata/tianchi_fresh_comp_train_user_online_partA.txt` 与 `partB.txt`
- 商品子集列表（P）：`rawdata/tianchi_fresh_comp_train_item_online.txt`

### 输出规范
- 原始日志：`data/processed/raw_behavior/event_day=YYYYMMDD/chunk_*.parquet`
- 商品子集：`data/processed/raw_item/items.parquet`
- 召回/精排数据：
  - MIND 召回：`data/processed/recall/`
    - `mind_user_history/mind_YYYYMMDD.parquet`
    - `mind_user_stats/user_stats_YYYYMMDD.parquet`
    - `mind_item_stats/item_stats_YYYYMMDD.parquet`
    - `mind_train.parquet` / `mind_eval.parquet` / `mind_predict_users.parquet`
  - DIN 精排：`data/processed/rank/din_*.parquet`
- 日志记录：关键数据量、异常计数写入 `logs/data_prep.log`

### 脚本规划
1. `convert_to_parquet.py`
   - 功能：分片读取 TXT，解析字段，追加 `event_day`、`event_hour`
   - 输出：按日期写入 Parquet，并生成校验摘要
2. `convert_items_to_parquet.py`
   - 功能：将官方商品子集 txt/csv 转为 Parquet，输出 `data/processed/raw_item/items.parquet`
   - 字段：至少包含 `item_id`，若原始文件含 `item_geohash`、`item_category` 也会一并保留
   - 用法示例：
     ```bash
     python scripts/data_prep/convert_items_to_parquet.py \
       --input-path rawdata/tianchi_fresh_comp_train_item_online.txt
     ```
3. `prepare_mind_data.py`
   - 功能：基于 `raw_behavior` 构造多阶段召回所需数据
     - 提取用户历史序列（按日分块写入 `mind_user_history/`）
     - 计算用户/物品滚动行为统计（写入 `mind_user_stats/`、`mind_item_stats/`）
     - 输出召回训练/验证/推理样本 (`mind_train` / `mind_eval` / `mind_predict_users`)
   - 关键参数：`--history-window-days`、`--stats-window-days`、`--item-subset-path`
   - 示例：
     ```bash
     python scripts/data_prep/prepare_mind_data.py \
       --behavior-dir data/processed/raw_behavior \
       --item-subset-path data/processed/raw_item/items.parquet \
       --history-window-days 7 --stats-window-days 7 \
       --train-start-day 20141125 --train-end-day 20141217 \
       --eval-day 20141218 --predict-day 20141219 \
       --output-dir data/processed/recall
     ```
4. `prepare_din_data.py`
   - 功能：读取召回候选与 MIND 产物，为 DIN 精排生成训练/验证/预测样本
   - 输入：
     - 召回候选 `outputs/.../recall_YYYYMMDD.parquet`
     - `mind_user_history/`、`mind_user_stats/`、`mind_item_stats/`
     - 正样本 `mind_train.parquet`、`mind_eval.parquet`
   - 输出：`data/processed/rank/din_train.parquet` 等
   - 示例：
     ```bash
     python scripts/data_prep/prepare_din_data.py \
       --recall-dir outputs/stage2_mind/recall \
       --mind-train data/processed/recall/mind_train.parquet \
       --mind-eval data/processed/recall/mind_eval.parquet \
       --mind-history-dir data/processed/recall/mind_user_history \
       --user-stats-dir data/processed/recall/mind_user_stats \
       --item-stats-dir data/processed/recall/mind_item_stats \
       --train-start-day 20141125 --train-end-day 20141217 \
       --eval-day 20141218 --predict-day 20141219 \
       --output-dir data/processed/rank
     ```

### 注意事项
- 确保空 `user_geohash`、`item_geohash` 使用占位符 `_NA_`
- 低内存环境建议在高配机器上运行 `prepare_mind_data.py`，或适当调小窗口/线程
- 召回候选需按日分块；若新增召回模型，请保证输出字段一致
- 处理完成后在 `docs/experiments.csv` 记录数据范围与特征版本

### 最佳实践（商品子集 P）
1. 转换商品子集：
   ```bash
   python scripts/data_prep/convert_items_to_parquet.py \
     --input-path /root/autodl-tmp/TorchEasyRec/tianchi_fresh_comp_train_item_online.txt
   ```
2. 生成召回数据：
   ```bash
   python scripts/data_prep/prepare_mind_data.py \
     --behavior-dir data/processed/raw_behavior \
     --item-subset-path data/processed/raw_item/items.parquet \
     --history-window-days 7 --stats-window-days 7 \
     --train-start-day 20141125 --train-end-day 20141217 \
     --eval-day 20141217 --predict-day 20141219 \
     --output-dir data/processed/recall
   ```
3. 生成精排数据：
   ```bash
   python scripts/data_prep/prepare_din_data.py \
     --recall-dir outputs/stage2_mind/recall \
     --mind-train data/processed/recall/mind_train.parquet \
     --mind-eval data/processed/recall/mind_eval.parquet \
     --mind-history-dir data/processed/recall/mind_user_history \
     --user-stats-dir data/processed/recall/mind_user_stats \
     --item-stats-dir data/processed/recall/mind_item_stats \
     --train-start-day 20141125 --train-end-day 20141217 \
     --eval-day 20141217 --predict-day 20141219 \
     --output-dir data/processed/rank
   ```
4. 召回/精排完成后，记录实验信息并进行模型训练/评估。
