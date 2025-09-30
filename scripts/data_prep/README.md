## 数据预处理说明

### 输入数据
- 用户行为明细：`rawdata/tianchi_fresh_comp_train_user_online_partA.txt` 与 `partB.txt`
- 商品子集列表（P）：`rawdata/tianchi_fresh_comp_train_item_online.txt`

### 输出规范
- 预处理产物写入 `data/processed/`，命名 `yyyymmdd_split.parquet`
- 列字段：行为聚合特征（点击/加购/收藏/购买）、地理/品类信息、时间衍生字段、标签 `is_buy`
- 日志记录：关键数据量、异常计数写入 `logs/data_prep.log`

### 脚本规划
1. `convert_to_parquet.py`
   - 功能：分片读取 TXT，解析字段，追加 `event_day`、`event_hour`
   - 输出：按日期写入 Parquet，并生成校验摘要
2. `convert_items_to_parquet.py`（新增）
   - 功能：将官方商品子集 txt/csv 转为 Parquet，输出 `data/processed/raw_item/items.parquet`
   - 字段：至少包含 `item_id`，若原始文件含 `item_geohash`、`item_category` 也会一并保留
   - 用法示例：
     ```bash
     python scripts/data_prep/convert_items_to_parquet.py \
       --input-path rawdata/tianchi_fresh_comp_train_item_online.txt
     ```
3. `generate_features.py`
   - 功能：基于 Parquet 聚合用户/商品行为统计、构造窗口特征（默认 7 日）
   - 读取策略：DuckDB SQL 聚合，避免 Python 层循环；支持 `--threads`、`--duckdb-temp-directory`、`--duckdb-memory-limit`
   - P 过滤：若存在 `data/processed/raw_item/items.parquet` 或通过 `--item-parquet` 指定，会在聚合前 JOIN 过滤 `item_id ∈ P`
   - 输出：`data/processed/features-7d/user_features_YYYYMMDD.parquet`、`item_features_YYYYMMDD.parquet`
4. `prepare_dataset.py`
   - 功能：拼接目标日、构造次日购买标签并导出 train/eval/predict
   - 输入：`data/processed/features-*` 与 `data/processed/raw_behavior`
   - P 过滤：
     - 参数 `--item-subset-path` 指定 Parquet/CSV，默认为 `data/processed/raw_item/items.parquet`
     - P 过滤同时作用于样本候选与次日购买标签（label 的 item 同样限定于 P）
   - 输出：`data/processed/20141218_train.parquet`、`20141218_eval.parquet`、`20141218_predict.parquet`

### 注意事项
- 确保空 `user_geohash`、`item_geohash` 使用占位符 `_NA_`
- 控制内存：采用分片大小 5M 行左右并行处理
- 处理完成后在 `docs/experiments.csv` 记录数据范围与特征版本

### 最佳实践（商品子集 P）
1. 先将官方 `tianchi_fresh_comp_train_item_online.txt` 转为 Parquet：
   ```bash
   python scripts/data_prep/convert_items_to_parquet.py \
     --input-path /root/autodl-tmp/TorchEasyRec/tianchi_fresh_comp_train_item_online.txt
   ```
2. 生成窗口特征（自动按 P 过滤，并可调节资源参数）：
   ```bash
   python scripts/data_prep/generate_features.py \
      --input-dir data/processed/raw_behavior \
      --output-dir data/processed/features-7d \
      --start-day 20141125 --end-day 20141218 --window 7 \
      --item-parquet data/processed/raw_item/items.parquet \
      --threads 4 --duckdb-temp-directory .tmp/duckdb_temp --duckdb-memory-limit 60GiB
   ```
3. 准备数据集（未显式传参时将自动使用 `raw_item/items.parquet` 作为 P）：
   ```bash
   python scripts/data_prep/prepare_dataset.py \
      --features-dir data/processed/features-7d \
      --item-subset-path data/processed/raw_item/items.parquet \
      --output-prefix 20141218
   ```
