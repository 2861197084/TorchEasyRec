## 数据预处理说明

### 输入数据
- 用户行为明细：`rawdata/tianchi_fresh_comp_train_user_online_partA.txt` 与 `partB.txt`
- 商品子集列表：`rawdata/tianchi_fresh_comp_train_item_online.txt`

### 输出规范
- 预处理产物写入 `data/processed/`，命名 `yyyymmdd_split.parquet`
- 列字段：行为聚合特征（点击/加购/收藏/购买）、地理/品类信息、时间衍生字段、标签 `is_buy`
- 日志记录：关键数据量、异常计数写入 `logs/data_prep.log`

### 脚本规划
1. `convert_to_parquet.py`
   - 功能：分片读取 TXT，解析字段，追加 `event_day`、`event_hour`
   - 输出：按日期写入 Parquet，并生成校验摘要
2. `generate_features.py`
   - 功能：基于 Parquet 聚合用户/商品行为统计、构造 7/3/1 日窗口特征
   - 读取策略：逐个读取 parquet 分片增量累加统计，避免一次性加载整窗数据造成内存爆炸；若环境安装 `tqdm`，命令行会显示分片进度条
   - 输出：合并特征后的样本数据，保存至 `data/processed/yyyymmdd_features.parquet`
3. `generate_features_duckdb.py`
   - 功能：利用 DuckDB 在 SQL 层并行聚合滚动窗口特征，处理全量数据更高效
   - 依赖：`pip install duckdb pyarrow`
   - 优势：直接对 `event_day=*/*.parquet` 做窗口聚合，减少 Python 层循环，适合 10 亿+ 行数据；如安装 `tqdm` 会显示 SQL 与写文件的进度条，可用 `--no-progress` 关闭
   - 参数补充：`--duckdb-path` 支持传入目录或文件路径决定是否落盘；默认内存数据库
   - 输出：与 `generate_features.py` 一致，可写入相同目录
   - 项目约定：当前主流程使用 DuckDB 版本生成 7 日窗口特征，输出目录为 `data/processed/features_duckdb`，并已与早期 `data/processed/features` 中的 20141125~20141127 结果逐列对齐验证
3. `prepare_dataset.py`
   - 功能：拼接目标日标签，划分 train/eval，导出推理集
   - 输入：`data/processed/features_duckdb/` 中的 7 日窗口特征（后续可扩展多窗口）、评分日 `20141219` 的购买标签
   - 输出：`data/processed/20241218_train.parquet`、`data/processed/20241217_eval.parquet`、`data/processed/20241219_predict.parquet`
   - 设计要点：
     1. 训练集：使用 `20141125-20141217` 的特征，并用 12-19 的购买行为作为标签 `is_buy`
     2. 验证集：例如选取 `20141218` 作为验证特征，对应 `is_buy` 取 12-19 的购买
     3. 预测集：保留没有标签的 `20141218` 或整理出的候选集，用于上线预测
     4. 标签来源：对评分日 `tianchi_mobile_recommend_train_user` 过滤 `time` 在 `2014-12-19` 且 `behavior_type=4` 的记录，生成 `user_id-item_id` 的购买对
     5. 负样本：对于训练/验证集中未在 12-19 购买的行为组合，`is_buy=0`；可选地在窗口内做负采样/平衡

### 注意事项
- 确保空 `user_geohash`、`item_geohash` 使用占位符 `_NA_`
- 控制内存：采用分片大小 5M 行左右并行处理
- 处理完成后在 `docs/experiments.csv` 记录数据范围与特征版本
