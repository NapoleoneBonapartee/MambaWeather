# 模型框架分析

本文档从 `run_model.py` 出发，分析了项目模型运行时的关键调用路径、模块依赖和整体框架设计。

## 1. 入口文件

- `run_model.py`
  - 作用：解析命令行参数，并调用 `libcity.pipeline.run_model`
  - 直接引用：
    - `from libcity.pipeline import run_model`
    - `from libcity.utils import str2bool, add_general_args`

## 2. 核心流程入口

- `libcity/pipeline/__init__.py`
  - 导出：`run_model`, `hyper_parameter`, `objective_function`
- `libcity/pipeline/pipeline.py`
  - 关键函数：`run_model(task, model_name, dataset_name, config_file, saved_model, train, other_args)`
  - 主要责任：
    - 加载配置
    - 生成 `exp_id`
    - 处理 resume 逻辑
    - 创建 logger
    - 加载数据集
    - 创建模型与执行器
    - 训练 / 保存 / 加载 / 评估
    - 复制日志并生成训练总结

## 3. 配置解析

- `libcity/config/config_parser.py`
  - 入口：`ConfigParser(task, model, dataset, config_file, saved_model, train, other_args, hyper_config_dict)`
  - 配置加载顺序：
    1. 命令行参数 / 外部参数
    2. 指定的 `./{config_file}.json`
    3. 默认配置文件：
       - `./libcity/config/task_config.json`
       - `./libcity/config/model/{task}/{model}.json`
       - `./libcity/config/data/{dataset_class}.json`
       - `./libcity/config/executor/{executor}.json`
       - `./libcity/config/evaluator/{evaluator}.json`
    4. 数据集元信息：`./raw_data/{dataset}/config.json`
  - 最终作用：
    - 确定 `dataset_class`, `executor`, `evaluator`
    - 填充任务特定参数，如 `traj_encoder`, `eta_encoder`
    - 初始化 `device`

## 4. 通用工具模块

- `libcity/utils/__init__.py`
  - 导出统一工具函数
- `libcity/utils/utils.py`
  - 关键函数：
    - `get_model(config, data_feature)`
    - `get_executor(config, model, data_feature)`
    - `get_evaluator(config)`
    - `get_logger(config, name=None)`
    - `set_random_seed(seed)`
    - `ensure_dir(dir_path)`
    - `preprocess_data(data, config)`
- `libcity/utils/argument_list.py`
  - 参数解析工具：`str2bool`, `add_general_args`, `add_hyper_args`

## 5. 数据加载路径

- `libcity/data/__init__.py`
  - 当前导出 `get_dataset`，映射到 `libcity.data.utils_optimized.get_dataset`
- `libcity/data/utils_optimized.py`
  - 动态加载 dataset 类：
    - `libcity.data.dataset`
    - `libcity.data.dataset.dataset_subclass`
  - 负责提供 `train_dataloader`, `eval_dataloader`, `test_dataloader`
- `libcity/data/list_dataset.py`
  - `ListDataset`
- `libcity/data/batch.py`
  - `Batch`, `BatchPAD`

## 6. 模型动态加载逻辑

- `libcity/utils/utils.py` 中的 `get_model`
  - 根据 `config['task']` 动态加载模型模块：
    - `traj_loc_pred` -> `libcity.model.trajectory_loc_prediction`
    - `traffic_state_pred` -> `libcity.model.traffic_flow_prediction`
      - fallback 加载：
        - `traffic_speed_prediction`
        - `traffic_demand_prediction`
        - `traffic_od_prediction`
        - `traffic_accident_prediction`
    - `map_matching` -> `libcity.model.map_matching`
    - `road_representation` -> `libcity.model.road_representation`
    - `eta` -> `libcity.model.eta`

## 7. 执行器动态加载逻辑

- `libcity/utils/utils.py` 中的 `get_executor`
  - 动态加载方式：`getattr(importlib.import_module('libcity.executor'), config['executor'])`
- `libcity/executor/__init__.py`
  - 导出执行器类，包括：
    - `TrafficStateExecutor`, `TrafficStateExecutorOptimized`, `DCRNNExecutor`, `MTGNNExecutor`, `GeoSANExecutor`,
      `TrajLocPredExecutor`, `ASTGNNExecutor`, `PDFormerExecutor`, `STSSLExecutor`, `TimeMixerExecutor`,
      `MegaCRNExecutor`, `TrafformerExecutor`, `SSTBANExecutor`, `STTSNetExecutor`, `FOGSExecutor`, `GEMLExecutor`,
      `ChebConvExecutor`, `LINEExecutor`, `ETAExecutor`, `GensimExecutor`, `TESTAMExecutor`, `HyperTuning`
- 基类：
  - `libcity/executor/abstract_executor.py`：定义接口 `train`, `evaluate`, `load_model`, `save_model`
  - `libcity/executor/abstract_tradition_executor.py`：实现传统模型的评估流程

## 8. `run_model` 的运行流程

1. `run_model.py` 解析参数后调用 `libcity.pipeline.run_model(...)`
2. `run_model` 加载并合并配置
3. 处理 `resume` 逻辑（可自动或指定 epoch 恢复训练）
4. 创建 logger
5. 设置随机种子
6. 加载数据集，并获取 `train_data`, `valid_data`, `test_data`, `data_feature`
7. 创建模型实例
8. 创建 executor 实例
9. 执行训练或加载模型
10. 执行评估
11. 复制日志并生成训练总结

## 9. 关键文件依赖关系

- `run_model.py`
  - -> `libcity.pipeline.run_model`
- `libcity/pipeline/pipeline.py`
  - -> `libcity.config.ConfigParser`
  - -> `libcity.utils.get_logger`
  - -> `libcity.utils.set_random_seed`
  - -> `libcity.data.get_dataset`
  - -> `dataset.get_data`
  - -> `dataset.get_data_feature`
  - -> `libcity.utils.get_model`
  - -> `libcity.utils.get_executor`
  - -> `executor.train`
  - -> `executor.save_model`
  - -> `executor.load_model`
  - -> `executor.evaluate`
- `libcity/config/config_parser.py`
  - -> `./libcity/config/task_config.json`
  - -> `./libcity/config/model/{task}/{model}.json`
  - -> `./libcity/config/data/{dataset_class}.json`
  - -> `./libcity/config/executor/{executor}.json`
  - -> `./libcity/config/evaluator/{evaluator}.json`
  - -> `./raw_data/{dataset}/config.json`
- `libcity/utils/utils.py`
  - -> `libcity.executor`
  - -> `libcity.model`
  - -> `libcity.evaluator`
- `libcity/data/utils_optimized.py`
  - -> `libcity.data.dataset`
  - -> `libcity.data.dataset.dataset_subclass`
  - -> `libcity.data.list_dataset`
  - -> `libcity.data.batch`

## 10. 框架设计结论

该项目采用“配置驱动 + 动态加载”设计：

- 用户只需指定 `task`、`model`、`dataset`，其余配置通过 JSON 自动补全。
- `ConfigParser` 负责合并用户参数与默认配置。
- `get_model` / `get_executor` 通过 `importlib` 动态加载类；`task` 决定模型模块的类别。
- 数据集、模型、执行器、评估器彼此解耦，便于扩展。

该架构适合快速增加新模型或任务，只需要新增配置项和相应类实现。
