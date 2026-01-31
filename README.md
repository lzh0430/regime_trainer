# 加密货币市场状态分类器

自动化训练系统，用于训练和维护加密货币市场状态分类模型。支持任意交易对，自动数据获取、特征工程、HMM标注和LSTM训练。

## 🚀 快速使用指南

### 方式1：启动 Flask API 服务器（推荐）

**同时运行 API 服务器和自动训练调度器**

```bash
python run_server.py
```

这将自动：
- ✅ 启动 HTTP API 服务器（端口 5000）
- ✅ 在后台启动训练调度器（自动执行增量训练）
  - 15m 模型：每 3 小时增量训练一次
  - 5m 模型：每 60 分钟增量训练一次

**API 端点示例：**
```bash
# 健康检查
curl http://localhost:5000/api/health

# 预测下一根K线
curl http://localhost:5000/api/predict/BTCUSDT?timeframe=15m

# 多步预测（推荐）
curl http://localhost:5000/api/predict_regimes/BTCUSDT?timeframe=15m

# 获取历史regime序列（新增）
curl "http://localhost:5000/api/history/BTCUSDT?timeframe=15m&lookback_hours=24"
curl "http://localhost:5000/api/history/BTCUSDT?timeframe=15m&start_date=2024-01-01&end_date=2024-01-31"
```

### 方式2：启动 React UI（可选）

**启动 React UI 前端界面**

在另一个终端窗口中运行：

```bash
cd ui
npm install  # 首次运行需要安装依赖
npm run dev
```

这将启动 React UI 开发服务器（端口 3000）

**访问地址：**
- React UI: http://localhost:3000
- Flask API: http://localhost:5000
- API Docs: http://localhost:5000/api/docs

### 方式3：仅运行训练调度器

**只启动自动增量训练，不提供 HTTP API**

```bash
python scheduler.py
```

这将：
- ✅ 在后台自动执行增量训练
- ✅ 15m 模型：每 3 小时训练一次
- ✅ 5m 模型：每 60 分钟训练一次

### 方式4：作为 Python 库使用

**在其他 Python 程序中直接调用**

```python
from model_api import ModelAPI, predict_regime

# 方式1：使用便捷函数
result = predict_regime("BTCUSDT", "15m")
print(result['most_likely_regime']['name'])

# 方式2：使用 ModelAPI 类
api = ModelAPI()
result = api.predict_next_regime("BTCUSDT", primary_timeframe="15m")

# 多步预测（推荐）
result = api.predict_regimes("BTCUSDT", primary_timeframe="15m")
print(result['predictions']['t+1']['most_likely'])
```

### 方式5：运行 HTTP 服务器（通过 model_api.py）

**使用 model_api.py 启动 HTTP 服务器和调度器**

```bash
python model_api.py --server
```

功能与 `run_server.py` 相同。

---

## 核心功能

- 🎯 **预测下一根K线的market regime概率分布**
- 🔄 **自动化训练**：增量训练（自动调度）+ 完整重训（手动执行）
- 📊 **6种市场状态**：Strong_Trend, Weak_Trend, Range, Choppy_High_Vol, Volatility_Spike, Squeeze
- 🔌 **简单API接口**：供其他项目调用（HTTP REST API 或 Python 库）
- ⚡ **多时间框架支持**：独立的 5m 和 15m 模型，可并行预测

**重要说明**：LSTM模型使用过去N根K线的特征序列，预测下一根K线的market regime。这是单步预测，不能直接预测多根K线。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 首次训练

```bash
# 15m 模型（默认）
python examples.py 1   # 训练单个交易对 15m 模型
python examples.py 2   # 批量训练所有交易对 15m 模型

# 5m 模型（快速决策）
python examples.py 7   # 训练单个交易对 5m 模型
python examples.py 13  # 批量训练所有交易对 5m 模型

# 或直接运行训练管道
python training_pipeline.py  # 训练所有交易对
```

### 3. 使用API预测

```python
from model_api import predict_regime

# 预测下一根15分钟K线的market regime
result = predict_regime("BTCUSDT", "15m")

print(f"最可能的状态: {result['most_likely_regime']['name']}")
print(f"概率: {result['most_likely_regime']['probability']:.2%}")

# 预测下一根5分钟K线的market regime（需要先训练5m模型）
result_5m = predict_regime("BTCUSDT", "5m")
```

## API 使用指南

### 基本用法

#### Request 示例

```python
from model_api import ModelAPI

# 初始化API
api = ModelAPI()

# 预测下一根15分钟K线的market regime
result = api.predict_next_regime(
    symbol="BTCUSDT",
    timeframe="15m"  # 必须与训练时的主时间框架一致
)
```

#### Response 示例

```python
{
    'symbol': 'BTCUSDT',
    'timeframe': '15m',
    'timestamp': datetime.datetime(2024, 1, 15, 10, 30, 0),
    'regime_probabilities': {
        'Strong_Trend': 0.35,
        'Weak_Trend': 0.25,
        'Range': 0.20,
        'Choppy_High_Vol': 0.10,
        'Volatility_Spike': 0.05,
        'Squeeze': 0.05
    },
    'most_likely_regime': {
        'id': 1,
        'name': 'Strong_Trend',
        'probability': 0.35
    },
    'confidence': 0.35,
    'is_uncertain': False,
    'model_info': {
        'primary_timeframe': '15m',
        'n_states': 6,
        'sequence_length': 64,  # 使用的历史K线数量
        'regime_mapping': {
            0: 'Choppy_High_Vol',
            1: 'Strong_Trend',
            2: 'Volatility_Spike',
            3: 'Weak_Trend',
            4: 'Range',
            5: 'Squeeze'
        }
    }
}
```

### 更多API方法

#### 1. 获取特定状态的概率

```python
from model_api import get_regime_probability

# Request
prob = get_regime_probability("BTCUSDT", "Strong_Trend")

# Response
# 0.35  (float, 0.0-1.0)
```

#### 2. 获取模型元数据

```python
api = ModelAPI()

# Request
metadata = api.get_model_metadata("BTCUSDT")

# Response
{
    'symbol': 'BTCUSDT',
    'primary_timeframe': '15m',
    'n_states': 6,
    'regime_mapping': {0: 'Choppy_High_Vol', 1: 'Strong_Trend', ...},
    'regime_names': ['Choppy_High_Vol', 'Strong_Trend', ...],
    'model_paths': {
        'lstm': 'models/BTCUSDT/lstm_model.h5',
        'hmm': 'models/BTCUSDT/hmm_model.pkl',
        'scaler': 'models/BTCUSDT/scaler.pkl'
    },
    'training_info': {
        'sequence_length': 64,
        'feature_count': 150
    }
}
```

#### 3. 列出可用模型

```python
api = ModelAPI()

# Request
available = api.list_available_models()

# Response
# ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
```

#### 4. 批量预测

```python
api = ModelAPI()

# Request
results = api.batch_predict(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframe="15m"
)

# Response
{
    'BTCUSDT': {
        'symbol': 'BTCUSDT',
        'most_likely_regime': {'name': 'Strong_Trend', ...},
        ...
    },
    'ETHUSDT': {
        'symbol': 'ETHUSDT',
        'most_likely_regime': {'name': 'Range', ...},
        ...
    }
}
```

#### 5. 多时间框架并行预测

```python
api = ModelAPI()

# Request - 同时获取 5m 和 15m 的预测结果
results = api.predict_multi_timeframe_regimes(
    symbol="BTCUSDT",
    timeframes=["5m", "15m"]
)

# Response
{
    '5m': {
        'symbol': 'BTCUSDT',
        'regime_name': 'Strong_Trend',
        'confidence': 0.42,
        'probabilities': {...},
        ...
    },
    '15m': {
        'symbol': 'BTCUSDT',
        'regime_name': 'Weak_Trend',
        'confidence': 0.38,
        'probabilities': {...},
        ...
    }
}
```

这在以下场景特别有用：
- **短期决策**：5m 模型捕捉快速变化，适合 3-5 分钟决策周期
- **趋势确认**：对比 5m 和 15m 状态是否一致，判断趋势强度
- **入场时机**：15m 确认大方向，5m 寻找精确入场点

#### 6. 获取历史 Market Regime 序列（新增）

**用于回测场景**：获取历史上的 market regime 序列，支持按回看小时数或日期范围查询。

```python
from datetime import datetime, timedelta

api = ModelAPI()

# 方式1: 按回看小时数查询（从当前时间往前回看）
result = api.get_regime_history(
    symbol="BTCUSDT",
    lookback_hours=24,  # 回看24小时
    primary_timeframe="15m"
)

# 方式2: 按日期范围查询（适合回测）
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # 最近30天

result = api.get_regime_history(
    symbol="BTCUSDT",
    start_date=start_date,
    end_date=end_date,
    primary_timeframe="15m"
)

# Response
{
    'symbol': 'BTCUSDT',
    'timeframe': '15m',
    'lookback_hours': 24,  # 或 None（如果使用日期范围）
    'start_date': None,     # 或 ISO 格式日期字符串
    'end_date': None,       # 或 ISO 格式日期字符串
    'timestamp': datetime(...),
    'count': 96,            # 记录数量
    'history': [
        {
            'timestamp': '2024-01-01T11:45:00',
            'regime_id': 0,
            'regime_name': 'Range',
            'confidence': 0.85,
            'is_uncertain': False,
            'original_regime': 'Range'
        },
        ...
    ]
}
```

**HTTP API 端点**：
```bash
# 按回看小时数
GET /api/history/BTCUSDT?timeframe=15m&lookback_hours=24

# 按日期范围
GET /api/history/BTCUSDT?timeframe=15m&start_date=2024-01-01&end_date=2024-01-31
```

**特性**：
- ✅ 支持按回看小时数或日期范围查询
- ✅ 优先从SQLite缓存读取历史K线数据
- ✅ 批量预测优化，适合长时间范围查询
- ✅ 最大支持30天回看或1年日期范围
- ✅ 适合回测场景，可获取任意历史时间段的regime序列

## 模型参数

### 多时间框架配置

系统支持独立的 5m 和 15m 模型，每个时间框架有优化的参数：

| 参数 | 5m 模型 | 15m 模型 | 说明 |
|------|---------|----------|------|
| `SEQUENCE_LENGTH` | 48 | 64 | 输入序列长度（K线数量） |
| `timeframes` | 1m, 5m, 15m | 5m, 15m, 1h | 特征使用的时间框架 |
| `LSTM_UNITS` | [96, 48] | [128, 64] | LSTM层单元数 |
| `覆盖时间` | 4小时 | 16小时 | 输入数据覆盖的时间范围 |

### HMM 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `N_STATES` | 6 | 市场状态数量 |
| `N_PCA_COMPONENTS` | 5 | PCA降维后的特征数 |

### LSTM 模型参数（默认值）

| 参数 | 值 | 说明 |
|------|-----|------|
| `SEQUENCE_LENGTH` | 64 | 输入序列长度（K线数量） |
| `LSTM_UNITS` | [128, 64] | LSTM层单元数 |
| `DENSE_UNITS` | [64] | 全连接层单元数 |
| `DROPOUT_RATE` | 0.35 | Dropout比率 |
| `L2_LAMBDA` | 1.5e-3 | L2正则化强度 |
| `LEARNING_RATE` | 1e-3 | Adam优化器学习率 |
| `EPOCHS` | 50 | 训练轮数 |
| `BATCH_SIZE` | 32 | 批次大小 |
| `USE_BATCH_NORM` | True | 是否使用BatchNormalization |
| `USE_CLASS_WEIGHT` | True | 是否使用类权重（处理类别不平衡） |

### 数据划分参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `TRAIN_RATIO` | 0.65 | 训练集比例 |
| `VAL_RATIO` | 0.20 | 验证集比例 |
| `TEST_RATIO` | 0.15 | 测试集比例 |
| `FULL_RETRAIN_DAYS` | 730 | 完整重训数据长度（天） |
| `INCREMENTAL_TRAIN_DAYS` | 30 | 增量训练数据长度（天） |

### 训练回调参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `EARLY_STOPPING_PATIENCE` | 8 | 早停耐心值（epoch数） |
| `LR_REDUCE_PATIENCE` | 5 | 学习率衰减耐心值 |
| `CONFIDENCE_THRESHOLD` | 0.4 | 置信度拒绝阈值 |

## 市场状态说明

系统识别6种market regime状态：

| 状态名称 | 特征描述 |
|---------|---------|
| **Strong_Trend** | 强趋势：高ADX，明显的趋势方向 |
| **Weak_Trend** | 弱趋势：中等ADX，有一定趋势 |
| **Range** | 区间震荡：低ADX，中等波动率 |
| **Choppy_High_Vol** | 高波动无方向：低ADX，高波动率 |
| **Volatility_Spike** | 波动率突增：极高波动率 |
| **Squeeze** | 低波动蓄势：极低波动率，低ADX |

## 自动化训练

### 启动调度器

```bash
python scheduler.py
```

调度器将自动执行：
- **每天 8:00 和 20:00 HKT**：增量训练（使用最近30天数据）
- **每周日 3:00 HKT**：完整重训（使用最近730天数据）

### 手动训练

```python
from training_pipeline import TrainingPipeline
from config import TrainingConfig

pipeline = TrainingPipeline(TrainingConfig)

# 15m 模型训练（默认）
result = pipeline.full_retrain("BTCUSDT")                    # 完整重训
result = pipeline.incremental_train("BTCUSDT")               # 增量训练

# 5m 模型训练
result = pipeline.full_retrain("BTCUSDT", primary_timeframe="5m")      # 完整重训
result = pipeline.incremental_train("BTCUSDT", primary_timeframe="5m") # 增量训练

# 训练所有时间框架的所有交易对
results = pipeline.train_all_multi_timeframe()
```

## 项目结构

```
regime_trainer/
├── config.py                 # 配置文件
├── data_fetcher.py           # Binance数据获取
├── feature_engineering.py    # 技术指标计算
├── hmm_trainer.py           # HMM状态标注
├── lstm_trainer.py          # LSTM训练
├── training_pipeline.py     # 训练管道
├── scheduler.py             # 定时任务调度
├── realtime_predictor.py   # 实时推理
├── model_api.py            # API接口（供其他项目使用）
├── examples.py             # 使用示例
├── test_api.py            # API测试脚本
├── API_USAGE.md           # 详细API文档
├── README.md              # 本文档
└── models/                  # 模型存储目录
    └── BTCUSDT/
        ├── 5m/              # 5m 时间框架模型
        │   ├── lstm_model.h5
        │   ├── hmm_model.pkl
        │   └── scaler.pkl
        └── 15m/             # 15m 时间框架模型
            ├── lstm_model.h5
            ├── hmm_model.pkl
            └── scaler.pkl
```

## 配置交易对

编辑 `config.py`：

```python
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    # 添加更多交易对...
]
```

## 常见问题

**Q: 如何知道哪些交易对有可用的模型？**

A: 使用 `api.list_available_models()` 方法，返回按时间框架分组的可用模型列表。

**Q: 预测结果中的概率分布是什么意思？**

A: 每个概率表示该状态在未来N根K线中出现的可能性。所有概率之和为1.0。

**Q: 可以预测其他时间框架吗？**

A: 系统支持 5m 和 15m 两种时间框架。需要分别训练对应的模型：
```python
# 训练 5m 模型
python examples.py 7

# 使用 5m 模型预测
result = predict_regime("BTCUSDT", "5m")
```

**Q: 5m 和 15m 模型有什么区别？应该用哪个？**

A: 
- **5m 模型**：更敏感，能快速捕捉市场变化，适合 3-5 分钟决策周期
- **15m 模型**：更稳定，过滤短期噪音，适合中期趋势判断

建议同时使用两个模型，用 `api.predict_multi_timeframe_regimes()` 并行预测，对比两者是否一致来判断信号强度。

**Q: 如何更新模型？**

A: 使用 `training_pipeline.py` 进行增量训练或完整重训。训练完成后，API会自动使用新的模型。

**Q: 如何同时获取 5m 和 15m 的预测？**

A: 使用多时间框架预测接口：
```python
from model_api import ModelAPI
api = ModelAPI()
results = api.predict_multi_timeframe_regimes("BTCUSDT", ["5m", "15m"])
```

## 示例脚本

运行 `python examples.py` 显示交互式菜单，或直接指定示例编号：

### 15m 时间框架（默认）

| 编号 | 功能 | 命令 |
|------|------|------|
| 1 | 训练单个交易对 (BTCUSDT) | `python examples.py 1` |
| 2 | 批量训练多个交易对 | `python examples.py 2` |
| 3 | 实时市场状态预测 | `python examples.py 3` |
| 4 | 查看历史市场状态变化 | `python examples.py 4` |
| 5 | 多交易对市场状态跟踪 | `python examples.py 5` |
| 6 | 增量训练 | `python examples.py 6` |

### 5m 时间框架（快速决策）

| 编号 | 功能 | 命令 |
|------|------|------|
| 7 | 训练单个交易对 5m 模型 | `python examples.py 7` |
| 8 | 5m 实时市场状态预测 | `python examples.py 8` |
| 9 | 5m 增量训练 | `python examples.py 9` |
| 10 | 5m 历史市场状态变化 | `python examples.py 10` |
| 12 | 5m 多交易对市场状态跟踪 | `python examples.py 12` |
| 13 | 批量训练多个交易对 5m 模型 | `python examples.py 13` |

### 多时间框架

| 编号 | 功能 | 命令 |
|------|------|------|
| 11 | 多时间框架并行预测 (5m + 15m) | `python examples.py 11` |

## 详细文档

- **API详细文档**: 查看 [API_USAGE.md](API_USAGE.md)
- **快速开始**: 查看 [QUICK_START.md](QUICK_START.md)
- **示例代码**: 运行 `python examples.py`

## License

MIT License
