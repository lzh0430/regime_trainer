# 加密货币市场状态分类器

自动化训练系统，用于训练和维护加密货币市场状态分类模型。支持任意交易对，自动数据获取、特征工程、HMM标注和LSTM训练。

## 核心功能

- 🎯 **预测下一根K线的market regime概率分布**
- 🔄 **自动化训练**：增量训练（每天2次）+ 完整重训（每周1次）
- 📊 **6种市场状态**：Strong_Trend, Weak_Trend, Range, Choppy_High_Vol, Volatility_Spike, Squeeze
- 🔌 **简单API接口**：供其他项目调用

**重要说明**：LSTM模型使用过去64根K线的特征序列，预测下一根K线的market regime。这是单步预测，不能直接预测多根K线。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 首次训练

```bash
python examples.py 1  # 训练单个交易对
# 或
python training_pipeline.py  # 训练所有交易对
```

### 3. 使用API预测

```python
from model_api import predict_regime

# 预测下一根15分钟K线的market regime
result = predict_regime("BTCUSDT", "15m")

print(f"最可能的状态: {result['most_likely_regime']['name']}")
print(f"概率: {result['most_likely_regime']['probability']:.2%}")
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

## 模型参数

### HMM 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `N_STATES` | 6 | 市场状态数量 |
| `N_PCA_COMPONENTS` | 5 | PCA降维后的特征数 |
| `PRIMARY_TIMEFRAME` | "15m" | 主时间框架 |

### LSTM 模型参数

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

# 完整重训
result = pipeline.full_retrain("BTCUSDT")

# 增量训练
result = pipeline.incremental_train("BTCUSDT")
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
└── README.md              # 本文档
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

A: 使用 `api.list_available_models()` 方法。

**Q: 预测结果中的概率分布是什么意思？**

A: 每个概率表示该状态在未来N根K线中出现的可能性。所有概率之和为1.0。

**Q: 可以预测其他时间框架吗？**

A: 目前只支持训练时使用的主时间框架（默认15m）。要支持其他时间框架，需要重新训练模型。

**Q: 如何更新模型？**

A: 使用 `training_pipeline.py` 进行增量训练或完整重训。训练完成后，API会自动使用新的模型。

## 详细文档

- **API详细文档**: 查看 [API_USAGE.md](API_USAGE.md)
- **快速开始**: 查看 [QUICK_START.md](QUICK_START.md)
- **示例代码**: 运行 `python examples.py`

## License

MIT License
