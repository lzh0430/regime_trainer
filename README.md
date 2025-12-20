# 加密货币市场状态分类器 - 自动化训练系统

这是一个完整的自动化训练系统，用于训练和维护加密货币市场状态分类模型。支持任意交易对，自动化数据获取、特征工程、HMM 标注和 LSTM 训练。

## 核心特性

- ✅ **支持任意交易对**：只需在配置文件中添加交易对即可
- ✅ **自动化数据获取**：每天自动从 Binance 获取最新数据
- ✅ **增量训练**：每天 2 次增量训练（HKT 8am & 8pm）
- ✅ **完整重训**：每周日完整重训（防止灾难性遗忘）
- ✅ **多时间框架分析**：5m, 15m, 1h 多时间框架技术指标
- ✅ **实时推理**：随时获取当前市场状态
- ✅ **模型版本管理**：每个交易对独立的模型

## 系统架构

```
regime_trainer/
├── config.py                 # 配置文件
├── data_fetcher.py          # Binance 数据获取
├── feature_engineering.py   # 技术指标计算
├── hmm_trainer.py          # HMM 状态标注
├── lstm_trainer.py         # LSTM 训练
├── training_pipeline.py    # 训练管道
├── scheduler.py            # 定时任务调度
├── realtime_predictor.py   # 实时推理
├── requirements.txt        # 依赖
└── README.md              # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.py`，设置你想要训练的交易对：

```python
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    # 添加更多...
]
```

### 3. 首次训练

首次使用时，需要对所有交易对进行完整重训：

```bash
python scheduler.py --init
```

这将：
- 获取最近 365 天的历史数据
- 计算技术指标
- 训练 HMM 模型进行状态标注
- 训练 LSTM 模型进行状态预测
- 保存模型到 `models/{symbol}/` 目录

### 4. 启动自动化调度

```bash
python scheduler.py
```

调度器将：
- **每天 8:00 和 20:00 HKT**：执行增量训练
- **每周日 3:00 HKT**：执行完整重训

### 5. 实时推理

```python
from realtime_predictor import RealtimeRegimePredictor
from config import TrainingConfig

# 创建预测器
predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)

# 获取当前市场状态
current = predictor.get_current_regime()
print(f"当前状态: {current['regime_name']}")
print(f"置信度: {current['confidence']:.2%}")
```

## 详细说明

### 增量训练 vs 完整重训

#### 增量训练（每天 2 次）
- **用途**：保持模型对最新市场的敏感性
- **数据**：最近 30 天
- **方式**：在现有模型基础上继续训练
- **优点**：快速、高效
- **缺点**：可能忘记旧的市场模式

#### 完整重训（每周 1 次）
- **用途**：防止"灾难性遗忘"，保持对各种市场环境的适应性
- **数据**：最近 365 天
- **方式**：从零开始重新训练 HMM 和 LSTM
- **优点**：记住所有重要的市场模式
- **缺点**：耗时较长

**为什么需要完整重训？**

加密市场虽然变化快，但某些市场模式（震荡、趋势、挤压）是周期性重复的。纯增量训练会让模型"忘记"几周前的市场模式。例如：

- 如果最近一个月都是单边上涨，增量训练会让模型过度适应趋势市场
- 当市场转为震荡时，模型可能无法正确识别（因为"忘记"了震荡的特征）
- 完整重训用 12 个月数据，确保模型记住牛市、熊市、震荡市的所有特征

### 训练任意交易对

要训练新的交易对，只需 3 步：

1. **在 `config.py` 中添加交易对**：
```python
SYMBOLS = [
    "BTCUSDT",
    "NEWTOKENUSDT",  # 新增
]
```

2. **运行训练**：
```bash
python training_pipeline.py
# 或者等待调度器自动训练
```

3. **开始使用**：
```python
predictor = RealtimeRegimePredictor("NEWTOKENUSDT", TrainingConfig)
```

**输出保证一致**：所有交易对使用相同的：
- 6 个市场状态定义
- 相同的技术指标
- 相同的模型架构
- 你的交易系统可以无缝切换交易对

### 6 个市场状态

| 状态 ID | 状态名称 | 特征 |
|--------|---------|------|
| 0 | Choppy_High_Vol | 高波动率震荡 |
| 1 | Strong_Trend | 强趋势 |
| 2 | Volatility_Spike | 波动率突然增加 |
| 3 | Weak_Trend | 弱趋势 |
| 4 | Range | 区间震荡 |
| 5 | Squeeze | 低波动率压缩 |

### 技术指标

系统自动计算 50+ 技术指标，包括：

**动量指标**
- RSI (7, 14)
- MACD
- Stochastic
- Williams %R
- ROC

**趋势指标**
- EMA (9, 21, 50, 200)
- SMA (20, 50)
- ADX

**波动率指标**
- ATR
- Bollinger Bands
- Keltner Channel

**成交量指标**
- OBV
- MFI
- Volume MA

**价格变化**
- Returns
- Log Returns
- High-Low Range
- Close Position

## 配置说明

### 数据配置

```python
# 时间框架
TIMEFRAMES = ["5m", "15m", "1h"]
PRIMARY_TIMEFRAME = "15m"  # 主要时间框架

# 数据长度
FULL_RETRAIN_DAYS = 365     # 完整重训使用 12 个月
INCREMENTAL_TRAIN_DAYS = 30  # 增量训练使用 30 天
```

### HMM 配置

```python
N_STATES = 6              # 市场状态数量
N_PCA_COMPONENTS = 4      # PCA 降维维度
```

### LSTM 配置

```python
SEQUENCE_LENGTH = 64      # 输入序列长度
LSTM_UNITS = [128, 64]   # LSTM 层配置
DROPOUT_RATE = 0.2       # Dropout 比率
EPOCHS = 50              # 训练轮数
BATCH_SIZE = 32          # 批次大小
```

### 调度配置

```python
# 增量训练时间（HKT）
INCREMENTAL_TRAIN_TIMES = ["08:00", "20:00"]

# 完整重训时间
FULL_RETRAIN_TIME = "03:00"  # 周日凌晨 3:00
FULL_RETRAIN_DAY = 6         # 0=周一, 6=周日
```

## 使用示例

### 单个交易对训练

```python
from training_pipeline import TrainingPipeline
from config import TrainingConfig

pipeline = TrainingPipeline(TrainingConfig)

# 完整重训
result = pipeline.full_retrain("BTCUSDT")
print(f"准确率: {result['accuracy']:.2%}")

# 增量训练
result = pipeline.incremental_train("BTCUSDT")
```

### 批量训练

```python
# 训练所有配置的交易对
results = pipeline.train_all_symbols(training_type='full')

for symbol, result in results.items():
    print(f"{symbol}: {result['accuracy']:.2%}")
```

### 实时推理

```python
from realtime_predictor import RealtimeRegimePredictor
from config import TrainingConfig

# 单个交易对
predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)

# 当前状态
current = predictor.get_current_regime()
print(f"状态: {current['regime_name']}")
print(f"置信度: {current['confidence']:.2%}")

# 概率分布
for regime, prob in current['probabilities'].items():
    print(f"{regime}: {prob:.2%}")

# 历史状态
history = predictor.get_regime_history(lookback_hours=24)
print(history)
```

### 多交易对跟踪

```python
from realtime_predictor import MultiSymbolRegimeTracker
from config import TrainingConfig

tracker = MultiSymbolRegimeTracker(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    config=TrainingConfig
)

# 所有交易对当前状态
all_regimes = tracker.get_all_regimes()

# 状态摘要
summary = tracker.get_regime_summary()
print(summary)
```

## 集成到交易系统

### 方式 1：定时查询

```python
from realtime_predictor import RealtimeRegimePredictor
from config import TrainingConfig

predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)

# 在你的交易循环中
while True:
    regime = predictor.get_current_regime()
    
    if regime['regime_name'] == 'Strong_Trend':
        # 执行趋势跟踪策略
        pass
    elif regime['regime_name'] == 'Range':
        # 执行区间交易策略
        pass
    
    time.sleep(300)  # 每 5 分钟查询一次
```

### 方式 2：批量预测

```python
# 获取最近 N 小时的状态变化
history = predictor.get_regime_history(lookback_hours=48)

# 分析状态持续时间
regime_changes = history['regime_name'].ne(history['regime_name'].shift()).cumsum()
regime_durations = history.groupby(regime_changes).size()
```

## 监控和维护

### 日志

系统会自动记录日志到：
- `training.log`：训练日志
- `scheduler.log`：调度日志

### 检查模型性能

```python
from lstm_trainer import LSTMRegimeClassifier
from config import TrainingConfig

# 加载模型
classifier = LSTMRegimeClassifier.load(
    TrainingConfig.get_model_path("BTCUSDT"),
    TrainingConfig.get_scaler_path("BTCUSDT")
)

# 评估（需要准备测试数据）
# results = classifier.evaluate(X_test, y_test)
# print(f"准确率: {results['accuracy']:.2%}")
```

### 模型文件位置

```
models/
├── BTCUSDT/
│   ├── lstm_model.h5      # LSTM 模型
│   ├── scaler.pkl        # 标准化器
│   └── hmm_model.pkl     # HMM 模型
├── ETHUSDT/
│   ├── lstm_model.h5
│   ├── scaler.pkl
│   └── hmm_model.pkl
...
```

## 性能优化建议

1. **GPU 加速**：安装 tensorflow-gpu 版本可大幅提升训练速度
2. **并行训练**：可以并行训练多个交易对
3. **特征缓存**：可以缓存特征工程结果以加快训练
4. **模型压缩**：可以使用模型量化减小模型大小

## 常见问题

### Q1: 增量训练会不会让模型忘记旧的市场模式？

A: 会的，这就是为什么我们设计了每周完整重训。增量训练保持对最新市场的敏感性，完整重训确保记住所有历史模式。

### Q2: 为什么不每天都完整重训？

A: 完整重训需要 1-2 小时（取决于硬件），而增量训练只需 5-10 分钟。每周完整重训在效率和性能之间取得平衡。

### Q3: 可以自定义市场状态数量吗？

A: 可以，修改 `config.py` 中的 `N_STATES`。但这会改变输出格式，你的交易系统也需要相应调整。

### Q4: 如何确保不同交易对的输出一致？

A: 所有交易对使用相同的：
- 状态数量（6个）
- 技术指标计算方法
- 模型架构
- 只是训练数据不同

### Q5: 训练失败怎么办？

A: 检查日志文件，常见问题：
- 数据不足（需要至少 2-3 个月历史数据）
- 网络问题（无法连接 Binance API）
- 内存不足（减少 `FULL_RETRAIN_DAYS`）

## 进阶功能

### 自定义技术指标

在 `feature_engineering.py` 的 `calculate_features` 方法中添加：

```python
# 添加自定义指标
features[f'{timeframe}_my_indicator'] = my_calculation(df)
```

### 调整训练参数

可以通过修改 `config.py` 来调整：
- 数据时间范围
- LSTM 结构
- 训练超参数
- 调度时间

### Hyperparameter Tuning

使用 Keras Tuner 进行超参数搜索：

```python
from tensorflow import keras
import keras_tuner

# 定义搜索空间
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.LSTM(
        hp.Int('units', 64, 256, step=64),
        ...
    ))
    return model

tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val))
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## License

MIT License
