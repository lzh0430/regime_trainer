# 加密货币市场状态分类器 - 自动化训练系统

## 📦 项目概述

这是一个完整的自动化训练系统，专门为加密货币永续合约市场设计，解决了你提出的所有问题：

### ✅ 解决的核心问题

1. **模型时效性问题** 
   - 每天 2 次增量训练（HKT 8am & 8pm）保持对最新市场的敏感性
   - 每周 1 次完整重训防止"灾难性遗忘"

2. **交易对通用性问题**
   - 支持任意交易对，只需在配置文件中添加
   - 所有交易对输出格式完全一致（6 个市场状态）

3. **自动化数据获取**
   - 自动从 Binance 获取最新 K线数据
   - 支持多时间框架（5m, 15m, 1h）

## 🎯 关键特性

- ✅ **完全自动化**：设置后无需人工干预
- ✅ **任意交易对**：BTCUSDT、ETHUSDT、SOLUSDT...
- ✅ **输出一致性**：所有交易对使用相同的 6 个状态定义
- ✅ **增量 + 完整训练**：平衡效率与性能
- ✅ **实时推理**：随时获取当前市场状态
- ✅ **模型版本管理**：每个交易对独立模型

## 📊 6 个市场状态

| 状态 ID | 状态名称 | 描述 |
|--------|---------|------|
| 0 | Choppy_High_Vol | 高波动率震荡 |
| 1 | Strong_Trend | 强趋势 |
| 2 | Volatility_Spike | 波动率突然增加 |
| 3 | Weak_Trend | 弱趋势 |
| 4 | Range | 区间震荡 |
| 5 | Squeeze | 低波动率压缩 |

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置交易对
编辑 `config.py`：
```python
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    # 添加你想交易的任意标的
]
```

### 3. 首次训练
```bash
python scheduler.py --init
```

### 4. 启动自动化调度
```bash
python scheduler.py
```

### 5. 实时推理
```python
from realtime_predictor import RealtimeRegimePredictor
from config import TrainingConfig

predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)
current = predictor.get_current_regime()

print(f"状态: {current['regime_name']}")
print(f"置信度: {current['confidence']:.2%}")
```

## 📚 文件说明

| 文件 | 功能 |
|------|------|
| `config.py` | 配置文件（交易对、训练参数等） |
| `data_fetcher.py` | Binance 数据获取 |
| `feature_engineering.py` | 技术指标计算（50+ 指标） |
| `hmm_trainer.py` | HMM 无监督状态标注 |
| `lstm_trainer.py` | LSTM 状态预测 |
| `training_pipeline.py` | 完整训练流程 |
| `scheduler.py` | 定时任务调度 |
| `realtime_predictor.py` | 实时推理 |
| `examples.py` | 使用示例 |
| `requirements.txt` | Python 依赖 |
| `README.md` | 详细文档 |

## 💡 关键概念解答

### Q1: 增量训练 vs 完整重训的区别？

**增量训练（每天 2 次）**
- 用最近 30 天数据
- 在现有模型基础上继续训练
- 速度快（5-10 分钟）
- 保持对最新市场的敏感性

**完整重训（每周 1 次）**
- 用最近 365 天数据
- 从零开始重新训练 HMM + LSTM
- 耗时较长（1-2 小时）
- 防止忘记旧的市场模式

**为什么需要完整重训？**

加密市场有周期性：
- 如果最近一个月都是趋势市场，增量训练会让模型过度适应趋势
- 当市场转为震荡时，模型可能无法识别（因为"忘记"了震荡的特征）
- 完整重训用 12 个月数据，确保记住牛市、熊市、震荡市的所有特征

### Q2: 如何训练任意交易对？

只需 3 步：

1. **在 `config.py` 添加交易对**
```python
SYMBOLS = ["BTCUSDT", "NEWTOKENUSDT"]
```

2. **运行训练**
```bash
python examples.py
# 选择示例 1 或 2
```

3. **开始使用**
```python
predictor = RealtimeRegimePredictor("NEWTOKENUSDT", TrainingConfig)
```

**输出保证一致**：
- 所有交易对都输出相同的 6 个状态
- 使用相同的技术指标
- 相同的模型架构
- 你的交易系统可以无缝切换交易对

### Q3: 训练时间安排

```
每天：
├── 08:00 HKT: 增量训练
└── 20:00 HKT: 增量训练

每周日：
└── 03:00 HKT: 完整重训
```

可以在 `config.py` 中自定义时间。

## 🔧 使用示例

### 示例 1: 单个交易对训练
```bash
python examples.py
# 选择 1
```

### 示例 2: 批量训练
```bash
python examples.py
# 选择 2
```

### 示例 3: 实时预测
```python
from realtime_predictor import RealtimeRegimePredictor
from config import TrainingConfig

predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)
current = predictor.get_current_regime()

print(f"状态: {current['regime_name']}")
print(f"置信度: {current['confidence']:.2%}")

# 查看所有状态的概率
for regime, prob in current['probabilities'].items():
    print(f"{regime}: {prob:.2%}")
```

### 示例 4: 集成到交易系统
```python
predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)

while True:
    regime = predictor.get_current_regime()
    
    if regime['regime_name'] == 'Strong_Trend':
        # 执行趋势跟踪策略
        execute_trend_strategy()
    elif regime['regime_name'] == 'Range':
        # 执行区间交易策略
        execute_range_strategy()
    
    time.sleep(300)  # 每 5 分钟查询一次
```

## 📈 技术指标（50+ 指标）

系统自动计算以下技术指标：

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

## 🎓 系统架构

```
1. 数据获取 (data_fetcher.py)
   ↓
2. 特征工程 (feature_engineering.py)
   ↓
3. HMM 标注 (hmm_trainer.py)
   ↓
4. LSTM 训练 (lstm_trainer.py)
   ↓
5. 模型保存
   ↓
6. 实时推理 (realtime_predictor.py)
```

## 🔒 安全性

- API Key 可选（读取公开数据不需要）
- 使用 `.env` 文件管理敏感信息
- 模型文件本地存储

## 📝 注意事项

1. **首次运行**：需要完整重训，大约 1-2 小时（取决于硬件）
2. **数据要求**：建议至少有 2-3 个月历史数据
3. **GPU 推荐**：使用 GPU 可大幅提升训练速度
4. **网络要求**：需要稳定的网络连接 Binance API

## 🐛 常见问题

**Q: 训练失败怎么办？**
A: 查看日志文件 `training.log`，常见问题：
- 数据不足（需要至少 2-3 个月）
- 网络问题（无法连接 Binance）
- 内存不足（减少 `FULL_RETRAIN_DAYS`）

**Q: 如何更新配置？**
A: 编辑 `config.py`，重启调度器即可生效

**Q: 如何查看训练进度？**
A: 查看日志文件 `training.log` 和 `scheduler.log`

## 📞 支持

- 详细文档：`README.md`
- 示例代码：`examples.py`
- 配置说明：`config.py`

## 📄 License

MIT License

---

**祝你交易顺利！🚀**
