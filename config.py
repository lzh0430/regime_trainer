"""
配置文件 - 市场趋势分类器自动化训练系统
"""
import os
import sys
import logging
from datetime import datetime, timedelta

class TrainingConfig:
    """训练配置"""
    
    # ============ 交易对配置 ============
    SYMBOLS = [
        # "BTCUSDT",
        # "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        # 添加更多交易对
    ]
    
    # ============ 数据配置 ============
    TIMEFRAMES = ["5m", "15m", "1h"]
    PRIMARY_TIMEFRAME = "15m"  # 主时间框架
    
    # 完整重训数据长度（天）
    FULL_RETRAIN_DAYS = 730  # 2年（2024-2025）- 增加数据量以改善类别不平衡
    
    # 增量训练数据长度（天）
    INCREMENTAL_TRAIN_DAYS = 30  # 最近30天
    
    # ============ HMM 配置 ============
    N_STATES = 6  # 市场状态数量（初始值，可被动态调整）
    N_PCA_COMPONENTS = 5  # PCA 降维后的特征数
    
    # ============ 动态状态数量调整 ============
    # 当验证/测试集状态分布异常时，自动尝试调整状态数量
    AUTO_ADJUST_N_STATES = True  # 是否启用动态调整
    N_STATES_MIN = 4  # 最小状态数量
    N_STATES_MAX = 8  # 最大状态数量
    
    # 触发调整的条件：
    # 1. 验证集中完全缺失的状态数量 >= 此阈值
    # 2. 或验证集中占比过低的状态数量 >= 此阈值
    MAX_MISSING_STATES_ALLOWED = 1  # 允许最多缺失 1 个状态，超过则触发调整
    MAX_LOW_RATIO_STATES_ALLOWED = 2  # 允许最多 2 个状态占比过低
    
    # 调整策略：
    # - "decrease_first": 优先减少状态数量（适合波动率低的币种）
    # - "bic_optimal": 使用 BIC 选择最优数量（更准确但更慢）
    N_STATES_ADJUST_STRATEGY = "decrease_first"
    
    # 状态名称（仅作为回退/参考）
    # 注意：实际的状态映射由 HMM 模型在训练时自动生成并保存
    # HMM 是无监督聚类模型，状态编号是任意的
    # auto_map_regimes() 方法会根据特征统计（ADX、波动率等）自动确定每个状态的语义名称
    # 这个配置仅在加载旧版本模型（没有保存映射）时作为回退使用
    # 
    # 当 AUTO_ADJUST_N_STATES=True 时，实际状态数量可能少于 6 个
    # 状态名称分配是完全数据驱动的：
    # - 减少 n_states 后，HMM 会重新聚类
    # - auto_map_regimes() 根据每个新聚类的特征（ADX、波动率等）分配最匹配的语义名称
    # - 不存在"优先级"：某个状态在验证集中缺失说明该市场条件在当前数据中不存在
    REGIME_NAMES = {
        0: "Choppy_High_Vol",   # 高波动无方向
        1: "Strong_Trend",      # 强趋势
        2: "Volatility_Spike",  # 波动率突增
        3: "Weak_Trend",        # 弱趋势
        4: "Range",             # 区间震荡
        5: "Squeeze"            # 低波动蓄势
    }
    
    # ============ 置信度与稳健性配置 ============
    # 置信度拒绝阈值：当最高概率低于此值时，输出 "Uncertain" 状态
    CONFIDENCE_THRESHOLD = 0.4
    
    # 映射稳定性检查：新旧映射差异超过此阈值时触发警告
    MAPPING_DIFF_THRESHOLD = 2  # 允许最多 2 个状态映射不同
    
    # BIC 验证：是否在训练时验证状态数量
    VALIDATE_N_STATES = False  # 设为 True 启用 BIC 验证（会增加训练时间）
    BIC_TEST_N_STATES = [4, 5, 6, 7, 8]  # 测试的状态数量范围
    
    # 转移矩阵监控：异常频繁切换的阈值（每小时切换次数）
    REGIME_SWITCH_WARNING_THRESHOLD = 10
    
    # ============ 状态分布检查配置 ============
    # 检测验证集/测试集是否缺失某些状态
    # 如果某状态在验证集中完全缺失，LSTM 的 early stopping 将无法评估该状态
    MIN_SAMPLES_PER_STATE = 10  # 每个状态在验证集中的最小样本数
    MIN_RATIO_PER_STATE = 0.01  # 每个状态在验证集中的最小占比 (1%)
    
    # ============ Regime 自动映射绝对阈值护栏 ============
    # 这些绝对阈值与相对阈值（基于中位数倍数）结合使用，
    # 防止在极端市场条件下（如所有状态都低波动或都高波动）出现误标记
    #
    # 波动率阈值（基于 hl_pct，即 (high-low)/close 的百分比）
    # - Volatility_Spike: 波动率必须 > 相对阈值 且 > MIN_VOL_FOR_SPIKE
    # - Squeeze: 波动率必须 < 相对阈值 且 < MAX_VOL_FOR_SQUEEZE
    REGIME_MIN_VOL_FOR_SPIKE = 0.02  # 波动率至少 2% 才能标记为 Volatility_Spike
    REGIME_MAX_VOL_FOR_SQUEEZE = 0.01  # 波动率最多 1% 才能标记为 Squeeze
    
    # ADX 阈值
    # - Strong_Trend: ADX 必须 > 相对阈值 且 > MIN_ADX_FOR_STRONG_TREND
    # - Squeeze: ADX 必须 < 相对阈值 且 < MAX_ADX_FOR_SQUEEZE
    REGIME_MIN_ADX_FOR_STRONG_TREND = 30  # ADX 至少 30 才能标记为 Strong_Trend
    REGIME_MAX_ADX_FOR_SQUEEZE = 20  # ADX 最多 20 才能标记为 Squeeze
    
    # ============ LSTM 配置 ============
    SEQUENCE_LENGTH = 64  # LSTM 输入序列长度
    LSTM_UNITS = [128, 64]  # LSTM 层配置
    DENSE_UNITS = [64]  # 全连接层配置
    DROPOUT_RATE = 0.35  # 增加dropout以缓解过拟合
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # ============ 数据划分配置（避免数据泄漏） ============
    # 使用 train/val/test 三分（而不是旧的 train/test 二分）
    # - 训练集：用于训练模型
    # - 验证集：用于早停和超参数调优
    # - 测试集：只用于最终评估，不参与任何模型选择
    # 注意：增大验证集比例可以提高验证集覆盖所有状态的概率
    TRAIN_RATIO = 0.65  # 训练集比例（从 0.70 降到 0.65）
    VAL_RATIO = 0.20    # 验证集比例（从 0.15 增到 0.20）
    TEST_RATIO = 0.15   # 测试集比例
    
    # 旧配置（保持向后兼容，但不推荐使用）
    VALIDATION_SPLIT = 0.2
    
    # ============ 正则化配置 ============
    L2_LAMBDA = 1.5e-3  # L2 正则化强度（权重衰减）; 已增大以缓解过拟合（从1e-4增加到1e-3）
    USE_BATCH_NORM = True  # 是否使用 BatchNormalization
    LEARNING_RATE = 1e-3  # Adam 优化器学习率
    USE_CLASS_WEIGHT = True  # 是否使用类权重（处理类别不平衡问题，让稀有状态的错误代价更高）
    
    # ============ 训练回调配置 ============
    EARLY_STOPPING_PATIENCE = 8  # 早停耐心值（验证损失不改善的epoch数）
    LR_REDUCE_PATIENCE = 5  # 学习率衰减耐心值（验证损失不改善的epoch数后降低学习率）
    
    # ============ 增量训练配置 ============
    INCREMENTAL_EPOCHS = 10  # 增量训练轮数
    INCREMENTAL_LEARNING_RATE = 1e-5  # 增量训练学习率（比完整训练小，避免破坏已学习的权重）
    INCREMENTAL_VALIDATION_SPLIT = 0.2  # 增量训练验证集比例
    INCREMENTAL_EARLY_STOPPING_PATIENCE = 3  # 增量训练早停耐心值（比完整训练小，更敏感）
    
    # ============ 训练调度配置 ============
    INCREMENTAL_TRAIN_TIMES = [
        "08:00",  # HKT 8am
        "20:00",  # HKT 8pm
    ]
    
    FULL_RETRAIN_TIME = "03:00"  # 周日凌晨3点完整重训
    FULL_RETRAIN_DAY = 6  # 0=周一, 6=周日
    
    # ============ 路径配置 ============
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # ============ Binance API 配置 ============
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    
    # ============ 数据缓存配置 ============
    CACHE_DB_PATH = os.path.join(BASE_DIR, "data", "cache.db")
    CACHE_ENABLED = True
    CACHE_COMPRESSION = True  # 是否压缩存储
    
    @classmethod
    def get_model_path(cls, symbol: str, model_type: str = "lstm") -> str:
        """获取模型保存路径"""
        return os.path.join(cls.MODELS_DIR, symbol, f"{model_type}_model.h5")
    
    @classmethod
    def get_scaler_path(cls, symbol: str) -> str:
        """获取标准化器保存路径"""
        return os.path.join(cls.MODELS_DIR, symbol, "scaler.pkl")
    
    @classmethod
    def get_hmm_path(cls, symbol: str) -> str:
        """获取 HMM 模型保存路径"""
        return os.path.join(cls.MODELS_DIR, symbol, "hmm_model.pkl")
    
    @classmethod
    def ensure_dirs(cls):
        """确保所有必要的目录存在"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        for symbol in cls.SYMBOLS:
            os.makedirs(os.path.join(cls.MODELS_DIR, symbol), exist_ok=True)
            os.makedirs(os.path.join(cls.DATA_DIR, symbol), exist_ok=True)


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    配置logging，解决Windows控制台编码问题
    跨平台兼容：Windows使用安全handler，Linux/macOS使用标准handler
    
    Args:
        log_file: 日志文件路径（可选）
        level: 日志级别
    """
    # 在Windows上需要特殊处理编码问题
    # 在Linux/macOS上，终端通常支持UTF-8，可以直接使用标准handler
    if sys.platform == 'win32':
        # Windows: 创建UTF-8安全的StreamHandler
        class SafeStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    stream = self.stream
                    # 尝试直接写入
                    stream.write(msg + self.terminator)
                    stream.flush()
                except UnicodeEncodeError:
                    # 如果编码失败，替换无法编码的字符
                    try:
                        msg = self.format(record)
                        # 将无法编码的字符替换为?
                        msg = msg.encode('ascii', errors='replace').decode('ascii')
                        stream.write(msg + self.terminator)
                        stream.flush()
                    except Exception:
                        # 如果还是失败，静默处理（避免无限循环）
                        pass
                except Exception:
                    self.handleError(record)
        
        stream_handler = SafeStreamHandler()
    else:
        # Linux/macOS: 使用标准StreamHandler（通常支持UTF-8）
        stream_handler = logging.StreamHandler()
    
    handlers = [stream_handler]
    
    # 添加文件handler（如果指定）
    # 文件handler在所有平台上都使用UTF-8编码
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
            handlers.append(file_handler)
        except Exception:
            # 如果UTF-8失败，尝试默认编码
            file_handler = logging.FileHandler(log_file, errors='replace')
            handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # 强制重新配置，覆盖之前的配置
    )
