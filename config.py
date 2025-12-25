"""
配置文件 - 市场趋势分类器自动化训练系统
"""
import os
from datetime import datetime, timedelta

class TrainingConfig:
    """训练配置"""
    
    # ============ 交易对配置 ============
    SYMBOLS = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        # 添加更多交易对
    ]
    
    # ============ 数据配置 ============
    TIMEFRAMES = ["5m", "15m", "1h"]
    PRIMARY_TIMEFRAME = "15m"  # 主时间框架
    
    # 完整重训数据长度（天）
    FULL_RETRAIN_DAYS = 365  # 12个月
    
    # 增量训练数据长度（天）
    INCREMENTAL_TRAIN_DAYS = 30  # 最近30天
    
    # ============ HMM 配置 ============
    N_STATES = 6  # 市场状态数量
    N_PCA_COMPONENTS = 4  # PCA 降维后的特征数
    
    # 状态名称（保持与原项目一致）
    REGIME_NAMES = {
        0: "Choppy_High_Vol",
        1: "Strong_Trend", 
        2: "Volatility_Spike",
        3: "Weak_Trend",
        4: "Range",
        5: "Squeeze"
    }
    
    # ============ LSTM 配置 ============
    SEQUENCE_LENGTH = 64  # LSTM 输入序列长度
    LSTM_UNITS = [128, 64]  # LSTM 层配置
    DROPOUT_RATE = 0.2
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
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
