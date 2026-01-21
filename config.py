"""
配置文件 - 市场趋势分类器自动化训练系统
"""
import os
import sys
import shutil
import logging
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
    PRIMARY_TIMEFRAME = "15m"  # 主时间框架（默认）
    
    # ============ 多时间框架模型配置 ============
    # 支持同时训练和使用多个主时间框架的模型
    # 每个配置定义一个独立的 regime 模型
    MODEL_CONFIGS = {
        "5m": {
            "primary_timeframe": "5m",
            "timeframes": ["1m", "5m", "15m"],  # 包含 1m 用于捕捉微观结构
            "sequence_length": 48,  # 48根5m K线 = 4小时
            "lstm_units": [96, 48],  # 增加容量以匹配更长的序列（48 vs 32）
            "dense_units": [48],  # 相应增加 Dense 层容量
            "epochs": 80,  # 5m模型需要更多epoch才能收敛（更大的模型容量）
            "early_stopping_patience": 12,  # 更大的patience，给5m模型更多训练机会
        },
        "15m": {
            "primary_timeframe": "15m",
            "timeframes": ["5m", "15m", "1h"],
            "sequence_length": 32,  # 32根15m K线 = 8小时
            "lstm_units": [64, 32],
            "dense_units": [32],
            "epochs": 50,  # 15m模型使用默认epoch数
            "early_stopping_patience": 8,  # 使用默认patience
        },
        "1h": {
            "primary_timeframe": "1h",
            "timeframes": ["15m", "1h", "4h"],  # 包含更高时间框架捕捉长期趋势
            "sequence_length": 24,  # 24根1h K线 = 24小时/1天
            "lstm_units": [64, 32],  # 与15m相同，序列长度相近
            "dense_units": [32],
            "epochs": 60,  # 比15m多，因为数据点更少需要更多训练
            "early_stopping_patience": 10,  # 介于5m和15m之间
        },
    }
    
    # 启用的模型列表
    ENABLED_MODELS = ["5m", "15m", "1h"]  # 同时启用 5m、15m 和 1h 模型
    
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
    SEQUENCE_LENGTH = 32  # LSTM 输入序列长度（32根15m K线 = 8小时，适合日内交易）
    LSTM_UNITS = [64, 32]  # LSTM 层配置（调整为匹配32根K线的序列长度）
    DENSE_UNITS = [32]  # 全连接层配置（调整为匹配更小的模型容量）
    DROPOUT_RATE = 0.25  # Dropout比率（降低正则化，给模型更多学习空间）
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # ============ 多步预测配置 ============
    # 预测未来多根 K 线的 market regime
    PREDICTION_HORIZONS = [1, 2, 3, 4]  # 预测 t+1 到 t+4
    
    # 软标签温度（>1 使分布更平滑，用于 t+2 及以后的标签生成）
    # 较高的温度会产生更 "软" 的标签，有助于处理长期预测的不确定性
    LABEL_TEMPERATURE = 1.5
    
    # 历史回看的 K 线数量（用于输出历史 regime 序列）
    # 16 根 15m K 线 = 4 小时
    HISTORY_LOOKBACK_BARS = 16
    
    # 各预测步数的损失权重
    # 远期预测的权重较低，因为不确定性更大
    HORIZON_LOSS_WEIGHTS = {
        't+1': 1.0,   # 主预测，权重最高
        't+2': 0.8,
        't+3': 0.6,
        't+4': 0.4,
    }
    
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
    L2_LAMBDA = 1e-3  # L2 正则化强度（权重衰减，降低以匹配更小的模型容量）
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
    def get_model_path(cls, symbol: str, model_type: str = "lstm", primary_timeframe: str = None) -> str:
        """
        获取模型保存路径
        
        Args:
            symbol: 交易对
            model_type: 模型类型（lstm/hmm）
            primary_timeframe: 主时间框架，如果为 None 则使用默认路径（向后兼容）
            
        Returns:
            模型文件路径
        """
        if primary_timeframe:
            return os.path.join(cls.MODELS_DIR, symbol, primary_timeframe, f"{model_type}_model.h5")
        else:
            # 向后兼容：检查新路径是否存在，如果不存在则使用旧路径
            new_path = os.path.join(cls.MODELS_DIR, symbol, cls.PRIMARY_TIMEFRAME, f"{model_type}_model.h5")
            old_path = os.path.join(cls.MODELS_DIR, symbol, f"{model_type}_model.h5")
            if os.path.exists(new_path):
                return new_path
            return old_path
    
    @classmethod
    def get_scaler_path(cls, symbol: str, primary_timeframe: str = None) -> str:
        """
        获取标准化器保存路径
        
        Args:
            symbol: 交易对
            primary_timeframe: 主时间框架，如果为 None 则使用默认路径（向后兼容）
            
        Returns:
            Scaler 文件路径
        """
        if primary_timeframe:
            return os.path.join(cls.MODELS_DIR, symbol, primary_timeframe, "scaler.pkl")
        else:
            new_path = os.path.join(cls.MODELS_DIR, symbol, cls.PRIMARY_TIMEFRAME, "scaler.pkl")
            old_path = os.path.join(cls.MODELS_DIR, symbol, "scaler.pkl")
            if os.path.exists(new_path):
                return new_path
            return old_path
    
    @classmethod
    def get_hmm_path(cls, symbol: str, primary_timeframe: str = None) -> str:
        """
        获取 HMM 模型保存路径
        
        Args:
            symbol: 交易对
            primary_timeframe: 主时间框架，如果为 None 则使用默认路径（向后兼容）
            
        Returns:
            HMM 模型文件路径
        """
        if primary_timeframe:
            return os.path.join(cls.MODELS_DIR, symbol, primary_timeframe, "hmm_model.pkl")
        else:
            new_path = os.path.join(cls.MODELS_DIR, symbol, cls.PRIMARY_TIMEFRAME, "hmm_model.pkl")
            old_path = os.path.join(cls.MODELS_DIR, symbol, "hmm_model.pkl")
            if os.path.exists(new_path):
                return new_path
            return old_path
    
    @classmethod
    def get_model_config(cls, primary_timeframe: str) -> dict:
        """
        获取指定时间框架的模型配置
        
        Args:
            primary_timeframe: 主时间框架（如 "5m" 或 "15m"）
            
        Returns:
            模型配置字典
        """
        if primary_timeframe not in cls.MODEL_CONFIGS:
            raise ValueError(f"不支持的时间框架: {primary_timeframe}，支持的值: {list(cls.MODEL_CONFIGS.keys())}")
        return cls.MODEL_CONFIGS[primary_timeframe]
    
    @classmethod
    def ensure_dirs(cls):
        """确保所有必要的目录存在"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        for symbol in cls.SYMBOLS:
            os.makedirs(os.path.join(cls.MODELS_DIR, symbol), exist_ok=True)
            os.makedirs(os.path.join(cls.DATA_DIR, symbol), exist_ok=True)
            # 为每个启用的模型创建子目录
            for tf in cls.ENABLED_MODELS:
                os.makedirs(os.path.join(cls.MODELS_DIR, symbol, tf), exist_ok=True)
    
    @classmethod
    def migrate_models_to_timeframe_dirs(cls, dry_run: bool = False) -> dict:
        """
        迁移旧版模型文件到时间框架子目录
        
        将 models/{symbol}/lstm_model.h5 等文件迁移到 models/{symbol}/15m/lstm_model.h5
        
        Args:
            dry_run: 如果为 True，只返回要迁移的文件列表，不实际执行迁移
            
        Returns:
            迁移结果字典：{"migrated": [...], "skipped": [...], "errors": [...]}
        """
        logger = logging.getLogger(__name__)
        result = {"migrated": [], "skipped": [], "errors": []}
        
        # 需要迁移的文件
        model_files = ["lstm_model.h5", "hmm_model.pkl", "scaler.pkl", "scaler_feature_names.pkl"]
        
        # 遍历所有交易对目录
        if not os.path.exists(cls.MODELS_DIR):
            return result
        
        for symbol_dir in os.listdir(cls.MODELS_DIR):
            symbol_path = os.path.join(cls.MODELS_DIR, symbol_dir)
            if not os.path.isdir(symbol_path):
                continue
            
            # 检查是否有旧版模型文件（直接在 symbol 目录下）
            old_files = []
            for model_file in model_files:
                old_path = os.path.join(symbol_path, model_file)
                if os.path.exists(old_path):
                    old_files.append(model_file)
            
            if not old_files:
                continue
            
            # 检查是否已经有新版目录结构
            new_dir = os.path.join(symbol_path, cls.PRIMARY_TIMEFRAME)
            new_dir_exists = os.path.exists(new_dir) and any(
                os.path.exists(os.path.join(new_dir, f)) for f in model_files
            )
            
            if new_dir_exists:
                # 新目录已存在且有文件，跳过迁移
                for f in old_files:
                    result["skipped"].append({
                        "file": os.path.join(symbol_path, f),
                        "reason": "新版目录已存在"
                    })
                continue
            
            # 执行迁移
            if not dry_run:
                os.makedirs(new_dir, exist_ok=True)
            
            for model_file in old_files:
                old_path = os.path.join(symbol_path, model_file)
                new_path = os.path.join(new_dir, model_file)
                
                try:
                    if not dry_run:
                        shutil.move(old_path, new_path)
                    result["migrated"].append({
                        "from": old_path,
                        "to": new_path
                    })
                    logger.info(f"{'[DRY-RUN] ' if dry_run else ''}迁移: {old_path} -> {new_path}")
                except Exception as e:
                    result["errors"].append({
                        "file": old_path,
                        "error": str(e)
                    })
                    logger.error(f"迁移失败: {old_path} -> {new_path}, 错误: {e}")
        
        if result["migrated"]:
            logger.info(f"{'[DRY-RUN] ' if dry_run else ''}迁移完成: {len(result['migrated'])} 个文件")
        if result["skipped"]:
            logger.info(f"跳过迁移: {len(result['skipped'])} 个文件（新版目录已存在）")
        if result["errors"]:
            logger.warning(f"迁移失败: {len(result['errors'])} 个文件")
        
        return result
    
    @classmethod
    def check_model_exists(cls, symbol: str, primary_timeframe: str = None) -> bool:
        """
        检查指定模型是否存在
        
        Args:
            symbol: 交易对
            primary_timeframe: 主时间框架
            
        Returns:
            模型是否存在
        """
        model_path = cls.get_model_path(symbol, "lstm", primary_timeframe)
        scaler_path = cls.get_scaler_path(symbol, primary_timeframe)
        return os.path.exists(model_path) and os.path.exists(scaler_path)


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
