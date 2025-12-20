"""
特征工程模块 - 计算技术指标和市场特征
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        pass
    
    def calculate_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        计算单个时间框架的技术指标
        
        Args:
            df: OHLCV DataFrame
            timeframe: 时间框架标识（用于列名前缀）
            
        Returns:
            包含技术指标的 DataFrame
        """
        features = pd.DataFrame(index=df.index)
        
        # ========== 价格动量指标 ==========
        # RSI
        features[f'{timeframe}_rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        features[f'{timeframe}_rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        features[f'{timeframe}_macd'] = macd.macd()
        features[f'{timeframe}_macd_signal'] = macd.macd_signal()
        features[f'{timeframe}_macd_diff'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        features[f'{timeframe}_stoch_k'] = stoch.stoch()
        features[f'{timeframe}_stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        features[f'{timeframe}_williams_r'] = ta.momentum.WilliamsRIndicator(
            df['high'], df['low'], df['close'], lbp=14
        ).williams_r()
        
        # ========== 趋势指标 ==========
        # EMA
        features[f'{timeframe}_ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        features[f'{timeframe}_ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        features[f'{timeframe}_ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        features[f'{timeframe}_ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        # SMA
        features[f'{timeframe}_sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        features[f'{timeframe}_sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        
        # ADX (趋势强度)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        features[f'{timeframe}_adx'] = adx.adx()
        features[f'{timeframe}_adx_pos'] = adx.adx_pos()
        features[f'{timeframe}_adx_neg'] = adx.adx_neg()
        
        # ========== 波动率指标 ==========
        # ATR (真实波动幅度)
        features[f'{timeframe}_atr_14'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        features[f'{timeframe}_bb_high'] = bollinger.bollinger_hband()
        features[f'{timeframe}_bb_mid'] = bollinger.bollinger_mavg()
        features[f'{timeframe}_bb_low'] = bollinger.bollinger_lband()
        features[f'{timeframe}_bb_width'] = bollinger.bollinger_wband()
        features[f'{timeframe}_bb_pct'] = bollinger.bollinger_pband()
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        features[f'{timeframe}_kc_high'] = keltner.keltner_channel_hband()
        features[f'{timeframe}_kc_low'] = keltner.keltner_channel_lband()
        features[f'{timeframe}_kc_width'] = keltner.keltner_channel_wband()
        
        # ========== 成交量指标 ==========
        # OBV (能量潮)
        features[f'{timeframe}_obv'] = ta.volume.OnBalanceVolumeIndicator(
            df['close'], df['volume']
        ).on_balance_volume()
        
        # MFI (资金流量指标)
        features[f'{timeframe}_mfi'] = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).money_flow_index()
        
        # Volume MA
        features[f'{timeframe}_volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # ========== 价格变化 ==========
        # Returns
        features[f'{timeframe}_returns'] = df['close'].pct_change()
        features[f'{timeframe}_log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low Range
        features[f'{timeframe}_hl_pct'] = (df['high'] - df['low']) / df['close']
        
        # Close位置 (在High-Low范围内的位置)
        features[f'{timeframe}_close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 价格相对于移动平均线的位置
        features[f'{timeframe}_price_vs_sma20'] = df['close'] / features[f'{timeframe}_sma_20']
        features[f'{timeframe}_price_vs_ema50'] = df['close'] / features[f'{timeframe}_ema_50']
        
        # ========== 动量指标 ==========
        # ROC (变化率)
        features[f'{timeframe}_roc_12'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
        
        # 移动平均线交叉
        features[f'{timeframe}_ema_cross'] = np.where(
            features[f'{timeframe}_ema_9'] > features[f'{timeframe}_ema_21'], 1, -1
        )
        
        logger.debug(f"计算了 {len(features.columns)} 个 {timeframe} 特征")
        return features
    
    def combine_timeframe_features(
        self, 
        data: Dict[str, pd.DataFrame],
        primary_timeframe: str = "15m"
    ) -> pd.DataFrame:
        """
        合并多个时间框架的特征
        
        Args:
            data: 时间框架到 DataFrame 的字典
            primary_timeframe: 主时间框架（其他时间框架会重采样到这个频率）
            
        Returns:
            合并后的特征 DataFrame
        """
        # 计算每个时间框架的特征
        all_features = {}
        for tf, df in data.items():
            features = self.calculate_features(df, tf)
            all_features[tf] = features
        
        # 以主时间框架为基准
        primary_features = all_features[primary_timeframe]
        combined = primary_features.copy()
        
        # 重采样其他时间框架到主时间框架
        for tf, features in all_features.items():
            if tf == primary_timeframe:
                continue
            
            # 前向填充（使用最近的值）
            resampled = features.reindex(primary_features.index, method='ffill')
            combined = pd.concat([combined, resampled], axis=1)
        
        # 删除包含 NaN 的行
        combined = combined.dropna()
        
        logger.info(f"合并后的特征数量: {len(combined.columns)}, 样本数: {len(combined)}")
        return combined
    
    def prepare_features_for_hmm(self, features: pd.DataFrame) -> np.ndarray:
        """
        准备用于 HMM 的特征（标准化）
        
        Args:
            features: 特征 DataFrame
            
        Returns:
            标准化后的特征数组
        """
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        return features_scaled
    
    def select_key_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        选择关键特征（可选：用于降维）
        
        这里可以添加特征选择逻辑，例如：
        - 基于方差的特征选择
        - 基于相关性的特征选择
        - 基于模型重要性的特征选择
        
        Args:
            features: 原始特征 DataFrame
            
        Returns:
            选择后的特征 DataFrame
        """
        # 简单示例：移除高度相关的特征
        # 在实际使用中，你可能想要更复杂的特征选择策略
        
        # 计算相关性矩阵
        corr_matrix = features.corr().abs()
        
        # 选择上三角矩阵
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 找到高度相关的特征（相关系数 > 0.95）
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        if to_drop:
            logger.info(f"移除 {len(to_drop)} 个高度相关的特征")
            features = features.drop(columns=to_drop)
        
        return features
