"""
特征工程模块 - 计算技术指标和市场特征
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, cache_manager=None):
        """
        初始化特征工程器
        
        Args:
            cache_manager: 数据缓存管理器（可选），如果提供则启用特征缓存
        """
        self.cache_manager = cache_manager
        if cache_manager:
            logger.debug("特征缓存已启用")
        else:
            logger.debug("特征缓存未启用")
    
    def calculate_features(
        self, 
        df: pd.DataFrame, 
        timeframe: str,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        计算单个时间框架的技术指标
        
        注意：技术指标需要窗口期，前 N 行会是 NaN（N = 最大窗口期）
        - EMA 200: 前 200 行是 NaN
        - EMA 50: 前 50 行是 NaN
        - SMA 50: 前 50 行是 NaN
        - RSI 14: 前 14 行是 NaN
        这是正常现象，不是 bug。
        
        Args:
            df: OHLCV DataFrame
            timeframe: 时间框架标识（用于列名前缀）
            symbol: 交易对（可选，用于特征缓存）
            
        Returns:
            包含技术指标的 DataFrame
        """
        if len(df) == 0:
            logger.warning(f"数据为空，无法计算特征: {timeframe}")
            return pd.DataFrame()
        
        # 检查特征缓存
        if self.cache_manager and symbol:
            start_date = df.index.min().date()
            end_date = df.index.max().date()
            
            # 获取缓存的特征
            cached_features = self.cache_manager.get_cached_features(
                symbol, timeframe, start_date, end_date
            )
            
            if not cached_features.empty:
                # 检查缓存是否覆盖了所有需要的数据
                cached_start = cached_features.index.min()
                cached_end = cached_features.index.max()
                data_start = df.index.min()
                data_end = df.index.max()
                
                # 如果缓存完全覆盖数据范围，直接返回缓存
                if cached_start <= data_start and cached_end >= data_end:
                    logger.debug(
                        f"使用缓存特征: {symbol} {timeframe} "
                        f"({len(cached_features)} 行, {len(cached_features.columns)} 列)"
                    )
                    # 只返回数据范围内的特征
                    return cached_features[(cached_features.index >= data_start) & 
                                          (cached_features.index <= data_end)]
                
                # 如果缓存部分覆盖，需要增量更新
                # 注意：由于技术指标依赖历史数据（如EMA 200需要200行历史），
                # 我们需要重新计算整个范围的特征，而不仅仅是新数据
                # 但为了性能优化，我们可以只计算新数据，然后合并
                # 这里为了简化，我们继续执行下面的计算逻辑，然后合并缓存和新特征
                if cached_end >= data_start:
                    logger.debug(
                        f"部分缓存命中: {symbol} {timeframe} "
                        f"缓存: {cached_start} 至 {cached_end}, "
                        f"需要: {data_start} 至 {data_end}"
                    )
                    # 继续执行下面的计算逻辑，计算完成后会合并
        
        # 检查数据量是否足够计算技术指标
        max_window = 200  # EMA 200 的最大窗口期
        if len(df) < max_window:
            logger.debug(
                f"数据量较少 ({len(df)} 行)，技术指标的前 {max_window} 行将是 NaN。"
                f"这是正常的，因为指标需要历史数据来计算。"
            )
        
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
        
        # 如果之前有部分缓存，合并缓存和新特征
        if self.cache_manager and symbol:
            start_date = df.index.min().date()
            end_date = df.index.max().date()
            cached_features = self.cache_manager.get_cached_features(
                symbol, timeframe, start_date, end_date
            )
            
            if not cached_features.empty:
                cached_end = cached_features.index.max()
                data_start = df.index.min()
                
                # 如果缓存部分覆盖，合并缓存和新特征
                if cached_end >= data_start and cached_end < df.index.max():
                    # 合并缓存和新特征
                    combined_features = pd.concat([cached_features, features], axis=0)
                    combined_features = combined_features.sort_index()
                    combined_features = combined_features[~combined_features.index.duplicated(keep='last')]
                    # 只返回数据范围内的特征
                    features = combined_features[(combined_features.index >= data_start) & 
                                                (combined_features.index <= df.index.max())]
                    logger.debug(
                        f"合并缓存和新特征: {symbol} {timeframe} "
                        f"最终特征数: {len(features)} 行"
                    )
        
        # 保存到缓存
        if self.cache_manager and symbol and not features.empty:
            self.cache_manager.save_features_range(symbol, timeframe, features)
        
        return features
    
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """将时间框架转换为分钟数"""
        timeframe_map = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '8h': 480,
            '12h': 720,
            '1d': 1440,
            '3d': 4320,
            '1w': 10080,
            '1M': 43200
        }
        return timeframe_map.get(timeframe, 15)
    
    def combine_timeframe_features(
        self, 
        data: Dict[str, pd.DataFrame],
        primary_timeframe: str = "15m",
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        合并多个时间框架的特征
        
        Args:
            data: 时间框架到 DataFrame 的字典
            primary_timeframe: 主时间框架（其他时间框架会重采样到这个频率）
            symbol: 交易对（可选，用于特征缓存）
            
        Returns:
            合并后的特征 DataFrame
        """
        # 检查合并特征的缓存
        combined_timeframe = f"combined_{primary_timeframe}"
        if self.cache_manager and symbol:
            # 获取数据的时间范围
            primary_df = data.get(primary_timeframe, pd.DataFrame())
            if not primary_df.empty:
                start_date = primary_df.index.min().date()
                end_date = primary_df.index.max().date()
                
                # 获取缓存的合并特征
                cached_combined = self.cache_manager.get_cached_features(
                    symbol, combined_timeframe, start_date, end_date
                )
                
                if not cached_combined.empty:
                    cached_start = cached_combined.index.min()
                    cached_end = cached_combined.index.max()
                    data_start = primary_df.index.min()
                    data_end = primary_df.index.max()
                    
                    # 如果缓存完全覆盖数据范围，直接返回缓存
                    if cached_start <= data_start and cached_end >= data_end:
                        logger.debug(
                            f"使用缓存合并特征: {symbol} {combined_timeframe} "
                            f"({len(cached_combined)} 行, {len(cached_combined.columns)} 列)"
                        )
                        # 只返回数据范围内的特征
                        return cached_combined[(cached_combined.index >= data_start) & 
                                              (cached_combined.index <= data_end)]
        
        # 计算每个时间框架的特征
        all_features = {}
        for tf, df in data.items():
            features = self.calculate_features(df, tf, symbol=symbol)
            all_features[tf] = features
        
        # 以主时间框架为基准
        primary_features = all_features[primary_timeframe]
        
        # 检查主时间框架是否有数据
        if len(primary_features) == 0:
            logger.warning(f"主时间框架 {primary_timeframe} 没有数据，无法合并特征")
            return pd.DataFrame()
        
        combined = primary_features.copy()
        
        # 重采样其他时间框架到主时间框架
        for tf, features in all_features.items():
            if tf == primary_timeframe:
                continue
            
            # 前向填充（使用最近的值）
            # 注意：如果 features 为空，reindex 会返回全 NaN
            if len(features) > 0:
                resampled = features.reindex(primary_features.index, method='ffill')
            else:
                # 如果 features 为空，创建全 NaN 的 DataFrame
                resampled = pd.DataFrame(index=primary_features.index, columns=features.columns)
            combined = pd.concat([combined, resampled], axis=1)
        
        # 处理 NaN 值
        # 注意：技术指标需要窗口期，前 N 行必然是 NaN（N = 最大窗口期，如 EMA 200 需要 200 行）
        # 这是正常现象，不是 bug
        if len(combined) > 0:
            # 只删除所有列都是 NaN 的行，而不是删除任何列包含 NaN 的行
            # 这样可以保留部分有效数据的行
            rows_before = len(combined)
            combined = combined.dropna(how='all')
            rows_after = len(combined)
            
            if rows_before > rows_after:
                logger.debug(
                    f"删除了 {rows_before - rows_after} 行全 NaN 数据 "
                    f"（技术指标窗口期导致前 {rows_before - rows_after} 行全为 NaN）"
                )
            
            # 如果删除后还有数据，使用前向填充填充剩余的 NaN
            if len(combined) > 0:
                # 使用 ffill() 方法（pandas 新版本推荐的方式）
                combined = combined.ffill().fillna(0)
                
                # 记录数据质量
                nan_count = combined.isna().sum().sum()
                total_cells = len(combined) * len(combined.columns)
                if nan_count > 0:
                    logger.warning(
                        f"特征中仍有 {nan_count}/{total_cells} ({nan_count/total_cells*100:.1f}%) 个 NaN 值，"
                        f"已用前向填充和0填充处理。"
                    )
        
        logger.info(
            f"合并后的特征数量: {len(combined.columns)}, 样本数: {len(combined)}"
        )
        
        # 如果之前有部分缓存，合并缓存和新特征
        if self.cache_manager and symbol and not combined.empty:
            primary_df = data.get(primary_timeframe, pd.DataFrame())
            if not primary_df.empty:
                start_date = primary_df.index.min().date()
                end_date = primary_df.index.max().date()
                cached_combined = self.cache_manager.get_cached_features(
                    symbol, combined_timeframe, start_date, end_date
                )
                
                if not cached_combined.empty:
                    cached_end = cached_combined.index.max()
                    data_start = primary_df.index.min()
                    
                    # 如果缓存部分覆盖，合并缓存和新特征
                    if cached_end >= data_start and cached_end < combined.index.max():
                        combined_features = pd.concat([cached_combined, combined], axis=0)
                        combined_features = combined_features.sort_index()
                        combined_features = combined_features[~combined_features.index.duplicated(keep='last')]
                        # 只返回数据范围内的特征
                        combined = combined_features[(combined_features.index >= data_start) & 
                                                    (combined_features.index <= combined.index.max())]
                        logger.debug(
                            f"合并缓存和新合并特征: {symbol} {combined_timeframe} "
                            f"最终特征数: {len(combined)} 行"
                        )
        
        # 保存合并后的特征到缓存
        if self.cache_manager and symbol and not combined.empty:
            self.cache_manager.save_features_range(symbol, combined_timeframe, combined)
        
        # 如果数据量不足，记录警告
        if len(combined) > 0:
            # 检查是否有足够的数据用于 LSTM（通常需要 sequence_length，约64行）
            # 但考虑到技术指标的窗口期，实际需要更多数据
            min_required_rows = 200  # EMA 200 的最大窗口期
            if len(combined) < min_required_rows:
                logger.warning(
                    f"数据量可能不足：只有 {len(combined)} 行，"
                    f"建议至少 {min_required_rows} 行以确保技术指标计算的准确性。"
                )
        
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
