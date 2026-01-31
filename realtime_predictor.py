"""
实时推理模块 - 用于实时市场状态预测
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from config import TrainingConfig, setup_logging
from data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngineer
from lstm_trainer import LSTMRegimeClassifier
from hmm_trainer import HMMRegimeLabeler

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeRegimePredictor:
    """实时市场状态预测器"""
    
    def __init__(self, symbol: str, config: TrainingConfig, primary_timeframe: str = None):
        """
        初始化
        
        Args:
            symbol: 交易对
            config: 配置
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
        """
        self.symbol = symbol
        self.config = config
        self.data_fetcher = BinanceDataFetcher()
        self.feature_engineer = FeatureEngineer(cache_manager=self.data_fetcher.cache_manager)
        
        # 获取模型配置
        if primary_timeframe is None:
            primary_timeframe = config.PRIMARY_TIMEFRAME
        
        self.primary_timeframe = primary_timeframe
        self.model_config = config.get_model_config(primary_timeframe)
        self.timeframes = self.model_config["timeframes"]
        
        # 加载 LSTM 模型（使用 PROD 版本路径）
        model_path = config.get_prod_model_path(symbol, "lstm", primary_timeframe)
        scaler_path = config.get_prod_scaler_path(symbol, primary_timeframe)
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"模型文件不存在，请先训练模型: {model_path}, {scaler_path}"
            )
        
        self.lstm_classifier = LSTMRegimeClassifier.load(model_path, scaler_path)
        logger.info(f"已加载 {symbol} 的 LSTM 模型 (primary_timeframe={primary_timeframe})")
        
        # 加载 HMM 模型以获取状态映射（使用 PROD 版本路径）
        hmm_path = config.get_prod_hmm_path(symbol, primary_timeframe)
        self.regime_mapping = {}  # 默认空映射
        
        if os.path.exists(hmm_path):
            try:
                hmm_labeler = HMMRegimeLabeler.load(hmm_path)
                self.regime_mapping = hmm_labeler.get_regime_mapping()
                logger.info(f"已加载 {symbol} 的状态映射: {self.regime_mapping}")
            except Exception as e:
                logger.warning(f"无法加载 HMM 模型的状态映射: {e}")
                logger.warning("将使用默认的状态名称（State_0, State_1, ...）")
        else:
            logger.warning(f"HMM 模型不存在: {hmm_path}")
            logger.warning("将使用默认的状态名称（State_0, State_1, ...）")
    
    def _get_regime_name(self, regime_id: int) -> str:
        """
        获取状态 ID 对应的语义名称
        
        优先使用 HMM 模型中保存的映射，如果没有则回退到 config 中的硬编码名称
        
        Args:
            regime_id: 状态 ID
            
        Returns:
            语义名称
        """
        # 优先使用 HMM 模型中的动态映射
        if self.regime_mapping and regime_id in self.regime_mapping:
            return self.regime_mapping[regime_id]
        
        # 回退到 config 中的硬编码名称（向后兼容）
        if hasattr(self.config, 'REGIME_NAMES') and regime_id in self.config.REGIME_NAMES:
            return self.config.REGIME_NAMES[regime_id]
        
        # 最后使用通用名称
        return f"State_{regime_id}"
    
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
    
    def _get_historical_regimes(
        self, 
        features: pd.DataFrame, 
        lookback_bars: int = 16
    ) -> Dict:
        """
        获取历史 regime 序列
        
        Args:
            features: 特征 DataFrame
            lookback_bars: 回看的 K 线数量（默认 16 根，对于 15m 约为 4 小时）
            
        Returns:
            历史 regime 字典:
            {
                'sequence': ['Range', 'Range', 'Weak_Trend', ...],
                'timestamps': [datetime, datetime, ...],
                'confidences': [0.72, 0.68, ...],
                'count': 16,
                'lookback_hours': 4.0
            }
        """
        sequence_length = self.lstm_classifier.sequence_length
        
        # 需要足够的数据来进行滑动窗口预测
        min_rows_needed = sequence_length + lookback_bars
        if len(features) < min_rows_needed:
            logger.warning(
                f"数据不足以生成历史序列：需要 {min_rows_needed} 行，当前 {len(features)} 行"
            )
            return {
                'sequence': [],
                'timestamps': [],
                'confidences': [],
                'count': 0,
                'lookback_hours': 0
            }
        
        # 对齐特征
        feature_names = self.lstm_classifier.feature_names_
        if feature_names is not None:
            features = features.reindex(columns=feature_names, fill_value=0)
        
        # 标准化
        features_scaled = self.lstm_classifier.scaler.transform(features)
        
        # 滑动窗口预测最近 lookback_bars 个时间点
        sequence = []
        timestamps = []
        confidences = []
        
        confidence_threshold = getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.4)
        
        # 从最早的开始，保证输出顺序是时间升序
        start_idx = len(features_scaled) - lookback_bars
        
        for i in range(start_idx, len(features_scaled)):
            X = np.array([features_scaled[i - sequence_length:i]])
            proba = self.lstm_classifier.predict_proba(X)[0]
            regime_id = np.argmax(proba)
            confidence = float(proba[regime_id])
            
            is_uncertain = confidence < confidence_threshold
            regime_name = "Uncertain" if is_uncertain else self._get_regime_name(regime_id)
            
            sequence.append(regime_name)
            timestamps.append(features.index[i])
            confidences.append(confidence)
        
        # 计算回看时间（小时）
        timeframe_minutes = self._get_timeframe_minutes(self.primary_timeframe)
        lookback_hours = lookback_bars * timeframe_minutes / 60
        
        return {
            'sequence': sequence,
            'timestamps': [t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in timestamps],
            'confidences': confidences,
            'count': len(sequence),
            'lookback_hours': lookback_hours
        }
    
    def get_current_regime(self) -> Dict:
        """
        获取当前市场状态（支持多步预测）
        
        优化说明：
        - 只获取最新的少量数据（最后几小时），而不是7天
        - 与缓存的特征合并（如果可用）
        - 只重新计算最新数据的特征，避免重复计算
        - 支持 t+1 到 t+4 的多步预测
        
        Returns:
            包含预测结果的字典
        """
        try:
            # 1. 获取最新数据
            # 增加到 7 天以确保所有时间框架都有足够的数据
            # （特别是 1h 和更高时间框架，需要至少 200+ 根 K 线来计算 EMA_200）
            days = 7
            
            data = self.data_fetcher.fetch_latest_data(
                symbol=self.symbol,
                timeframes=self.timeframes,
                days=days
            )
            
            # 检查数据是否足够
            primary_df = data.get(self.primary_timeframe, pd.DataFrame())
            if len(primary_df) < self.lstm_classifier.sequence_length:
                logger.warning(
                    f"数据量不足：只有 {len(primary_df)} 行，需要至少 {self.lstm_classifier.sequence_length} 行。"
                    f"建议获取更多历史数据或等待数据积累。"
                )
            
            # 2. 计算特征
            features = self.feature_engineer.combine_timeframe_features(
                data,
                primary_timeframe=self.primary_timeframe,
                symbol=self.symbol
            )
            
            # 检查特征质量
            if len(features) > 0:
                nan_percentage = (features.isna().sum().sum() / (len(features) * len(features.columns))) * 100
                logger.debug(
                    f"特征质量：{len(features)} 行 × {len(features.columns)} 列，"
                    f"NaN 比例: {nan_percentage:.1f}%"
                )
                if nan_percentage > 50:
                    logger.warning(
                        f"特征中 NaN 比例较高 ({nan_percentage:.1f}%)，"
                        f"这可能是因为数据量不足或技术指标窗口期较大。"
                    )
            
            # 3. 准备推理数据
            X = self.lstm_classifier.prepare_live_data(features)
            
            # 4. 多步预测
            confidence_threshold = getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.4)
            
            # 多步模型：返回 t+1 到 t+4 的预测
            multistep_proba = self.lstm_classifier.predict_multistep(X)
            
            # t+1 预测（主要预测）
            proba_t1 = multistep_proba['t+1'][0]
            regime_id = np.argmax(proba_t1)
            confidence = float(proba_t1[regime_id])
            
            # 构建多步预测结果
            predictions = {}
            for horizon, proba in multistep_proba.items():
                proba = proba[0]  # 取第一个样本
                pred_regime_id = np.argmax(proba)
                pred_confidence = float(proba[pred_regime_id])
                is_uncertain = pred_confidence < confidence_threshold
                
                predictions[horizon] = {
                    'regime_id': int(pred_regime_id),
                    'regime_name': "Uncertain" if is_uncertain else self._get_regime_name(pred_regime_id),
                    'confidence': pred_confidence,
                    'is_uncertain': is_uncertain,
                    'probabilities': {
                        self._get_regime_name(i): float(p)
                        for i, p in enumerate(proba)
                    }
                }
            
            # 5. 置信度拒绝机制（针对 t+1）
            is_uncertain = confidence < confidence_threshold
            
            if is_uncertain:
                regime_name = "Uncertain"
                logger.warning(
                    f"{self.symbol} t+1 置信度过低 ({confidence:.2%} < {confidence_threshold:.0%})，"
                    f"标记为 Uncertain（原判断: {self._get_regime_name(regime_id)}）"
                )
            else:
                regime_name = self._get_regime_name(regime_id)
            
            # 6. 获取历史 regime 序列
            history_lookback_bars = getattr(self.config, 'HISTORY_LOOKBACK_BARS', 16)
            historical_regimes = self._get_historical_regimes(features, lookback_bars=history_lookback_bars)
            
            # 7. 返回结果
            result = {
                'symbol': self.symbol,
                'primary_timeframe': self.primary_timeframe,
                'timestamp': datetime.now(),
                # 主预测（t+1，向后兼容）
                'regime_id': int(regime_id),
                'regime_name': regime_name,
                'confidence': confidence,
                'is_uncertain': is_uncertain,
                'confidence_threshold': confidence_threshold,
                'original_regime': self._get_regime_name(regime_id),
                'probabilities': predictions['t+1']['probabilities'],
                # 多步预测
                'is_multistep': True,
                'predictions': predictions,
                # 历史 regime 序列
                'historical_regimes': historical_regimes,
            }
            
            if not is_uncertain:
                logger.info(f"{self.symbol} 当前状态: {regime_name} (置信度: {confidence:.2%})")
                for h in ['t+2', 't+3', 't+4']:
                    if h in predictions:
                        pred = predictions[h]
                        logger.debug(
                            f"  {h}: {pred['regime_name']} (置信度: {pred['confidence']:.2%})"
                        )
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}", exc_info=True)
            raise
    
    def get_regime_history(
        self, 
        lookback_hours: int = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        获取历史市场状态（滑动窗口预测）
        
        支持两种查询方式：
        1. 按回看小时数：指定 lookback_hours，从当前时间往前回看
        2. 按日期范围：指定 start_date 和 end_date，获取指定时间范围的regime
        
        Args:
            lookback_hours: 回看小时数（如果指定，则从当前时间往前回看）
            start_date: 开始日期时间（如果指定，则获取指定时间范围）
            end_date: 结束日期时间（如果指定，则获取指定时间范围）
            
        Returns:
            包含历史预测的 DataFrame，列包括：
            - timestamp: 时间戳
            - regime_id: 状态ID
            - regime_name: 状态名称
            - confidence: 置信度
            - is_uncertain: 是否不确定
            - original_regime: 原始状态名称
        """
        try:
            # 确定查询方式
            if start_date is not None and end_date is not None:
                # 按日期范围查询
                if start_date >= end_date:
                    raise ValueError("start_date 必须早于 end_date")
                
                # 计算需要的天数（多获取一些以确保有足够数据计算特征）
                days_needed = (end_date - start_date).days + 7  # 额外7天用于特征计算
                start_date_for_fetch = start_date - timedelta(days=7)
                
                # 从缓存获取数据（优先使用缓存）
                if self.data_fetcher.cache_enabled and self.data_fetcher.cache_manager:
                    cached_data = {}
                    for tf in self.timeframes:
                        cached_df = self.data_fetcher.cache_manager.get_cached_data(
                            symbol=self.symbol,
                            timeframe=tf,
                            start_date=start_date_for_fetch.date(),
                            end_date=end_date.date()
                        )
                        if not cached_df.empty:
                            cached_data[tf] = cached_df
                    
                    # 如果缓存中有数据，使用缓存数据
                    if cached_data:
                        logger.info(
                            f"从缓存获取历史数据: {self.symbol} "
                            f"{start_date.date()} 至 {end_date.date()}"
                        )
                        data = cached_data
                    else:
                        # 缓存中没有数据，从API获取
                        logger.info(
                            f"缓存中没有数据，从API获取: {self.symbol} "
                            f"{start_date_for_fetch.date()} 至 {end_date.date()}"
                        )
                        data = self.data_fetcher.fetch_latest_data(
                            symbol=self.symbol,
                            timeframes=self.timeframes,
                            days=days_needed
                        )
                else:
                    # 缓存未启用，从API获取
                    data = self.data_fetcher.fetch_latest_data(
                        symbol=self.symbol,
                        timeframes=self.timeframes,
                        days=days_needed
                    )
                
                # 计算特征
                features = self.feature_engineer.combine_timeframe_features(
                    data,
                    primary_timeframe=self.primary_timeframe,
                    symbol=self.symbol
                )
                
                # 过滤到指定日期范围
                features = features[(features.index >= start_date) & (features.index <= end_date)]
                
            elif lookback_hours is not None:
                # 按回看小时数查询（原有逻辑）
                days = max(7, lookback_hours // 24 + 1)
                
                data = self.data_fetcher.fetch_latest_data(
                    symbol=self.symbol,
                    timeframes=self.timeframes,
                    days=days
                )
                
                # 计算特征
                features = self.feature_engineer.combine_timeframe_features(
                    data,
                    primary_timeframe=self.primary_timeframe,
                    symbol=self.symbol
                )
                
                # 只取最近的数据（根据时间框架计算需要的行数）
                timeframe_minutes = self._get_timeframe_minutes(self.primary_timeframe)
                rows_needed = (lookback_hours * 60) // timeframe_minutes
                # 加上 sequence_length 以确保有足够的数据进行滑动窗口预测
                min_rows = rows_needed + self.lstm_classifier.sequence_length
                features = features.tail(min_rows)
            else:
                # 默认：回看24小时
                days = 7
                data = self.data_fetcher.fetch_latest_data(
                    symbol=self.symbol,
                    timeframes=self.timeframes,
                    days=days
                )
                features = self.feature_engineer.combine_timeframe_features(
                    data,
                    primary_timeframe=self.primary_timeframe,
                    symbol=self.symbol
                )
                timeframe_minutes = self._get_timeframe_minutes(self.primary_timeframe)
                rows_needed = (24 * 60) // timeframe_minutes
                min_rows = rows_needed + self.lstm_classifier.sequence_length
                features = features.tail(min_rows)
            
            # 对齐特征名称（与训练时保持一致）
            feature_names = self.lstm_classifier.feature_names_
            if feature_names is None and hasattr(self.lstm_classifier.scaler, 'feature_names_in_'):
                feature_names = list(self.lstm_classifier.scaler.feature_names_in_)
            
            if feature_names is not None:
                missing_features = set(feature_names) - set(features.columns)
                extra_features = set(features.columns) - set(feature_names)
                
                if missing_features or extra_features:
                    # 对齐特征：添加缺失的特征（填充0），移除多余的特征
                    features = features.reindex(columns=feature_names, fill_value=0)
                else:
                    # 特征名称一致，但需要确保顺序一致
                    features = features[feature_names]
            elif hasattr(self.lstm_classifier.scaler, 'n_features_in_'):
                # 旧版本模型：只检查特征数量
                if len(features.columns) != self.lstm_classifier.scaler.n_features_in_:
                    raise ValueError(
                        f"特征数量不匹配！训练时: {self.lstm_classifier.scaler.n_features_in_} 个特征, "
                        f"当前: {len(features.columns)} 个特征"
                    )
            
            # 检查数据是否足够
            sequence_length = self.lstm_classifier.sequence_length
            if len(features) < sequence_length:
                logger.warning(
                    f"数据量不足：只有 {len(features)} 行，需要至少 {sequence_length} 行才能进行历史预测。"
                    f"建议获取更多历史数据。"
                )
                # 返回空 DataFrame，但包含正确的列
                return pd.DataFrame(columns=['timestamp', 'regime_id', 'regime_name', 'confidence', 'is_uncertain', 'original_regime'])
            
            # 批量预测（优化性能）
            features_scaled = self.lstm_classifier.scaler.transform(features)
            confidence_threshold = getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.4)
            
            # 准备批量输入数据
            batch_X = []
            batch_indices = []
            
            for i in range(sequence_length, len(features_scaled)):
                batch_X.append(features_scaled[i-sequence_length:i])
                batch_indices.append(i)
            
            if not batch_X:
                logger.warning("没有生成任何预测结果")
                return pd.DataFrame(columns=['timestamp', 'regime_id', 'regime_name', 'confidence', 'is_uncertain', 'original_regime'])
            
            # 批量预测（一次性预测所有样本，提高性能）
            batch_X = np.array(batch_X)
            batch_proba = self.lstm_classifier.predict_proba(batch_X)
            
            # 处理预测结果
            predictions = []
            for idx, proba in zip(batch_indices, batch_proba):
                regime_id = np.argmax(proba)
                confidence = float(proba[regime_id])
                
                # 置信度拒绝
                is_uncertain = confidence < confidence_threshold
                regime_name = "Uncertain" if is_uncertain else self._get_regime_name(regime_id)
                
                predictions.append({
                    'timestamp': features.index[idx],
                    'regime_id': regime_id,
                    'regime_name': regime_name,
                    'confidence': confidence,
                    'is_uncertain': is_uncertain,
                    'original_regime': self._get_regime_name(regime_id)
                })
            
            if not predictions:
                logger.warning("没有生成任何预测结果")
                return pd.DataFrame(columns=['timestamp', 'regime_id', 'regime_name', 'confidence', 'is_uncertain', 'original_regime'])
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            logger.error(f"获取历史状态失败: {e}", exc_info=True)
            raise


class MultiTimeframeRegimePredictor:
    """多时间框架市场状态预测器
    
    同时加载和预测多个时间框架的 regime，例如同时返回 5m 和 15m 的 regime。
    """
    
    def __init__(self, symbol: str, config: TrainingConfig, timeframes: list = None):
        """
        初始化
        
        Args:
            symbol: 交易对
            config: 配置
            timeframes: 要加载的时间框架列表（如 ["5m", "15m"]），如果为 None 则使用 ENABLED_MODELS
        """
        self.symbol = symbol
        self.config = config
        
        if timeframes is None:
            timeframes = config.ENABLED_MODELS
        
        self.timeframes = timeframes
        self.predictors = {}
        
        # 为每个时间框架创建预测器
        for tf in timeframes:
            try:
                self.predictors[tf] = RealtimeRegimePredictor(symbol, config, primary_timeframe=tf)
                logger.info(f"已加载 {symbol} 的 {tf} 模型")
            except FileNotFoundError as e:
                logger.warning(f"无法为 {symbol} 加载 {tf} 模型: {e}")
    
    def get_current_regimes(self) -> Dict:
        """
        获取所有时间框架的当前市场状态
        
        Returns:
            {timeframe: prediction_result} 格式的字典
        """
        results = {
            'symbol': self.symbol,
            'timestamp': datetime.now(),
            'regimes': {}
        }
        
        for tf, predictor in self.predictors.items():
            try:
                results['regimes'][tf] = predictor.get_current_regime()
            except Exception as e:
                logger.error(f"获取 {self.symbol} 的 {tf} 状态失败: {e}")
                results['regimes'][tf] = {'error': str(e)}
        
        return results
    
    def get_regime(self, timeframe: str) -> Dict:
        """
        获取指定时间框架的当前市场状态
        
        Args:
            timeframe: 时间框架（如 "5m" 或 "15m"）
            
        Returns:
            预测结果字典
        """
        if timeframe not in self.predictors:
            raise ValueError(f"时间框架 {timeframe} 未加载，可用的时间框架: {list(self.predictors.keys())}")
        
        return self.predictors[timeframe].get_current_regime()
    
    def has_timeframe(self, timeframe: str) -> bool:
        """检查是否加载了指定时间框架的模型"""
        return timeframe in self.predictors


class MultiSymbolRegimeTracker:
    """多交易对市场状态跟踪器"""
    
    def __init__(self, symbols: list, config: TrainingConfig, primary_timeframe: str = None):
        """
        初始化
        
        Args:
            symbols: 交易对列表
            config: 配置
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
        """
        self.symbols = symbols
        self.config = config
        self.primary_timeframe = primary_timeframe
        self.predictors = {}
        
        # 为每个交易对创建预测器
        for symbol in symbols:
            try:
                self.predictors[symbol] = RealtimeRegimePredictor(symbol, config, primary_timeframe)
            except FileNotFoundError as e:
                logger.warning(f"无法为 {symbol} 创建预测器: {e}")
    
    def get_all_regimes(self) -> Dict:
        """
        获取所有交易对的当前市场状态
        
        Returns:
            交易对到预测结果的字典
        """
        results = {}
        
        for symbol, predictor in self.predictors.items():
            try:
                results[symbol] = predictor.get_current_regime()
            except Exception as e:
                logger.error(f"获取 {symbol} 状态失败: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def get_regime_summary(self) -> pd.DataFrame:
        """
        获取市场状态摘要
        
        Returns:
            摘要 DataFrame
        """
        results = self.get_all_regimes()
        
        summary = []
        for symbol, result in results.items():
            if 'error' not in result:
                summary.append({
                    'symbol': symbol,
                    'regime': result['regime_name'],
                    'confidence': result['confidence'],
                    'timestamp': result['timestamp']
                })
        
        return pd.DataFrame(summary)


def main():
    """主函数 - 示例用法"""
    # 单个交易对预测
    predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)
    
    # 获取当前状态
    current = predictor.get_current_regime()
    print("\n当前市场状态:")
    print(f"交易对: {current['symbol']}")
    print(f"状态: {current['regime_name']}")
    print(f"置信度: {current['confidence']:.2%}")
    print("\n所有状态概率:")
    for regime, prob in current['probabilities'].items():
        print(f"  {regime}: {prob:.2%}")
    
    # 获取历史状态
    history = predictor.get_regime_history(lookback_hours=24)
    print(f"\n最近24小时状态变化:")
    print(history.tail(10))
    
    # 多交易对跟踪
    tracker = MultiSymbolRegimeTracker(
        symbols=["BTCUSDT", "ETHUSDT"],
        config=TrainingConfig
    )
    
    summary = tracker.get_regime_summary()
    print("\n市场状态摘要:")
    print(summary)


if __name__ == "__main__":
    main()
