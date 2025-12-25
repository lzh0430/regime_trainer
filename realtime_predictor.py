"""
实时推理模块 - 用于实时市场状态预测
"""
import logging
import os
from datetime import datetime
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from config import TrainingConfig
from data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngineer
from lstm_trainer import LSTMRegimeClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealtimeRegimePredictor:
    """实时市场状态预测器"""
    
    def __init__(self, symbol: str, config: TrainingConfig):
        """
        初始化
        
        Args:
            symbol: 交易对
            config: 配置
        """
        self.symbol = symbol
        self.config = config
        self.data_fetcher = BinanceDataFetcher()
        self.feature_engineer = FeatureEngineer(cache_manager=self.data_fetcher.cache_manager)
        
        # 加载模型
        model_path = config.get_model_path(symbol)
        scaler_path = config.get_scaler_path(symbol)
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"模型文件不存在，请先训练模型: {model_path}, {scaler_path}"
            )
        
        self.lstm_classifier = LSTMRegimeClassifier.load(model_path, scaler_path)
        logger.info(f"已加载 {symbol} 的模型")
    
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
    
    def get_current_regime(self) -> Dict:
        """
        获取当前市场状态
        
        优化说明：
        - 只获取最新的少量数据（最后几小时），而不是7天
        - 与缓存的特征合并（如果可用）
        - 只重新计算最新数据的特征，避免重复计算
        
        Returns:
            包含预测结果的字典
        """
        try:
            # 1. 获取最新数据
            # 优化：只获取最后几小时的数据，而不是7天
            # LSTM 只需要 sequence_length (64) 行数据，加上技术指标的窗口期（最大200行）
            # 所以只需要获取最后约 300 行数据（约2-3天）
            # 但为了安全，我们获取最近3天的数据
            days = 3
            
            data = self.data_fetcher.fetch_latest_data(
                symbol=self.symbol,
                timeframes=self.config.TIMEFRAMES,
                days=days
            )
            
            # 检查数据是否足够
            primary_df = data.get(self.config.PRIMARY_TIMEFRAME, pd.DataFrame())
            if len(primary_df) < self.lstm_classifier.sequence_length:
                logger.warning(
                    f"数据量不足：只有 {len(primary_df)} 行，需要至少 {self.lstm_classifier.sequence_length} 行。"
                    f"建议获取更多历史数据或等待数据积累。"
                )
            
            # 2. 计算特征
            features = self.feature_engineer.combine_timeframe_features(
                data,
                primary_timeframe=self.config.PRIMARY_TIMEFRAME,
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
                        f"建议获取更多历史数据（至少 {max(200, self.lstm_classifier.sequence_length * 2)} 行）。"
                    )
            
            # 3. 准备推理数据
            X = self.lstm_classifier.prepare_live_data(features)
            
            # 4. 预测
            proba = self.lstm_classifier.predict_proba(X)[0]
            regime_id = np.argmax(proba)
            regime_name = self.config.REGIME_NAMES.get(regime_id, f"State_{regime_id}")
            
            # 5. 返回结果
            result = {
                'symbol': self.symbol,
                'timestamp': datetime.now(),
                'regime_id': int(regime_id),
                'regime_name': regime_name,
                'confidence': float(proba[regime_id]),
                'probabilities': {
                    self.config.REGIME_NAMES.get(i, f"State_{i}"): float(p)
                    for i, p in enumerate(proba)
                }
            }
            
            logger.info(f"{self.symbol} 当前状态: {regime_name} (置信度: {proba[regime_id]:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}", exc_info=True)
            raise
    
    def get_regime_history(self, lookback_hours: int = 24) -> pd.DataFrame:
        """
        获取历史市场状态（滑动窗口预测）
        
        Args:
            lookback_hours: 回看小时数
            
        Returns:
            包含历史预测的 DataFrame
        """
        try:
            # 获取足够的历史数据
            days = max(7, lookback_hours // 24 + 1)
            
            data = self.data_fetcher.fetch_latest_data(
                symbol=self.symbol,
                timeframes=self.config.TIMEFRAMES,
                days=days
            )
            
            # 计算特征
            features = self.feature_engineer.combine_timeframe_features(
                data,
                primary_timeframe=self.config.PRIMARY_TIMEFRAME,
                symbol=self.symbol
            )
            
            # 只取最近的数据（根据时间框架计算需要的行数）
            # 主时间框架是 15m，所以 24 小时 = 24 * 60 / 15 = 96 行
            # 但为了安全，我们获取更多数据以确保有足够的数据进行预测
            timeframe_minutes = self._get_timeframe_minutes(self.config.PRIMARY_TIMEFRAME)
            rows_needed = (lookback_hours * 60) // timeframe_minutes
            # 加上 sequence_length 以确保有足够的数据进行滑动窗口预测
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
                return pd.DataFrame(columns=['timestamp', 'regime_id', 'regime_name', 'confidence'])
            
            # 滑动窗口预测
            predictions = []
            features_scaled = self.lstm_classifier.scaler.transform(features)
            
            for i in range(sequence_length, len(features_scaled)):
                X = np.array([features_scaled[i-sequence_length:i]])
                proba = self.lstm_classifier.predict_proba(X)[0]
                regime_id = np.argmax(proba)
                
                predictions.append({
                    'timestamp': features.index[i],
                    'regime_id': regime_id,
                    'regime_name': self.config.REGIME_NAMES.get(regime_id, f"State_{regime_id}"),
                    'confidence': proba[regime_id]
                })
            
            if not predictions:
                logger.warning("没有生成任何预测结果")
                return pd.DataFrame(columns=['timestamp', 'regime_id', 'regime_name', 'confidence'])
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            logger.error(f"获取历史状态失败: {e}", exc_info=True)
            raise


class MultiSymbolRegimeTracker:
    """多交易对市场状态跟踪器"""
    
    def __init__(self, symbols: list, config: TrainingConfig):
        """
        初始化
        
        Args:
            symbols: 交易对列表
            config: 配置
        """
        self.symbols = symbols
        self.config = config
        self.predictors = {}
        
        # 为每个交易对创建预测器
        for symbol in symbols:
            try:
                self.predictors[symbol] = RealtimeRegimePredictor(symbol, config)
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
