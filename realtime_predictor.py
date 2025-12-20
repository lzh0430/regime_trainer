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
        self.feature_engineer = FeatureEngineer()
        
        # 加载模型
        model_path = config.get_model_path(symbol)
        scaler_path = config.get_scaler_path(symbol)
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"模型文件不存在，请先训练模型: {model_path}, {scaler_path}"
            )
        
        self.lstm_classifier = LSTMRegimeClassifier.load(model_path, scaler_path)
        logger.info(f"已加载 {symbol} 的模型")
    
    def get_current_regime(self) -> Dict:
        """
        获取当前市场状态
        
        Returns:
            包含预测结果的字典
        """
        try:
            # 1. 获取最新数据（需要足够的历史数据来构建序列）
            # 增加 buffer 以确保有足够数据计算技术指标
            days = max(7, self.config.INCREMENTAL_TRAIN_DAYS // 4)
            
            data = self.data_fetcher.fetch_latest_data(
                symbol=self.symbol,
                timeframes=self.config.TIMEFRAMES,
                days=days
            )
            
            # 2. 计算特征
            features = self.feature_engineer.combine_timeframe_features(
                data,
                primary_timeframe=self.config.PRIMARY_TIMEFRAME
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
                primary_timeframe=self.config.PRIMARY_TIMEFRAME
            )
            
            # 只取最近的数据
            features = features.tail(lookback_hours)
            
            # 滑动窗口预测
            predictions = []
            sequence_length = self.lstm_classifier.sequence_length
            
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
