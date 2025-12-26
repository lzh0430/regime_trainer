"""
模型 API 模块 - 为其他项目提供简单的预测接口

这个模块提供了简单的接口，让其他项目可以方便地：
1. 预测未来N根K线的market regime概率分布
2. 获取模型元数据（状态映射、时间框架等）
3. 查询可用的交易对和模型信息
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config import TrainingConfig, setup_logging
from realtime_predictor import RealtimeRegimePredictor

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAPI:
    """
    模型 API - 提供简单的预测接口
    
    使用示例:
        api = ModelAPI()
        
        # 预测未来6根15分钟K线的market regime
        result = api.predict_future_regimes(
            symbol="BTCUSDT",
            timeframe="15m",
            n_bars=6
        )
        
        # 获取模型元数据
        metadata = api.get_model_metadata("BTCUSDT")
    """
    
    def __init__(self, config: TrainingConfig = None):
        """
        初始化 API
        
        Args:
            config: 训练配置，如果为None则使用默认配置
        """
        self.config = config or TrainingConfig
        self._predictors = {}  # 缓存预测器，避免重复加载模型
    
    def _get_predictor(self, symbol: str) -> RealtimeRegimePredictor:
        """
        获取或创建预测器（带缓存）
        
        Args:
            symbol: 交易对
            
        Returns:
            预测器实例
        """
        if symbol not in self._predictors:
            try:
                self._predictors[symbol] = RealtimeRegimePredictor(symbol, self.config)
            except FileNotFoundError as e:
                logger.error(f"无法加载 {symbol} 的模型: {e}")
                raise ValueError(f"模型文件不存在，请先训练 {symbol} 的模型")
        
        return self._predictors[symbol]
    
    def predict_future_regimes(
        self,
        symbol: str,
        timeframe: str = "15m",
        n_bars: int = 6
    ) -> Dict:
        """
        预测未来N根K线的market regime概率分布
        
        注意：LSTM模型训练时预测的是"下一根K线"的状态。
        对于未来N根K线的预测，我们返回当前预测的状态概率分布，
        这代表了基于当前市场状态，未来N根K线最可能的market regime。
        
        Args:
            symbol: 交易对（如 "BTCUSDT"）
            timeframe: 时间框架（如 "15m", "1h"），必须与训练时的主时间框架一致
            n_bars: 要预测的K线数量（默认6根）
            
        Returns:
            包含预测结果的字典:
            {
                'symbol': str,  # 交易对
                'timeframe': str,  # 时间框架
                'n_bars': int,  # 预测的K线数量
                'timestamp': datetime,  # 预测时间
                'regime_probabilities': {  # 各状态的概率分布
                    'State_0': float,  # 或语义名称如 'Strong_Trend'
                    'State_1': float,
                    ...
                },
                'most_likely_regime': {  # 最可能的状态
                    'id': int,
                    'name': str,
                    'probability': float
                },
                'model_info': {  # 模型信息
                    'primary_timeframe': str,
                    'n_states': int,
                    'regime_mapping': dict
                }
            }
        """
        # 验证时间框架
        if timeframe != self.config.PRIMARY_TIMEFRAME:
            logger.warning(
                f"请求的时间框架 {timeframe} 与训练时的主时间框架 "
                f"{self.config.PRIMARY_TIMEFRAME} 不一致。"
                f"将使用训练时的主时间框架 {self.config.PRIMARY_TIMEFRAME}"
            )
            timeframe = self.config.PRIMARY_TIMEFRAME
        
        # 获取预测器
        predictor = self._get_predictor(symbol)
        
        # 获取当前市场状态预测
        current_regime = predictor.get_current_regime()
        
        # 提取概率分布
        regime_probs = current_regime['probabilities']
        
        # 找到最可能的状态
        most_likely_id = current_regime['regime_id']
        most_likely_name = current_regime['regime_name']
        most_likely_prob = current_regime['confidence']
        
        # 获取模型元数据
        model_info = self._get_model_info(predictor)
        
        # 构建结果
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'n_bars': n_bars,
            'timestamp': datetime.now(),
            'regime_probabilities': regime_probs,
            'most_likely_regime': {
                'id': int(most_likely_id),
                'name': most_likely_name,
                'probability': float(most_likely_prob)
            },
            'confidence': float(current_regime['confidence']),
            'is_uncertain': current_regime.get('is_uncertain', False),
            'model_info': model_info
        }
        
        logger.info(
            f"{symbol} 未来{n_bars}根{timeframe}K线预测: "
            f"{most_likely_name} (概率: {most_likely_prob:.2%})"
        )
        
        return result
    
    def get_model_metadata(self, symbol: str) -> Dict:
        """
        获取模型元数据
        
        Args:
            symbol: 交易对
            
        Returns:
            模型元数据字典:
            {
                'symbol': str,
                'primary_timeframe': str,
                'n_states': int,
                'regime_mapping': dict,  # {state_id: regime_name}
                'regime_names': list,  # 所有状态名称列表
                'model_paths': {
                    'lstm': str,
                    'hmm': str,
                    'scaler': str
                },
                'training_info': {
                    'sequence_length': int,
                    'feature_count': int
                }
            }
        """
        predictor = self._get_predictor(symbol)
        model_info = self._get_model_info(predictor)
        
        # 获取模型路径
        model_paths = {
            'lstm': self.config.get_model_path(symbol, 'lstm'),
            'hmm': self.config.get_hmm_path(symbol),
            'scaler': self.config.get_scaler_path(symbol)
        }
        
        # 获取训练信息
        training_info = {
            'sequence_length': predictor.lstm_classifier.sequence_length,
            'feature_count': len(predictor.lstm_classifier.feature_names_) 
                if predictor.lstm_classifier.feature_names_ else None
        }
        
        result = {
            'symbol': symbol,
            'primary_timeframe': model_info['primary_timeframe'],
            'n_states': model_info['n_states'],
            'regime_mapping': model_info['regime_mapping'],
            'regime_names': list(model_info['regime_mapping'].values()),
            'model_paths': model_paths,
            'training_info': training_info
        }
        
        return result
    
    def _get_model_info(self, predictor: RealtimeRegimePredictor) -> Dict:
        """
        从预测器提取模型信息
        
        Args:
            predictor: 预测器实例
            
        Returns:
            模型信息字典
        """
        regime_mapping = predictor.regime_mapping or {}
        
        # 如果没有映射，使用默认状态名称
        if not regime_mapping:
            n_states = predictor.lstm_classifier.n_states
            regime_mapping = {i: f"State_{i}" for i in range(n_states)}
        
        return {
            'primary_timeframe': self.config.PRIMARY_TIMEFRAME,
            'n_states': predictor.lstm_classifier.n_states,
            'regime_mapping': regime_mapping
        }
    
    def list_available_models(self) -> List[str]:
        """
        列出所有可用的模型（已训练的交易对）
        
        Returns:
            交易对列表
        """
        available = []
        
        for symbol in self.config.SYMBOLS:
            model_path = self.config.get_model_path(symbol, 'lstm')
            scaler_path = self.config.get_scaler_path(symbol)
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                available.append(symbol)
        
        return available
    
    def batch_predict(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        n_bars: int = 6
    ) -> Dict[str, Dict]:
        """
        批量预测多个交易对
        
        Args:
            symbols: 交易对列表
            timeframe: 时间框架
            n_bars: 要预测的K线数量
            
        Returns:
            {symbol: prediction_result} 字典
        """
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.predict_future_regimes(
                    symbol=symbol,
                    timeframe=timeframe,
                    n_bars=n_bars
                )
            except Exception as e:
                logger.error(f"预测 {symbol} 失败: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def get_regime_probability(
        self,
        symbol: str,
        regime_name: str,
        timeframe: str = "15m",
        n_bars: int = 6
    ) -> float:
        """
        获取特定状态的概率（便捷方法）
        
        Args:
            symbol: 交易对
            regime_name: 状态名称（如 "Strong_Trend"）
            timeframe: 时间框架
            n_bars: 要预测的K线数量
            
        Returns:
            该状态的概率（0.0-1.0）
        """
        result = self.predict_future_regimes(
            symbol=symbol,
            timeframe=timeframe,
            n_bars=n_bars
        )
        
        regime_probs = result['regime_probabilities']
        
        # 尝试直接匹配
        if regime_name in regime_probs:
            return regime_probs[regime_name]
        
        # 尝试不区分大小写匹配
        for name, prob in regime_probs.items():
            if name.lower() == regime_name.lower():
                return prob
        
        # 如果找不到，返回0.0
        logger.warning(f"未找到状态 '{regime_name}'，可用状态: {list(regime_probs.keys())}")
        return 0.0


# ==================== 便捷函数 ====================

def predict_regime(
    symbol: str,
    timeframe: str = "15m",
    n_bars: int = 6,
    config: TrainingConfig = None
) -> Dict:
    """
    便捷函数：预测未来N根K线的market regime
    
    Args:
        symbol: 交易对
        timeframe: 时间框架
        n_bars: 要预测的K线数量
        config: 配置（可选）
        
    Returns:
        预测结果字典
        
    示例:
        result = predict_regime("BTCUSDT", "15m", 6)
        print(result['most_likely_regime']['name'])
        print(result['regime_probabilities'])
    """
    api = ModelAPI(config)
    return api.predict_future_regimes(symbol, timeframe, n_bars)


def get_regime_probability(
    symbol: str,
    regime_name: str,
    timeframe: str = "15m",
    n_bars: int = 6,
    config: TrainingConfig = None
) -> float:
    """
    便捷函数：获取特定状态的概率
    
    Args:
        symbol: 交易对
        regime_name: 状态名称
        timeframe: 时间框架
        n_bars: 要预测的K线数量
        config: 配置（可选）
        
    Returns:
        该状态的概率（0.0-1.0）
        
    示例:
        prob = get_regime_probability("BTCUSDT", "Strong_Trend")
        print(f"Strong_Trend 概率: {prob:.2%}")
    """
    api = ModelAPI(config)
    return api.get_regime_probability(symbol, regime_name, timeframe, n_bars)


# ==================== 主函数（示例） ====================

def main():
    """示例用法"""
    api = ModelAPI()
    
    # 列出可用的模型
    available = api.list_available_models()
    print(f"\n可用的模型: {available}")
    
    if not available:
        print("\n⚠️  没有可用的模型，请先训练模型")
        return
    
    # 使用第一个可用的交易对
    symbol = available[0]
    
    # 预测未来6根15分钟K线的market regime
    print(f"\n预测 {symbol} 未来6根15分钟K线的market regime:")
    print("=" * 70)
    
    result = api.predict_future_regimes(symbol, "15m", 6)
    
    print(f"交易对: {result['symbol']}")
    print(f"时间框架: {result['timeframe']}")
    print(f"预测K线数: {result['n_bars']}")
    print(f"预测时间: {result['timestamp']}")
    print(f"\n最可能的状态: {result['most_likely_regime']['name']}")
    print(f"概率: {result['most_likely_regime']['probability']:.2%}")
    print(f"置信度: {result['confidence']:.2%}")
    
    print(f"\n所有状态概率分布:")
    print("-" * 70)
    for regime_name, prob in sorted(
        result['regime_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "█" * int(prob * 50)
        print(f"{regime_name:25s} {prob:6.2%} {bar}")
    
    # 获取模型元数据
    print(f"\n模型元数据:")
    print("=" * 70)
    metadata = api.get_model_metadata(symbol)
    print(f"状态数量: {metadata['n_states']}")
    print(f"状态映射: {metadata['regime_mapping']}")
    print(f"主时间框架: {metadata['primary_timeframe']}")
    
    # 使用便捷函数
    print(f"\n使用便捷函数:")
    print("=" * 70)
    prob = get_regime_probability(symbol, "Strong_Trend")
    print(f"Strong_Trend 概率: {prob:.2%}")


if __name__ == "__main__":
    main()

