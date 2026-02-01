"""
模型 API 模块 - 为其他项目提供简单的预测接口

这个模块提供了简单的接口，让其他项目可以方便地：
1. 预测未来多根K线的market regime概率分布（t+1 到 t+4）
2. 获取历史regime序列（过去N根K线）
3. 获取模型元数据（状态映射、时间框架等）
4. 查询可用的交易对和模型信息

注意：模型支持多步预测（t+1 到 t+4），同时提供历史regime序列。
"""
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config import TrainingConfig, setup_logging
from realtime_predictor import RealtimeRegimePredictor, MultiTimeframeRegimePredictor
from model_registry import get_prod_info, set_prod, list_versions
from forward_testing import trigger_all_pending_forward_tests, ForwardTestCronManager, get_campaign_accuracy
from config_registry import (
    init_from_config_file, create_config_version, get_config_version,
    list_config_versions, update_config_version, delete_config_version,
    get_config_for_model, get_models_for_config, get_config_or_default,
    config_dict_to_object, get_default_config
)
from training_pipeline import TrainingPipeline

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAPI:
    """
    模型 API - 提供简单的预测接口
    
    使用示例:
        api = ModelAPI()
        
        # 预测下一根15分钟K线的market regime
        result = api.predict_next_regime(
            symbol="BTCUSDT",
            timeframe="15m"
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
        self._predictors = {}  # 缓存预测器，避免重复加载模型 {(symbol, timeframe): predictor}
        self._predictors_by_version = {}  # {(symbol, timeframe, version_id): predictor}
        self._multi_tf_predictors = {}  # 缓存多时间框架预测器 {symbol: predictor}
    
    def _get_predictor(self, symbol: str, primary_timeframe: str = None) -> RealtimeRegimePredictor:
        """
        获取或创建预测器（带缓存）
        
        Args:
            symbol: 交易对
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
            
        Returns:
            预测器实例
        """
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        cache_key = (symbol, primary_timeframe)
        
        if cache_key not in self._predictors:
            try:
                self._predictors[cache_key] = RealtimeRegimePredictor(
                    symbol, self.config, primary_timeframe
                )
            except FileNotFoundError as e:
                logger.error(f"无法加载 {symbol} ({primary_timeframe}) 的模型: {e}")
                raise ValueError(f"模型文件不存在，请先训练 {symbol} 的 {primary_timeframe} 模型")
        
        return self._predictors[cache_key]
    
    def _get_predictor_for_version(self, symbol: str, primary_timeframe: str, version_id: str) -> RealtimeRegimePredictor:
        """获取指定版本的预测器（用于 forward testing），缓存键 (symbol, timeframe, version_id)。"""
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        cache_key = (symbol, primary_timeframe, version_id)
        if cache_key not in self._predictors_by_version:
            try:
                self._predictors_by_version[cache_key] = RealtimeRegimePredictor(
                    symbol, self.config, primary_timeframe, version_id=version_id
                )
            except FileNotFoundError as e:
                logger.error(f"无法加载 {symbol} ({primary_timeframe}) 版本 {version_id} 的模型: {e}")
                raise ValueError(f"模型文件不存在: {symbol} {primary_timeframe} version {version_id}")
        return self._predictors_by_version[cache_key]
    
    def _get_multi_tf_predictor(self, symbol: str, timeframes: list = None) -> MultiTimeframeRegimePredictor:
        """
        获取或创建多时间框架预测器（带缓存）
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表
            
        Returns:
            多时间框架预测器实例
        """
        if timeframes is None:
            timeframes = self.config.ENABLED_MODELS
        
        cache_key = symbol
        
        if cache_key not in self._multi_tf_predictors:
            self._multi_tf_predictors[cache_key] = MultiTimeframeRegimePredictor(
                symbol, self.config, timeframes
            )
        
        return self._multi_tf_predictors[cache_key]
    
    def predict_regimes(
        self,
        symbol: str,
        primary_timeframe: str = None,
        include_history: bool = True,
        history_bars: int = 16
    ) -> Dict:
        """
        预测未来多根 K 线的 market regime 概率分布（支持 t+1 到 t+4）
        
        这是推荐的新 API 方法，同时返回：
        - 历史 regime 序列（过去 N 根 K 线的状态）
        - 未来 4 步预测（t+1 到 t+4 的概率分布）
        
        Args:
            symbol: 交易对（如 "BTCUSDT"）
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
            include_history: 是否包含历史 regime 序列
            history_bars: 历史序列的 K 线数量（默认 16 根，对于 15m 约为 4 小时）
            
        Returns:
            包含预测结果的字典:
            {
                'symbol': str,
                'timeframe': str,
                'timestamp': datetime,
                'historical_regimes': {  # 历史 regime 序列
                    'sequence': ['Range', 'Range', 'Weak_Trend', ...],
                    'timestamps': [...],
                    'confidences': [...],
                    'count': 16,
                    'lookback_hours': 4.0
                },
                'predictions': {  # 多步预测
                    't+1': {
                        'probabilities': {...},
                        'most_likely': str,
                        'confidence': float,
                        'is_uncertain': bool
                    },
                    't+2': {...},
                    't+3': {...},
                    't+4': {...}
                },
                'is_multistep': bool,  # 是否多步模型
                'model_info': {...}
            }
        """
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        # 获取预测器
        predictor = self._get_predictor(symbol, primary_timeframe)
        
        # 获取完整预测结果（包括多步预测和历史序列）
        current_regime = predictor.get_current_regime()
        
        # 获取模型元数据
        model_info = self._get_model_info(predictor)
        model_info['sequence_length'] = predictor.lstm_classifier.sequence_length
        model_info['is_multistep'] = True  # 现在总是多步预测
        model_info['prediction_horizons'] = predictor.lstm_classifier.prediction_horizons
        
        # 构建预测结果
        predictions = {}
        for horizon, pred in current_regime.get('predictions', {}).items():
            predictions[horizon] = {
                'probabilities': pred['probabilities'],
                'most_likely': pred['regime_name'],
                'regime_id': pred['regime_id'],
                'confidence': pred['confidence'],
                'is_uncertain': pred['is_uncertain']
            }
        
        # 构建结果
        result = {
            'symbol': symbol,
            'timeframe': primary_timeframe,
            'timestamp': datetime.now(),
            'predictions': predictions,
            'is_multistep': True,  # 现在总是多步预测
            'model_info': model_info
        }
        
        # 添加历史 regime 序列
        if include_history:
            result['historical_regimes'] = current_regime.get('historical_regimes', {})
        
        # 日志输出
        if predictions.get('t+1'):
            t1 = predictions['t+1']
            logger.info(
                f"{symbol} 多步预测: t+1={t1['most_likely']} ({t1['confidence']:.2%})"
            )
            for h in ['t+2', 't+3', 't+4']:
                if h in predictions:
                    p = predictions[h]
                    logger.debug(f"  {h}: {p['most_likely']} ({p['confidence']:.2%})")
        
        return result
    
    def predict_next_regime(
        self,
        symbol: str,
        timeframe: str = None,
        primary_timeframe: str = None
    ) -> Dict:
        """
        预测下一根K线的market regime概率分布（向后兼容的接口）
        
        注意：此方法保留用于向后兼容。推荐使用 predict_regimes() 方法以获取多步预测。
        
        对于多步模型，此方法只返回 t+1 的预测结果。
        
        Args:
            symbol: 交易对（如 "BTCUSDT"）
            timeframe: [已废弃] 使用 primary_timeframe 代替
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
            
        Returns:
            包含预测结果的字典（只包含 t+1 预测，向后兼容格式）
        """
        # 处理 timeframe 参数（向后兼容）
        if primary_timeframe is None:
            if timeframe is not None and timeframe in self.config.MODEL_CONFIGS:
                primary_timeframe = timeframe
            else:
                primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        # 获取预测器
        predictor = self._get_predictor(symbol, primary_timeframe)
        timeframe = primary_timeframe  # 用于返回结果
        
        # 获取当前市场状态预测
        current_regime = predictor.get_current_regime()
        
        # 提取概率分布（t+1）
        regime_probs = current_regime['probabilities']
        
        # 找到最可能的状态
        most_likely_id = current_regime['regime_id']
        most_likely_name = current_regime['regime_name']
        most_likely_prob = current_regime['confidence']
        
        # 获取模型元数据
        model_info = self._get_model_info(predictor)
        model_info['sequence_length'] = predictor.lstm_classifier.sequence_length
        
        # 构建结果（向后兼容格式）
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
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
            f"{symbol} 下一根{timeframe}K线预测: "
            f"{most_likely_name} (概率: {most_likely_prob:.2%})"
        )
        
        return result
    
    def predict_next_regime_for_version(
        self,
        symbol: str,
        primary_timeframe: str,
        version_id: str
    ) -> Dict:
        """
        使用指定版本模型预测下一根K线的 regime（用于 forward testing）。
        返回格式与 predict_next_regime 相同。
        """
        predictor = self._get_predictor_for_version(symbol, primary_timeframe, version_id)
        tf = primary_timeframe
        current_regime = predictor.get_current_regime()
        regime_probs = current_regime['probabilities']
        most_likely_id = current_regime['regime_id']
        most_likely_name = current_regime['regime_name']
        most_likely_prob = current_regime['confidence']
        model_info = self._get_model_info(predictor)
        model_info['sequence_length'] = predictor.lstm_classifier.sequence_length
        return {
            'symbol': symbol,
            'timeframe': tf,
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
    
    # 保持向后兼容（已废弃，建议使用 predict_next_regime）
    def predict_future_regimes(
        self,
        symbol: str,
        timeframe: str = "15m",
        n_bars: int = 1
    ) -> Dict:
        """
        [已废弃] 预测未来N根K线的market regime概率分布
        
        注意：此方法已废弃。请使用 predict_regimes() 方法以获取多步预测（t+1 到 t+4）。
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            n_bars: 已废弃，将被忽略
            
        Returns:
            预测结果（只返回 t+1 预测）
        """
        if n_bars != 1:
            logger.warning(
                f"predict_future_regimes() 已废弃。"
                f"请使用 predict_regimes() 方法以获取多步预测（t+1 到 t+4）。"
            )
        
        return self.predict_next_regime(symbol, timeframe)
    
    def get_model_metadata(self, symbol: str, primary_timeframe: str = None) -> Dict:
        """
        获取模型元数据
        
        Args:
            symbol: 交易对
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
            
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
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        predictor = self._get_predictor(symbol, primary_timeframe)
        model_info = self._get_model_info(predictor)
        
        # 获取模型路径（PROD 版本）
        model_paths = {
            'lstm': self.config.get_prod_model_path(symbol, 'lstm', primary_timeframe),
            'hmm': self.config.get_prod_hmm_path(symbol, primary_timeframe),
            'scaler': self.config.get_prod_scaler_path(symbol, primary_timeframe)
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
            'primary_timeframe': predictor.primary_timeframe,
            'n_states': predictor.lstm_classifier.n_states,
            'regime_mapping': regime_mapping
        }
    
    def predict_multi_timeframe_regimes(
        self,
        symbol: str,
        timeframes: List[str] = None,
        include_history: bool = True
    ) -> Dict:
        """
        同时预测多个时间框架的 market regime（多步预测）
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表（如 ["5m", "15m"]），如果为 None 则使用 ENABLED_MODELS
            include_history: 是否包含历史regime序列
            
        Returns:
            包含多个时间框架预测结果的字典:
            {
                'symbol': str,
                'timestamp': datetime,
                'regimes': {
                    '5m': {...多步预测结果...},
                    '15m': {...多步预测结果...}
                }
            }
        """
        if timeframes is None:
            timeframes = self.config.ENABLED_MODELS
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'regimes': {}
        }
        
        for tf in timeframes:
            try:
                result = self.predict_regimes(
                    symbol=symbol,
                    primary_timeframe=tf,
                    include_history=include_history
                )
                results['regimes'][tf] = result
            except Exception as e:
                logger.error(f"预测 {symbol} 的 {tf} regime 失败: {e}")
                results['regimes'][tf] = {'error': str(e)}
        
        return results
    
    def predict_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = None
    ) -> Dict:
        """
        [已废弃] 同时预测多个时间框架的 market regime
        
        注意：此方法已废弃，请使用 predict_multi_timeframe_regimes() 以获取多步预测。
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表（如 ["5m", "15m"]），如果为 None 则使用 ENABLED_MODELS
            
        Returns:
            包含多个时间框架预测结果的字典（只包含 t+1 预测）
        """
        return self.predict_multi_timeframe_regimes(symbol, timeframes, include_history=False)
    
    def list_available_models(self, primary_timeframe: str = None) -> List[str]:
        """
        列出所有可用的模型（已训练的交易对）
        
        Args:
            primary_timeframe: 主时间框架，如果为 None 则检查所有启用的时间框架
            
        Returns:
            交易对列表
        """
        available = []
        
        if primary_timeframe:
            timeframes_to_check = [primary_timeframe]
        else:
            timeframes_to_check = self.config.ENABLED_MODELS
        
        for symbol in self.config.SYMBOLS:
            for tf in timeframes_to_check:
                model_path = self.config.get_prod_model_path(symbol, 'lstm', tf)
                scaler_path = self.config.get_prod_scaler_path(symbol, tf)
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    available.append(symbol)
                    break  # 只要有一个时间框架的模型存在就认为可用
        
        return available
    
    def list_available_models_by_timeframe(self) -> Dict[str, List[str]]:
        """
        列出每个时间框架可用的模型
        
        Returns:
            {timeframe: [symbol, ...]} 格式的字典
        """
        result = {}
        
        for tf in self.config.MODEL_CONFIGS.keys():
            result[tf] = []
            for symbol in self.config.SYMBOLS:
                model_path = self.config.get_prod_model_path(symbol, 'lstm', tf)
                scaler_path = self.config.get_prod_scaler_path(symbol, tf)
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    result[tf].append(symbol)
        
        return result
    
    def batch_predict(
        self,
        symbols: List[str],
        primary_timeframe: str = None
    ) -> Dict[str, Dict]:
        """
        批量预测多个交易对的下一根K线
        
        Args:
            symbols: 交易对列表
            primary_timeframe: 主时间框架
            
        Returns:
            {symbol: prediction_result} 字典
        """
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.predict_next_regime(
                    symbol=symbol,
                    primary_timeframe=primary_timeframe
                )
            except Exception as e:
                logger.error(f"预测 {symbol} 失败: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def get_regime_probability(
        self,
        symbol: str,
        regime_name: str,
        primary_timeframe: str = None
    ) -> float:
        """
        获取下一根K线特定状态的概率（便捷方法）
        
        Args:
            symbol: 交易对
            regime_name: 状态名称（如 "Strong_Trend"）
            primary_timeframe: 主时间框架
            
        Returns:
            该状态的概率（0.0-1.0）
        """
        result = self.predict_next_regime(
            symbol=symbol,
            primary_timeframe=primary_timeframe
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
    
    def get_regime_history(
        self,
        symbol: str,
        lookback_hours: int = None,
        start_date: datetime = None,
        end_date: datetime = None,
        primary_timeframe: str = None
    ) -> Dict:
        """
        获取历史上的 market regime 序列
        
        支持两种查询方式：
        1. 按回看小时数：指定 lookback_hours，从当前时间往前回看
        2. 按日期范围：指定 start_date 和 end_date，获取指定时间范围的regime
        
        Args:
            symbol: 交易对（如 "BTCUSDT"）
            lookback_hours: 回看小时数（如果指定，则从当前时间往前回看）
            start_date: 开始日期时间（如果指定，则获取指定时间范围）
            end_date: 结束日期时间（如果指定，则获取指定时间范围）
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
            
        Returns:
            包含历史regime序列的字典:
            {
                'symbol': str,
                'timeframe': str,
                'lookback_hours': int or None,
                'start_date': datetime or None,
                'end_date': datetime or None,
                'timestamp': datetime,
                'history': [
                    {
                        'timestamp': datetime,
                        'regime_id': int,
                        'regime_name': str,
                        'confidence': float,
                        'is_uncertain': bool,
                        'original_regime': str
                    },
                    ...
                ],
                'count': int
            }
        """
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        # 如果没有指定任何参数，默认回看24小时
        if lookback_hours is None and start_date is None and end_date is None:
            lookback_hours = 24
        
        # 获取预测器
        predictor = self._get_predictor(symbol, primary_timeframe)
        
        # 获取历史regime序列
        history_df = predictor.get_regime_history(
            lookback_hours=lookback_hours,
            start_date=start_date,
            end_date=end_date
        )
        
        # 转换为字典格式
        history_list = []
        for _, row in history_df.iterrows():
            history_list.append({
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'regime_id': int(row['regime_id']),
                'regime_name': str(row['regime_name']),
                'confidence': float(row['confidence']),
                'is_uncertain': bool(row.get('is_uncertain', False)),
                'original_regime': str(row.get('original_regime', row['regime_name']))
            })
        
        result = {
            'symbol': symbol,
            'timeframe': primary_timeframe,
            'lookback_hours': lookback_hours,
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None,
            'timestamp': datetime.now(),
            'history': history_list,
            'count': len(history_list)
        }
        
        if start_date and end_date:
            logger.info(
                f"获取 {symbol} ({primary_timeframe}) 的历史regime: "
                f"{start_date.date()} 至 {end_date.date()}，共 {len(history_list)} 条记录"
            )
        else:
            logger.info(
                f"获取 {symbol} ({primary_timeframe}) 的历史regime: "
                f"回看 {lookback_hours} 小时，共 {len(history_list)} 条记录"
            )
        
        return result


# ==================== 便捷函数 ====================

def predict_regime(
    symbol: str,
    primary_timeframe: str = None,
    config: TrainingConfig = None
) -> Dict:
    """
    便捷函数：预测下一根K线的market regime
    
    Args:
        symbol: 交易对
        primary_timeframe: 主时间框架（如 "5m", "15m"）
        config: 配置（可选）
        
    Returns:
        预测结果字典
        
    示例:
        result = predict_regime("BTCUSDT", "15m")
        print(result['most_likely_regime']['name'])
        print(result['regime_probabilities'])
    """
    api = ModelAPI(config)
    return api.predict_next_regime(symbol, primary_timeframe=primary_timeframe)


def predict_multi_timeframe(
    symbol: str,
    timeframes: List[str] = None,
    config: TrainingConfig = None
) -> Dict:
    """
    便捷函数：同时预测多个时间框架的market regime
    
    Args:
        symbol: 交易对
        timeframes: 时间框架列表（如 ["5m", "15m"]）
        config: 配置（可选）
        
    Returns:
        多时间框架预测结果
        
    示例:
        result = predict_multi_timeframe("BTCUSDT", ["5m", "15m"])
        print(result['regimes']['5m']['most_likely_regime']['name'])
        print(result['regimes']['15m']['most_likely_regime']['name'])
    """
    api = ModelAPI(config)
    return api.predict_multi_timeframe(symbol, timeframes)


def get_regime_probability(
    symbol: str,
    regime_name: str,
    primary_timeframe: str = None,
    config: TrainingConfig = None
) -> float:
    """
    便捷函数：获取下一根K线特定状态的概率
    
    Args:
        symbol: 交易对
        regime_name: 状态名称
        primary_timeframe: 主时间框架
        config: 配置（可选）
        
    Returns:
        该状态的概率（0.0-1.0）
        
    示例:
        prob = get_regime_probability("BTCUSDT", "Strong_Trend")
        print(f"Strong_Trend 概率: {prob:.2%}")
    """
    api = ModelAPI(config)
    return api.get_regime_probability(symbol, regime_name, primary_timeframe)


# ==================== HTTP 服务器（可选） ====================

def create_app(api_instance: ModelAPI = None):
    """
    创建 Flask 应用（可选功能）
    
    如果安装了 flask 和 flask-cors，可以使用此功能提供 HTTP REST API
    
    使用方式:
        python model_api.py --server
        或
        from model_api import create_app
        app = create_app()
        app.run(port=5000)
    """
    try:
        from flask import Flask, jsonify, request
        from flask_cors import CORS
        from flasgger import Swagger
    except ImportError:
        raise ImportError(
            "需要安装 flask、flask-cors 和 flasgger 才能使用 HTTP 服务器功能:\n"
            "pip install flask flask-cors flasgger"
        )
    
    app = Flask(__name__)
    CORS(app)
    
    # Swagger configuration
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/api/docs",
    }
    
    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "Regime Trainer API",
            "description": "API for market regime prediction using LSTM and HMM models",
            "version": "1.0.0",
            "contact": {
                "name": "API Support"
            }
        },
        "basePath": "/api",
        "schemes": ["http", "https"],
        "tags": [
            {"name": "Health", "description": "Health check endpoints"},
            {"name": "Prediction", "description": "Market regime prediction endpoints"},
            {"name": "Models", "description": "Model management endpoints"},
            {"name": "Forward Testing", "description": "Forward testing endpoints"},
            {"name": "History", "description": "Historical data endpoints"},
            {"name": "Config", "description": "Training configuration management endpoints"},
            {"name": "Training", "description": "Model training endpoints"},
        ]
    }
    
    swagger = Swagger(app, config=swagger_config, template=swagger_template)
    
    api = api_instance or ModelAPI()
    
    def datetime_to_str(obj):
        """将 datetime 对象转换为字符串"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: datetime_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [datetime_to_str(item) for item in obj]
        return obj
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """
        健康检查端点
        ---
        tags:
          - Health
        summary: Health check endpoint
        description: Returns the health status of the API
        responses:
          200:
            description: API is healthy
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: healthy
                timestamp:
                  type: string
                  format: date-time
                  example: "2024-01-01T12:00:00"
        """
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/predict/<symbol>', methods=['GET'])
    def predict(symbol: str):
        """
        预测下一根K线的market regime
        ---
        tags:
          - Prediction
        summary: Predict next market regime
        description: Predicts the market regime for the next K-line candle
        parameters:
          - name: symbol
            in: path
            type: string
            required: true
            description: Trading pair symbol (e.g., BTCUSDT)
            example: BTCUSDT
          - name: timeframe
            in: query
            type: string
            required: false
            default: 15m
            enum: [5m, 15m, 1h]
            description: Timeframe for prediction
        responses:
          200:
            description: Successful prediction
            schema:
              type: object
              properties:
                symbol:
                  type: string
                timeframe:
                  type: string
                most_likely_regime:
                  type: object
                  properties:
                    name:
                      type: string
                    probability:
                      type: number
                all_regimes:
                  type: array
                  items:
                    type: object
          400:
            description: Invalid timeframe
          404:
            description: Model not found
          500:
            description: Server error
        """
        try:
            timeframe = request.args.get('timeframe', '15m')
            if timeframe not in api.config.MODEL_CONFIGS.keys():
                return jsonify({'error': f'不支持的时间框架: {timeframe}，支持的值: {list(api.config.MODEL_CONFIGS.keys())}'}), 400
            result = api.predict_next_regime(symbol, primary_timeframe=timeframe)
            # Transform regime_probabilities dict to all_regimes array for frontend compatibility
            if 'regime_probabilities' in result and 'all_regimes' not in result:
                result['all_regimes'] = [
                    {'name': name, 'probability': prob}
                    for name, prob in result['regime_probabilities'].items()
                ]
            return jsonify(datetime_to_str(result))
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"预测失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predict_regimes/<symbol>', methods=['GET'])
    def predict_regimes(symbol: str):
        """
        多步预测（t+1 到 t+4）
        ---
        tags:
          - Prediction
        summary: Multi-step regime prediction
        description: Predicts market regimes for next 4 time steps (t+1 to t+4)
        parameters:
          - name: symbol
            in: path
            type: string
            required: true
            description: Trading pair symbol (e.g., BTCUSDT)
            example: BTCUSDT
          - name: timeframe
            in: query
            type: string
            required: false
            default: 15m
            enum: [5m, 15m, 1h]
            description: Timeframe for prediction
          - name: include_history
            in: query
            type: boolean
            required: false
            default: true
            description: Whether to include historical regime sequence
        responses:
          200:
            description: Successful prediction
            schema:
              type: object
              properties:
                symbol:
                  type: string
                timeframe:
                  type: string
                predictions:
                  type: array
                  items:
                    type: object
                history:
                  type: array
                  items:
                    type: object
          400:
            description: Invalid timeframe
          404:
            description: Model not found
          500:
            description: Server error
        """
        try:
            timeframe = request.args.get('timeframe', '15m')
            include_history = request.args.get('include_history', 'true').lower() == 'true'
            if timeframe not in api.config.MODEL_CONFIGS.keys():
                return jsonify({'error': f'不支持的时间框架: {timeframe}，支持的值: {list(api.config.MODEL_CONFIGS.keys())}'}), 400
            result = api.predict_regimes(symbol, primary_timeframe=timeframe, include_history=include_history)
            
            # Transform predictions object to array format for frontend compatibility
            if 'predictions' in result and isinstance(result['predictions'], dict):
                predictions_array = []
                for horizon, pred_data in result['predictions'].items():
                    # Extract step number from horizon (e.g., 't+1' -> 1)
                    step_num = int(horizon.split('+')[1]) if '+' in horizon else 1
                    # Convert probabilities dict to regimes array
                    regimes = [
                        {'name': name, 'probability': prob}
                        for name, prob in pred_data.get('probabilities', {}).items()
                    ]
                    predictions_array.append({
                        'step': step_num,
                        'horizon': horizon,
                        'regimes': regimes,
                        'most_likely': pred_data.get('most_likely'),
                        'confidence': pred_data.get('confidence'),
                        'is_uncertain': pred_data.get('is_uncertain', False)
                    })
                # Sort by step number
                predictions_array.sort(key=lambda x: x['step'])
                result['predictions'] = predictions_array
            
            # Transform history if present
            if 'historical_regimes' in result and isinstance(result['historical_regimes'], dict):
                hist = result['historical_regimes']
                if 'sequence' in hist:
                    result['history'] = [
                        {
                            'timestamp': hist.get('timestamps', [])[i] if i < len(hist.get('timestamps', [])) else None,
                            'regime': regime
                        }
                        for i, regime in enumerate(hist.get('sequence', []))
                    ]
            
            return jsonify(datetime_to_str(result))
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"多步预测失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/metadata/<symbol>', methods=['GET'])
    def get_metadata(symbol: str):
        """
        获取模型元数据
        ---
        tags:
          - Models
        summary: Get model metadata
        description: Returns metadata for a specific model including regime mappings and configuration
        parameters:
          - name: symbol
            in: path
            type: string
            required: true
            description: Trading pair symbol (e.g., BTCUSDT)
            example: BTCUSDT
          - name: timeframe
            in: query
            type: string
            required: false
            default: 15m
            enum: [5m, 15m, 1h]
            description: Timeframe for the model
        responses:
          200:
            description: Model metadata
            schema:
              type: object
              properties:
                symbol:
                  type: string
                timeframe:
                  type: string
                regime_mapping:
                  type: object
                model_config:
                  type: object
          400:
            description: Invalid timeframe
          404:
            description: Model not found
          500:
            description: Server error
        """
        try:
            timeframe = request.args.get('timeframe', '15m')
            if timeframe not in api.config.MODEL_CONFIGS.keys():
                return jsonify({'error': f'不支持的时间框架: {timeframe}，支持的值: {list(api.config.MODEL_CONFIGS.keys())}'}), 400
            metadata = api.get_model_metadata(symbol, primary_timeframe=timeframe)
            return jsonify(datetime_to_str(metadata))
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"获取元数据失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/available', methods=['GET'])
    def list_available():
        """
        列出可用模型
        ---
        tags:
          - Models
        summary: List available models
        description: Returns a list of all available models, optionally filtered by timeframe
        parameters:
          - name: timeframe
            in: query
            type: string
            required: false
            enum: [5m, 15m, 1h]
            description: Optional timeframe filter
        responses:
          200:
            description: List of available models
            schema:
              type: object
              properties:
                available_models:
                  type: array
                  items:
                    type: string
                count:
                  type: integer
          400:
            description: Invalid timeframe
          500:
            description: Server error
        """
        try:
            timeframe = request.args.get('timeframe')
            if timeframe:
                if timeframe not in api.config.MODEL_CONFIGS.keys():
                    return jsonify({'error': f'不支持的时间框架: {timeframe}，支持的值: {list(api.config.MODEL_CONFIGS.keys())}'}), 400
                models = api.list_available_models(primary_timeframe=timeframe)
            else:
                models = api.list_available_models()
            return jsonify({'available_models': models, 'count': len(models)})
        except Exception as e:
            logger.error(f"列出可用模型失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/by_timeframe', methods=['GET'])
    def list_by_timeframe():
        """
        按时间框架列出模型
        ---
        tags:
          - Models
        summary: List models by timeframe
        description: Returns models grouped by timeframe
        responses:
          200:
            description: Models grouped by timeframe
            schema:
              type: object
              additionalProperties:
                type: array
                items:
                  type: string
          500:
            description: Server error
        """
        try:
            models_by_tf = api.list_available_models_by_timeframe()
            return jsonify(models_by_tf)
        except Exception as e:
            logger.error(f"按时间框架列出模型失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/prod', methods=['GET'])
    def get_prod():
        """
        获取 PROD 指针
        ---
        tags:
          - Models
        summary: Get production version pointer
        description: Returns the production version ID for a given symbol and timeframe
        parameters:
          - name: symbol
            in: query
            type: string
            required: true
            description: Trading pair symbol (e.g., BTCUSDT)
            example: BTCUSDT
          - name: timeframe
            in: query
            type: string
            required: false
            default: 15m
            enum: [5m, 15m, 1h]
            description: Timeframe for the model
        responses:
          200:
            description: Production version information
            schema:
              type: object
              properties:
                symbol:
                  type: string
                timeframe:
                  type: string
                version_id:
                  type: string
                updated_at:
                  type: string
                  format: date-time
                note:
                  type: string
                config_version_id:
                  type: string
                  description: Config version ID used to train this PROD model (if available)
          400:
            description: Missing symbol or invalid timeframe
          500:
            description: Server error
        """
        try:
            symbol = request.args.get('symbol')
            timeframe = request.args.get('timeframe', '15m')
            if not symbol:
                return jsonify({'error': '缺少参数 symbol'}), 400
            if timeframe not in api.config.MODEL_CONFIGS.keys():
                return jsonify({'error': f'不支持的时间框架: {timeframe}'}), 400
            info = get_prod_info(symbol, timeframe, models_dir=api.config.MODELS_DIR)
            if info is None:
                # 无显式 PROD 指针时返回当前生效的版本（latest 或 legacy）
                from model_registry import get_prod_version
                version_id = get_prod_version(symbol, timeframe, models_dir=api.config.MODELS_DIR)
                result = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'version_id': version_id,
                    'updated_at': None,
                    'note': 'fallback (no prod_pointer row)'
                }
                # Add config version if available
                if version_id:
                    config_info = get_config_for_model(version_id, symbol, timeframe)
                    if config_info:
                        result['config_version_id'] = config_info.get('config_version_id')
                return jsonify(result)
            # Add config version to PROD info
            config_info = get_config_for_model(info['version_id'], symbol, timeframe)
            if config_info:
                info['config_version_id'] = config_info.get('config_version_id')
            return jsonify(info)
        except Exception as e:
            logger.error(f"获取 PROD 失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/prod', methods=['POST', 'PUT'])
    def set_prod_route():
        """
        设置 PROD 指针
        ---
        tags:
          - Models
        summary: Set production version pointer
        description: Sets the production version ID for a given symbol and timeframe
        parameters:
          - name: body
            in: body
            required: true
            schema:
              type: object
              required:
                - symbol
                - version_id
              properties:
                symbol:
                  type: string
                  description: Trading pair symbol (e.g., BTCUSDT)
                  example: BTCUSDT
                timeframe:
                  type: string
                  description: Timeframe for the model
                  default: 15m
                  enum: [5m, 15m, 1h]
                version_id:
                  type: string
                  description: Version ID to set as production
                  example: "2024-01-01-1"
        responses:
          200:
            description: Production version information after update
            schema:
              type: object
              properties:
                symbol:
                  type: string
                timeframe:
                  type: string
                version_id:
                  type: string
                updated_at:
                  type: string
                  format: date-time
          400:
            description: Missing required fields or invalid timeframe or path not found
          500:
            description: Server error
        """
        try:
            data = request.get_json()
            if not data or 'symbol' not in data or 'version_id' not in data:
                return jsonify({'error': '请求体必须包含 symbol 和 version_id'}), 400
            symbol = data['symbol']
            timeframe = data.get('timeframe', '15m')
            version_id = data['version_id']
            if timeframe not in api.config.MODEL_CONFIGS.keys():
                return jsonify({'error': f'不支持的时间框架: {timeframe}'}), 400
            ok = set_prod(symbol, timeframe, version_id, models_dir=api.config.MODELS_DIR)
            if not ok:
                return jsonify({'error': f'路径不存在: models/{version_id}/{symbol}/{timeframe}/'}), 400
            info = get_prod_info(symbol, timeframe, models_dir=api.config.MODELS_DIR)
            return jsonify(info)
        except Exception as e:
            logger.error(f"设置 PROD 失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/versions', methods=['GET'])
    def list_versions_route():
        """
        列出所有版本及每个版本包含的 symbol/timeframe；标注 is_prod 和 config_version_id
        ---
        tags:
          - Models
        summary: List all model versions
        description: Returns all registered model versions with their contents (symbols/timeframes), production status, and config version used for training
        responses:
          200:
            description: List of all versions
            schema:
              type: object
              properties:
                versions:
                  type: array
                  items:
                    type: object
                    properties:
                      version_id:
                        type: string
                      created_at:
                        type: string
                        format: date-time
                      contents:
                        type: array
                        items:
                          type: object
                          properties:
                            symbol:
                              type: string
                            timeframe:
                              type: string
                            is_prod:
                              type: boolean
                            config_version_id:
                              type: string
                              description: Config version ID used to train this model (if available)
          500:
            description: Server error
        """
        try:
            versions = list_versions(models_dir=api.config.MODELS_DIR)
            return jsonify({'versions': versions})
        except Exception as e:
            logger.error(f"列出版本失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/forward_test/trigger_all', methods=['POST'])
    def trigger_all_forward_tests():
        """
        触发所有待执行的 forward test（手动触发所有 pending campaigns）
        ---
        tags:
          - Forward Testing
        summary: Trigger all pending forward tests
        description: Manually triggers forward tests for all pending campaigns (active campaigns that still need runs)
        responses:
          200:
            description: Forward test execution summary
            schema:
              type: object
              properties:
                total_campaigns:
                  type: integer
                  description: Total number of pending campaigns
                successful_runs:
                  type: integer
                  description: Number of successful test runs
                failed_runs:
                  type: integer
                  description: Number of failed runs
                skipped_runs:
                  type: integer
                  description: Number of skipped runs
                results:
                  type: array
                  items:
                    type: object
                    properties:
                      campaign_id:
                        type: integer
                      version_id:
                        type: string
                      symbol:
                        type: string
                      timeframe:
                        type: string
                      status:
                        type: string
                        enum: [success, skipped, error]
          500:
            description: Server error
        """
        try:
            # Try to use cron manager if available, otherwise use default
            cron_mgr = ForwardTestCronManager._instance
            if cron_mgr is not None:
                summary = cron_mgr.trigger_all_pending()
            else:
                summary = trigger_all_pending_forward_tests(config=api.config)
            return jsonify(datetime_to_str(summary))
        except Exception as e:
            logger.error(f"触发 forward test 失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/forward_test/accuracy', methods=['GET'])
    def get_campaign_accuracy_route():
        """
        获取 forward test campaign 的准确率
        ---
        tags:
          - Forward Testing
        summary: Get campaign accuracy
        description: Returns accuracy metrics for a forward test campaign
        parameters:
          - name: version_id
            in: query
            type: string
            required: true
            description: Version ID
            example: "2024-01-01-1"
          - name: symbol
            in: query
            type: string
            required: true
            description: Trading pair symbol (e.g., BTCUSDT)
            example: BTCUSDT
          - name: timeframe
            in: query
            type: string
            required: false
            default: 15m
            enum: [5m, 15m, 1h]
            description: Timeframe for the model
        responses:
          200:
            description: Campaign accuracy metrics
            schema:
              type: object
              properties:
                version_id:
                  type: string
                symbol:
                  type: string
                timeframe:
                  type: string
                accuracy:
                  type: number
                total_runs:
                  type: integer
                runs_with_golden:
                  type: integer
                matches:
                  type: integer
                auto_promoted:
                  type: boolean
          404:
            description: Campaign not found
          500:
            description: Server error
        """
        try:
            version_id = request.args.get('version_id')
            symbol = request.args.get('symbol')
            timeframe = request.args.get('timeframe', '15m')
            
            if not version_id or not symbol:
                return jsonify({'error': 'Missing required parameters: version_id and symbol'}), 400
            
            result = get_campaign_accuracy(version_id, symbol, timeframe)
            if result is None:
                return jsonify({'error': 'Campaign not found'}), 404
            
            return jsonify(datetime_to_str(result))
        except Exception as e:
            logger.error(f"获取 campaign accuracy 失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/forward_test/status', methods=['GET'])
    def get_forward_test_status():
        """
        获取 forward test cron 状态
        ---
        tags:
          - Forward Testing
        summary: Get forward test cron manager status
        description: Returns the status of the forward test cron manager including registered jobs
        responses:
          200:
            description: Cron manager status
            schema:
              type: object
              properties:
                is_running:
                  type: boolean
                  description: Whether the cron scheduler is running
                thread_alive:
                  type: boolean
                  description: Whether the scheduler thread is alive
                registered_jobs:
                  type: array
                  items:
                    type: object
                    properties:
                      version_id:
                        type: string
                      symbol:
                        type: string
                      timeframe:
                        type: string
                      interval_minutes:
                        type: integer
                      next_run:
                        type: string
                total_jobs:
                  type: integer
          404:
            description: Cron manager not initialized
        """
        try:
            cron_mgr = ForwardTestCronManager._instance
            if cron_mgr is None:
                return jsonify({
                    'error': 'Cron manager not initialized',
                    'is_running': False,
                    'registered_jobs': [],
                    'total_jobs': 0
                }), 404
            
            jobs = cron_mgr.list_registered_jobs()
            thread_alive = cron_mgr._scheduler_thread.is_alive() if cron_mgr._scheduler_thread else False
            
            return jsonify({
                'is_running': cron_mgr._is_running,
                'thread_alive': thread_alive,
                'registered_jobs': jobs,
                'total_jobs': len(jobs)
            })
        except Exception as e:
            logger.error(f"获取 forward test 状态失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    # ============ Config Management Endpoints ============
    
    @app.route('/api/configs', methods=['GET'])
    def list_configs():
        """
        列出所有配置版本
        ---
        tags:
          - Config
        summary: List all config versions
        description: Returns a list of all training config versions
        parameters:
          - name: include_inactive
            in: query
            type: boolean
            required: false
            default: false
            description: Include inactive (deleted) configs
          - name: limit
            in: query
            type: integer
            required: false
            default: 50
            description: Maximum number of configs to return
          - name: offset
            in: query
            type: integer
            required: false
            default: 0
            description: Number of configs to skip
        responses:
          200:
            description: List of config versions
            schema:
              type: object
              properties:
                configs:
                  type: array
                  items:
                    type: object
                    properties:
                      config_version_id:
                        type: string
                      created_at:
                        type: string
                      description:
                        type: string
                      is_active:
                        type: boolean
                      model_count:
                        type: integer
                total:
                  type: integer
        """
        try:
            include_inactive = request.args.get('include_inactive', 'false').lower() == 'true'
            limit = int(request.args.get('limit', 50))
            offset = int(request.args.get('offset', 0))
            
            configs = list_config_versions(include_inactive=include_inactive)
            total = len(configs)
            configs = configs[offset:offset + limit]
            
            return jsonify({
                'configs': configs,
                'total': total
            })
        except Exception as e:
            logger.error(f"列出配置版本失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/configs/defaults', methods=['GET'])
    def get_default_config_endpoint():
        """
        获取默认配置值
        ---
        tags:
          - Config
        summary: Get default config values from TrainingConfig
        description: Returns the default config values from TrainingConfig class without creating a version
        responses:
          200:
            description: Default config values
            schema:
              type: object
              additionalProperties:
                type: string
        """
        try:
            default_config = get_default_config()
            # Parse JSON values back to Python objects for response
            parsed_config = {}
            for key, value_json in default_config.items():
                try:
                    parsed_config[key] = json.loads(value_json)
                except (json.JSONDecodeError, TypeError):
                    parsed_config[key] = value_json
            return jsonify(parsed_config)
        except Exception as e:
            logger.error(f"获取默认配置失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/configs/<config_version_id>', methods=['GET'])
    def get_config(config_version_id):
        """
        获取特定配置版本
        ---
        tags:
          - Config
        summary: Get specific config version
        description: Returns the full configuration for a specific version
        parameters:
          - name: config_version_id
            in: path
            type: string
            required: true
            description: Config version ID
        responses:
          200:
            description: Config version data
            schema:
              type: object
              additionalProperties:
                type: string
          404:
            description: Config version not found
        """
        try:
            config_dict = get_config_version(config_version_id)
            if config_dict is None:
                return jsonify({'error': f'Config version {config_version_id} not found'}), 404
            
            # Parse JSON values back to Python objects for response
            parsed_config = {}
            for key, value_json in config_dict.items():
                try:
                    parsed_config[key] = json.loads(value_json)
                except (json.JSONDecodeError, TypeError):
                    parsed_config[key] = value_json
            
            return jsonify(parsed_config)
        except Exception as e:
            logger.error(f"获取配置版本失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/configs/init', methods=['POST'])
    def init_config():
        """
        初始化/迁移配置
        ---
        tags:
          - Config
        summary: Initialize config from TrainingConfig file
        description: Imports current TrainingConfig class values into database as a new version. Always creates a new version.
        parameters:
          - name: body
            in: body
            required: false
            schema:
              type: object
              properties:
                description:
                  type: string
                  description: Description for the config version
                  default: "Initial config from TrainingConfig"
        responses:
          201:
            description: Config initialized (new version created)
            schema:
              type: object
              properties:
                config_version_id:
                  type: string
                message:
                  type: string
          400:
            description: Invalid request
        """
        try:
            data = request.get_json() or {}
            description = data.get('description', 'Initial config from TrainingConfig')
            
            config_version_id = init_from_config_file(description=description)
            
            return jsonify({
                'config_version_id': config_version_id,
                'message': f'Config initialized: {config_version_id}'
            }), 201
        except Exception as e:
            logger.error(f"初始化配置失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/configs', methods=['POST'])
    def create_config():
        """
        创建新配置版本
        ---
        tags:
          - Config
        summary: Create new config version
        description: Creates a new config version from provided config dict
        parameters:
          - name: body
            in: body
            required: true
            schema:
              type: object
              required:
                - config
              properties:
                config:
                  type: object
                  description: Config dictionary (can be nested or flattened)
                description:
                  type: string
                  description: Optional description
        responses:
          201:
            description: Config version created
            schema:
              type: object
              properties:
                config_version_id:
                  type: string
                message:
                  type: string
          400:
            description: Invalid request
        """
        try:
            data = request.get_json()
            if not data or 'config' not in data:
                return jsonify({'error': 'Missing required field: config'}), 400
            
            config_dict = data['config']
            description = data.get('description')
            
            config_version_id = create_config_version(config_dict, description=description)
            
            return jsonify({
                'config_version_id': config_version_id,
                'message': f'Config version created: {config_version_id}'
            }), 201
        except Exception as e:
            logger.error(f"创建配置版本失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/configs/<config_version_id>', methods=['PUT'])
    def update_config(config_version_id):
        """
        更新配置版本（创建新版本）
        ---
        tags:
          - Config
        summary: Update config version (creates new version)
        description: Creates a new config version with updates from the specified version
        parameters:
          - name: config_version_id
            in: path
            type: string
            required: true
            description: Source config version ID
          - name: body
            in: body
            required: true
            schema:
              type: object
              required:
                - updates
              properties:
                updates:
                  type: object
                  description: Dict of updates (can be nested or flattened)
                description:
                  type: string
                  description: Optional description for new version
        responses:
          201:
            description: New config version created
            schema:
              type: object
              properties:
                config_version_id:
                  type: string
                message:
                  type: string
          404:
            description: Source config version not found
        """
        try:
            data = request.get_json()
            if not data or 'updates' not in data:
                return jsonify({'error': 'Missing required field: updates'}), 400
            
            updates_dict = data['updates']
            description = data.get('description')
            
            new_config_version_id = update_config_version(
                config_version_id,
                updates_dict,
                description=description
            )
            
            return jsonify({
                'config_version_id': new_config_version_id,
                'message': f'New config version created: {new_config_version_id} (from {config_version_id})'
            }), 201
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"更新配置版本失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/configs/<config_version_id>', methods=['DELETE'])
    def delete_config(config_version_id):
        """
        删除配置版本（软删除）
        ---
        tags:
          - Config
        summary: Delete config version (soft delete)
        description: Soft deletes a config version by setting is_active=0
        parameters:
          - name: config_version_id
            in: path
            type: string
            required: true
            description: Config version ID to delete
        responses:
          200:
            description: Config version deleted
            schema:
              type: object
              properties:
                message:
                  type: string
          404:
            description: Config version not found
        """
        try:
            success = delete_config_version(config_version_id)
            if not success:
                return jsonify({'error': f'Config version {config_version_id} not found'}), 404
            
            return jsonify({
                'message': f'Config version {config_version_id} deleted'
            })
        except Exception as e:
            logger.error(f"删除配置版本失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/configs/<config_version_id>/models', methods=['GET'])
    def get_models_for_config_endpoint(config_version_id):
        """
        获取使用指定配置训练的模型列表
        ---
        tags:
          - Config
        summary: List models trained with config version
        description: Returns all models that were trained using the specified config version
        parameters:
          - name: config_version_id
            in: path
            type: string
            required: true
            description: Config version ID
        responses:
          200:
            description: List of models
            schema:
              type: object
              properties:
                models:
                  type: array
                  items:
                    type: object
                    properties:
                      model_version_id:
                        type: string
                      symbol:
                        type: string
                      timeframe:
                        type: string
                      created_at:
                        type: string
        """
        try:
            models = get_models_for_config(config_version_id)
            return jsonify({'models': models})
        except Exception as e:
            logger.error(f"获取配置关联的模型失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    # ============ Training Endpoints ============
    
    @app.route('/api/training/train', methods=['POST'])
    def trigger_training():
        """
        触发模型训练
        ---
        tags:
          - Training
        summary: Trigger model training with config version
        description: Triggers model training for a symbol/timeframe using a specific config version
        parameters:
          - name: body
            in: body
            required: true
            schema:
              type: object
              required:
                - symbol
                - timeframe
                - training_type
              properties:
                symbol:
                  type: string
                  description: Trading pair symbol
                  example: BTCUSDT
                timeframe:
                  type: string
                  description: Primary timeframe
                  enum: [5m, 15m, 1h]
                  example: 15m
                config_version_id:
                  type: string
                  description: Config version ID (optional, uses defaults if not provided)
                training_type:
                  type: string
                  enum: [full, incremental]
                  description: Training type
                  example: full
        responses:
          200:
            description: Training started
            schema:
              type: object
              properties:
                job_id:
                  type: string
                message:
                  type: string
                status:
                  type: string
          400:
            description: Invalid request
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing request body'}), 400
            
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            training_type = data.get('training_type')
            config_version_id = data.get('config_version_id')  # Optional
            
            if not symbol or not timeframe or not training_type:
                return jsonify({
                    'error': 'Missing required fields: symbol, timeframe, training_type'
                }), 400
            
            if training_type not in ['full', 'incremental']:
                return jsonify({
                    'error': 'training_type must be "full" or "incremental"'
                }), 400
            
            # Create training pipeline with config version
            pipeline = TrainingPipeline(config_version_id=config_version_id)
            
            # Trigger training in background thread
            import threading
            def train_worker():
                try:
                    if training_type == 'full':
                        result = pipeline.full_retrain(symbol, timeframe)
                    else:
                        result = pipeline.incremental_train(symbol, timeframe)
                    logger.info(f"Training completed: {result}")
                except Exception as e:
                    logger.error(f"Training failed: {e}", exc_info=True)
            
            thread = threading.Thread(target=train_worker, daemon=True)
            thread.start()
            
            job_id = f"{symbol}-{timeframe}-{datetime.now().isoformat()}"
            
            return jsonify({
                'job_id': job_id,
                'message': f'Training {training_type} started for {symbol} {timeframe}',
                'status': 'started',
                'config_version_id': config_version_id or 'default'
            })
        except Exception as e:
            logger.error(f"触发训练失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/<model_version_id>/config', methods=['GET'])
    def get_model_config(model_version_id):
        """
        获取模型使用的配置
        ---
        tags:
          - Models
        summary: Get config used for model
        description: Returns the config version used to train a specific model
        parameters:
          - name: model_version_id
            in: path
            type: string
            required: true
            description: Model version ID
          - name: symbol
            in: query
            type: string
            required: true
            description: Trading pair symbol
          - name: timeframe
            in: query
            type: string
            required: true
            description: Timeframe
        responses:
          200:
            description: Config version info
            schema:
              type: object
              properties:
                config_version_id:
                  type: string
                config:
                  type: object
                  additionalProperties:
                    type: string
          404:
            description: Model or config not found
        """
        try:
            symbol = request.args.get('symbol')
            timeframe = request.args.get('timeframe')
            
            if not symbol or not timeframe:
                return jsonify({
                    'error': 'Missing required query parameters: symbol, timeframe'
                }), 400
            
            config_dict = get_config_for_model(model_version_id, symbol, timeframe)
            
            if config_dict is None:
                return jsonify({
                    'config_version_id': 'default',
                    'message': 'Model not linked to any config version, using defaults'
                })
            
            # Parse JSON values
            parsed_config = {}
            for key, value_json in config_dict.items():
                try:
                    parsed_config[key] = json.loads(value_json)
                except (json.JSONDecodeError, TypeError):
                    parsed_config[key] = value_json
            
            # Find config_version_id from model_config_links
            from config_registry import _default_db_path, _get_conn, _init_database
            db_path = _default_db_path()
            _init_database(db_path)
            with _get_conn(db_path) as conn:
                cur = conn.execute(
                    "SELECT config_version_id FROM model_config_links WHERE model_version_id = ? AND symbol = ? AND timeframe = ?",
                    (model_version_id, symbol, timeframe)
                )
                row = cur.fetchone()
                config_version_id = row[0] if row else 'default'
            
            return jsonify({
                'config_version_id': config_version_id,
                'config': parsed_config
            })
        except Exception as e:
            logger.error(f"获取模型配置失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/batch_predict', methods=['POST'])
    def batch_predict():
        """
        批量预测
        ---
        tags:
          - Prediction
        summary: Batch prediction for multiple symbols
        description: Predicts market regimes for multiple symbols in a single request
        parameters:
          - name: body
            in: body
            required: true
            schema:
              type: object
              required:
                - symbols
              properties:
                symbols:
                  type: array
                  items:
                    type: string
                  description: List of trading pair symbols
                  example: ["BTCUSDT", "ETHUSDT"]
                timeframe:
                  type: string
                  description: Timeframe for prediction
                  default: 15m
                  enum: [5m, 15m, 1h]
        responses:
          200:
            description: Batch prediction results
            schema:
              type: object
              additionalProperties:
                type: object
          400:
            description: Missing symbols or invalid timeframe
          500:
            description: Server error
        """
        try:
            data = request.get_json()
            if not data or 'symbols' not in data:
                return jsonify({'error': '请求体必须包含 symbols 字段'}), 400
            symbols = data['symbols']
            timeframe = data.get('timeframe', '15m')
            if not isinstance(symbols, list):
                return jsonify({'error': 'symbols 必须是列表'}), 400
            if timeframe not in api.config.MODEL_CONFIGS.keys():
                return jsonify({'error': f'不支持的时间框架: {timeframe}，支持的值: {list(api.config.MODEL_CONFIGS.keys())}'}), 400
            results = api.batch_predict(symbols, primary_timeframe=timeframe)
            return jsonify(datetime_to_str(results))
        except Exception as e:
            logger.error(f"批量预测失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/history/<symbol>', methods=['GET'])
    def get_history(symbol: str):
        """
        获取历史上的market regime序列
        ---
        tags:
          - History
        summary: Get historical regime sequence
        description: |
          Returns historical market regime sequence for a symbol.
          
          Supports two query modes:
          1. By lookback hours: ?timeframe=15m&lookback_hours=24
          2. By date range: ?timeframe=15m&start_date=2024-01-01&end_date=2024-01-31
          
          Date format: ISO 8601 (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
        parameters:
          - name: symbol
            in: path
            type: string
            required: true
            description: Trading pair symbol (e.g., BTCUSDT)
            example: BTCUSDT
          - name: timeframe
            in: query
            type: string
            required: false
            default: 15m
            enum: [5m, 15m, 1h]
            description: Timeframe for historical data
          - name: lookback_hours
            in: query
            type: integer
            required: false
            description: Number of hours to look back (alternative to date range)
            example: 24
          - name: start_date
            in: query
            type: string
            required: false
            description: Start date in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
            example: "2024-01-01"
          - name: end_date
            in: query
            type: string
            required: false
            description: End date in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
            example: "2024-01-31"
        responses:
          200:
            description: Historical regime sequence
            schema:
              type: object
              properties:
                symbol:
                  type: string
                timeframe:
                  type: string
                history:
                  type: array
                  items:
                    type: object
                    properties:
                      timestamp:
                        type: string
                        format: date-time
                      regime:
                        type: string
          400:
            description: Invalid timeframe or date format
          404:
            description: Model not found
          500:
            description: Server error
        """
        try:
            timeframe = request.args.get('timeframe', '15m')
            lookback_hours = request.args.get('lookback_hours', type=int)
            start_date_str = request.args.get('start_date')
            end_date_str = request.args.get('end_date')
            
            if timeframe not in api.config.MODEL_CONFIGS.keys():
                return jsonify({'error': f'不支持的时间框架: {timeframe}，支持的值: {list(api.config.MODEL_CONFIGS.keys())}'}), 400
            
            # 解析日期
            start_date = None
            end_date = None
            if start_date_str:
                try:
                    start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
                except ValueError:
                    return jsonify({'error': f'无效的 start_date 格式: {start_date_str}，请使用 ISO 8601 格式'}), 400
            
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                except ValueError:
                    return jsonify({'error': f'无效的 end_date 格式: {end_date_str}，请使用 ISO 8601 格式'}), 400
            
            # 验证参数
            if start_date_str and not end_date_str:
                return jsonify({'error': '如果指定了 start_date，必须同时指定 end_date'}), 400
            
            if end_date_str and not start_date_str:
                return jsonify({'error': '如果指定了 end_date，必须同时指定 start_date'}), 400
            
            if start_date and end_date:
                if start_date >= end_date:
                    return jsonify({'error': 'start_date 必须早于 end_date'}), 400
                
                # 限制最大日期范围（1年）
                max_days = 365
                if (end_date - start_date).days > max_days:
                    return jsonify({
                        'error': f'日期范围不能超过 {max_days} 天（1年）'
                    }), 400
            
            if lookback_hours is not None:
                if lookback_hours <= 0:
                    return jsonify({'error': 'lookback_hours 必须大于 0'}), 400
                
                # 限制最大回看时间（30天）
                max_lookback_hours = 720  # 30天
                if lookback_hours > max_lookback_hours:
                    return jsonify({
                        'error': f'lookback_hours 不能超过 {max_lookback_hours} 小时（30天）'
                    }), 400
            
            # 如果同时指定了日期范围和回看小时数，优先使用日期范围
            if start_date and end_date:
                result = api.get_regime_history(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    primary_timeframe=timeframe
                )
            elif lookback_hours is not None:
                result = api.get_regime_history(
                    symbol=symbol,
                    lookback_hours=lookback_hours,
                    primary_timeframe=timeframe
                )
            else:
                # 默认回看24小时
                result = api.get_regime_history(
                    symbol=symbol,
                    lookback_hours=24,
                    primary_timeframe=timeframe
                )
            
            return jsonify(datetime_to_str(result))
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"获取历史regime失败: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    return app


# ==================== 主函数（示例） ====================

def main():
    """主函数 - 支持多种运行模式"""
    import sys
    
    # 检查是否要运行 HTTP 服务器
    if '--server' in sys.argv or '--http' in sys.argv:
        # 运行 HTTP 服务器模式
        import threading
        from scheduler import TrainingScheduler
        
        TrainingConfig.ensure_dirs()
        
        # 启动调度器（后台线程）
        scheduler = TrainingScheduler(TrainingConfig)
        scheduler_thread = threading.Thread(target=scheduler.run, daemon=True)
        scheduler_thread.start()
        logger.info("训练调度器已在后台线程启动")
        
        # 创建并运行 Flask 应用
        app = create_app()
        host = '0.0.0.0'
        port = 5000
        
        logger.info("="*80)
        logger.info("API 服务器启动")
        logger.info(f"监听地址: http://{host}:{port}")
        logger.info("API 端点:")
        logger.info("  GET  /api/health")
        logger.info("  GET  /api/predict/<symbol>?timeframe=15m")
        logger.info("  GET  /api/predict_regimes/<symbol>?timeframe=15m")
        logger.info("  GET  /api/history/<symbol>?timeframe=15m&lookback_hours=24")
        logger.info("  GET  /api/history/<symbol>?timeframe=15m&start_date=2024-01-01&end_date=2024-01-31")
        logger.info("  GET  /api/metadata/<symbol>?timeframe=15m")
        logger.info("  GET  /api/models/available")
        logger.info("  GET  /api/models/by_timeframe")
        logger.info("  POST /api/batch_predict")
        logger.info("="*80)
        
        app.run(host=host, port=port, debug=False, threaded=True)
        return
    
    # 默认：运行示例代码
    api = ModelAPI()
    
    # 列出可用的模型
    available = api.list_available_models()
    print(f"\n可用的模型: {available}")
    
    if not available:
        print("\n⚠️  没有可用的模型，请先训练模型")
        return
    
    # 使用第一个可用的交易对
    symbol = available[0]
    
    # 预测下一根15分钟K线的market regime
    print(f"\n预测 {symbol} 下一根15分钟K线的market regime:")
    print("=" * 70)
    
    result = api.predict_next_regime(symbol, "15m")
    
    print(f"交易对: {result['symbol']}")
    print(f"时间框架: {result['timeframe']}")
    print(f"预测时间: {result['timestamp']}")
    print(f"使用历史K线数: {result['model_info']['sequence_length']}")
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

