"""
主训练管道 - 协调数据获取、特征工程、HMM 和 LSTM 训练
"""
import logging
import os
from datetime import datetime
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from config import TrainingConfig
from data_fetcher import BinanceDataFetcher, save_data
from feature_engineering import FeatureEngineer
from hmm_trainer import HMMRegimeLabeler, create_labeled_dataset
from lstm_trainer import LSTMRegimeClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """训练管道"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_fetcher = BinanceDataFetcher(
            api_key=config.BINANCE_API_KEY,
            api_secret=config.BINANCE_API_SECRET
        )
        self.feature_engineer = FeatureEngineer()
    
    def full_retrain(self, symbol: str) -> Dict:
        """
        完整重训（从零开始）
        
        Args:
            symbol: 交易对
            
        Returns:
            训练结果
        """
        logger.info(f"="*80)
        logger.info(f"开始完整重训: {symbol}")
        logger.info(f"="*80)
        
        # 1. 获取数据
        logger.info("步骤 1/5: 获取历史数据...")
        data = self.data_fetcher.fetch_full_training_data(
            symbol=symbol,
            timeframes=self.config.TIMEFRAMES,
            days=self.config.FULL_RETRAIN_DAYS
        )
        save_data(data, symbol, self.config)
        
        # 2. 特征工程
        logger.info("步骤 2/5: 计算技术指标...")
        features = self.feature_engineer.combine_timeframe_features(
            data,
            primary_timeframe=self.config.PRIMARY_TIMEFRAME
        )
        
        # 可选：特征选择
        features = self.feature_engineer.select_key_features(features)
        
        logger.info(f"特征数量: {len(features.columns)}, 样本数: {len(features)}")
        
        # 3. HMM 标注
        logger.info("步骤 3/5: HMM 状态标注...")
        hmm_labeler = HMMRegimeLabeler(
            n_states=self.config.N_STATES,
            n_components=self.config.N_PCA_COMPONENTS
        )
        
        states = hmm_labeler.fit(features)
        
        # 保存 HMM 模型
        hmm_path = self.config.get_hmm_path(symbol)
        hmm_labeler.save(hmm_path)
        
        # 分析市场状态
        regime_analysis = hmm_labeler.analyze_regimes(features, states)
        logger.info(f"\n市场状态分析:\n{regime_analysis}")
        
        # 4. 准备 LSTM 训练数据
        logger.info("步骤 4/5: 准备 LSTM 训练数据...")
        lstm_classifier = LSTMRegimeClassifier(
            n_states=self.config.N_STATES,
            sequence_length=self.config.SEQUENCE_LENGTH,
            lstm_units=self.config.LSTM_UNITS,
            dropout_rate=self.config.DROPOUT_RATE
        )
        
        X_train, X_test, y_train, y_test = lstm_classifier.prepare_data(
            features, states, test_size=self.config.VALIDATION_SPLIT
        )
        
        # 5. 训练 LSTM
        logger.info("步骤 5/5: 训练 LSTM 模型...")
        model_path = self.config.get_model_path(symbol)
        
        history = lstm_classifier.train(
            X_train, y_train,
            X_test, y_test,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            model_path=model_path
        )
        
        # 评估模型
        eval_results = lstm_classifier.evaluate(X_test, y_test)
        
        # 保存模型和标准化器
        scaler_path = self.config.get_scaler_path(symbol)
        lstm_classifier.save(model_path, scaler_path)
        
        logger.info(f"完整重训完成: {symbol}")
        logger.info(f"模型准确率: {eval_results['accuracy']:.4f}")
        
        return {
            'symbol': symbol,
            'training_type': 'full_retrain',
            'timestamp': datetime.now(),
            'accuracy': eval_results['accuracy'],
            'loss': eval_results['loss'],
            'regime_analysis': regime_analysis,
            'history': history
        }
    
    def incremental_train(self, symbol: str) -> Dict:
        """
        增量训练（在现有模型基础上）
        
        Args:
            symbol: 交易对
            
        Returns:
            训练结果
        """
        logger.info(f"="*80)
        logger.info(f"开始增量训练: {symbol}")
        logger.info(f"="*80)
        
        # 1. 获取最新数据
        logger.info("步骤 1/4: 获取最新数据...")
        data = self.data_fetcher.fetch_latest_data(
            symbol=symbol,
            timeframes=self.config.TIMEFRAMES,
            days=self.config.INCREMENTAL_TRAIN_DAYS
        )
        
        # 2. 特征工程
        logger.info("步骤 2/4: 计算技术指标...")
        features = self.feature_engineer.combine_timeframe_features(
            data,
            primary_timeframe=self.config.PRIMARY_TIMEFRAME
        )
        
        # 3. 加载 HMM 模型并标注
        logger.info("步骤 3/4: HMM 状态标注...")
        hmm_path = self.config.get_hmm_path(symbol)
        
        if not os.path.exists(hmm_path):
            logger.warning(f"HMM 模型不存在，将执行完整重训: {hmm_path}")
            return self.full_retrain(symbol)
        
        hmm_labeler = HMMRegimeLabeler.load(hmm_path)
        states = hmm_labeler.predict(features)
        
        # 4. 加载 LSTM 模型并增量训练
        logger.info("步骤 4/4: LSTM 增量训练...")
        model_path = self.config.get_model_path(symbol)
        scaler_path = self.config.get_scaler_path(symbol)
        
        if not os.path.exists(model_path):
            logger.warning(f"LSTM 模型不存在，将执行完整重训: {model_path}")
            return self.full_retrain(symbol)
        
        lstm_classifier = LSTMRegimeClassifier.load(model_path, scaler_path)
        
        # 准备增量训练数据
        features_scaled = lstm_classifier.scaler.transform(features)
        
        X, y = [], []
        for i in range(len(features_scaled) - lstm_classifier.sequence_length):
            X.append(features_scaled[i:i+lstm_classifier.sequence_length])
            y.append(states[i+lstm_classifier.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # 执行增量训练
        lstm_classifier.incremental_train(
            X, y,
            epochs=10,  # 增量训练使用较少的 epoch
            batch_size=self.config.BATCH_SIZE
        )
        
        # 保存更新后的模型
        lstm_classifier.save(model_path, scaler_path)
        
        logger.info(f"增量训练完成: {symbol}")
        
        return {
            'symbol': symbol,
            'training_type': 'incremental',
            'timestamp': datetime.now(),
            'samples_used': len(X)
        }
    
    def train_all_symbols(self, training_type: str = 'full') -> Dict:
        """
        训练所有交易对
        
        Args:
            training_type: 'full' 或 'incremental'
            
        Returns:
            所有交易对的训练结果
        """
        results = {}
        
        for symbol in self.config.SYMBOLS:
            try:
                if training_type == 'full':
                    result = self.full_retrain(symbol)
                else:
                    result = self.incremental_train(symbol)
                
                results[symbol] = result
                
            except Exception as e:
                logger.error(f"训练 {symbol} 时出错: {e}", exc_info=True)
                results[symbol] = {'error': str(e)}
        
        return results


def main():
    """主函数"""
    # 确保目录存在
    TrainingConfig.ensure_dirs()
    
    # 创建训练管道
    pipeline = TrainingPipeline(TrainingConfig)
    
    # 示例：完整重训 BTC
    result = pipeline.full_retrain("BTCUSDT")
    
    logger.info(f"\n训练结果: {result}")


if __name__ == "__main__":
    main()
