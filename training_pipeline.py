"""
主训练管道 - 协调数据获取、特征工程、HMM 和 LSTM 训练

修复数据泄漏问题：
1. 先按时间划分数据为 train/val/test
2. HMM 只在训练集上拟合（scaler, PCA, HMM 参数）
3. 用训练好的 HMM 对验证集和测试集进行预测（无数据泄漏）
4. LSTM 使用独立的验证集和测试集
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
from hmm_trainer import HMMRegimeLabeler
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
        self.feature_engineer = FeatureEngineer(cache_manager=self.data_fetcher.cache_manager)
    
    def _split_data_by_time(
        self, 
        features: pd.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序划分数据为 train/val/test
        
        Args:
            features: 完整特征 DataFrame
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            (train_features, val_features, test_features)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio 必须等于 1.0"
        
        n = len(features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_features = features.iloc[:train_end]
        val_features = features.iloc[train_end:val_end]
        test_features = features.iloc[val_end:]
        
        logger.info(f"时间序列数据划分:")
        logger.info(f"  训练集: {len(train_features)} 行 ({train_ratio:.0%})")
        logger.info(f"  验证集: {len(val_features)} 行 ({val_ratio:.0%})")
        logger.info(f"  测试集: {len(test_features)} 行 ({test_ratio:.0%})")
        
        if len(train_features) > 0 and len(test_features) > 0:
            logger.info(f"  训练集时间范围: {train_features.index.min()} ~ {train_features.index.max()}")
            logger.info(f"  验证集时间范围: {val_features.index.min()} ~ {val_features.index.max()}")
            logger.info(f"  测试集时间范围: {test_features.index.min()} ~ {test_features.index.max()}")
        
        return train_features, val_features, test_features
    
    def full_retrain(self, symbol: str) -> Dict:
        """
        完整重训（从零开始）
        
        修复数据泄漏问题：
        1. 先按时间划分数据为 train/val/test
        2. HMM 只在训练集上拟合
        3. LSTM 使用独立的验证集和测试集
        
        Args:
            symbol: 交易对
            
        Returns:
            训练结果
        """
        logger.info(f"="*80)
        logger.info(f"开始完整重训: {symbol}")
        logger.info(f"="*80)
        
        # 1. 获取数据
        logger.info("步骤 1/6: 获取历史数据...")
        data = self.data_fetcher.fetch_full_training_data(
            symbol=symbol,
            timeframes=self.config.TIMEFRAMES,
            days=self.config.FULL_RETRAIN_DAYS
        )
        # 注意：数据已自动保存到 SQLite 缓存中，无需额外保存
        
        # 输出 API 统计信息
        stats = self.data_fetcher.get_api_stats()
        logger.info(f"API 请求统计: {stats}")
        
        # 2. 特征工程
        logger.info("步骤 2/6: 计算技术指标...")
        features = self.feature_engineer.combine_timeframe_features(
            data,
            primary_timeframe=self.config.PRIMARY_TIMEFRAME,
            symbol=symbol
        )
        
        # 可选：特征选择
        features = self.feature_engineer.select_key_features(features)
        
        logger.info(f"特征数量: {len(features.columns)}, 样本数: {len(features)}")
        
        # 3. 按时间划分数据（关键步骤：避免数据泄漏）
        logger.info("步骤 3/6: 按时间划分数据...")
        train_features, val_features, test_features = self._split_data_by_time(
            features,
            train_ratio=self.config.TRAIN_RATIO,
            val_ratio=self.config.VAL_RATIO,
            test_ratio=self.config.TEST_RATIO
        )
        
        # 4. HMM 标注（只在训练集上拟合，避免数据泄漏）
        logger.info("步骤 4/6: HMM 状态标注（只在训练集上拟合）...")
        hmm_labeler = HMMRegimeLabeler(
            n_states=self.config.N_STATES,
            n_components=self.config.N_PCA_COMPONENTS
        )
        
        # 使用新方法：在训练集上拟合，分别预测各数据集的标签
        train_states, val_states, test_states = hmm_labeler.fit_predict_split(
            train_features=train_features,
            val_features=val_features,
            test_features=test_features
        )
        
        # 保存 HMM 模型
        hmm_path = self.config.get_hmm_path(symbol)
        hmm_labeler.save(hmm_path)
        
        # 分析市场状态（只用训练集分析，避免泄漏）
        regime_analysis = hmm_labeler.analyze_regimes(train_features, train_states)
        logger.info(f"\n训练集市场状态分析:\n{regime_analysis}")
        
        # 5. 准备 LSTM 训练数据
        logger.info("步骤 5/6: 准备 LSTM 训练数据...")
        lstm_classifier = LSTMRegimeClassifier(
            n_states=self.config.N_STATES,
            sequence_length=self.config.SEQUENCE_LENGTH,
            lstm_units=self.config.LSTM_UNITS,
            dense_units=self.config.DENSE_UNITS,
            dropout_rate=self.config.DROPOUT_RATE,
            l2_lambda=self.config.L2_LAMBDA,
            use_batch_norm=self.config.USE_BATCH_NORM,
            learning_rate=self.config.LEARNING_RATE
        )
        
        # 使用新方法：支持 train/val/test 三分
        X_train, y_train, X_val, y_val, X_test, y_test = lstm_classifier.prepare_data_split(
            train_features=train_features,
            train_labels=train_states,
            val_features=val_features,
            val_labels=val_states,
            test_features=test_features,
            test_labels=test_states
        )
        
        # 6. 训练 LSTM
        logger.info("步骤 6/6: 训练 LSTM 模型...")
        model_path = self.config.get_model_path(symbol)
        
        # 使用验证集进行早停和模型选择
        history = lstm_classifier.train(
            X_train, y_train,
            X_val, y_val,  # 验证集用于早停
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            early_stopping_patience=self.config.EARLY_STOPPING_PATIENCE,
            lr_reduce_patience=self.config.LR_REDUCE_PATIENCE,
            model_path=model_path,
            use_class_weight=self.config.USE_CLASS_WEIGHT
        )
        
        # 在独立测试集上评估模型（这才是真实的泛化性能）
        logger.info("在独立测试集上评估模型...")
        if X_test is not None and y_test is not None:
            eval_results = lstm_classifier.evaluate(X_test, y_test)
            logger.info(f"🎯 测试集准确率: {eval_results['accuracy']:.4f} (这是真实的泛化性能)")
        else:
            # 如果没有测试集，使用验证集评估（不推荐）
            eval_results = lstm_classifier.evaluate(X_val, y_val)
            logger.warning("⚠️ 没有独立测试集，使用验证集评估（结果可能偏乐观）")
        
        # 同时输出验证集准确率作为参考
        val_eval = lstm_classifier.evaluate(X_val, y_val)
        logger.info(f"验证集准确率: {val_eval['accuracy']:.4f}")
        
        # 保存模型和标准化器
        scaler_path = self.config.get_scaler_path(symbol)
        lstm_classifier.save(model_path, scaler_path)
        
        logger.info(f"完整重训完成: {symbol}")
        logger.info(f"测试集准确率: {eval_results['accuracy']:.4f}")
        
        return {
            'symbol': symbol,
            'training_type': 'full_retrain',
            'timestamp': datetime.now(),
            'test_accuracy': eval_results['accuracy'],
            'val_accuracy': val_eval['accuracy'],
            'test_loss': eval_results['loss'],
            'regime_analysis': regime_analysis,
            'history': history,
            'data_split': {
                'train_samples': len(train_features),
                'val_samples': len(val_features),
                'test_samples': len(test_features)
            }
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
        
        # 输出 API 统计信息
        stats = self.data_fetcher.get_api_stats()
        logger.info(f"API 请求统计: {stats}")
        
        # 2. 特征工程
        logger.info("步骤 2/4: 计算技术指标...")
        features = self.feature_engineer.combine_timeframe_features(
            data,
            primary_timeframe=self.config.PRIMARY_TIMEFRAME,
            symbol=symbol
        )
        
        # 3. 加载 HMM 模型并标注
        logger.info("步骤 3/4: HMM 状态标注...")
        hmm_path = self.config.get_hmm_path(symbol)
        
        if not os.path.exists(hmm_path):
            logger.warning(f"HMM 模型不存在，将执行完整重训: {hmm_path}")
            return self.full_retrain(symbol)
        
        hmm_labeler = HMMRegimeLabeler.load(hmm_path)
        
        # 应用特征选择（与完整训练保持一致）
        features_before_selection = features.copy()
        features = self.feature_engineer.select_key_features(features)
        
        # 如果模型保存了特征名称，确保特征一致
        if hmm_labeler.feature_names_ is not None:
            # 检查特征是否匹配
            missing_features = set(hmm_labeler.feature_names_) - set(features.columns)
            extra_features = set(features.columns) - set(hmm_labeler.feature_names_)
            
            if missing_features or extra_features:
                logger.warning(
                    f"特征选择结果不一致！\n"
                    f"  训练时特征数: {len(hmm_labeler.feature_names_)}\n"
                    f"  当前特征数: {len(features.columns)}\n"
                    f"  缺少特征: {len(missing_features)} 个\n"
                    f"  多余特征: {len(extra_features)} 个"
                )
                # predict 方法会自动处理特征对齐
        else:
            # 旧版本模型：检查特征数量
            expected_features = (
                hmm_labeler.scaler.n_features_in_ 
                if hasattr(hmm_labeler.scaler, 'n_features_in_') 
                else None
            )
            if expected_features and len(features.columns) != expected_features:
                logger.error(
                    f"特征数量不匹配！训练时: {expected_features} 个特征, "
                    f"当前: {len(features.columns)} 个特征\n"
                    f"这是旧版本模型，建议重新训练模型（运行示例 1）以保存特征名称。"
                )
                raise ValueError(
                    f"特征数量不匹配。请重新训练模型（运行示例 1）以确保特征一致性。"
                )
        
        states = hmm_labeler.predict(features)
        
        # 4. 加载 LSTM 模型并增量训练
        logger.info("步骤 4/4: LSTM 增量训练...")
        model_path = self.config.get_model_path(symbol)
        scaler_path = self.config.get_scaler_path(symbol)
        
        if not os.path.exists(model_path):
            logger.warning(f"LSTM 模型不存在，将执行完整重训: {model_path}")
            return self.full_retrain(symbol)
        
        lstm_classifier = LSTMRegimeClassifier.load(model_path, scaler_path)
        
        # 对齐特征（确保与训练时一致）
        # 优先使用保存的特征名称，如果没有则使用 scaler 的 feature_names_in_
        feature_names = lstm_classifier.feature_names_
        if feature_names is None and hasattr(lstm_classifier.scaler, 'feature_names_in_'):
            feature_names = list(lstm_classifier.scaler.feature_names_in_)
            logger.info(f"使用 scaler 的特征名称: {len(feature_names)} 个特征")
        
        if feature_names is not None:
            # 确保特征顺序和数量与训练时一致
            missing_features = set(feature_names) - set(features.columns)
            extra_features = set(features.columns) - set(feature_names)
            
            if missing_features or extra_features:
                logger.warning(
                    f"特征不一致！\n"
                    f"  训练时特征数: {len(feature_names)}\n"
                    f"  当前特征数: {len(features.columns)}\n"
                    f"  缺少特征: {len(missing_features)} 个\n"
                    f"  多余特征: {len(extra_features)} 个"
                )
                if missing_features:
                    logger.warning(f"  缺少的特征: {missing_features}")
                if extra_features:
                    logger.warning(f"  多余的特征: {extra_features}")
                
                # 对齐特征：添加缺失的特征（填充0），移除多余的特征
                features_aligned = features.reindex(columns=feature_names, fill_value=0)
                logger.info(f"特征已对齐: {len(features_aligned.columns)} 个特征")
            else:
                # 特征名称一致，但需要确保顺序一致
                features_aligned = features[feature_names]
        else:
            # 旧版本模型：只检查特征数量
            expected_features = (
                lstm_classifier.scaler.n_features_in_ 
                if hasattr(lstm_classifier.scaler, 'n_features_in_') 
                else None
            )
            if expected_features and len(features.columns) != expected_features:
                logger.error(
                    f"特征数量不匹配！训练时: {expected_features} 个特征, "
                    f"当前: {len(features.columns)} 个特征\n"
                    f"这是旧版本模型，建议重新训练模型（运行示例 1）以保存特征名称。"
                )
                raise ValueError(
                    f"特征数量不匹配。请重新训练模型（运行示例 1）以确保特征一致性。"
                )
            features_aligned = features
        
        # 准备增量训练数据
        features_scaled = lstm_classifier.scaler.transform(features_aligned)
        
        X, y = [], []
        for i in range(len(features_scaled) - lstm_classifier.sequence_length):
            X.append(features_scaled[i:i+lstm_classifier.sequence_length])
            y.append(states[i+lstm_classifier.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # 执行增量训练（带验证集和早停）
        lstm_classifier.incremental_train(
            X, y,
            epochs=self.config.INCREMENTAL_EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.INCREMENTAL_LEARNING_RATE,
            validation_split=self.config.INCREMENTAL_VALIDATION_SPLIT,
            early_stopping_patience=self.config.INCREMENTAL_EARLY_STOPPING_PATIENCE,
            use_class_weight=self.config.USE_CLASS_WEIGHT
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
