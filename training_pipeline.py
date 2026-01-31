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

from config import TrainingConfig, setup_logging
from data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngineer
from hmm_trainer import HMMRegimeLabeler
from lstm_trainer import LSTMRegimeClassifier
from model_registry import allocate_version_id, register_version
from forward_testing import on_training_finished as forward_test_on_training_finished, ForwardTestCronManager

setup_logging(log_file='training.log', level=logging.INFO)
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
    
    def _log_state_adjustment_summary(
        self, 
        original_n_states: int, 
        new_n_states: int, 
        new_mapping: Dict[int, str]
    ):
        """
        输出状态调整的详细总结
        
        Args:
            original_n_states: 原始状态数量
            new_n_states: 调整后的状态数量
            new_mapping: 新的状态映射
        """
        # 完整的 6 个语义名称
        all_regime_names = {
            "Strong_Trend", "Weak_Trend", "Range", 
            "Choppy_High_Vol", "Volatility_Spike", "Squeeze"
        }
        
        # 当前保留的语义名称
        current_names = set(new_mapping.values())
        
        # 被删除的语义名称
        removed_names = all_regime_names - current_names
        
        logger.info("=" * 70)
        logger.info("📊 状态数量调整总结")
        logger.info("=" * 70)
        logger.info(f"  原始状态数量: {original_n_states}")
        logger.info(f"  调整后数量:   {new_n_states}")
        logger.info(f"  ")
        logger.info(f"  ✅ 保留的状态 ({len(current_names)} 个):")
        for name in sorted(current_names):
            logger.info(f"     - {name}")
        
        if removed_names:
            logger.info(f"  ")
            logger.info(f"  ❌ 删除的状态 ({len(removed_names)} 个):")
            for name in sorted(removed_names):
                logger.info(f"     - {name} (该市场状态在验证/测试期未出现)")
        
        logger.info("=" * 70)
    
    def full_retrain(self, symbol: str, primary_timeframe: str = None, version_id: str = None) -> Dict:
        """
        完整重训（从零开始）
        
        修复数据泄漏问题：
        1. 先按时间划分数据为 train/val/test
        2. HMM 只在训练集上拟合
        3. LSTM 使用独立的验证集和测试集
        
        Args:
            symbol: 交易对
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
            version_id: 版本目录 id（如 2025-01-31-1）；若为 None 则自动分配
            
        Returns:
            训练结果
        """
        # 获取模型配置
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        if version_id is None:
            version_id = allocate_version_id(models_dir=self.config.MODELS_DIR)
        # 确保版本目录存在：models/{version_id}/{symbol}/{timeframe}/
        version_dir = self.config.get_version_dir(version_id)
        symbol_tf_dir = os.path.join(version_dir, symbol, primary_timeframe)
        os.makedirs(symbol_tf_dir, exist_ok=True)
        
        model_config = self.config.get_model_config(primary_timeframe)
        timeframes = model_config["timeframes"]
        sequence_length = model_config["sequence_length"]
        lstm_units = model_config.get("lstm_units", self.config.LSTM_UNITS)
        dense_units = model_config.get("dense_units", self.config.DENSE_UNITS)
        dropout_rate = model_config.get("dropout_rate", self.config.DROPOUT_RATE)
        
        logger.info(f"="*80)
        logger.info(f"开始完整重训: {symbol} (primary_timeframe={primary_timeframe})")
        logger.info(f"  时间框架: {timeframes}")
        logger.info(f"  序列长度: {sequence_length}")
        logger.info(f"  LSTM 单元: {lstm_units}, Dense 单元: {dense_units}")
        logger.info(f"  Dropout: {dropout_rate}")
        logger.info(f"="*80)
        
        # 1. 获取数据
        logger.info("步骤 1/6: 获取历史数据...")
        data = self.data_fetcher.fetch_full_training_data(
            symbol=symbol,
            timeframes=timeframes,
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
            primary_timeframe=primary_timeframe,
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
        
        # 加载旧模型的映射（用于比对）- 使用 PROD 路径
        old_hmm_path = self.config.get_prod_hmm_path(symbol, primary_timeframe)
        old_mapping = None
        if os.path.exists(old_hmm_path):
            try:
                old_hmm = HMMRegimeLabeler.load(old_hmm_path)
                old_mapping = old_hmm.get_regime_mapping()
                logger.info(f"已加载旧模型映射用于比对: {old_mapping}")
            except Exception as e:
                logger.warning(f"无法加载旧模型: {e}")
        
        # 保存路径使用版本目录
        hmm_path = self.config.get_hmm_path_for_version(version_id, symbol, primary_timeframe)
        
        hmm_labeler = HMMRegimeLabeler(
            n_states=self.config.N_STATES,
            n_components=self.config.N_PCA_COMPONENTS,
            primary_timeframe=primary_timeframe
        )
        
        # 可选：BIC 验证状态数量是否合理
        bic_validation = None
        if getattr(self.config, 'VALIDATE_N_STATES', False):
            logger.info("执行 BIC 验证（验证状态数量是否合理）...")
            bic_test_states = getattr(self.config, 'BIC_TEST_N_STATES', [4, 5, 6, 7, 8])
            bic_validation = hmm_labeler.validate_n_states(
                train_features, 
                n_states_range=bic_test_states
            )
            logger.info(f"BIC 验证结果: {bic_validation['recommendation']}")
        
        # 使用新方法：在训练集上拟合，分别预测各数据集的标签
        train_states, val_states, test_states = hmm_labeler.fit_predict_split(
            train_features=train_features,
            val_features=val_features,
            test_features=test_features
        )
        
        # ========== 多步预测标签生成 ==========
        # 使用 forward-only filtering（无 look-ahead bias）
        prediction_horizons = getattr(self.config, 'PREDICTION_HORIZONS', [1, 2, 3, 4])
        label_temperature = getattr(self.config, 'LABEL_TEMPERATURE', 1.5)
        
        logger.info(f"生成多步预测标签: horizons={prediction_horizons}, temperature={label_temperature}")
        
        # Forward filter 生成滤波后验概率
        train_posteriors = hmm_labeler.forward_filter(train_features)
        val_posteriors = hmm_labeler.forward_filter(val_features)
        test_posteriors = hmm_labeler.forward_filter(test_features) if test_features is not None else None
        
        # 生成多步标签
        train_multistep_labels = hmm_labeler.generate_multistep_labels(
            train_posteriors, horizons=prediction_horizons, temperature=label_temperature
        )
        val_multistep_labels = hmm_labeler.generate_multistep_labels(
            val_posteriors, horizons=prediction_horizons, temperature=label_temperature
        )
        test_multistep_labels = None
        if test_posteriors is not None:
            test_multistep_labels = hmm_labeler.generate_multistep_labels(
                test_posteriors, horizons=prediction_horizons, temperature=label_temperature
            )
        
        # 自动映射 HMM 状态到语义名称（关键步骤！）
        # 使用配置中的绝对阈值护栏参数
        regime_mapping = hmm_labeler.auto_map_regimes(
            train_features, 
            train_states,
            min_vol_for_spike=getattr(self.config, 'REGIME_MIN_VOL_FOR_SPIKE', 0.02),
            max_vol_for_squeeze=getattr(self.config, 'REGIME_MAX_VOL_FOR_SQUEEZE', 0.01),
            min_adx_for_strong_trend=getattr(self.config, 'REGIME_MIN_ADX_FOR_STRONG_TREND', 30),
            max_adx_for_squeeze=getattr(self.config, 'REGIME_MAX_ADX_FOR_SQUEEZE', 20)
        )
        logger.info(f"HMM 状态到语义名称的映射: {regime_mapping}")
        
        # 检查状态分布是否健康（验证集/测试集是否缺失某些状态）
        state_distribution_check = hmm_labeler.check_state_distribution(
            train_states=train_states,
            val_states=val_states,
            test_states=test_states,
            min_samples_per_state=getattr(self.config, 'MIN_SAMPLES_PER_STATE', 10),
            min_ratio_per_state=getattr(self.config, 'MIN_RATIO_PER_STATE', 0.01)
        )
        
        # ========== 动态调整状态数量 ==========
        # 如果状态分布不健康且启用了自动调整，尝试优化状态数量
        n_states_optimization = None
        auto_adjust_enabled = getattr(self.config, 'AUTO_ADJUST_N_STATES', False)
        
        if auto_adjust_enabled and not state_distribution_check['healthy']:
            missing_val = len(state_distribution_check['missing_states']['val'])
            low_ratio_val = len(state_distribution_check['low_sample_states']['val'])
            max_missing = getattr(self.config, 'MAX_MISSING_STATES_ALLOWED', 1)
            max_low_ratio = getattr(self.config, 'MAX_LOW_RATIO_STATES_ALLOWED', 2)
            
            if missing_val > max_missing or low_ratio_val > max_low_ratio:
                logger.info(f"🔄 状态分布不健康（缺失: {missing_val}, 低占比: {low_ratio_val}），尝试自动优化状态数量...")
                
                n_states_optimization = hmm_labeler.auto_optimize_n_states(
                    train_features=train_features,
                    val_features=val_features,
                    test_features=test_features,
                    n_states_min=getattr(self.config, 'N_STATES_MIN', 4),
                    n_states_max=getattr(self.config, 'N_STATES_MAX', 8),
                    max_missing_allowed=max_missing,
                    max_low_ratio_allowed=max_low_ratio,
                    strategy=getattr(self.config, 'N_STATES_ADJUST_STRATEGY', 'decrease_first'),
                    min_samples_per_state=getattr(self.config, 'MIN_SAMPLES_PER_STATE', 10),
                    min_ratio_per_state=getattr(self.config, 'MIN_RATIO_PER_STATE', 0.01)
                )
                
                # 如果状态数量被调整，需要重新训练和映射
                if n_states_optimization['adjusted']:
                    new_n_states = n_states_optimization['optimal_n_states']
                    logger.info(f"使用优化后的状态数量 {new_n_states} 重新训练...")
                    
                    # 重新训练
                    train_states, val_states, test_states = hmm_labeler.retrain_with_n_states(
                        n_states=new_n_states,
                        train_features=train_features,
                        val_features=val_features,
                        test_features=test_features
                    )
                    
                    # 重新映射（使用优先级选择名称）
                    regime_mapping = hmm_labeler.auto_map_regimes(
                        train_features, 
                        train_states,
                        min_vol_for_spike=getattr(self.config, 'REGIME_MIN_VOL_FOR_SPIKE', 0.02),
                        max_vol_for_squeeze=getattr(self.config, 'REGIME_MAX_VOL_FOR_SQUEEZE', 0.01),
                        min_adx_for_strong_trend=getattr(self.config, 'REGIME_MIN_ADX_FOR_STRONG_TREND', 30),
                        max_adx_for_squeeze=getattr(self.config, 'REGIME_MAX_ADX_FOR_SQUEEZE', 20)
                    )
                    logger.info(f"优化后的状态映射: {regime_mapping}")
                    
                    # 重新检查分布
                    state_distribution_check = hmm_labeler.check_state_distribution(
                        train_states=train_states,
                        val_states=val_states,
                        test_states=test_states,
                        min_samples_per_state=getattr(self.config, 'MIN_SAMPLES_PER_STATE', 10),
                        min_ratio_per_state=getattr(self.config, 'MIN_RATIO_PER_STATE', 0.01)
                    )
                    
                    # ⚠️ 关键修复：状态数量调整后，需要重新生成多步预测标签
                    # 因为标签的维度必须与新的状态数量匹配
                    logger.info(f"重新生成多步预测标签（状态数量已从 {n_states_optimization['original_n_states']} 调整为 {new_n_states}）...")
                    train_posteriors = hmm_labeler.forward_filter(train_features)
                    val_posteriors = hmm_labeler.forward_filter(val_features)
                    test_posteriors = hmm_labeler.forward_filter(test_features) if test_features is not None else None
                    
                    train_multistep_labels = hmm_labeler.generate_multistep_labels(
                        train_posteriors, horizons=prediction_horizons, temperature=label_temperature
                    )
                    val_multistep_labels = hmm_labeler.generate_multistep_labels(
                        val_posteriors, horizons=prediction_horizons, temperature=label_temperature
                    )
                    test_multistep_labels = None
                    if test_posteriors is not None:
                        test_multistep_labels = hmm_labeler.generate_multistep_labels(
                            test_posteriors, horizons=prediction_horizons, temperature=label_temperature
                        )
                    
                    # 输出状态调整总结
                    self._log_state_adjustment_summary(
                        original_n_states=n_states_optimization['original_n_states'],
                        new_n_states=new_n_states,
                        new_mapping=regime_mapping
                    )
        
        # 新旧映射比对（检测语义漂移）
        mapping_comparison = None
        if old_mapping is not None:
            mapping_diff_threshold = getattr(self.config, 'MAPPING_DIFF_THRESHOLD', 2)
            mapping_comparison = hmm_labeler.compare_mapping(old_mapping, threshold=mapping_diff_threshold)
            logger.info(f"映射比对结果: {mapping_comparison['message']}")
        
        # 分析 regime 稳定性（检测异常频繁切换）
        switch_threshold = getattr(self.config, 'REGIME_SWITCH_WARNING_THRESHOLD', 10)
        stability_analysis = hmm_labeler.analyze_regime_stability(train_states, switch_threshold)
        
        # 计算驻留时间分布
        dwell_times = hmm_labeler.compute_dwell_times(train_states)
        logger.info(f"状态驻留时间分布: {dwell_times}")
        
        # 保存 HMM 模型（包含状态映射、profiles、转移矩阵等）
        hmm_labeler.save(hmm_path)
        
        # 分析市场状态（只用训练集分析，避免泄漏）
        regime_analysis = hmm_labeler.analyze_regimes(train_features, train_states)
        # 添加语义名称到分析结果
        regime_analysis['regime_name'] = regime_analysis['state'].map(regime_mapping)
        logger.info(f"\n训练集市场状态分析:\n{regime_analysis}")
        
        # 5. 准备 LSTM 训练数据
        logger.info("步骤 5/6: 准备 LSTM 多步预测训练数据...")
        
        # 获取损失权重配置
        horizon_loss_weights = getattr(self.config, 'HORIZON_LOSS_WEIGHTS', {
            't+1': 1.0, 't+2': 0.8, 't+3': 0.6, 't+4': 0.4
        })
        
        lstm_classifier = LSTMRegimeClassifier(
            n_states=hmm_labeler.n_states,  # 使用 HMM 的实际状态数量（可能被动态调整）
            sequence_length=sequence_length,  # 使用模型配置的序列长度
            lstm_units=lstm_units,  # 使用模型配置的 LSTM 单元数
            dense_units=dense_units,  # 使用模型配置的 Dense 单元数
            dropout_rate=dropout_rate,
            l2_lambda=self.config.L2_LAMBDA,
            use_batch_norm=self.config.USE_BATCH_NORM,
            learning_rate=self.config.LEARNING_RATE,
            prediction_horizons=prediction_horizons,
            horizon_loss_weights=horizon_loss_weights
        )
        
        # 使用多步数据准备方法
        X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict = lstm_classifier.prepare_multistep_data_split(
            train_features=train_features,
            train_labels=train_multistep_labels,
            val_features=val_features,
            val_labels=val_multistep_labels,
            test_features=test_features,
            test_labels=test_multistep_labels
        )
        
        # 6. 训练 LSTM
        logger.info("步骤 6/6: 训练 LSTM 多步预测模型...")
        model_path = self.config.get_model_path_for_version(version_id, symbol, "lstm", primary_timeframe)
        
        # 使用多步训练方法
        history = lstm_classifier.train_multistep(
            X_train, y_train_dict,
            X_val, y_val_dict,  # 验证集用于早停
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            early_stopping_patience=self.config.EARLY_STOPPING_PATIENCE,
            lr_reduce_patience=self.config.LR_REDUCE_PATIENCE,
            model_path=model_path,
            use_class_weight=self.config.USE_CLASS_WEIGHT
        )
        
        # 在独立测试集上评估模型（这才是真实的泛化性能）
        logger.info("在独立测试集上评估多步预测模型...")
        eval_results = {}
        val_eval = {}
        
        if X_test is not None and y_test_dict is not None:
            # 评估 t+1 预测（主要指标）
            y_test_t1 = y_test_dict['t+1']
            y_pred_t1 = lstm_classifier.predict(X_test)
            from sklearn.metrics import accuracy_score
            test_acc_t1 = accuracy_score(y_test_t1, y_pred_t1)
            eval_results['accuracy'] = test_acc_t1
            eval_results['loss'] = 0.0  # 需要从模型获取
            logger.info(f"🎯 测试集 t+1 准确率: {test_acc_t1:.4f} (这是真实的泛化性能)")
            
            # 评估其他 horizon（如果存在）
            multistep_predictions = lstm_classifier.predict_multistep(X_test)
            for h in prediction_horizons:
                if h > 1:
                    # 对于软标签，比较 argmax
                    y_true_h = np.argmax(y_test_dict[f't+{h}'], axis=1)
                    y_pred_h = np.argmax(multistep_predictions[f't+{h}'], axis=1)
                    acc_h = accuracy_score(y_true_h, y_pred_h)
                    logger.info(f"    测试集 t+{h} 准确率: {acc_h:.4f}")
        else:
            # 使用验证集评估
            y_val_t1 = y_val_dict['t+1']
            y_pred_val_t1 = lstm_classifier.predict(X_val)
            from sklearn.metrics import accuracy_score
            val_acc_t1 = accuracy_score(y_val_t1, y_pred_val_t1)
            eval_results['accuracy'] = val_acc_t1
            eval_results['loss'] = 0.0
            logger.warning("⚠️ 没有独立测试集，使用验证集评估（结果可能偏乐观）")
        
        # 验证集准确率作为参考
        y_val_t1 = y_val_dict['t+1']
        y_pred_val = lstm_classifier.predict(X_val)
        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(y_val_t1, y_pred_val)
        val_eval['accuracy'] = val_acc
        logger.info(f"验证集 t+1 准确率: {val_acc:.4f}")
        
        # 保存模型和标准化器
        scaler_path = self.config.get_scaler_path_for_version(version_id, symbol, primary_timeframe)
        lstm_classifier.save(model_path, scaler_path)
        
        register_version(version_id, db_path=os.path.join(self.config.DATA_DIR, "model_registry.db"))
        try:
            cron_mgr = ForwardTestCronManager._instance
            forward_test_on_training_finished(symbol, primary_timeframe, version_id, self.config, cron_manager=cron_mgr)
        except Exception as e:
            logger.warning(f"Forward test enrollment failed (training result unchanged): {e}")
        logger.info(f"完整重训完成: {symbol} (primary_timeframe={primary_timeframe}) version_id={version_id}")
        logger.info(f"测试集准确率: {eval_results['accuracy']:.4f}")
        
        return {
            'symbol': symbol,
            'primary_timeframe': primary_timeframe,  # 主时间框架
            'version_id': version_id,
            'training_type': 'full_retrain',
            'timestamp': datetime.now(),
            'test_accuracy': eval_results['accuracy'],
            'val_accuracy': val_eval['accuracy'],
            'test_loss': eval_results.get('loss', 0.0),
            'regime_analysis': regime_analysis,
            'regime_mapping': regime_mapping,  # HMM 状态到语义名称的映射
            'mapping_comparison': mapping_comparison,  # 新旧映射比对结果
            'stability_analysis': stability_analysis,  # regime 稳定性分析
            'state_distribution_check': state_distribution_check,  # 状态分布健康检查
            'dwell_times': dwell_times,  # 状态驻留时间分布
            'training_bic': hmm_labeler.training_bic_,  # HMM 训练的 BIC 值
            'bic_validation': bic_validation,  # BIC 状态数量验证结果
            'n_states_optimization': n_states_optimization,  # 动态状态数量优化结果
            'final_n_states': hmm_labeler.n_states,  # 最终使用的状态数量
            'sequence_length': sequence_length,  # 序列长度
            'prediction_horizons': prediction_horizons,  # 多步预测步数
            'is_multistep': lstm_classifier.is_multistep,  # 是否多步模型
            'history': history,
            'data_split': {
                'train_samples': len(train_features),
                'val_samples': len(val_features),
                'test_samples': len(test_features)
            }
        }
    
    def incremental_train(self, symbol: str, primary_timeframe: str = None, version_id: str = None) -> Dict:
        """
        增量训练（在现有模型基础上）
        
        Args:
            symbol: 交易对
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
            version_id: 版本目录 id；若为 None 则自动分配
            
        Returns:
            训练结果
        """
        # 获取模型配置
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        if version_id is None:
            version_id = allocate_version_id(models_dir=self.config.MODELS_DIR)
        # 确保版本目录存在
        symbol_tf_dir = os.path.join(self.config.get_version_dir(version_id), symbol, primary_timeframe)
        os.makedirs(symbol_tf_dir, exist_ok=True)
        
        model_config = self.config.get_model_config(primary_timeframe)
        timeframes = model_config["timeframes"]
        
        logger.info(f"="*80)
        logger.info(f"开始增量训练: {symbol} (primary_timeframe={primary_timeframe}) version_id={version_id}")
        logger.info(f"="*80)
        
        # 1. 获取最新数据
        logger.info("步骤 1/4: 获取最新数据...")
        data = self.data_fetcher.fetch_latest_data(
            symbol=symbol,
            timeframes=timeframes,
            days=self.config.INCREMENTAL_TRAIN_DAYS
        )
        
        # 输出 API 统计信息
        stats = self.data_fetcher.get_api_stats()
        logger.info(f"API 请求统计: {stats}")
        
        # 2. 特征工程
        logger.info("步骤 2/4: 计算技术指标...")
        features = self.feature_engineer.combine_timeframe_features(
            data,
            primary_timeframe=primary_timeframe,
            symbol=symbol
        )
        
        # 3. 加载 HMM 模型并标注（从 PROD 路径加载）
        logger.info("步骤 3/4: HMM 状态标注...")
        hmm_path_load = self.config.get_prod_hmm_path(symbol, primary_timeframe)
        
        if not os.path.exists(hmm_path_load):
            logger.warning(f"HMM 模型不存在，将执行完整重训: {hmm_path_load}")
            return self.full_retrain(symbol, primary_timeframe, version_id=version_id)
        
        hmm_labeler = HMMRegimeLabeler.load(hmm_path_load)
        
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
        
        # 4. 加载 LSTM 模型并增量训练（从 PROD 路径加载）
        logger.info("步骤 4/4: LSTM 增量训练...")
        model_path_load = self.config.get_prod_model_path(symbol, "lstm", primary_timeframe)
        scaler_path_load = self.config.get_prod_scaler_path(symbol, primary_timeframe)
        
        if not os.path.exists(model_path_load):
            logger.warning(f"LSTM 模型不存在，将执行完整重训: {model_path_load}")
            return self.full_retrain(symbol, primary_timeframe, version_id=version_id)
        
        lstm_classifier = LSTMRegimeClassifier.load(model_path_load, scaler_path_load)
        
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
        
        # 保存更新后的模型到版本目录
        model_path = self.config.get_model_path_for_version(version_id, symbol, "lstm", primary_timeframe)
        scaler_path = self.config.get_scaler_path_for_version(version_id, symbol, primary_timeframe)
        lstm_classifier.save(model_path, scaler_path)
        
        register_version(version_id, db_path=os.path.join(self.config.DATA_DIR, "model_registry.db"))
        try:
            cron_mgr = ForwardTestCronManager._instance
            forward_test_on_training_finished(symbol, primary_timeframe, version_id, self.config, cron_manager=cron_mgr)
        except Exception as e:
            logger.warning(f"Forward test enrollment failed (training result unchanged): {e}")
        logger.info(f"增量训练完成: {symbol} (primary_timeframe={primary_timeframe}) version_id={version_id}")
        
        return {
            'symbol': symbol,
            'primary_timeframe': primary_timeframe,
            'version_id': version_id,
            'training_type': 'incremental',
            'timestamp': datetime.now(),
            'samples_used': len(X)
        }
    
    def train_all_symbols(self, training_type: str = 'full', primary_timeframe: str = None) -> Dict:
        """
        训练所有交易对（一个 version_id 对应本次调用的所有 symbol）
        
        Args:
            training_type: 'full' 或 'incremental'
            primary_timeframe: 主时间框架（如 "5m", "15m" 或 "1h"），如果为 None 则使用默认配置
            
        Returns:
            所有交易对的训练结果
        """
        version_id = allocate_version_id(models_dir=self.config.MODELS_DIR)
        results = {}
        
        for symbol in self.config.SYMBOLS:
            try:
                if training_type == 'full':
                    result = self.full_retrain(symbol, primary_timeframe, version_id=version_id)
                else:
                    result = self.incremental_train(symbol, primary_timeframe, version_id=version_id)
                
                results[symbol] = result
                
            except Exception as e:
                logger.error(f"训练 {symbol} 时出错: {e}", exc_info=True)
                results[symbol] = {'error': str(e)}
        
        return results
    
    def train_multi_timeframe_models(
        self, 
        symbol: str, 
        timeframes: list = None, 
        training_type: str = 'full',
        version_id: str = None
    ) -> Dict:
        """
        为单个交易对训练多个时间框架的模型（一个 version_id 对应本次调用的所有 timeframe）
        
        Args:
            symbol: 交易对
            timeframes: 要训练的时间框架列表（如 ["5m", "15m", "1h"]），如果为 None 则使用 ENABLED_MODELS
            training_type: 'full' 或 'incremental'
            version_id: 版本目录 id；若为 None 则自动分配
            
        Returns:
            各时间框架的训练结果
        """
        if timeframes is None:
            timeframes = self.config.ENABLED_MODELS
        
        if version_id is None:
            version_id = allocate_version_id(models_dir=self.config.MODELS_DIR)
        
        results = {}
        
        for tf in timeframes:
            logger.info(f"\n{'='*80}")
            logger.info(f"训练 {symbol} 的 {tf} 模型... (version_id={version_id})")
            logger.info(f"{'='*80}\n")
            
            try:
                if training_type == 'full':
                    result = self.full_retrain(symbol, primary_timeframe=tf, version_id=version_id)
                else:
                    result = self.incremental_train(symbol, primary_timeframe=tf, version_id=version_id)
                
                results[tf] = result
                
            except Exception as e:
                logger.error(f"训练 {symbol} 的 {tf} 模型时出错: {e}", exc_info=True)
                results[tf] = {'error': str(e)}
        
        return results
    
    def train_all_multi_timeframe(
        self, 
        timeframes: list = None, 
        training_type: str = 'full'
    ) -> Dict:
        """
        为所有交易对训练多个时间框架的模型（一个 version_id 对应本次调用的全部 symbol×timeframe）
        
        Args:
            timeframes: 要训练的时间框架列表（如 ["5m", "15m", "1h"]），如果为 None 则使用 ENABLED_MODELS
            training_type: 'full' 或 'incremental'
            
        Returns:
            {symbol: {timeframe: result}} 格式的训练结果
        """
        if timeframes is None:
            timeframes = self.config.ENABLED_MODELS
        
        version_id = allocate_version_id(models_dir=self.config.MODELS_DIR)
        results = {}
        
        for symbol in self.config.SYMBOLS:
            logger.info(f"\n{'#'*80}")
            logger.info(f"开始训练 {symbol} 的所有时间框架模型: {timeframes} (version_id={version_id})")
            logger.info(f"{'#'*80}\n")
            
            results[symbol] = self.train_multi_timeframe_models(
                symbol, timeframes, training_type, version_id=version_id
            )
        
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
