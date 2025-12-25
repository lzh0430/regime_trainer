"""
HMM 训练模块 - 用于无监督市场状态标注

修复数据泄漏问题：
- fit() 只在训练集上拟合 scaler, PCA, HMM
- predict() 使用训练好的模型对新数据进行预测（不泄漏未来信息）
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class HMMRegimeLabeler:
    """HMM 市场状态标注器"""
    
    def __init__(self, n_states: int = 6, n_components: int = 4):
        """
        初始化
        
        Args:
            n_states: 隐藏状态数量（市场状态数）
            n_components: PCA 降维后的特征数
        """
        self.n_states = n_states
        self.n_components = n_components
        self.hmm_model = None
        self.pca = None
        self.scaler = None
        self.feature_names_ = None  # 保存训练时使用的特征名称
    
    def fit(self, features: pd.DataFrame, n_iter: int = 100) -> np.ndarray:
        """
        训练 HMM 并标注市场状态
        
        注意：此方法只应在训练集上调用，避免数据泄漏。
        对于验证集和测试集，应使用 predict() 方法。
        
        Args:
            features: 特征 DataFrame（应只包含训练集数据）
            n_iter: HMM 训练迭代次数
            
        Returns:
            状态标签数组（训练集的标签）
        """
        logger.info(f"开始 HMM 训练，特征维度: {features.shape}")
        logger.info("注意：HMM 只在训练集上拟合，避免数据泄漏")
        
        # 保存特征名称（用于后续预测时确保特征一致）
        self.feature_names_ = list(features.columns)
        
        # 1. 标准化（只在训练集上 fit）
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # 2. PCA 降维（只在训练集上 fit）
        self.pca = PCA(n_components=self.n_components)
        features_pca = self.pca.fit_transform(features_scaled)
        
        logger.info(f"PCA 解释方差比: {self.pca.explained_variance_ratio_}")
        logger.info(f"PCA 累计解释方差: {np.cumsum(self.pca.explained_variance_ratio_)}")
        
        # 3. 训练 HMM（只在训练集上 fit）
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            verbose=True
        )
        
        self.hmm_model.fit(features_pca)
        
        # 4. 预测训练集的状态
        states = self.hmm_model.predict(features_pca)
        
        logger.info(f"HMM 训练完成，BIC: {self.hmm_model.bic(features_pca):.2f}")
        logger.info(f"训练集状态分布: {np.bincount(states)}")
        
        return states
    
    def fit_predict_split(
        self, 
        train_features: pd.DataFrame, 
        val_features: Optional[pd.DataFrame] = None,
        test_features: Optional[pd.DataFrame] = None,
        n_iter: int = 100
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        在训练集上拟合 HMM，并分别预测训练/验证/测试集的状态标签
        
        这是推荐的方法，可以避免数据泄漏：
        - Scaler 只在训练集上 fit
        - PCA 只在训练集上 fit  
        - HMM 只在训练集上 fit
        - 验证集和测试集只做 transform 和 predict
        
        Args:
            train_features: 训练集特征
            val_features: 验证集特征（可选）
            test_features: 测试集特征（可选）
            n_iter: HMM 训练迭代次数
            
        Returns:
            (train_states, val_states, test_states) - 各数据集的状态标签
        """
        # 在训练集上拟合
        train_states = self.fit(train_features, n_iter=n_iter)
        
        # 预测验证集状态（使用训练集拟合的 scaler/PCA/HMM）
        val_states = None
        if val_features is not None and len(val_features) > 0:
            val_states = self.predict(val_features)
            logger.info(f"验证集状态分布: {np.bincount(val_states, minlength=self.n_states)}")
        
        # 预测测试集状态
        test_states = None
        if test_features is not None and len(test_features) > 0:
            test_states = self.predict(test_features)
            logger.info(f"测试集状态分布: {np.bincount(test_states, minlength=self.n_states)}")
        
        return train_states, val_states, test_states
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        预测市场状态
        
        Args:
            features: 特征 DataFrame
            
        Returns:
            状态标签数组
        """
        if self.hmm_model is None:
            raise ValueError("模型尚未训练，请先调用 fit()")
        
        # 确保特征列与训练时一致
        if self.feature_names_ is not None:
            # 检查是否有缺失的特征
            missing_features = set(self.feature_names_) - set(features.columns)
            if missing_features:
                # 尝试添加缺失的特征（填充为0）
                logger.warning(
                    f"缺少 {len(missing_features)} 个训练时使用的特征，将填充为0: "
                    f"{list(missing_features)[:5]}..."
                )
                for feat in missing_features:
                    features[feat] = 0.0
            
            # 检查是否有额外的特征
            extra_features = set(features.columns) - set(self.feature_names_)
            if extra_features:
                logger.debug(f"移除 {len(extra_features)} 个训练时未使用的特征")
            
            # 只选择训练时使用的特征，并按照训练时的顺序排列
            features = features[self.feature_names_]
        else:
            # 向后兼容：如果没有保存特征名称，检查特征数量
            expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None
            if expected_features and len(features.columns) != expected_features:
                raise ValueError(
                    f"特征数量不匹配！训练时: {expected_features} 个特征, "
                    f"当前: {len(features.columns)} 个特征\n"
                    f"这是旧版本模型，请重新训练模型以保存特征名称。"
                )
            logger.warning(
                "模型中没有保存特征名称（旧版本模型）。"
                "建议重新训练模型以确保特征一致性。"
            )
        
        # 应用相同的预处理
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        states = self.hmm_model.predict(features_pca)
        return states
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        预测市场状态的概率分布
        
        Args:
            features: 特征 DataFrame
            
        Returns:
            状态概率矩阵 (n_samples, n_states)
        """
        if self.hmm_model is None:
            raise ValueError("模型尚未训练，请先调用 fit()")
        
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        # 计算后验概率
        log_prob, posteriors = self.hmm_model.score_samples(features_pca)
        
        return posteriors
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'hmm_model': self.hmm_model,
            'pca': self.pca,
            'scaler': self.scaler,
            'n_states': self.n_states,
            'n_components': self.n_components,
            'feature_names': self.feature_names_  # 保存特征名称
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"HMM 模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HMMRegimeLabeler':
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        labeler = cls(
            n_states=model_data['n_states'],
            n_components=model_data['n_components']
        )
        labeler.hmm_model = model_data['hmm_model']
        labeler.pca = model_data['pca']
        labeler.scaler = model_data['scaler']
        labeler.feature_names_ = model_data.get('feature_names')  # 加载特征名称（向后兼容）
        
        logger.info(f"HMM 模型已从 {filepath} 加载")
        if labeler.feature_names_:
            logger.info(f"训练时使用的特征数: {len(labeler.feature_names_)}")
        return labeler
    
    def analyze_regimes(self, features: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
        """
        分析不同市场状态的特征
        
        Args:
            features: 特征 DataFrame
            states: 状态标签
            
        Returns:
            每个状态的统计信息
        """
        regime_stats = []
        
        for state in range(self.n_states):
            mask = states == state
            state_features = features[mask]
            
            stats = {
                'state': state,
                'count': mask.sum(),
                'percentage': mask.sum() / len(states) * 100,
            }
            
            # 计算一些关键指标的平均值
            if len(state_features) > 0:
                for col in features.columns:
                    if 'returns' in col.lower():
                        stats[f'{col}_mean'] = state_features[col].mean()
                    elif 'atr' in col.lower() or 'volatility' in col.lower():
                        stats[f'{col}_mean'] = state_features[col].mean()
                    elif 'adx' in col.lower():
                        stats[f'{col}_mean'] = state_features[col].mean()
            
            regime_stats.append(stats)
        
        return pd.DataFrame(regime_stats)
