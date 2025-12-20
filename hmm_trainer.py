"""
HMM 训练模块 - 用于无监督市场状态标注
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from typing import Tuple

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
    
    def fit(self, features: pd.DataFrame, n_iter: int = 100) -> np.ndarray:
        """
        训练 HMM 并标注市场状态
        
        Args:
            features: 特征 DataFrame
            n_iter: HMM 训练迭代次数
            
        Returns:
            状态标签数组
        """
        logger.info(f"开始 HMM 训练，特征维度: {features.shape}")
        
        # 1. 标准化
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # 2. PCA 降维
        self.pca = PCA(n_components=self.n_components)
        features_pca = self.pca.fit_transform(features_scaled)
        
        logger.info(f"PCA 解释方差比: {self.pca.explained_variance_ratio_}")
        logger.info(f"PCA 累计解释方差: {np.cumsum(self.pca.explained_variance_ratio_)}")
        
        # 3. 训练 HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            verbose=True
        )
        
        self.hmm_model.fit(features_pca)
        
        # 4. 预测状态
        states = self.hmm_model.predict(features_pca)
        
        logger.info(f"HMM 训练完成，BIC: {self.hmm_model.bic(features_pca):.2f}")
        logger.info(f"状态分布: {np.bincount(states)}")
        
        return states
    
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
            'n_components': self.n_components
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
        
        logger.info(f"HMM 模型已从 {filepath} 加载")
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


def create_labeled_dataset(
    features: pd.DataFrame,
    states: np.ndarray,
    sequence_length: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建带标签的序列数据集（用于 LSTM 训练）
    
    Args:
        features: 特征 DataFrame
        states: HMM 状态标签
        sequence_length: 序列长度
        
    Returns:
        (X, y) - X 是序列特征，y 是对应的状态标签
    """
    X, y = [], []
    
    features_array = features.values
    
    for i in range(len(features_array) - sequence_length):
        X.append(features_array[i:i+sequence_length])
        y.append(states[i+sequence_length])
    
    return np.array(X), np.array(y)
