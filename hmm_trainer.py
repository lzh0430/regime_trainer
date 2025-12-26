"""
HMM 训练模块 - 用于无监督市场状态标注

修复数据泄漏问题：
- fit() 只在训练集上拟合 scaler, PCA, HMM
- predict() 使用训练好的模型对新数据进行预测（不泄漏未来信息）

自动映射功能：
- auto_map_regimes() 根据特征统计自动将 HMM 状态映射到语义名称
- 解决了 HMM 状态编号任意性的问题
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)

# 默认的 regime 名称（按优先级排序）
DEFAULT_REGIME_NAMES = [
    "Strong_Trend",      # 高 ADX, 高趋势强度
    "Weak_Trend",        # 中等 ADX, 有一定趋势
    "Range",             # 低 ADX, 中等波动率
    "Choppy_High_Vol",   # 低 ADX, 高波动率
    "Volatility_Spike",  # 极高波动率
    "Squeeze",           # 极低波动率, 低 ADX
]

class HMMRegimeLabeler:
    """HMM 市场状态标注器"""
    
    def __init__(self, n_states: int = 6, n_components: int = 4, primary_timeframe: str = "15m"):
        """
        初始化
        
        Args:
            n_states: 隐藏状态数量（市场状态数）
            n_components: PCA 降维后的特征数
            primary_timeframe: 主时间框架（用于优先匹配特征列）
        """
        self.n_states = n_states
        self.n_components = n_components
        self.primary_timeframe = primary_timeframe
        self.hmm_model = None
        self.pca = None
        self.scaler = None
        self.feature_names_ = None  # 保存训练时使用的特征名称
        self.regime_mapping_ = None  # 状态 ID 到语义名称的映射 {state_id: regime_name}
        self.state_profiles_ = None  # 每个状态的特征 profile（用于审计）
        self.training_bic_ = None  # 训练时的 BIC 值
        self.transition_matrix_ = None  # 状态转移矩阵
    
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
        
        # 5. 保存 BIC 值和转移矩阵
        self.training_bic_ = self.hmm_model.bic(features_pca)
        self.compute_transition_matrix(states)
        
        logger.info(f"HMM 训练完成，BIC: {self.training_bic_:.2f}")
        logger.info(f"训练集状态分布: {np.bincount(states, minlength=self.n_states).tolist()}")
        logger.debug(
            "注意：HMM 在训练集上 fit，会自动将数据分配到 6 个状态，因此训练集一定有所有状态。"
            "验证/测试集用训练好的模型 predict，如果那段时间市场没有某种状态，就不会出现该状态。"
        )
        
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
            logger.info(f"验证集状态分布: {np.bincount(val_states, minlength=self.n_states).tolist()}")
        
        # 预测测试集状态
        test_states = None
        if test_features is not None and len(test_features) > 0:
            test_states = self.predict(test_features)
            logger.info(f"测试集状态分布: {np.bincount(test_states, minlength=self.n_states).tolist()}")
        
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
        """保存模型（包括状态映射、特征 profile、BIC 等完整信息）"""
        model_data = {
            'hmm_model': self.hmm_model,
            'pca': self.pca,
            'scaler': self.scaler,
            'n_states': self.n_states,
            'n_components': self.n_components,
            'primary_timeframe': self.primary_timeframe,  # 保存主时间框架
            'feature_names': self.feature_names_,  # 保存特征名称
            'regime_mapping': self.regime_mapping_,  # 保存状态到语义名称的映射
            'state_profiles': self.state_profiles_,  # 保存特征 profile（用于审计）
            'training_bic': self.training_bic_,  # 保存训练时的 BIC 值
            'transition_matrix': self.transition_matrix_,  # 保存转移矩阵
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"HMM 模型已保存到 {filepath}")
        if self.regime_mapping_:
            logger.info(f"已保存的状态映射: {self.regime_mapping_}")
        if self.training_bic_:
            logger.info(f"已保存的 BIC 值: {self.training_bic_:.2f}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HMMRegimeLabeler':
        """加载模型（包括状态映射、特征 profile、BIC 等完整信息）"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        labeler = cls(
            n_states=model_data['n_states'],
            n_components=model_data['n_components'],
            primary_timeframe=model_data.get('primary_timeframe', '15m')  # 向后兼容
        )
        labeler.hmm_model = model_data['hmm_model']
        labeler.pca = model_data['pca']
        labeler.scaler = model_data['scaler']
        labeler.feature_names_ = model_data.get('feature_names')  # 加载特征名称（向后兼容）
        labeler.regime_mapping_ = model_data.get('regime_mapping')  # 加载状态映射（向后兼容）
        labeler.state_profiles_ = model_data.get('state_profiles')  # 加载特征 profile
        labeler.training_bic_ = model_data.get('training_bic')  # 加载 BIC 值
        labeler.transition_matrix_ = model_data.get('transition_matrix')  # 加载转移矩阵
        
        logger.info(f"HMM 模型已从 {filepath} 加载")
        if labeler.feature_names_:
            logger.info(f"训练时使用的特征数: {len(labeler.feature_names_)}")
        if labeler.regime_mapping_:
            logger.info(f"已加载的状态映射: {labeler.regime_mapping_}")
        else:
            logger.warning(
                "模型中没有保存状态映射（旧版本模型）。"
                "建议重新训练模型以生成状态映射。"
            )
        if labeler.training_bic_:
            logger.info(f"训练时的 BIC 值: {labeler.training_bic_:.2f}")
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
    
    # ==================== 自动映射功能 ====================
    
    def _find_feature_column(self, features: pd.DataFrame, pattern: str) -> Optional[str]:
        """
        根据模式查找特征列名（优先匹配 primary timeframe）
        
        当存在多个时间框架的特征（如 5m_adx, 15m_adx, 1h_adx）时，
        优先返回 primary timeframe 的列，避免随机拾取导致命名不稳定。
        
        Args:
            features: 特征 DataFrame
            pattern: 要匹配的模式（不区分大小写）
            
        Returns:
            匹配的列名，如果没有找到则返回 None
        """
        matching_cols = []
        for col in features.columns:
            if pattern.lower() in col.lower():
                matching_cols.append(col)
        
        if not matching_cols:
            return None
        
        # 如果只有一个匹配，直接返回
        if len(matching_cols) == 1:
            return matching_cols[0]
        
        # 优先返回 primary timeframe 的列
        primary_tf = self.primary_timeframe.lower()
        for col in matching_cols:
            if col.lower().startswith(primary_tf):
                return col
        
        # 如果没有找到 primary timeframe 的列，记录警告并返回第一个
        logger.debug(
            f"未找到 {self.primary_timeframe} 时间框架的 '{pattern}' 特征，"
            f"使用第一个匹配的列: {matching_cols[0]}"
        )
        return matching_cols[0]
    
    def _safe_mean(self, df: pd.DataFrame, pattern: str) -> float:
        """
        安全地计算包含某模式的列的均值
        
        Args:
            df: DataFrame
            pattern: 列名模式
            
        Returns:
            均值，如果列不存在则返回 0.0
        """
        col = self._find_feature_column(df, pattern)
        if col is not None and col in df.columns:
            return df[col].mean()
        return 0.0
    
    def _calc_trend_strength(self, df: pd.DataFrame) -> float:
        """
        计算趋势强度（基于收益的方向一致性）
        
        Args:
            df: DataFrame
            
        Returns:
            趋势强度值
        """
        returns_col = self._find_feature_column(df, 'returns')
        if returns_col is not None and returns_col in df.columns:
            returns = df[returns_col]
            # 方向一致性：绝对收益均值 * 方向符号
            if len(returns) > 0:
                return abs(returns.mean()) * 1000  # 放大以便比较
        return 0.0
    
    def _calc_state_profile(self, features: pd.DataFrame, states: np.ndarray, state: int) -> Dict:
        """
        计算单个状态的特征 profile
        
        Args:
            features: 特征 DataFrame
            states: 状态标签数组
            state: 要分析的状态 ID
            
        Returns:
            状态的特征 profile 字典
        """
        mask = states == state
        state_features = features[mask]
        
        if len(state_features) == 0:
            return {
                'state': state,
                'count': 0,
                'adx_mean': 0.0,
                'atr_pct_mean': 0.0,
                'bb_width_mean': 0.0,
                'volatility_score': 0.0,
                'trend_strength': 0.0,
                'returns_abs_mean': 0.0,
            }
        
        # 计算关键指标
        adx_mean = self._safe_mean(state_features, 'adx')
        
        # 波动率相关（ATR 百分比、BB 宽度）
        atr_14 = self._safe_mean(state_features, 'atr_14')
        bb_width = self._safe_mean(state_features, 'bb_width')
        hl_pct = self._safe_mean(state_features, 'hl_pct')
        
        # 计算相对于价格的 ATR 百分比
        # 注意：atr_14 是绝对值，需要标准化
        # 使用 hl_pct 作为波动率的代理指标
        volatility_score = hl_pct if hl_pct > 0 else bb_width
        
        # 趋势强度
        trend_strength = self._calc_trend_strength(state_features)
        
        # 绝对收益均值
        returns_col = self._find_feature_column(state_features, 'returns')
        returns_abs_mean = 0.0
        if returns_col is not None:
            returns_abs_mean = state_features[returns_col].abs().mean()
        
        return {
            'state': state,
            'count': mask.sum(),
            'adx_mean': adx_mean,
            'atr_pct_mean': hl_pct,  # 使用 hl_pct 作为波动率指标
            'bb_width_mean': bb_width,
            'volatility_score': volatility_score,
            'trend_strength': trend_strength,
            'returns_abs_mean': returns_abs_mean,
        }
    
    def _select_best_fallback_name(
        self, 
        profile: Dict, 
        available_names: set, 
        adx_median: float, 
        vol_median: float
    ) -> str:
        """
        根据状态特征选择最合适的 fallback 名称
        
        不是随机选择，而是根据特征与各 regime 的典型特征进行匹配。
        
        典型特征：
        - Strong_Trend: 高 ADX (> median), 高趋势强度
        - Weak_Trend: 中等 ADX
        - Range: 低 ADX, 中等波动率
        - Choppy_High_Vol: 低 ADX, 高波动率
        - Volatility_Spike: 极高波动率
        - Squeeze: 极低波动率, 低 ADX
        
        Args:
            profile: 状态特征 profile
            available_names: 可用的 regime 名称集合
            adx_median: ADX 中位数
            vol_median: 波动率中位数
            
        Returns:
            最合适的 regime 名称
        """
        adx = profile['adx_mean']
        vol = profile['volatility_score']
        trend = profile['trend_strength']
        
        # 计算每个可用名称的匹配分数（越高越匹配）
        scores = {}
        
        for name in available_names:
            score = 0
            
            if name == 'Strong_Trend':
                # 高 ADX + 高趋势强度
                score = (adx / adx_median) + (trend * 10)
                
            elif name == 'Weak_Trend':
                # 中等 ADX
                if adx_median * 0.6 < adx < adx_median * 1.4:
                    score = 1.0 - abs(adx - adx_median) / adx_median
                else:
                    score = 0.1
                    
            elif name == 'Range':
                # 低 ADX + 中等波动率
                if adx < adx_median:
                    score = (1 - adx / adx_median) * 0.5
                    if vol_median * 0.5 < vol < vol_median * 1.5:
                        score += 0.5
                        
            elif name == 'Choppy_High_Vol':
                # 低 ADX + 高波动率
                if adx < adx_median and vol > vol_median:
                    score = (vol / vol_median) * (1 - adx / adx_median)
                    
            elif name == 'Volatility_Spike':
                # 极高波动率
                score = vol / vol_median if vol_median > 0 else 0
                
            elif name == 'Squeeze':
                # 极低波动率 + 低 ADX
                if vol < vol_median and adx < adx_median:
                    score = (1 - vol / vol_median) * (1 - adx / adx_median)
            
            scores[name] = score
        
        # 选择分数最高的名称
        best_name = max(scores, key=scores.get)
        
        logger.debug(
            f"Fallback 名称选择: ADX={adx:.2f}, vol={vol:.4f}, "
            f"scores={scores}, best={best_name}"
        )
        
        return best_name
    
    def auto_map_regimes(
        self, 
        features: pd.DataFrame, 
        states: np.ndarray,
        min_vol_for_spike: float = 0.02,
        max_vol_for_squeeze: float = 0.01,
        min_adx_for_strong_trend: float = 30,
        max_adx_for_squeeze: float = 20
    ) -> Dict[int, str]:
        """
        根据特征统计自动映射 HMM 状态到语义名称
        
        使用**相对阈值 + 绝对阈值护栏**的组合判断策略：
        - 相对阈值：基于所有状态的中位数倍数（适应不同市场条件）
        - 绝对阈值护栏：防止在极端市场条件下（如所有状态都低波动）出现误标记
        
        例如：Volatility_Spike 必须同时满足：
        - 相对条件：波动率 > 中位数 * 1.5
        - 绝对条件：波动率 > min_vol_for_spike (默认 0.02)
        
        Args:
            features: 特征 DataFrame
            states: HMM 预测的状态数组
            min_vol_for_spike: Volatility_Spike 的最小波动率阈值
            max_vol_for_squeeze: Squeeze 的最大波动率阈值
            min_adx_for_strong_trend: Strong_Trend 的最小 ADX 阈值
            max_adx_for_squeeze: Squeeze 的最大 ADX 阈值
            
        Returns:
            {state_id: regime_name} 映射字典
        """
        logger.info("开始自动映射 HMM 状态到语义名称...")
        logger.info(
            f"绝对阈值护栏: min_vol_spike={min_vol_for_spike}, max_vol_squeeze={max_vol_for_squeeze}, "
            f"min_adx_strong={min_adx_for_strong_trend}, max_adx_squeeze={max_adx_for_squeeze}"
        )
        
        # 计算每个状态的特征 profile
        profiles = []
        for state in range(self.n_states):
            profile = self._calc_state_profile(features, states, state)
            profiles.append(profile)
            logger.debug(
                f"State {state}: count={profile['count']}, "
                f"ADX={profile['adx_mean']:.2f}, "
                f"volatility={profile['volatility_score']:.4f}, "
                f"trend_strength={profile['trend_strength']:.4f}"
            )
        
        # 过滤掉空状态
        valid_profiles = [p for p in profiles if p['count'] > 0]
        
        if not valid_profiles:
            logger.warning("没有有效的状态，使用默认映射")
            return {i: f"State_{i}" for i in range(self.n_states)}
        
        # 计算统计量用于归一化比较（相对阈值基准）
        all_adx = [p['adx_mean'] for p in valid_profiles]
        all_vol = [p['volatility_score'] for p in valid_profiles]
        
        adx_median = np.median(all_adx) if all_adx else 25
        vol_median = np.median(all_vol) if all_vol else 0.01
        
        logger.info(f"相对阈值基准: ADX_median={adx_median:.2f}, vol_median={vol_median:.4f}")
        
        # 分配 regime 名称
        mapping = {}
        used_names = set()
        
        # 按优先级分配名称
        # 1. Volatility_Spike: 波动率最高的状态
        #    条件：波动率 > vol_median * 1.5 且 > min_vol_for_spike（绝对护栏）
        vol_sorted = sorted(valid_profiles, key=lambda x: x['volatility_score'], reverse=True)
        if vol_sorted:
            candidate = vol_sorted[0]
            relative_ok = candidate['volatility_score'] > vol_median * 1.5
            absolute_ok = candidate['volatility_score'] > min_vol_for_spike
            
            if relative_ok and absolute_ok:
                mapping[candidate['state']] = 'Volatility_Spike'
                used_names.add('Volatility_Spike')
                logger.info(f"State {candidate['state']} -> Volatility_Spike (volatility={candidate['volatility_score']:.4f})")
            elif relative_ok and not absolute_ok:
                logger.info(
                    f"State {candidate['state']} 满足相对条件但不满足绝对护栏 "
                    f"(volatility={candidate['volatility_score']:.4f} < {min_vol_for_spike})，不标记为 Volatility_Spike"
                )
        
        # 2. Squeeze: 波动率最低 + ADX 最低
        #    条件：波动率 < vol_median * 0.7 且 < max_vol_for_squeeze（绝对护栏）
        #          ADX < adx_median 且 < max_adx_for_squeeze（绝对护栏）
        remaining = [p for p in valid_profiles if p['state'] not in mapping]
        if remaining:
            squeeze_sorted = sorted(remaining, key=lambda x: x['volatility_score'] + x['adx_mean'] / 100)
            candidate = squeeze_sorted[0]
            
            vol_relative_ok = candidate['volatility_score'] < vol_median * 0.7
            vol_absolute_ok = candidate['volatility_score'] < max_vol_for_squeeze
            adx_relative_ok = candidate['adx_mean'] < adx_median
            adx_absolute_ok = candidate['adx_mean'] < max_adx_for_squeeze
            
            if vol_relative_ok and vol_absolute_ok and adx_relative_ok and adx_absolute_ok:
                mapping[candidate['state']] = 'Squeeze'
                used_names.add('Squeeze')
                logger.info(f"State {candidate['state']} -> Squeeze (volatility={candidate['volatility_score']:.4f}, ADX={candidate['adx_mean']:.2f})")
            elif (vol_relative_ok or adx_relative_ok) and not (vol_absolute_ok and adx_absolute_ok):
                logger.info(
                    f"State {candidate['state']} 满足相对条件但不满足绝对护栏，不标记为 Squeeze "
                    f"(volatility={candidate['volatility_score']:.4f}, ADX={candidate['adx_mean']:.2f})"
                )
        
        # 3. Strong_Trend: ADX 最高 + 趋势强度高
        #    条件：ADX > adx_median * 1.2 且 > min_adx_for_strong_trend（绝对护栏）
        remaining = [p for p in valid_profiles if p['state'] not in mapping]
        if remaining:
            trend_sorted = sorted(remaining, key=lambda x: x['adx_mean'] + x['trend_strength'] * 10, reverse=True)
            candidate = trend_sorted[0]
            
            relative_ok = candidate['adx_mean'] > adx_median * 1.2
            absolute_ok = candidate['adx_mean'] > min_adx_for_strong_trend
            
            if relative_ok and absolute_ok:
                mapping[candidate['state']] = 'Strong_Trend'
                used_names.add('Strong_Trend')
                logger.info(f"State {candidate['state']} -> Strong_Trend (ADX={candidate['adx_mean']:.2f})")
            elif relative_ok and not absolute_ok:
                logger.info(
                    f"State {candidate['state']} 满足相对条件但不满足绝对护栏 "
                    f"(ADX={candidate['adx_mean']:.2f} < {min_adx_for_strong_trend})，不标记为 Strong_Trend"
                )
        
        # 4. Choppy_High_Vol: 高波动 + 低 ADX（不需要绝对护栏，已由 Volatility_Spike 过滤）
        remaining = [p for p in valid_profiles if p['state'] not in mapping]
        if remaining:
            choppy_sorted = sorted(remaining, key=lambda x: x['volatility_score'] - x['adx_mean'] / 100, reverse=True)
            candidate = choppy_sorted[0]
            if candidate['volatility_score'] > vol_median and candidate['adx_mean'] < adx_median:
                mapping[candidate['state']] = 'Choppy_High_Vol'
                used_names.add('Choppy_High_Vol')
                logger.info(f"State {candidate['state']} -> Choppy_High_Vol (volatility={candidate['volatility_score']:.4f}, ADX={candidate['adx_mean']:.2f})")
        
        # 5. Weak_Trend: 中等 ADX
        remaining = [p for p in valid_profiles if p['state'] not in mapping]
        if remaining:
            weak_trend_sorted = sorted(remaining, key=lambda x: x['adx_mean'], reverse=True)
            candidate = weak_trend_sorted[0]
            if candidate['adx_mean'] > adx_median * 0.8:
                mapping[candidate['state']] = 'Weak_Trend'
                used_names.add('Weak_Trend')
                logger.info(f"State {candidate['state']} -> Weak_Trend (ADX={candidate['adx_mean']:.2f})")
        
        # 6. Range: 剩余的状态
        remaining = [p for p in valid_profiles if p['state'] not in mapping]
        for p in remaining:
            if 'Range' not in used_names:
                mapping[p['state']] = 'Range'
                used_names.add('Range')
                logger.info(f"State {p['state']} -> Range")
            else:
                # 如果还有剩余，根据特征选择最接近的名称（不是随机分配！）
                available_names = set(DEFAULT_REGIME_NAMES) - used_names
                if available_names:
                    # 根据特征选择最合适的名称
                    best_name = self._select_best_fallback_name(
                        p, available_names, adx_median, vol_median
                    )
                    mapping[p['state']] = best_name
                    used_names.add(best_name)
                    logger.info(
                        f"State {p['state']} -> {best_name} (fallback, "
                        f"ADX={p['adx_mean']:.2f}, vol={p['volatility_score']:.4f})"
                    )
                else:
                    mapping[p['state']] = f"State_{p['state']}"
                    logger.info(f"State {p['state']} -> State_{p['state']} (no available names)")
        
        # 确保所有状态都有映射
        for state in range(self.n_states):
            if state not in mapping:
                mapping[state] = f"State_{state}"
        
        # 映射合理性检查：验证分配的名称是否与特征一致
        self._validate_mapping(mapping, profiles, adx_median, vol_median)
        
        # 保存映射和 profiles
        self.regime_mapping_ = mapping
        self.state_profiles_ = profiles  # 保存特征 profile（用于审计）
        
        logger.info(f"自动映射完成: {mapping}")
        return mapping
    
    def _validate_mapping(
        self, 
        mapping: Dict[int, str], 
        profiles: List[Dict],
        adx_median: float,
        vol_median: float
    ):
        """
        验证映射结果是否合理
        
        检查分配的语义名称是否与状态特征一致，不一致时记录警告。
        """
        profile_dict = {p['state']: p for p in profiles}
        
        for state, name in mapping.items():
            if state not in profile_dict:
                continue
            p = profile_dict[state]
            
            # Strong_Trend 应该有较高的 ADX
            if name == 'Strong_Trend' and p['adx_mean'] < adx_median * 0.8:
                logger.warning(
                    f"⚠️ 映射可能不合理: State {state} 被映射为 {name}，"
                    f"但 ADX={p['adx_mean']:.2f} 低于中位数*0.8={adx_median*0.8:.2f}"
                )
            
            # Squeeze 应该有较低的波动率
            if name == 'Squeeze' and p['volatility_score'] > vol_median * 1.5:
                logger.warning(
                    f"⚠️ 映射可能不合理: State {state} 被映射为 {name}，"
                    f"但波动率={p['volatility_score']:.4f} 高于中位数*1.5={vol_median*1.5:.4f}"
                )
            
            # Volatility_Spike 应该有较高的波动率
            if name == 'Volatility_Spike' and p['volatility_score'] < vol_median * 0.8:
                logger.warning(
                    f"⚠️ 映射可能不合理: State {state} 被映射为 {name}，"
                    f"但波动率={p['volatility_score']:.4f} 低于中位数*0.8={vol_median*0.8:.4f}"
                )
    
    def get_regime_name(self, state_id: int) -> str:
        """
        获取状态 ID 对应的语义名称
        
        Args:
            state_id: 状态 ID
            
        Returns:
            语义名称
        """
        if self.regime_mapping_ is None:
            return f"State_{state_id}"
        return self.regime_mapping_.get(state_id, f"State_{state_id}")
    
    def get_regime_mapping(self) -> Dict[int, str]:
        """
        获取完整的状态映射
        
        Returns:
            {state_id: regime_name} 映射字典
        """
        if self.regime_mapping_ is None:
            return {i: f"State_{i}" for i in range(self.n_states)}
        return self.regime_mapping_.copy()
    
    # ==================== BIC 验证功能 ====================
    
    def validate_n_states(
        self, 
        features: pd.DataFrame, 
        n_states_range: List[int] = None,
        n_iter: int = 100
    ) -> Dict:
        """
        使用 BIC 验证状态数量是否合理
        
        Args:
            features: 特征 DataFrame
            n_states_range: 要测试的状态数量范围，默认 [4, 5, 6, 7, 8]
            n_iter: HMM 训练迭代次数
            
        Returns:
            包含各状态数量 BIC 值的字典
        """
        if n_states_range is None:
            n_states_range = [4, 5, 6, 7, 8]
        
        logger.info(f"开始 BIC 验证，测试状态数量: {n_states_range}")
        
        # 预处理数据
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=self.n_components)
        features_pca = pca.fit_transform(features_scaled)
        
        results = {}
        best_n_states = None
        best_bic = float('inf')
        
        for n_states in n_states_range:
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=n_iter,
                    random_state=42
                )
                model.fit(features_pca)
                bic = model.bic(features_pca)
                results[n_states] = {
                    'bic': bic,
                    'converged': model.monitor_.converged
                }
                
                if bic < best_bic:
                    best_bic = bic
                    best_n_states = n_states
                
                logger.info(f"  n_states={n_states}: BIC={bic:.2f}, converged={model.monitor_.converged}")
                
            except Exception as e:
                logger.warning(f"  n_states={n_states}: 训练失败 - {e}")
                results[n_states] = {'bic': None, 'error': str(e)}
        
        results['best_n_states'] = best_n_states
        results['best_bic'] = best_bic
        results['current_n_states'] = self.n_states
        results['recommendation'] = (
            f"建议使用 {best_n_states} 个状态（BIC={best_bic:.2f}）" 
            if best_n_states != self.n_states 
            else f"当前 {self.n_states} 个状态是最优选择"
        )
        
        logger.info(f"BIC 验证完成: {results['recommendation']}")
        return results
    
    # ==================== 转移矩阵分析 ====================
    
    def compute_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        计算状态转移矩阵（经验估计）
        
        Args:
            states: 状态序列
            
        Returns:
            转移矩阵 (n_states x n_states)
        """
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            transition_counts[from_state, to_state] += 1
        
        # 归一化为概率
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除以零
        transition_matrix = transition_counts / row_sums
        
        self.transition_matrix_ = transition_matrix
        return transition_matrix
    
    def compute_dwell_times(self, states: np.ndarray) -> Dict[int, Dict]:
        """
        计算每个状态的驻留时间分布
        
        Args:
            states: 状态序列
            
        Returns:
            每个状态的驻留时间统计
        """
        dwell_times = {i: [] for i in range(self.n_states)}
        
        if len(states) == 0:
            return {i: {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0} 
                    for i in range(self.n_states)}
        
        current_state = states[0]
        current_dwell = 1
        
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_dwell += 1
            else:
                dwell_times[current_state].append(current_dwell)
                current_state = states[i]
                current_dwell = 1
        
        # 记录最后一个状态的驻留时间
        dwell_times[current_state].append(current_dwell)
        
        # 计算统计量
        result = {}
        for state, times in dwell_times.items():
            if times:
                result[state] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
            else:
                result[state] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        return result
    
    def analyze_regime_stability(
        self, 
        states: np.ndarray, 
        switch_threshold: int = 10
    ) -> Dict:
        """
        分析 regime 稳定性（检测异常频繁切换）
        
        Args:
            states: 状态序列
            switch_threshold: 每小时切换次数警告阈值（假设数据是15分钟频率）
            
        Returns:
            稳定性分析结果
        """
        if len(states) < 2:
            return {'switches': 0, 'switch_rate': 0, 'warning': False}
        
        # 计算状态切换次数
        switches = np.sum(states[1:] != states[:-1])
        
        # 假设数据是15分钟频率，计算每小时切换次数
        # 每小时有 4 个15分钟周期
        hours = len(states) / 4
        switch_rate = switches / hours if hours > 0 else 0
        
        # 检查是否异常频繁切换
        warning = switch_rate > switch_threshold
        
        result = {
            'total_switches': int(switches),
            'switch_rate_per_hour': switch_rate,
            'warning': warning,
            'message': (
                f"⚠️ 异常频繁切换: {switch_rate:.1f} 次/小时 > {switch_threshold}" 
                if warning 
                else f"✓ 正常: {switch_rate:.1f} 次/小时"
            )
        }
        
        if warning:
            logger.warning(result['message'])
        else:
            logger.info(result['message'])
        
        return result
    
    # ==================== 状态分布检查功能 ====================
    
    def check_state_distribution(
        self,
        train_states: np.ndarray,
        val_states: Optional[np.ndarray],
        test_states: Optional[np.ndarray],
        min_samples_per_state: int = 10,
        min_ratio_per_state: float = 0.01
    ) -> Dict:
        """
        检查各数据集的状态分布是否健康
        
        检测以下问题：
        1. 某状态在验证集/测试集中完全缺失（样本数为 0）
        2. 某状态在验证集/测试集中样本过少（低于阈值）
        3. 训练集和验证集/测试集的分布差异过大
        
        Args:
            train_states: 训练集状态
            val_states: 验证集状态（可选）
            test_states: 测试集状态（可选）
            min_samples_per_state: 每个状态的最小样本数
            min_ratio_per_state: 每个状态的最小占比
            
        Returns:
            检查结果，包含 warnings 和 recommendations
        """
        result = {
            'healthy': True,
            'warnings': [],
            'missing_states': {
                'val': [],
                'test': []
            },
            'low_sample_states': {
                'val': [],
                'test': []
            },
            'distributions': {},
            'recommendations': []
        }
        
        # 计算训练集分布
        train_dist = np.bincount(train_states, minlength=self.n_states)
        train_total = len(train_states)
        train_ratios = train_dist / train_total
        result['distributions']['train'] = {
            'counts': train_dist.tolist(),
            'ratios': train_ratios.tolist()
        }
        
        # 检查验证集
        if val_states is not None and len(val_states) > 0:
            val_dist = np.bincount(val_states, minlength=self.n_states)
            val_total = len(val_states)
            val_ratios = val_dist / val_total
            result['distributions']['val'] = {
                'counts': val_dist.tolist(),
                'ratios': val_ratios.tolist()
            }
            
            # 检查缺失状态
            for state in range(self.n_states):
                state_name = self.get_regime_name(state)
                
                if val_dist[state] == 0:
                    result['healthy'] = False
                    result['missing_states']['val'].append(state)
                    warning = (
                        f"⚠️ 状态 {state} ({state_name}) 在验证集中完全缺失！"
                        f"训练集有 {train_dist[state]} 个样本 ({train_ratios[state]:.1%})"
                    )
                    result['warnings'].append(warning)
                    logger.warning(warning)
                    
                elif val_dist[state] < min_samples_per_state:
                    result['low_sample_states']['val'].append(state)
                    warning = (
                        f"⚠️ 状态 {state} ({state_name}) 在验证集中样本过少："
                        f"{val_dist[state]} 个 (< {min_samples_per_state})"
                    )
                    result['warnings'].append(warning)
                    logger.warning(warning)
                    
                elif val_ratios[state] < min_ratio_per_state:
                    result['low_sample_states']['val'].append(state)
                    warning = (
                        f"⚠️ 状态 {state} ({state_name}) 在验证集中占比过低："
                        f"{val_ratios[state]:.2%} (< {min_ratio_per_state:.1%})"
                    )
                    result['warnings'].append(warning)
                    logger.warning(warning)
        
        # 检查测试集
        if test_states is not None and len(test_states) > 0:
            test_dist = np.bincount(test_states, minlength=self.n_states)
            test_total = len(test_states)
            test_ratios = test_dist / test_total
            result['distributions']['test'] = {
                'counts': test_dist.tolist(),
                'ratios': test_ratios.tolist()
            }
            
            # 检查缺失状态
            for state in range(self.n_states):
                state_name = self.get_regime_name(state)
                
                if test_dist[state] == 0:
                    result['missing_states']['test'].append(state)
                    warning = (
                        f"⚠️ 状态 {state} ({state_name}) 在测试集中完全缺失！"
                        f"训练集有 {train_dist[state]} 个样本 ({train_ratios[state]:.1%})"
                    )
                    result['warnings'].append(warning)
                    logger.warning(warning)
        
        # 生成建议
        if result['missing_states']['val']:
            result['recommendations'].append(
                "验证集缺失某些状态，这是时间序列按时间划分的正常现象。"
                "HMM 在训练集上 fit 时会自动发现 6 个状态，但验证/测试时间段内"
                "可能没有出现某些市场状态（如 Volatility Spike 或 Squeeze）。"
                "可以尝试：1) 增加数据天数；2) 调整验证集比例；3) 接受这是真实市场的情况。"
            )
        
        if result['low_sample_states']['val']:
            result['recommendations'].append(
                "建议：验证集某些状态样本过少，early stopping 可能无法准确评估这些状态。"
                "考虑增大验证集比例（如从 15% 增加到 20%）。"
            )
        
        # 打印按语义名称排序的分布（便于跨训练比较）
        self._print_distribution_by_regime_name(
            train_states, val_states, test_states
        )
        
        # 打印总结
        if result['healthy']:
            logger.info("✓ 状态分布检查通过：所有状态在各数据集中都有足够样本")
        else:
            missing_val = len(result['missing_states']['val'])
            missing_test = len(result['missing_states']['test'])
            logger.warning(
                f"状态分布检查发现问题：验证集缺失 {missing_val} 个状态，测试集缺失 {missing_test} 个状态"
            )
            logger.info(
                "  📝 解释：这是时间序列按时间划分的正常现象。"
                "HMM 在训练集上 fit 时会自动聚类出 6 个状态，但验证/测试时间段内"
                "可能没有出现某些市场状态（如极端波动或极低波动期）。"
            )
            for rec in result['recommendations']:
                logger.info(f"  💡 {rec}")
        
        return result
    
    def _print_distribution_by_regime_name(
        self,
        train_states: np.ndarray,
        val_states: Optional[np.ndarray],
        test_states: Optional[np.ndarray]
    ):
        """
        按语义名称顺序打印状态分布（便于跨训练比较）
        
        语义名称顺序固定为：
        Choppy_High_Vol, Strong_Trend, Volatility_Spike, Weak_Trend, Range, Squeeze
        
        这样无论 HMM 状态编号如何变化，相同语义的状态总是在同一位置显示。
        
        注意：如果 regime_mapping_ 未初始化（如在 auto_optimize_n_states 中），
        此方法会跳过打印，避免输出全 0 的误导性信息。
        """
        # 如果没有 regime_mapping，跳过按语义名称打印（避免全 0 输出）
        if self.regime_mapping_ is None:
            logger.debug("跳过按语义名称打印（regime_mapping 未初始化）")
            return
        
        # 定义语义名称的固定顺序（与 config.py REGIME_NAMES 一致）
        SEMANTIC_ORDER = [
            "Choppy_High_Vol",   # 高波动无方向
            "Strong_Trend",      # 强趋势
            "Volatility_Spike",  # 波动率突增
            "Weak_Trend",        # 弱趋势
            "Range",             # 区间震荡
            "Squeeze"            # 低波动蓄势
        ]
        
        # 构建语义名称到状态编号的反向映射
        # 注意：到达这里时 regime_mapping_ 一定不是 None（已在上面检查）
        name_to_state = {name: state for state, name in self.regime_mapping_.items()}
        
        # 计算各数据集的分布
        train_dist = np.bincount(train_states, minlength=self.n_states)
        val_dist = np.bincount(val_states, minlength=self.n_states) if val_states is not None else None
        test_dist = np.bincount(test_states, minlength=self.n_states) if test_states is not None else None
        
        # 按语义名称顺序构建分布（转换为 Python int，避免打印 np.int64）
        train_by_name = []
        val_by_name = []
        test_by_name = []
        
        for name in SEMANTIC_ORDER:
            state = name_to_state.get(name)
            if state is not None and state < len(train_dist):
                train_by_name.append(int(train_dist[state]))
                if val_dist is not None:
                    val_by_name.append(int(val_dist[state]))
                if test_dist is not None:
                    test_by_name.append(int(test_dist[state]))
            else:
                train_by_name.append(0)
                if val_dist is not None:
                    val_by_name.append(0)
                if test_dist is not None:
                    test_by_name.append(0)
        
        # 打印按语义名称排序的分布
        logger.info("=" * 70)
        logger.info("状态分布（按语义名称顺序，便于跨训练比较）:")
        logger.info(f"  语义名称顺序: {SEMANTIC_ORDER}")
        logger.info(f"  训练集分布:   {train_by_name}")
        if val_dist is not None:
            logger.info(f"  验证集分布:   {val_by_name}")
        if test_dist is not None:
            logger.info(f"  测试集分布:   {test_by_name}")
        logger.info("=" * 70)
    
    # ==================== 映射比对功能 ====================
    
    def compare_mapping(self, old_mapping: Dict[int, str], threshold: int = 2) -> Dict:
        """
        比较新旧映射的差异（基于语义名称集合，而非 state id）
        
        注意：HMM 状态编号是任意的，两次训练即使发现相同的市场状态，
        编号也可能不同。因此我们比较**语义名称集合**而非按 state id 比较。
        
        例如：
        - 旧模型：{0: Strong_Trend, 1: Range}
        - 新模型：{0: Range, 1: Strong_Trend}
        这两个映射的语义是一致的，不应该报告差异。
        
        Args:
            old_mapping: 旧的状态映射
            threshold: 允许的语义名称差异数量（新增或消失的名称）
            
        Returns:
            比对结果
        """
        if self.regime_mapping_ is None:
            return {
                'identical': False,
                'semantic_diff_count': -1,
                'message': "当前模型没有状态映射",
                'needs_review': True
            }
        
        new_mapping = self.regime_mapping_
        
        # 提取语义名称集合（忽略 state id）
        old_names = set(old_mapping.values())
        new_names = set(new_mapping.values())
        
        # 计算语义差异
        names_added = new_names - old_names  # 新增的名称
        names_removed = old_names - new_names  # 消失的名称
        names_unchanged = old_names & new_names  # 保持不变的名称
        
        semantic_diff_count = len(names_added) + len(names_removed)
        semantic_identical = semantic_diff_count == 0
        needs_review = semantic_diff_count > threshold
        
        # 同时记录 state id 级别的变化（仅供参考）
        state_id_changes = []
        for state in range(self.n_states):
            old_name = old_mapping.get(state, f"State_{state}")
            new_name = new_mapping.get(state, f"State_{state}")
            if old_name != new_name:
                state_id_changes.append({
                    'state': state,
                    'old': old_name,
                    'new': new_name
                })
        
        result = {
            'semantic_identical': semantic_identical,
            'semantic_diff_count': semantic_diff_count,
            'names_added': list(names_added),
            'names_removed': list(names_removed),
            'names_unchanged': list(names_unchanged),
            'state_id_changes': state_id_changes,  # 仅供参考，不用于判断
            'threshold': threshold,
            'needs_review': needs_review,
            'message': self._build_comparison_message(
                semantic_identical, semantic_diff_count, 
                names_added, names_removed, state_id_changes, threshold
            )
        }
        
        if needs_review:
            logger.warning(result['message'])
        else:
            logger.info(result['message'])
        
        return result
    
    def _build_comparison_message(
        self, 
        semantic_identical: bool,
        semantic_diff_count: int,
        names_added: set,
        names_removed: set,
        state_id_changes: List[Dict],
        threshold: int
    ) -> str:
        """构建映射比对的消息"""
        if semantic_identical:
            if state_id_changes:
                return (
                    f"✓ 语义一致（state id 有 {len(state_id_changes)} 处重排，"
                    f"这是 HMM 正常行为，不影响语义）"
                )
            return "✓ 映射完全一致"
        
        parts = []
        if names_added:
            parts.append(f"新增: {names_added}")
        if names_removed:
            parts.append(f"消失: {names_removed}")
        
        if semantic_diff_count > threshold:
            return f"⚠️ 语义变化较大（{', '.join(parts)}），建议人工复核"
        else:
            return f"语义有 {semantic_diff_count} 处差异（{', '.join(parts)}），在可接受范围内"
    
    def get_state_profiles(self) -> Optional[List[Dict]]:
        """
        获取保存的状态特征 profile
        
        Returns:
            状态特征 profile 列表
        """
        return self.state_profiles_
    
    # ==================== 动态状态数量优化 ====================
    
    def auto_optimize_n_states(
        self,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        test_features: Optional[pd.DataFrame] = None,
        n_states_min: int = 4,
        n_states_max: int = 8,
        max_missing_allowed: int = 1,
        max_low_ratio_allowed: int = 2,
        strategy: str = "decrease_first",
        min_samples_per_state: int = 10,
        min_ratio_per_state: float = 0.01,
        n_iter: int = 100
    ) -> Dict:
        """
        自动优化状态数量，确保验证/测试集分布健康
        
        当验证/测试集中某些状态完全缺失或占比过低时，
        自动尝试调整状态数量，找到一个使分布更健康的值。
        
        Args:
            train_features: 训练集特征
            val_features: 验证集特征
            test_features: 测试集特征（可选）
            n_states_min: 最小状态数量
            n_states_max: 最大状态数量
            max_missing_allowed: 允许的最大缺失状态数量
            max_low_ratio_allowed: 允许的最大低占比状态数量
            strategy: 调整策略
                - "decrease_first": 优先减少状态数量
                - "bic_optimal": 使用 BIC 选择最优数量
            min_samples_per_state: 判断"样本过少"的阈值
            min_ratio_per_state: 判断"占比过低"的阈值
            n_iter: HMM 训练迭代次数
            
        Returns:
            优化结果，包含最优 n_states、各尝试的结果等
        """
        logger.info("=" * 70)
        logger.info("开始自动优化状态数量...")
        logger.info(f"  策略: {strategy}")
        logger.info(f"  状态数量范围: {n_states_min} - {n_states_max}")
        logger.info(f"  允许缺失状态数: {max_missing_allowed}")
        logger.info("=" * 70)
        
        original_n_states = self.n_states
        results = {}
        best_n_states = None
        best_score = float('-inf')
        
        # 确定尝试的状态数量顺序
        if strategy == "decrease_first":
            # 从当前值开始，优先向下尝试
            n_states_to_try = []
            for n in range(original_n_states, n_states_min - 1, -1):
                n_states_to_try.append(n)
            for n in range(original_n_states + 1, n_states_max + 1):
                n_states_to_try.append(n)
        else:
            # BIC 策略：尝试所有可能的值
            n_states_to_try = list(range(n_states_min, n_states_max + 1))
        
        for n_states in n_states_to_try:
            logger.info(f"\n尝试 n_states = {n_states}...")
            
            try:
                # 创建临时 HMM 实例
                temp_hmm = HMMRegimeLabeler(
                    n_states=n_states,
                    n_components=self.n_components,
                    primary_timeframe=self.primary_timeframe
                )
                
                # 训练并预测
                train_states, val_states, test_states = temp_hmm.fit_predict_split(
                    train_features, val_features, test_features, n_iter=n_iter
                )
                
                # 检查分布健康度
                dist_check = temp_hmm.check_state_distribution(
                    train_states, val_states, test_states,
                    min_samples_per_state, min_ratio_per_state
                )
                
                # 计算评分
                missing_val = len(dist_check['missing_states']['val'])
                missing_test = len(dist_check['missing_states']['test'])
                low_ratio_val = len(dist_check['low_sample_states']['val'])
                
                # 健康度评分（越高越好）
                # - 每个缺失状态扣 10 分
                # - 每个低占比状态扣 3 分
                # - BIC 越低加分（归一化到 0-5 分）
                health_score = 100 - (missing_val * 10) - (missing_test * 5) - (low_ratio_val * 3)
                
                # BIC 评分（需要在相同数据上比较才有意义）
                bic = temp_hmm.training_bic_ if temp_hmm.training_bic_ else float('inf')
                
                result = {
                    'n_states': n_states,
                    'bic': bic,
                    'missing_val': missing_val,
                    'missing_test': missing_test,
                    'low_ratio_val': low_ratio_val,
                    'health_score': health_score,
                    'is_healthy': dist_check['healthy'],
                    'train_dist': dist_check['distributions']['train']['counts'],
                    'val_dist': dist_check['distributions'].get('val', {}).get('counts', []),
                }
                
                results[n_states] = result
                
                logger.info(f"  BIC: {bic:.2f}")
                logger.info(f"  验证集缺失: {missing_val}, 测试集缺失: {missing_test}")
                logger.info(f"  健康评分: {health_score}")
                
                # 检查是否满足条件
                if missing_val <= max_missing_allowed and low_ratio_val <= max_low_ratio_allowed:
                    if health_score > best_score:
                        best_score = health_score
                        best_n_states = n_states
                        
                        # 如果是 decrease_first 策略且找到满足条件的，立即返回
                        if strategy == "decrease_first" and dist_check['healthy']:
                            logger.info(f"✓ 找到健康的状态数量: {n_states}")
                            break
                            
            except Exception as e:
                logger.warning(f"  n_states={n_states} 训练失败: {e}")
                results[n_states] = {'error': str(e)}
        
        # 如果没有找到完全健康的配置，选择最佳的
        if best_n_states is None:
            # 选择缺失最少的
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                best_n_states = min(
                    valid_results.keys(),
                    key=lambda k: (valid_results[k]['missing_val'], valid_results[k]['missing_test'])
                )
                logger.warning(f"未找到完全健康的配置，选择最佳: n_states={best_n_states}")
            else:
                best_n_states = original_n_states
                logger.warning(f"所有配置都失败，保持原值: n_states={best_n_states}")
        
        # 更新当前实例的 n_states
        if best_n_states != original_n_states:
            logger.info(f"🔄 状态数量调整: {original_n_states} -> {best_n_states}")
            self.n_states = best_n_states
        else:
            logger.info(f"✓ 保持原状态数量: {best_n_states}")
        
        optimization_result = {
            'original_n_states': original_n_states,
            'optimal_n_states': best_n_states,
            'adjusted': best_n_states != original_n_states,
            'strategy': strategy,
            'all_results': results,
            'best_result': results.get(best_n_states, {}),
            'message': (
                f"状态数量从 {original_n_states} 调整为 {best_n_states}"
                if best_n_states != original_n_states
                else f"状态数量 {best_n_states} 已是最优"
            )
        }
        
        logger.info("=" * 70)
        logger.info(f"状态数量优化完成: {optimization_result['message']}")
        logger.info("=" * 70)
        
        return optimization_result
    
    def retrain_with_n_states(
        self,
        n_states: int,
        train_features: pd.DataFrame,
        val_features: Optional[pd.DataFrame] = None,
        test_features: Optional[pd.DataFrame] = None,
        n_iter: int = 100
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        使用新的状态数量重新训练
        
        Args:
            n_states: 新的状态数量
            train_features: 训练集特征
            val_features: 验证集特征
            test_features: 测试集特征
            n_iter: HMM 训练迭代次数
            
        Returns:
            (train_states, val_states, test_states)
        """
        logger.info(f"使用 n_states={n_states} 重新训练...")
        
        self.n_states = n_states
        
        # 重置模型
        self.hmm_model = None
        self.pca = None
        self.scaler = None
        self.regime_mapping_ = None
        self.state_profiles_ = None
        
        return self.fit_predict_split(train_features, val_features, test_features, n_iter)
