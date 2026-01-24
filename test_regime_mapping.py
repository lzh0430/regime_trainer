#!/usr/bin/env python3
"""
Regime 映射逻辑单元测试

测试 HMM 聚类到 market regime 的映射逻辑是否正确。
包括：
1. 各 regime 的典型特征映射测试
2. 边界条件测试
3. Fallback 逻辑测试
4. 验证逻辑测试
"""
import unittest
import numpy as np
import pandas as pd
import logging
from unittest.mock import patch, MagicMock
from hmm_trainer import HMMRegimeLabeler, DEFAULT_REGIME_NAMES

# 设置日志级别为 WARNING 以减少测试输出
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TestTrendStrengthCalculation(unittest.TestCase):
    """测试趋势强度计算"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_strong_uptrend_direction_consistency(self):
        """测试强上升趋势的方向一致性"""
        # 创建全正收益的 DataFrame（强上升趋势）
        df = pd.DataFrame({
            '15m_returns': [0.01, 0.02, 0.015, 0.01, 0.02, 0.015, 0.01, 0.02]
        })
        
        strength = self.labeler._calc_trend_strength(df)
        
        # 方向一致性 = 1.0（全正），所以趋势强度应该较高
        self.assertGreater(strength, 0)
        
    def test_strong_downtrend_direction_consistency(self):
        """测试强下降趋势的方向一致性"""
        # 创建全负收益的 DataFrame（强下降趋势）
        df = pd.DataFrame({
            '15m_returns': [-0.01, -0.02, -0.015, -0.01, -0.02, -0.015, -0.01, -0.02]
        })
        
        strength = self.labeler._calc_trend_strength(df)
        
        # 方向一致性 = -1.0（全负），绝对值为 1，趋势强度应该较高
        self.assertGreater(strength, 0)
        
    def test_choppy_no_direction(self):
        """测试震荡行情的方向一致性（无明确方向）"""
        # 创建交替正负收益的 DataFrame（震荡）
        df = pd.DataFrame({
            '15m_returns': [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
        })
        
        strength = self.labeler._calc_trend_strength(df)
        
        # 方向一致性 ≈ 0（交替），趋势强度应该很低
        self.assertLess(strength, 0.1)
        
    def test_empty_returns(self):
        """测试空收益数据"""
        df = pd.DataFrame({
            '15m_returns': []
        })
        
        strength = self.labeler._calc_trend_strength(df)
        self.assertEqual(strength, 0.0)


class TestRegimeMappingBasic(unittest.TestCase):
    """测试基本的 regime 映射逻辑"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def _create_mock_features_and_states(self, profiles_data):
        """
        创建模拟的特征数据和状态数组
        
        Args:
            profiles_data: list of dicts，每个 dict 包含 state, adx, vol, trend, count
        """
        # 创建状态数组
        states = []
        features_data = []
        
        for profile in profiles_data:
            state = profile['state']
            count = profile.get('count', 100)
            adx = profile['adx']
            vol = profile['vol']
            trend = profile.get('trend', 0.5)
            
            # 为每个状态创建 count 个样本
            for _ in range(count):
                states.append(state)
                features_data.append({
                    '15m_adx': adx + np.random.normal(0, 1),
                    '15m_hl_pct': vol + np.random.normal(0, 0.001),
                    '15m_bb_width': vol * 2,
                    '15m_returns': trend / 1000 * (1 if np.random.random() > 0.5 else -1),
                })
        
        states = np.array(states)
        features = pd.DataFrame(features_data)
        
        return features, states
    

class TestVolatilitySpikeMapping(unittest.TestCase):
    """测试 Volatility_Spike 映射"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_high_volatility_maps_to_spike(self):
        """高波动率应该映射到 Volatility_Spike"""
        # 创建一个高波动率的状态 profile
        profiles = [
            {'state': 0, 'adx': 25, 'vol': 0.05, 'count': 100},  # 极高波动
            {'state': 1, 'adx': 30, 'vol': 0.01, 'count': 100},  # 正常
            {'state': 2, 'adx': 20, 'vol': 0.01, 'count': 100},
            {'state': 3, 'adx': 15, 'vol': 0.008, 'count': 100},
            {'state': 4, 'adx': 35, 'vol': 0.012, 'count': 100},
            {'state': 5, 'adx': 22, 'vol': 0.009, 'count': 100},
        ]
        
        features, states = self._create_mock_data(profiles)
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # State 0 应该被映射为 Volatility_Spike
        self.assertEqual(mapping[0], 'Volatility_Spike')
    
    def test_absolute_threshold_guardrail(self):
        """测试绝对阈值护栏：波动率低于 0.02 不应标记为 Volatility_Spike"""
        # 所有状态波动率都很接近，这样相对条件（> median * 1.5）不会满足
        # 即使 state 0 波动率最高，但相对中位数不够高
        profiles = [
            {'state': 0, 'adx': 25, 'vol': 0.012, 'count': 100},  # 相对最高，但 < 0.02 且 < median * 1.5
            {'state': 1, 'adx': 30, 'vol': 0.010, 'count': 100},
            {'state': 2, 'adx': 20, 'vol': 0.009, 'count': 100},
            {'state': 3, 'adx': 15, 'vol': 0.008, 'count': 100},
            {'state': 4, 'adx': 35, 'vol': 0.011, 'count': 100},
            {'state': 5, 'adx': 22, 'vol': 0.010, 'count': 100},
        ]
        
        features, states = self._create_mock_data(profiles)
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # State 0 不应该被映射为 Volatility_Spike
        # 因为波动率 0.012 < 0.02（绝对护栏）且 0.012 < median(0.01) * 1.5 = 0.015（相对条件）
        self.assertNotEqual(mapping[0], 'Volatility_Spike')
    
    def _create_mock_data(self, profiles):
        """创建模拟数据"""
        states = []
        features_data = []
        
        for profile in profiles:
            state = profile['state']
            count = profile.get('count', 100)
            adx = profile['adx']
            vol = profile['vol']
            
            for _ in range(count):
                states.append(state)
                features_data.append({
                    '15m_adx': adx + np.random.normal(0, 0.5),
                    '15m_hl_pct': vol + np.random.normal(0, 0.0005),
                    '15m_bb_width': vol * 2,
                    '15m_returns': np.random.normal(0, 0.001),
                })
        
        return pd.DataFrame(features_data), np.array(states)


class TestSqueezeMapping(unittest.TestCase):
    """测试 Squeeze 映射"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_low_volatility_low_adx_maps_to_squeeze(self):
        """低波动 + 低 ADX 应该映射到 Squeeze"""
        profiles = [
            {'state': 0, 'adx': 15, 'vol': 0.005, 'count': 100},  # 低波动 + 低 ADX
            {'state': 1, 'adx': 35, 'vol': 0.015, 'count': 100},  # 高 ADX
            {'state': 2, 'adx': 25, 'vol': 0.025, 'count': 100},  # 高波动
            {'state': 3, 'adx': 28, 'vol': 0.012, 'count': 100},
            {'state': 4, 'adx': 22, 'vol': 0.018, 'count': 100},
            {'state': 5, 'adx': 30, 'vol': 0.014, 'count': 100},
        ]
        
        features, states = self._create_mock_data(profiles)
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # State 0 应该被映射为 Squeeze
        self.assertEqual(mapping[0], 'Squeeze')
    
    def _create_mock_data(self, profiles):
        """创建模拟数据"""
        states = []
        features_data = []
        
        for profile in profiles:
            state = profile['state']
            count = profile.get('count', 100)
            adx = profile['adx']
            vol = profile['vol']
            
            for _ in range(count):
                states.append(state)
                features_data.append({
                    '15m_adx': adx + np.random.normal(0, 0.5),
                    '15m_hl_pct': vol + np.random.normal(0, 0.0005),
                    '15m_bb_width': vol * 2,
                    '15m_returns': np.random.normal(0, 0.001),
                })
        
        return pd.DataFrame(features_data), np.array(states)


class TestStrongTrendMapping(unittest.TestCase):
    """测试 Strong_Trend 映射"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_high_adx_high_trend_maps_to_strong_trend(self):
        """高 ADX + 高趋势强度应该映射到 Strong_Trend"""
        # 创建数据，state 1 有高 ADX 和高趋势强度（连续同向收益）
        states = []
        features_data = []
        
        # State 0: 中等
        for _ in range(100):
            states.append(0)
            features_data.append({
                '15m_adx': 25 + np.random.normal(0, 1),
                '15m_hl_pct': 0.01,
                '15m_bb_width': 0.02,
                '15m_returns': np.random.normal(0, 0.005),
            })
        
        # State 1: 高 ADX + 高趋势强度（全正收益）
        for _ in range(100):
            states.append(1)
            features_data.append({
                '15m_adx': 40 + np.random.normal(0, 1),
                '15m_hl_pct': 0.015,
                '15m_bb_width': 0.03,
                '15m_returns': 0.01 + np.random.normal(0, 0.001),  # 全正
            })
        
        # State 2-5: 其他状态
        for state in range(2, 6):
            for _ in range(100):
                states.append(state)
                features_data.append({
                    '15m_adx': 20 + np.random.normal(0, 1),
                    '15m_hl_pct': 0.01,
                    '15m_bb_width': 0.02,
                    '15m_returns': np.random.normal(0, 0.005),
                })
        
        features = pd.DataFrame(features_data)
        states = np.array(states)
        
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # State 1 应该被映射为 Strong_Trend
        self.assertEqual(mapping[1], 'Strong_Trend')


class TestRangeMapping(unittest.TestCase):
    """测试 Range 映射"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_low_adx_medium_vol_no_direction_maps_to_range(self):
        """低 ADX + 中等波动 + 无方向应该映射到 Range"""
        states = []
        features_data = []
        
        # State 0: Range 特征 - 低 ADX，中等波动（接近中位数），无方向（交替收益）
        for i in range(100):
            states.append(0)
            features_data.append({
                '15m_adx': 18 + np.random.normal(0, 0.5),  # 低 ADX
                '15m_hl_pct': 0.014,  # 中等波动，接近中位数
                '15m_bb_width': 0.028,
                '15m_returns': 0.002 * (1 if i % 2 == 0 else -1),  # 交替，方向一致性低
            })
        
        # State 1: Strong_Trend - 高 ADX
        for _ in range(100):
            states.append(1)
            features_data.append({
                '15m_adx': 42 + np.random.normal(0, 0.5),
                '15m_hl_pct': 0.016,
                '15m_bb_width': 0.032,
                '15m_returns': 0.008,  # 全正，高方向一致性
            })
        
        # State 2: Volatility_Spike - 高波动
        for _ in range(100):
            states.append(2)
            features_data.append({
                '15m_adx': 26 + np.random.normal(0, 0.5),
                '15m_hl_pct': 0.04,
                '15m_bb_width': 0.08,
                '15m_returns': np.random.normal(0, 0.01),
            })
        
        # State 3: Squeeze - 低波动 + 低 ADX
        for _ in range(100):
            states.append(3)
            features_data.append({
                '15m_adx': 12 + np.random.normal(0, 0.5),
                '15m_hl_pct': 0.005,
                '15m_bb_width': 0.01,
                '15m_returns': np.random.normal(0, 0.0005),
            })
        
        # State 4: Choppy_High_Vol - 高波动 + 低 ADX（波动率高于 Range）
        for _ in range(100):
            states.append(4)
            features_data.append({
                '15m_adx': 16 + np.random.normal(0, 0.5),
                '15m_hl_pct': 0.025,  # 高于中位数
                '15m_bb_width': 0.05,
                '15m_returns': np.random.normal(0, 0.008),
            })
        
        # State 5: Weak_Trend - 中等 ADX + 有方向性
        for _ in range(100):
            states.append(5)
            features_data.append({
                '15m_adx': 28 + np.random.normal(0, 0.5),
                '15m_hl_pct': 0.015,
                '15m_bb_width': 0.03,
                '15m_returns': 0.004,  # 全正，有方向性
            })
        
        features = pd.DataFrame(features_data)
        states = np.array(states)
        
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # State 0 应该被映射为 Range（低 ADX + 中等波动 + 无方向）
        self.assertEqual(mapping[0], 'Range')


class TestWeakTrendMapping(unittest.TestCase):
    """测试 Weak_Trend 映射"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_medium_adx_with_direction_maps_to_weak_trend(self):
        """中等 ADX + 有方向性应该映射到 Weak_Trend"""
        states = []
        features_data = []
        
        # State 0: Weak_Trend - 中等 ADX，有方向性
        for _ in range(100):
            states.append(0)
            features_data.append({
                '15m_adx': 26 + np.random.normal(0, 1),  # 中等 ADX（接近中位数）
                '15m_hl_pct': 0.012,
                '15m_bb_width': 0.024,
                '15m_returns': 0.003,  # 全正，有方向性
            })
        
        # State 1: Strong_Trend - 高 ADX
        for _ in range(100):
            states.append(1)
            features_data.append({
                '15m_adx': 42 + np.random.normal(0, 1),
                '15m_hl_pct': 0.015,
                '15m_bb_width': 0.03,
                '15m_returns': 0.01,
            })
        
        # State 2: Volatility_Spike
        for _ in range(100):
            states.append(2)
            features_data.append({
                '15m_adx': 25 + np.random.normal(0, 1),
                '15m_hl_pct': 0.04,
                '15m_bb_width': 0.08,
                '15m_returns': np.random.normal(0, 0.01),
            })
        
        # State 3: Squeeze
        for _ in range(100):
            states.append(3)
            features_data.append({
                '15m_adx': 12 + np.random.normal(0, 1),
                '15m_hl_pct': 0.005,
                '15m_bb_width': 0.01,
                '15m_returns': np.random.normal(0, 0.001),
            })
        
        # State 4: Choppy_High_Vol
        for _ in range(100):
            states.append(4)
            features_data.append({
                '15m_adx': 18 + np.random.normal(0, 1),
                '15m_hl_pct': 0.025,
                '15m_bb_width': 0.05,
                '15m_returns': np.random.normal(0, 0.008),
            })
        
        # State 5: Range
        for i in range(100):
            states.append(5)
            features_data.append({
                '15m_adx': 20 + np.random.normal(0, 1),
                '15m_hl_pct': 0.011,
                '15m_bb_width': 0.022,
                '15m_returns': 0.002 * (1 if i % 2 == 0 else -1),  # 无方向
            })
        
        features = pd.DataFrame(features_data)
        states = np.array(states)
        
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # State 0 应该被映射为 Weak_Trend
        self.assertEqual(mapping[0], 'Weak_Trend')


class TestChoppyHighVolMapping(unittest.TestCase):
    """测试 Choppy_High_Vol 映射"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_high_vol_low_adx_maps_to_choppy(self):
        """高波动 + 低 ADX 应该映射到 Choppy_High_Vol"""
        states = []
        features_data = []
        
        # State 0: Choppy_High_Vol - 高波动，低 ADX
        for _ in range(100):
            states.append(0)
            features_data.append({
                '15m_adx': 18 + np.random.normal(0, 1),  # 低 ADX
                '15m_hl_pct': 0.025,  # 高波动（但不及 Volatility_Spike）
                '15m_bb_width': 0.05,
                '15m_returns': np.random.normal(0, 0.008),
            })
        
        # State 1: Strong_Trend
        for _ in range(100):
            states.append(1)
            features_data.append({
                '15m_adx': 42 + np.random.normal(0, 1),
                '15m_hl_pct': 0.015,
                '15m_bb_width': 0.03,
                '15m_returns': 0.01,
            })
        
        # State 2: Volatility_Spike - 最高波动
        for _ in range(100):
            states.append(2)
            features_data.append({
                '15m_adx': 25 + np.random.normal(0, 1),
                '15m_hl_pct': 0.05,
                '15m_bb_width': 0.1,
                '15m_returns': np.random.normal(0, 0.015),
            })
        
        # State 3: Squeeze
        for _ in range(100):
            states.append(3)
            features_data.append({
                '15m_adx': 12 + np.random.normal(0, 1),
                '15m_hl_pct': 0.005,
                '15m_bb_width': 0.01,
                '15m_returns': np.random.normal(0, 0.001),
            })
        
        # State 4, 5: 其他
        for state in [4, 5]:
            for _ in range(100):
                states.append(state)
                features_data.append({
                    '15m_adx': 24 + np.random.normal(0, 1),
                    '15m_hl_pct': 0.012,
                    '15m_bb_width': 0.024,
                    '15m_returns': np.random.normal(0, 0.004),
                })
        
        features = pd.DataFrame(features_data)
        states = np.array(states)
        
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # State 0 应该被映射为 Choppy_High_Vol
        self.assertEqual(mapping[0], 'Choppy_High_Vol')


class TestFallbackLogic(unittest.TestCase):
    """测试 fallback 逻辑"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_select_best_fallback_name(self):
        """测试 fallback 名称选择逻辑"""
        # 测试高 ADX 高趋势强度的 profile 应该选择 Strong_Trend
        profile = {
            'state': 0,
            'adx_mean': 40,
            'volatility_score': 0.015,
            'trend_strength': 5.0,
        }
        
        available_names = {'Strong_Trend', 'Weak_Trend', 'Range'}
        best = self.labeler._select_best_fallback_name(
            profile, available_names, adx_median=25, vol_median=0.01, trend_median=1.0
        )
        
        self.assertEqual(best, 'Strong_Trend')
    
    def test_fallback_squeeze_selection(self):
        """测试 fallback 选择 Squeeze"""
        # Squeeze 特征：极低波动 + 低 ADX
        # 为了确保 Squeeze 得分最高，波动率和 ADX 都要远低于中位数
        profile = {
            'state': 0,
            'adx_mean': 10,  # 远低于中位数 25 (adx_norm = 0.4)
            'volatility_score': 0.003,  # 远低于中位数 0.01 (vol_norm = 0.3)
            'trend_strength': 0.1,
        }
        
        # Squeeze vs Choppy_High_Vol vs Weak_Trend
        # 对于这个 profile:
        # - Squeeze: (1 - 0.3) * (1 - 0.4) = 0.7 * 0.6 = 0.42
        # - Choppy_High_Vol: 需要 vol > median，所以得 0
        # - Weak_Trend: adx_norm 不在 0.6-1.4 范围，得 0.1
        available_names = {'Squeeze', 'Choppy_High_Vol', 'Weak_Trend'}
        best = self.labeler._select_best_fallback_name(
            profile, available_names, adx_median=25, vol_median=0.01, trend_median=1.0
        )
        
        self.assertEqual(best, 'Squeeze')


class TestValidationWarnings(unittest.TestCase):
    """测试映射验证警告"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_strong_trend_low_adx_warning(self):
        """测试 Strong_Trend 低 ADX 警告"""
        mapping = {0: 'Strong_Trend'}
        profiles = [{'state': 0, 'adx_mean': 15, 'volatility_score': 0.01, 'trend_strength': 1.0}]
        
        with patch('hmm_trainer.logger') as mock_logger:
            self.labeler._validate_mapping(mapping, profiles, adx_median=25, vol_median=0.01, trend_median=1.0)
            # 应该触发警告
            mock_logger.warning.assert_called()
    
    def test_range_high_adx_warning(self):
        """测试 Range 高 ADX 警告"""
        mapping = {0: 'Range'}
        profiles = [{'state': 0, 'adx_mean': 35, 'volatility_score': 0.01, 'trend_strength': 0.5}]
        
        with patch('hmm_trainer.logger') as mock_logger:
            self.labeler._validate_mapping(mapping, profiles, adx_median=25, vol_median=0.01, trend_median=1.0)
            # 应该触发警告
            mock_logger.warning.assert_called()


class TestAllRegimesCovered(unittest.TestCase):
    """测试所有 6 种 regime 都能被正确映射"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_all_six_regimes_mapped(self):
        """测试完整的 6 状态映射"""
        states = []
        features_data = []
        
        # State 0: Volatility_Spike - 极高波动
        for _ in range(100):
            states.append(0)
            features_data.append({
                '15m_adx': 28 + np.random.normal(0, 1),
                '15m_hl_pct': 0.05,
                '15m_bb_width': 0.1,
                '15m_returns': np.random.normal(0, 0.015),
            })
        
        # State 1: Squeeze - 极低波动 + 低 ADX
        for _ in range(100):
            states.append(1)
            features_data.append({
                '15m_adx': 12 + np.random.normal(0, 1),
                '15m_hl_pct': 0.004,
                '15m_bb_width': 0.008,
                '15m_returns': np.random.normal(0, 0.0005),
            })
        
        # State 2: Strong_Trend - 高 ADX + 高趋势强度
        for _ in range(100):
            states.append(2)
            features_data.append({
                '15m_adx': 45 + np.random.normal(0, 1),
                '15m_hl_pct': 0.018,
                '15m_bb_width': 0.036,
                '15m_returns': 0.008,  # 全正，高方向一致性
            })
        
        # State 3: Choppy_High_Vol - 高波动 + 低 ADX
        for _ in range(100):
            states.append(3)
            features_data.append({
                '15m_adx': 16 + np.random.normal(0, 1),
                '15m_hl_pct': 0.028,
                '15m_bb_width': 0.056,
                '15m_returns': np.random.normal(0, 0.01),
            })
        
        # State 4: Range - 低 ADX + 中等波动 + 无方向
        for i in range(100):
            states.append(4)
            features_data.append({
                '15m_adx': 18 + np.random.normal(0, 1),
                '15m_hl_pct': 0.012,
                '15m_bb_width': 0.024,
                '15m_returns': 0.002 * (1 if i % 2 == 0 else -1),  # 无方向
            })
        
        # State 5: Weak_Trend - 中等 ADX + 有方向性
        for _ in range(100):
            states.append(5)
            features_data.append({
                '15m_adx': 26 + np.random.normal(0, 1),
                '15m_hl_pct': 0.014,
                '15m_bb_width': 0.028,
                '15m_returns': 0.004,  # 全正，有方向性
            })
        
        features = pd.DataFrame(features_data)
        states = np.array(states)
        
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # 验证所有 6 种 regime 都被分配
        assigned_regimes = set(mapping.values())
        expected_regimes = set(DEFAULT_REGIME_NAMES)
        
        self.assertEqual(assigned_regimes, expected_regimes)


class TestExtremeCases(unittest.TestCase):
    """测试极端情况"""
    
    def setUp(self):
        self.labeler = HMMRegimeLabeler(n_states=6)
    
    def test_all_low_volatility_market(self):
        """测试所有状态都低波动的市场"""
        states = []
        features_data = []
        
        # 所有状态波动率都很低
        for state in range(6):
            for _ in range(100):
                states.append(state)
                features_data.append({
                    '15m_adx': 20 + state * 3 + np.random.normal(0, 1),
                    '15m_hl_pct': 0.005 + state * 0.001,  # 都很低
                    '15m_bb_width': 0.01 + state * 0.002,
                    '15m_returns': np.random.normal(0, 0.001),
                })
        
        features = pd.DataFrame(features_data)
        states = np.array(states)
        
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # 确保没有 crash，所有状态都有映射
        self.assertEqual(len(mapping), 6)
        for state in range(6):
            self.assertIn(state, mapping)
    
    def test_all_high_volatility_market(self):
        """测试所有状态都高波动的市场"""
        states = []
        features_data = []
        
        # 所有状态波动率都很高
        for state in range(6):
            for _ in range(100):
                states.append(state)
                features_data.append({
                    '15m_adx': 20 + state * 3 + np.random.normal(0, 1),
                    '15m_hl_pct': 0.03 + state * 0.005,  # 都很高
                    '15m_bb_width': 0.06 + state * 0.01,
                    '15m_returns': np.random.normal(0, 0.01),
                })
        
        features = pd.DataFrame(features_data)
        states = np.array(states)
        
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # 确保没有 crash，所有状态都有映射
        self.assertEqual(len(mapping), 6)
        for state in range(6):
            self.assertIn(state, mapping)
    
    def test_empty_state(self):
        """测试某些状态没有样本的情况"""
        states = []
        features_data = []
        
        # 只有 state 0, 1, 2 有数据
        for state in range(3):
            for _ in range(100):
                states.append(state)
                features_data.append({
                    '15m_adx': 20 + state * 10 + np.random.normal(0, 1),
                    '15m_hl_pct': 0.01 + state * 0.01,
                    '15m_bb_width': 0.02 + state * 0.02,
                    '15m_returns': np.random.normal(0, 0.005),
                })
        
        features = pd.DataFrame(features_data)
        states = np.array(states)
        
        mapping = self.labeler.auto_map_regimes(features, states)
        
        # 所有 6 个状态都应该有映射（即使是空的）
        self.assertEqual(len(mapping), 6)


def main():
    """运行所有测试"""
    import sys
    import io
    
    # 设置 stdout 编码为 UTF-8（解决 Windows 控制台编码问题）
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("=" * 80)
    print("Regime Mapping Unit Tests")
    print("=" * 80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestTrendStrengthCalculation,
        TestVolatilitySpikeMapping,
        TestSqueezeMapping,
        TestStrongTrendMapping,
        TestRangeMapping,
        TestWeakTrendMapping,
        TestChoppyHighVolMapping,
        TestFallbackLogic,
        TestValidationWarnings,
        TestAllRegimesCovered,
        TestExtremeCases,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[PASS] All tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
