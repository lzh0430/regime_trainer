"""
快速开始示例脚本
支持 5m 和 15m 两种时间框架的训练和预测
支持多步预测 (t+1 到 t+4)
"""
import sys
import logging
from config import TrainingConfig, setup_logging
from training_pipeline import TrainingPipeline
from realtime_predictor import RealtimeRegimePredictor, MultiSymbolRegimeTracker
from model_api import ModelAPI

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认时间框架
DEFAULT_TIMEFRAME = "15m"


def _print_multistep_results(result: dict):
    """打印多步预测训练结果的辅助函数"""
    print(f"\n📊 多步预测信息:")
    print(f"  预测步数: {result.get('prediction_horizons', [1, 2, 3, 4])}")
    
    # 显示各步的损失权重
    from config import TrainingConfig
    print(f"  损失权重: {TrainingConfig.HORIZON_LOSS_WEIGHTS}")

def example_1_single_symbol_training():
    """示例 1: 训练单个交易对"""
    print("\n" + "="*80)
    print("示例 1: 训练单个交易对 (BTCUSDT)")
    print("="*80 + "\n")
    
    print("⚠️  注意：完整训练可能需要较长时间（10-30分钟）")
    print("   包括：数据获取、特征工程、HMM训练、LSTM训练")
    print("   请耐心等待...\n")
    
    import sys
    sys.stdout.flush()  # 确保输出立即显示
    
    try:
        # 创建训练管道
        pipeline = TrainingPipeline(TrainingConfig)
        
        # 完整重训
        logger.info("开始完整重训...")
        result = pipeline.full_retrain("BTCUSDT")
        
        print(f"\n✅ 训练完成！")
        print(f"测试集准确率 (t+1): {result['test_accuracy']:.2%}")
        print(f"测试集损失: {result['test_loss']:.4f}")
        if 'val_accuracy' in result:
            print(f"验证集准确率 (t+1): {result['val_accuracy']:.2%}")
        
        # 显示多步预测信息
        _print_multistep_results(result)
        
        # 显示动态状态数量优化结果
        if result.get('n_states_optimization'):
            opt = result['n_states_optimization']
            if opt['adjusted']:
                print(f"\n🔄 状态数量已自动调整: {opt['original_n_states']} -> {opt['optimal_n_states']}")
                
                # 显示保留和删除的状态
                all_names = {"Strong_Trend", "Weak_Trend", "Range", 
                            "Choppy_High_Vol", "Volatility_Spike", "Squeeze"}
                current_names = set(result.get('regime_mapping', {}).values())
                removed_names = all_names - current_names
                
                print(f"   保留的状态: {sorted(current_names)}")
                if removed_names:
                    print(f"   删除的状态: {sorted(removed_names)}")
            else:
                print(f"\n✓ 状态数量保持不变: {opt['optimal_n_states']}")
        
        print(f"最终状态数量: {result.get('final_n_states', 6)}")
        
        # 显示状态分布检查结果
        if 'state_distribution_check' in result:
            dist_check = result['state_distribution_check']
            if not dist_check['healthy']:
                print(f"\n⚠️  状态分布警告:")
                for warning in dist_check['warnings']:
                    print(f"   {warning}")
                if dist_check['recommendations']:
                    print(f"\n💡 建议:")
                    for rec in dist_check['recommendations']:
                        print(f"   {rec}")
            else:
                print(f"\n✓ 状态分布健康：所有状态在各数据集中都有足够样本")
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        raise
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        raise


def example_2_multiple_symbols_training():
    """示例 2: 批量训练多个交易对"""
    print("\n" + "="*80)
    print("示例 2: 批量训练多个交易对")
    print("="*80 + "\n")
    
    # 临时设置要训练的交易对
    symbols = TrainingConfig.SYMBOLS
    
    pipeline = TrainingPipeline(TrainingConfig)
    
    # 批量完整重训
    logger.info(f"开始批量训练 {len(symbols)} 个交易对...")
    
    # 临时修改配置
    original_symbols = TrainingConfig.SYMBOLS
    TrainingConfig.SYMBOLS = symbols
    
    results = pipeline.train_all_symbols(training_type='full')
    
    # 恢复配置
    TrainingConfig.SYMBOLS = original_symbols
    
    print("\n训练结果汇总:")
    for symbol, result in results.items():
        if 'error' in result:
            print(f"{symbol}: 失败 - {result['error']}")
        else:
            print(f"{symbol}: 测试集准确率 {result['test_accuracy']:.2%}")
            # 显示动态调整信息
            if result.get('n_states_optimization') and result['n_states_optimization']['adjusted']:
                opt = result['n_states_optimization']
                print(f"  🔄 状态数量调整: {opt['original_n_states']} -> {opt['optimal_n_states']}")
                
                # 显示保留的状态名称
                current_names = set(result.get('regime_mapping', {}).values())
                print(f"  保留的状态: {sorted(current_names)}")
            if 'state_distribution_check' in result:
                dist_check = result['state_distribution_check']
                if not dist_check['healthy']:
                    print(f"  ⚠️ 警告: 验证集缺失 {len(dist_check['missing_states']['val'])} 个状态")


def example_3_realtime_prediction():
    """示例 3: 实时市场状态预测（支持多步预测 t+1 到 t+4）"""
    print("\n" + "="*80)
    print("示例 3: 实时市场状态预测（多步预测）")
    print("="*80 + "\n")
    
    try:
        # 创建预测器
        predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)
        
        # 获取当前市场状态（包括多步预测）
        current = predictor.get_current_regime()
        
        print(f"\n{current['symbol']} 当前市场状态 ({current['primary_timeframe']}):")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"时间: {current['timestamp']}")
        
        # 显示多步预测结果
        predictions = current.get('predictions', {})
        if predictions:
            print(f"\n📈 多步预测结果:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for horizon in ['t+1', 't+2', 't+3', 't+4']:
                if horizon in predictions:
                    pred = predictions[horizon]
                    bar = "█" * int(pred['confidence'] * 30)
                    uncertain_mark = " ⚠️" if pred.get('is_uncertain', False) else ""
                    print(f"  {horizon}: {pred['regime_name']:20s} {pred['confidence']:6.2%} {bar}{uncertain_mark}")
        
        # 显示 t+1 的详细概率分布
        print(f"\nt+1 状态概率分布详情:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for regime, prob in sorted(
            current['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            bar = "█" * int(prob * 50)
            print(f"  {regime:20s} {prob:6.2%} {bar}")
        
        # 显示历史 regime 序列
        historical = current.get('historical_regimes', {})
        if historical and historical.get('sequence'):
            print(f"\n📜 历史 Regime 序列 (过去 {historical.get('lookback_hours', 4)} 小时):")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            seq = historical['sequence']
            # 只显示最近 8 个
            recent = seq[-8:] if len(seq) > 8 else seq
            print(f"  最近 {len(recent)} 根 K 线: {' -> '.join(recent)}")
            
            # 统计历史分布
            from collections import Counter
            counts = Counter(seq)
            print(f"  分布统计: {dict(counts)}")
        
    except FileNotFoundError:
        print("\n❌ 模型文件不存在，请先运行训练（示例 1 或 2）")


def example_4_regime_history():
    """示例 4: 查看历史市场状态变化"""
    print("\n" + "="*80)
    print("示例 4: 历史市场状态变化")
    print("="*80 + "\n")
    
    try:
        predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)
        
        # 获取最近 24 小时的状态变化
        history = predictor.get_regime_history(lookback_hours=24)
        
        print(f"\n最近 24 小时的市场状态变化:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        if history.empty:
            print("⚠️  没有足够的历史数据进行分析。")
            print("   可能的原因：")
            print("   1. 数据量不足（需要至少 64 行数据）")
            print("   2. 特征计算失败")
            print("   建议：获取更多历史数据或检查数据获取是否正常")
        else:
            print(history.tail(20))
            
            # 统计各状态出现次数
            print(f"\n状态分布:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            regime_counts = history['regime_name'].value_counts()
            for regime, count in regime_counts.items():
                percentage = count / len(history) * 100
                print(f"{regime:20s} {count:4d} 次 ({percentage:5.1f}%)")
        
    except FileNotFoundError:
        print("\n❌ 模型文件不存在，请先运行训练（示例 1 或 2）")


def example_5_multi_symbol_tracking():
    """示例 5: 多交易对市场状态跟踪"""
    print("\n" + "="*80)
    print("示例 5: 多交易对市场状态跟踪")
    print("="*80 + "\n")
    
    try:
        # 创建多交易对跟踪器
        tracker = MultiSymbolRegimeTracker(
            symbols=["BTCUSDT", "ETHUSDT"],
            config=TrainingConfig
        )
        
        # 获取所有交易对的当前状态
        all_regimes = tracker.get_all_regimes()
        
        print(f"\n所有交易对当前状态:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for symbol, result in all_regimes.items():
            if 'error' not in result:
                print(f"{symbol:12s} {result['regime_name']:20s} 置信度: {result['confidence']:.2%}")
            else:
                print(f"{symbol:12s} ❌ {result['error']}")
        
        # 获取状态摘要
        summary = tracker.get_regime_summary()
        if not summary.empty:
            print(f"\n市场状态摘要表:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(summary.to_string(index=False))
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("请确保至少有一个交易对已完成训练")


def example_6_incremental_training():
    """示例 6: 增量训练"""
    print("\n" + "="*80)
    print("示例 6: 增量训练（在现有模型基础上）")
    print("="*80 + "\n")
    
    print("⚠️  注意：增量训练通常需要 2-5 分钟")
    print("   包括：获取最新数据、特征工程、模型更新")
    print("   请耐心等待...\n")
    
    import sys
    sys.stdout.flush()  # 确保输出立即显示
    
    try:
        pipeline = TrainingPipeline(TrainingConfig)
        
        # 执行增量训练
        logger.info("开始增量训练...")
        result = pipeline.incremental_train("BTCUSDT")
        
        print(f"\n✅ 增量训练完成！")
        print(f"使用样本数: {result['samples_used']}")
        print(f"训练时间: {result['timestamp']}")
        
    except FileNotFoundError as e:
        print(f"\n❌ 模型文件不存在: {e}")
        print("   请先运行完整训练（示例 1）")
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        raise
    except Exception as e:
        print(f"\n❌ 增量训练失败: {e}")
        raise


# ============================================================================
# 5m 时间框架专用示例
# ============================================================================

def example_7_5m_single_symbol_training():
    """示例 7: 训练单个交易对的 5m 模型"""
    print("\n" + "="*80)
    print("示例 7: 训练单个交易对的 5m 模型 (BTCUSDT)")
    print("="*80 + "\n")
    
    print("⚠️  注意：完整训练可能需要较长时间（10-30分钟）")
    print("   包括：数据获取、特征工程、HMM训练、LSTM训练")
    print("   5m 模型使用更短的时间框架进行更快速的决策")
    print("   请耐心等待...\n")
    
    sys.stdout.flush()
    
    try:
        pipeline = TrainingPipeline(TrainingConfig)
        
        # 完整重训 5m 模型
        logger.info("开始训练 5m 模型...")
        result = pipeline.full_retrain("BTCUSDT", primary_timeframe="5m")
        
        print(f"\n✅ 5m 模型训练完成！")
        print(f"测试集准确率 (t+1): {result['test_accuracy']:.2%}")
        print(f"测试集损失: {result['test_loss']:.4f}")
        if 'val_accuracy' in result:
            print(f"验证集准确率 (t+1): {result['val_accuracy']:.2%}")
        
        # 显示多步预测信息
        _print_multistep_results(result)
        
        # 显示动态状态数量优化结果
        if result.get('n_states_optimization'):
            opt = result['n_states_optimization']
            if opt['adjusted']:
                print(f"\n🔄 状态数量已自动调整: {opt['original_n_states']} -> {opt['optimal_n_states']}")
                current_names = set(result.get('regime_mapping', {}).values())
                print(f"   保留的状态: {sorted(current_names)}")
            else:
                print(f"\n✓ 状态数量保持不变: {opt['optimal_n_states']}")
        
        print(f"最终状态数量: {result.get('final_n_states', 6)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        raise
    except Exception as e:
        print(f"\n❌ 5m 模型训练失败: {e}")
        raise


def example_8_5m_realtime_prediction():
    """示例 8: 5m 实时市场状态预测（支持多步预测）"""
    print("\n" + "="*80)
    print("示例 8: 5m 实时市场状态预测（多步预测）")
    print("="*80 + "\n")
    
    try:
        # 创建 5m 预测器
        predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig, primary_timeframe="5m")
        
        # 获取当前市场状态（包括多步预测）
        current = predictor.get_current_regime()
        
        print(f"\n{current['symbol']} 当前市场状态 (5m 时间框架):")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"时间: {current['timestamp']}")
        
        # 显示多步预测结果
        predictions = current.get('predictions', {})
        if predictions:
            print(f"\n📈 多步预测结果 (5m):")
            print(f"  (每步代表 5 分钟，t+4 = 20 分钟后)")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for horizon in ['t+1', 't+2', 't+3', 't+4']:
                if horizon in predictions:
                    pred = predictions[horizon]
                    bar = "█" * int(pred['confidence'] * 30)
                    uncertain_mark = " ⚠️" if pred.get('is_uncertain', False) else ""
                    print(f"  {horizon}: {pred['regime_name']:20s} {pred['confidence']:6.2%} {bar}{uncertain_mark}")
        
        # 显示 t+1 的详细概率分布
        print(f"\nt+1 状态概率分布详情:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for regime, prob in sorted(
            current['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            bar = "█" * int(prob * 50)
            print(f"  {regime:20s} {prob:6.2%} {bar}")
        
        # 显示历史 regime 序列
        historical = current.get('historical_regimes', {})
        if historical and historical.get('sequence'):
            print(f"\n📜 历史 Regime 序列 (过去 {historical.get('lookback_hours', 4)} 小时):")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            seq = historical['sequence']
            recent = seq[-8:] if len(seq) > 8 else seq
            print(f"  最近 {len(recent)} 根 K 线: {' -> '.join(recent)}")
        
    except FileNotFoundError:
        print("\n❌ 5m 模型文件不存在，请先运行 5m 训练（示例 7）")


def example_9_5m_incremental_training():
    """示例 9: 5m 增量训练"""
    print("\n" + "="*80)
    print("示例 9: 5m 增量训练（在现有 5m 模型基础上）")
    print("="*80 + "\n")
    
    print("⚠️  注意：增量训练通常需要 2-5 分钟")
    print("   包括：获取最新数据、特征工程、模型更新")
    print("   请耐心等待...\n")
    
    sys.stdout.flush()
    
    try:
        pipeline = TrainingPipeline(TrainingConfig)
        
        # 执行 5m 增量训练
        logger.info("开始 5m 增量训练...")
        result = pipeline.incremental_train("BTCUSDT", primary_timeframe="5m")
        
        print(f"\n✅ 5m 增量训练完成！")
        print(f"使用样本数: {result['samples_used']}")
        print(f"训练时间: {result['timestamp']}")
        
    except FileNotFoundError as e:
        print(f"\n❌ 5m 模型文件不存在: {e}")
        print("   请先运行 5m 完整训练（示例 7）")
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        raise
    except Exception as e:
        print(f"\n❌ 5m 增量训练失败: {e}")
        raise


def example_10_5m_regime_history():
    """示例 10: 查看 5m 历史市场状态变化"""
    print("\n" + "="*80)
    print("示例 10: 5m 历史市场状态变化")
    print("="*80 + "\n")
    
    try:
        predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig, primary_timeframe="5m")
        
        # 获取最近 4 小时的状态变化（5m 模型适合更短周期）
        history = predictor.get_regime_history(lookback_hours=4)
        
        print(f"\n最近 4 小时的 5m 市场状态变化:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        if history.empty:
            print("⚠️  没有足够的历史数据进行分析。")
        else:
            print(history.tail(20))
            
            # 统计各状态出现次数
            print(f"\n状态分布:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            regime_counts = history['regime_name'].value_counts()
            for regime, count in regime_counts.items():
                percentage = count / len(history) * 100
                print(f"{regime:20s} {count:4d} 次 ({percentage:5.1f}%)")
        
    except FileNotFoundError:
        print("\n❌ 5m 模型文件不存在，请先运行 5m 训练（示例 7）")


def example_11_multi_timeframe_prediction():
    """示例 11: 多时间框架并行预测 (5m + 15m + 1h)"""
    print("\n" + "="*80)
    print("示例 11: 多时间框架并行预测 (5m + 15m + 1h)")
    print("="*80 + "\n")
    
    try:
        api = ModelAPI()
        
        # 同时获取所有启用时间框架的预测
        timeframes = api.config.ENABLED_MODELS
        results = api.predict_multi_timeframe_regimes("BTCUSDT", timeframes)
        
        print(f"\nBTCUSDT 多时间框架市场状态:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        regimes = results.get('regimes', {})
        for tf, result in regimes.items():
            if 'error' in result:
                print(f"\n{tf} 时间框架: ❌ {result['error']}")
            else:
                print(f"\n{tf} 时间框架:")
                # 获取 t+1 预测
                t1_pred = result.get('predictions', {}).get('t+1', {})
                if t1_pred:
                    print(f"  t+1 状态: {t1_pred['most_likely']}")
                    print(f"  t+1 置信度: {t1_pred['confidence']:.2%}")
                    print(f"  概率分布:")
                    for regime, prob in sorted(
                        t1_pred['probabilities'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]:  # 只显示前3个
                        bar = "█" * int(prob * 30)
                        print(f"    {regime:20s} {prob:6.2%} {bar}")
        
        # 比较所有时间框架的状态
        valid_regimes = {tf: regimes[tf] for tf in timeframes if 'error' not in regimes.get(tf, {})}
        if len(valid_regimes) >= 2:
            print(f"\n📊 时间框架对比 (t+1):")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            t1_predictions = {}
            for tf, result in valid_regimes.items():
                t1_predictions[tf] = result.get('predictions', {}).get('t+1', {}).get('most_likely', 'N/A')
            
            # 检查是否所有预测一致
            unique_predictions = set(t1_predictions.values())
            if len(unique_predictions) == 1:
                print(f"✓ 所有时间框架的 t+1 状态一致: {list(unique_predictions)[0]}")
            else:
                print(f"⚠️ 不同时间框架的 t+1 状态不一致:")
                for tf, pred in t1_predictions.items():
                    print(f"   {tf}: {pred}")
                print(f"   这可能表示市场正在发生不同时间尺度的变化")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("请确保至少有一个时间框架的模型已完成训练")


def example_12_5m_multi_symbol_tracking():
    """示例 12: 5m 多交易对市场状态跟踪"""
    print("\n" + "="*80)
    print("示例 12: 5m 多交易对市场状态跟踪")
    print("="*80 + "\n")
    
    try:
        # 创建 5m 多交易对跟踪器
        tracker = MultiSymbolRegimeTracker(
            symbols=["BTCUSDT", "ETHUSDT"],
            config=TrainingConfig,
            primary_timeframe="5m"
        )
        
        # 获取所有交易对的当前状态
        all_regimes = tracker.get_all_regimes()
        
        print(f"\n所有交易对当前 5m 状态:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for symbol, result in all_regimes.items():
            if 'error' not in result:
                print(f"{symbol:12s} {result['regime_name']:20s} 置信度: {result['confidence']:.2%}")
            else:
                print(f"{symbol:12s} ❌ {result['error']}")
        
        # 获取状态摘要
        summary = tracker.get_regime_summary()
        if not summary.empty:
            print(f"\n5m 市场状态摘要表:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(summary.to_string(index=False))
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("请确保至少有一个交易对的 5m 模型已完成训练")


def example_13_batch_5m_training():
    """示例 13: 批量训练多个交易对的 5m 模型"""
    print("\n" + "="*80)
    print("示例 13: 批量训练多个交易对的 5m 模型")
    print("="*80 + "\n")
    
    symbols = TrainingConfig.SYMBOLS
    
    pipeline = TrainingPipeline(TrainingConfig)
    
    logger.info(f"开始批量训练 {len(symbols)} 个交易对的 5m 模型...")
    
    results = pipeline.train_all_symbols(training_type='full', primary_timeframe="5m")
    
    print("\n5m 模型训练结果汇总:")
    for symbol, result in results.items():
        if 'error' in result:
            print(f"{symbol}: 失败 - {result['error']}")
        else:
            print(f"{symbol}: 测试集准确率 {result['test_accuracy']:.2%}")
            if result.get('n_states_optimization') and result['n_states_optimization']['adjusted']:
                opt = result['n_states_optimization']
                print(f"  🔄 状态数量调整: {opt['original_n_states']} -> {opt['optimal_n_states']}")


# ============================================================================
# 多步预测 API 测试示例
# ============================================================================

def example_14_multistep_api_15m():
    """示例 14: 使用 API 进行 15m 多步预测"""
    print("\n" + "="*80)
    print("示例 14: 使用 predict_regimes() API 进行 15m 多步预测")
    print("="*80 + "\n")
    
    try:
        api = ModelAPI()
        
        # 使用新的 predict_regimes API
        result = api.predict_regimes(
            symbol="BTCUSDT",
            primary_timeframe="15m",
            include_history=True,
            history_bars=16  # 16 根 15m K 线 = 4 小时
        )
        
        print(f"\n{result['symbol']} 多步预测结果 ({result['timeframe']}):")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"时间: {result['timestamp']}")
        # 现在总是多步预测
        
        # 模型信息
        model_info = result.get('model_info', {})
        print(f"\n📊 模型信息:")
        print(f"  序列长度: {model_info.get('sequence_length', 'N/A')}")
        print(f"  状态数量: {model_info.get('n_states', 'N/A')}")
        print(f"  预测步数: {model_info.get('prediction_horizons', 'N/A')}")
        
        # 多步预测
        predictions = result.get('predictions', {})
        if predictions:
            print(f"\n📈 多步预测 (15m):")
            print(f"  (每步代表 15 分钟，t+4 = 1 小时后)")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for horizon in ['t+1', 't+2', 't+3', 't+4']:
                if horizon in predictions:
                    pred = predictions[horizon]
                    bar = "█" * int(pred['confidence'] * 30)
                    uncertain_mark = " ⚠️" if pred.get('is_uncertain', False) else ""
                    print(f"  {horizon}: {pred['most_likely']:20s} {pred['confidence']:6.2%} {bar}{uncertain_mark}")
        
        # 历史序列
        historical = result.get('historical_regimes', {})
        if historical and historical.get('sequence'):
            print(f"\n📜 历史 Regime 序列 (过去 {historical.get('lookback_hours', 4):.1f} 小时):")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            seq = historical['sequence']
            recent = seq[-8:] if len(seq) > 8 else seq
            print(f"  最近 {len(recent)} 根 K 线: {' -> '.join(recent)}")
            
            # 统计
            from collections import Counter
            counts = Counter(seq)
            print(f"  历史分布: {dict(counts)}")
        
    except FileNotFoundError:
        print("\n❌ 15m 模型文件不存在，请先运行训练（示例 1）")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def example_15_multistep_api_5m():
    """示例 15: 使用 API 进行 5m 多步预测"""
    print("\n" + "="*80)
    print("示例 15: 使用 predict_regimes() API 进行 5m 多步预测")
    print("="*80 + "\n")
    
    try:
        api = ModelAPI()
        
        # 使用新的 predict_regimes API
        result = api.predict_regimes(
            symbol="BTCUSDT",
            primary_timeframe="5m",
            include_history=True,
            history_bars=24  # 24 根 5m K 线 = 2 小时
        )
        
        print(f"\n{result['symbol']} 多步预测结果 ({result['timeframe']}):")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"时间: {result['timestamp']}")
        # 现在总是多步预测
        
        # 模型信息
        model_info = result.get('model_info', {})
        print(f"\n📊 模型信息:")
        print(f"  序列长度: {model_info.get('sequence_length', 'N/A')}")
        print(f"  状态数量: {model_info.get('n_states', 'N/A')}")
        print(f"  预测步数: {model_info.get('prediction_horizons', 'N/A')}")
        
        # 多步预测
        predictions = result.get('predictions', {})
        if predictions:
            print(f"\n📈 多步预测 (5m):")
            print(f"  (每步代表 5 分钟，t+4 = 20 分钟后)")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for horizon in ['t+1', 't+2', 't+3', 't+4']:
                if horizon in predictions:
                    pred = predictions[horizon]
                    bar = "█" * int(pred['confidence'] * 30)
                    uncertain_mark = " ⚠️" if pred.get('is_uncertain', False) else ""
                    print(f"  {horizon}: {pred['most_likely']:20s} {pred['confidence']:6.2%} {bar}{uncertain_mark}")
        
        # 历史序列
        historical = result.get('historical_regimes', {})
        if historical and historical.get('sequence'):
            print(f"\n📜 历史 Regime 序列 (过去 {historical.get('lookback_hours', 2):.1f} 小时):")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            seq = historical['sequence']
            recent = seq[-8:] if len(seq) > 8 else seq
            print(f"  最近 {len(recent)} 根 K 线: {' -> '.join(recent)}")
        
    except FileNotFoundError:
        print("\n❌ 5m 模型文件不存在，请先运行 5m 训练（示例 7）")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def example_16_compare_timeframes():
    """示例 16: 对比多个时间框架的多步预测"""
    print("\n" + "="*80)
    print("示例 16: 对比多个时间框架的多步预测")
    print("="*80 + "\n")
    
    try:
        api = ModelAPI()
        
        # 获取所有启用时间框架的预测
        timeframes = api.config.ENABLED_MODELS
        results = {}
        for tf in timeframes:
            try:
                results[tf] = api.predict_regimes(
                    symbol="BTCUSDT",
                    primary_timeframe=tf,
                    include_history=True
                )
            except FileNotFoundError:
                results[tf] = {'error': f'{tf} 模型不存在'}
        
        print(f"\nBTCUSDT 多时间框架多步预测对比:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # 动态生成表头
        header = f"{'Horizon':<10}"
        for tf in timeframes:
            header += f" {tf:^25}"
        print(f"\n{header}")
        print("-" * (10 + 26 * len(timeframes)))
        
        # 显示每个 horizon 的预测
        for horizon in ['t+1', 't+2', 't+3', 't+4']:
            row = f"{horizon:<10}"
            for tf in timeframes:
                if 'error' not in results.get(tf, {}):
                    pred = results[tf].get('predictions', {}).get(horizon, {})
                    if pred:
                        row += f" {pred['most_likely'][:23]:^25}"
                    else:
                        row += f" {'N/A':^25}"
                else:
                    row += f" {'N/A':^25}"
            print(row)
        
        # 时间对应关系
        print(f"\n⏱️ 时间对应关系:")
        timeframe_minutes = {
            '5m': 5,
            '15m': 15,
            '1h': 60
        }
        for tf in timeframes:
            minutes = timeframe_minutes.get(tf, 15)
            print(f"  {tf}: t+1={minutes}分钟, t+2={minutes*2}分钟, t+3={minutes*3}分钟, t+4={minutes*4}分钟")
        
        # 一致性分析（只分析有结果的时间框架）
        valid_results = {tf: results[tf] for tf in timeframes if 'error' not in results.get(tf, {})}
        if len(valid_results) >= 2:
            print(f"\n📊 t+1 一致性分析:")
            t1_predictions = {}
            for tf, result in valid_results.items():
                t1_predictions[tf] = result.get('predictions', {}).get('t+1', {}).get('most_likely')
            
            # 检查是否所有预测一致
            unique_predictions = set(t1_predictions.values())
            if len(unique_predictions) == 1:
                print(f"  ✓ 所有时间框架的 t+1 预测一致: {list(unique_predictions)[0]}")
            else:
                print(f"  ⚠️ 不同时间框架的 t+1 预测不一致:")
                for tf, pred in t1_predictions.items():
                    print(f"    {tf}: {pred}")
                print(f"     这可能表示市场正在发生不同时间尺度的变化")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


# ============================================================================
# 1h 时间框架专用示例
# ============================================================================

def example_17_1h_single_symbol_training():
    """示例 17: 训练单个交易对的 1h 模型"""
    print("\n" + "="*80)
    print("示例 17: 训练单个交易对的 1h 模型 (BTCUSDT)")
    print("="*80 + "\n")
    
    print("⚠️  注意：完整训练可能需要较长时间（15-40分钟）")
    print("   包括：数据获取、特征工程、HMM训练、LSTM训练")
    print("   1h 模型用于捕捉更长期的市场趋势")
    print("   请耐心等待...\n")
    
    sys.stdout.flush()
    
    try:
        pipeline = TrainingPipeline(TrainingConfig)
        
        # 完整重训 1h 模型
        logger.info("开始训练 1h 模型...")
        result = pipeline.full_retrain("BTCUSDT", primary_timeframe="1h")
        
        print(f"\n✅ 1h 模型训练完成！")
        print(f"测试集准确率 (t+1): {result['test_accuracy']:.2%}")
        print(f"测试集损失: {result['test_loss']:.4f}")
        if 'val_accuracy' in result:
            print(f"验证集准确率 (t+1): {result['val_accuracy']:.2%}")
        
        # 显示多步预测信息
        _print_multistep_results(result)
        
        # 显示动态状态数量优化结果
        if result.get('n_states_optimization'):
            opt = result['n_states_optimization']
            if opt['adjusted']:
                print(f"\n🔄 状态数量已自动调整: {opt['original_n_states']} -> {opt['optimal_n_states']}")
                current_names = set(result.get('regime_mapping', {}).values())
                print(f"   保留的状态: {sorted(current_names)}")
            else:
                print(f"\n✓ 状态数量保持不变: {opt['optimal_n_states']}")
        
        print(f"最终状态数量: {result.get('final_n_states', 6)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        raise
    except Exception as e:
        print(f"\n❌ 1h 模型训练失败: {e}")
        raise


def example_18_1h_realtime_prediction():
    """示例 18: 1h 实时市场状态预测（支持多步预测）"""
    print("\n" + "="*80)
    print("示例 18: 1h 实时市场状态预测（多步预测）")
    print("="*80 + "\n")
    
    try:
        # 创建 1h 预测器
        predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig, primary_timeframe="1h")
        
        # 获取当前市场状态（包括多步预测）
        current = predictor.get_current_regime()
        
        print(f"\n{current['symbol']} 当前市场状态 (1h 时间框架):")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"时间: {current['timestamp']}")
        
        # 显示多步预测结果
        predictions = current.get('predictions', {})
        if predictions:
            print(f"\n📈 多步预测结果 (1h):")
            print(f"  (每步代表 1 小时，t+4 = 4 小时后)")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for horizon in ['t+1', 't+2', 't+3', 't+4']:
                if horizon in predictions:
                    pred = predictions[horizon]
                    bar = "█" * int(pred['confidence'] * 30)
                    uncertain_mark = " ⚠️" if pred.get('is_uncertain', False) else ""
                    print(f"  {horizon}: {pred['regime_name']:20s} {pred['confidence']:6.2%} {bar}{uncertain_mark}")
        
        # 显示 t+1 的详细概率分布
        print(f"\nt+1 状态概率分布详情:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for regime, prob in sorted(
            current['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            bar = "█" * int(prob * 50)
            print(f"  {regime:20s} {prob:6.2%} {bar}")
        
        # 显示历史 regime 序列
        historical = current.get('historical_regimes', {})
        if historical and historical.get('sequence'):
            print(f"\n📜 历史 Regime 序列 (过去 {historical.get('lookback_hours', 16):.1f} 小时):")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            seq = historical['sequence']
            recent = seq[-8:] if len(seq) > 8 else seq
            print(f"  最近 {len(recent)} 根 K 线: {' -> '.join(recent)}")
        
    except FileNotFoundError:
        print("\n❌ 1h 模型文件不存在，请先运行 1h 训练（示例 17）")


def example_19_1h_incremental_training():
    """示例 19: 1h 增量训练"""
    print("\n" + "="*80)
    print("示例 19: 1h 增量训练（在现有 1h 模型基础上）")
    print("="*80 + "\n")
    
    print("⚠️  注意：增量训练通常需要 2-5 分钟")
    print("   包括：获取最新数据、特征工程、模型更新")
    print("   请耐心等待...\n")
    
    sys.stdout.flush()
    
    try:
        pipeline = TrainingPipeline(TrainingConfig)
        
        # 执行 1h 增量训练
        logger.info("开始 1h 增量训练...")
        result = pipeline.incremental_train("BTCUSDT", primary_timeframe="1h")
        
        print(f"\n✅ 1h 增量训练完成！")
        print(f"使用样本数: {result['samples_used']}")
        print(f"训练时间: {result['timestamp']}")
        
    except FileNotFoundError as e:
        print(f"\n❌ 1h 模型文件不存在: {e}")
        print("   请先运行 1h 完整训练（示例 17）")
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        raise
    except Exception as e:
        print(f"\n❌ 1h 增量训练失败: {e}")
        raise


def example_20_1h_regime_history():
    """示例 20: 查看 1h 历史市场状态变化"""
    print("\n" + "="*80)
    print("示例 20: 1h 历史市场状态变化")
    print("="*80 + "\n")
    
    try:
        predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig, primary_timeframe="1h")
        
        # 获取最近 7 天的状态变化（1h 模型适合更长周期）
        history = predictor.get_regime_history(lookback_hours=168)  # 7天
        
        print(f"\n最近 7 天的 1h 市场状态变化:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        if history.empty:
            print("⚠️  没有足够的历史数据进行分析。")
        else:
            print(history.tail(20))
            
            # 统计各状态出现次数
            print(f"\n状态分布:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            regime_counts = history['regime_name'].value_counts()
            for regime, count in regime_counts.items():
                percentage = count / len(history) * 100
                print(f"{regime:20s} {count:4d} 次 ({percentage:5.1f}%)")
        
    except FileNotFoundError:
        print("\n❌ 1h 模型文件不存在，请先运行 1h 训练（示例 17）")


def example_21_batch_1h_training():
    """示例 21: 批量训练多个交易对的 1h 模型"""
    print("\n" + "="*80)
    print("示例 21: 批量训练多个交易对的 1h 模型")
    print("="*80 + "\n")
    
    symbols = TrainingConfig.SYMBOLS
    
    pipeline = TrainingPipeline(TrainingConfig)
    
    logger.info(f"开始批量训练 {len(symbols)} 个交易对的 1h 模型...")
    
    results = pipeline.train_all_symbols(training_type='full', primary_timeframe="1h")
    
    print("\n1h 模型训练结果汇总:")
    for symbol, result in results.items():
        if 'error' in result:
            print(f"{symbol}: 失败 - {result['error']}")
        else:
            print(f"{symbol}: 测试集准确率 {result['test_accuracy']:.2%}")
            if result.get('n_states_optimization') and result['n_states_optimization']['adjusted']:
                opt = result['n_states_optimization']
                print(f"  🔄 状态数量调整: {opt['original_n_states']} -> {opt['optimal_n_states']}")


def example_23_batch_all_timeframes_training():
    """示例 23: 批量训练所有交易对的所有时间框架 (5m + 15m + 1h)"""
    print("\n" + "="*80)
    print("示例 23: 批量训练所有交易对的所有时间框架 (5m + 15m + 1h)")
    print("="*80 + "\n")
    
    print("⚠️  注意：此操作将训练所有交易对的所有时间框架模型")
    print(f"   交易对: {TrainingConfig.SYMBOLS}")
    print(f"   时间框架: {TrainingConfig.ENABLED_MODELS}")
    print(f"   总计: {len(TrainingConfig.SYMBOLS)} 个交易对 × {len(TrainingConfig.ENABLED_MODELS)} 个时间框架 = {len(TrainingConfig.SYMBOLS) * len(TrainingConfig.ENABLED_MODELS)} 个模型")
    print("   预计耗时: 1-2 小时（取决于数据获取速度）")
    print("   请耐心等待...\n")
    
    import sys
    sys.stdout.flush()
    
    try:
        pipeline = TrainingPipeline(TrainingConfig)
        
        # 训练所有交易对的所有时间框架
        logger.info("开始批量训练所有交易对的所有时间框架...")
        results = pipeline.train_all_multi_timeframe(training_type='full')
        
        print("\n" + "="*80)
        print("训练结果汇总")
        print("="*80)
        
        # 按交易对汇总
        for symbol, symbol_results in results.items():
            print(f"\n{symbol}:")
            print("-" * 60)
            for timeframe, result in symbol_results.items():
                if 'error' in result:
                    print(f"  {timeframe}: ❌ {result['error']}")
                else:
                    accuracy = result.get('test_accuracy', 0)
                    print(f"  {timeframe}: ✅ 测试集准确率 {accuracy:.2%}")
                    
                    # 显示状态数量调整信息
                    if result.get('n_states_optimization') and result['n_states_optimization']['adjusted']:
                        opt = result['n_states_optimization']
                        print(f"         🔄 状态数量调整: {opt['original_n_states']} -> {opt['optimal_n_states']}")
                    
                    # 显示状态分布警告
                    if 'state_distribution_check' in result:
                        dist_check = result['state_distribution_check']
                        if not dist_check['healthy']:
                            missing = len(dist_check['missing_states']['val'])
                            if missing > 0:
                                print(f"         ⚠️ 验证集缺失 {missing} 个状态")
        
        # 统计成功和失败
        total_models = sum(len(symbol_results) for symbol_results in results.values())
        successful = sum(
            1 for symbol_results in results.values()
            for result in symbol_results.values()
            if 'error' not in result
        )
        failed = total_models - successful
        
        print("\n" + "="*80)
        print("训练统计")
        print("="*80)
        print(f"总模型数: {total_models}")
        print(f"成功: {successful} ✅")
        print(f"失败: {failed} ❌")
        print(f"成功率: {successful/total_models*100:.1f}%")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        raise
    except Exception as e:
        print(f"\n❌ 批量训练失败: {e}")
        raise


def example_22_multistep_api_1h():
    """示例 22: 使用 API 进行 1h 多步预测"""
    print("\n" + "="*80)
    print("示例 22: 使用 predict_regimes() API 进行 1h 多步预测")
    print("="*80 + "\n")
    
    try:
        api = ModelAPI()
        
        # 使用新的 predict_regimes API
        result = api.predict_regimes(
            symbol="BTCUSDT",
            primary_timeframe="1h",
            include_history=True,
            history_bars=24  # 24 根 1h K 线 = 24 小时/1天
        )
        
        print(f"\n{result['symbol']} 多步预测结果 ({result['timeframe']}):")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"时间: {result['timestamp']}")
        
        # 模型信息
        model_info = result.get('model_info', {})
        print(f"\n📊 模型信息:")
        print(f"  序列长度: {model_info.get('sequence_length', 'N/A')}")
        print(f"  状态数量: {model_info.get('n_states', 'N/A')}")
        print(f"  预测步数: {model_info.get('prediction_horizons', 'N/A')}")
        
        # 多步预测
        predictions = result.get('predictions', {})
        if predictions:
            print(f"\n📈 多步预测 (1h):")
            print(f"  (每步代表 1 小时，t+4 = 4 小时后)")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for horizon in ['t+1', 't+2', 't+3', 't+4']:
                if horizon in predictions:
                    pred = predictions[horizon]
                    bar = "█" * int(pred['confidence'] * 30)
                    uncertain_mark = " ⚠️" if pred.get('is_uncertain', False) else ""
                    print(f"  {horizon}: {pred['most_likely']:20s} {pred['confidence']:6.2%} {bar}{uncertain_mark}")
        
        # 历史序列
        historical = result.get('historical_regimes', {})
        if historical and historical.get('sequence'):
            print(f"\n📜 历史 Regime 序列 (过去 {historical.get('lookback_hours', 24):.1f} 小时):")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            seq = historical['sequence']
            recent = seq[-8:] if len(seq) > 8 else seq
            print(f"  最近 {len(recent)} 根 K 线: {' -> '.join(recent)}")
            
            # 统计
            from collections import Counter
            counts = Counter(seq)
            print(f"  历史分布: {dict(counts)}")
        
    except FileNotFoundError:
        print("\n❌ 1h 模型文件不存在，请先运行 1h 训练（示例 17）")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def print_menu():
    """打印菜单"""
    print("\n" + "="*80)
    print("加密货币市场状态分类器 - 示例脚本")
    print("支持多步预测 (t+1 到 t+4)")
    print("="*80)
    print("\n选择要运行的示例：")
    
    print("\n" + "-"*40)
    print("📊 15m 时间框架 (默认)")
    print("-"*40)
    print("  训练相关:")
    print("    1. 训练单个交易对 (BTCUSDT) [多步预测]")
    print("    2. 批量训练多个交易对")
    print("    6. 增量训练")
    print("  推理相关:")
    print("    3. 实时市场状态预测 [多步预测 t+1~t+4]")
    print("    4. 查看历史市场状态变化")
    print("    5. 多交易对市场状态跟踪")
    print("   14. 🆕 使用 predict_regimes() API 多步预测")
    
    print("\n" + "-"*40)
    print("⚡ 5m 时间框架 (快速决策)")
    print("-"*40)
    print("  训练相关:")
    print("    7. 训练单个交易对 5m 模型 [多步预测]")
    print("   13. 批量训练多个交易对 5m 模型")
    print("    9. 5m 增量训练")
    print("  推理相关:")
    print("    8. 5m 实时市场状态预测 [多步预测 t+1~t+4]")
    print("   10. 5m 历史市场状态变化")
    print("   12. 5m 多交易对市场状态跟踪")
    print("   15. 🆕 使用 predict_regimes() API 5m 多步预测")
    
    print("\n" + "-"*40)
    print("📈 1h 时间框架 (长期趋势)")
    print("-"*40)
    print("  训练相关:")
    print("   17. 训练单个交易对 1h 模型 [多步预测]")
    print("   21. 批量训练多个交易对 1h 模型")
    print("   19. 1h 增量训练")
    print("  推理相关:")
    print("   18. 1h 实时市场状态预测 [多步预测 t+1~t+4]")
    print("   20. 1h 历史市场状态变化")
    print("   22. 🆕 使用 predict_regimes() API 1h 多步预测")
    
    print("\n" + "-"*40)
    print("🔄 多时间框架")
    print("-"*40)
    print("   11. 多时间框架并行预测 (5m + 15m + 1h)")
    print("   16. 🆕 对比多个时间框架的多步预测 (5m + 15m + 1h)")
    print("   23. 🆕 批量训练所有交易对的所有时间框架 (5m + 15m + 1h)")
    
    print("\n" + "-"*40)
    print("其他:")
    print("    0. 退出")
    print("="*80)


def main():
    """主函数"""
    # 确保目录存在
    TrainingConfig.ensure_dirs()
    
    # 运行选定的示例
    examples = {
        # 15m 时间框架
        1: example_1_single_symbol_training,
        2: example_2_multiple_symbols_training,
        3: example_3_realtime_prediction,
        4: example_4_regime_history,
        5: example_5_multi_symbol_tracking,
        6: example_6_incremental_training,
        # 5m 时间框架
        7: example_7_5m_single_symbol_training,
        8: example_8_5m_realtime_prediction,
        9: example_9_5m_incremental_training,
        10: example_10_5m_regime_history,
        11: example_11_multi_timeframe_prediction,
        12: example_12_5m_multi_symbol_tracking,
        13: example_13_batch_5m_training,
        # 多步预测 API 测试
        14: example_14_multistep_api_15m,
        15: example_15_multistep_api_5m,
        16: example_16_compare_timeframes,
        # 1h 时间框架
        17: example_17_1h_single_symbol_training,
        18: example_18_1h_realtime_prediction,
        19: example_19_1h_incremental_training,
        20: example_20_1h_regime_history,
        21: example_21_batch_1h_training,
        22: example_22_multistep_api_1h,
        # 多时间框架批量训练
        23: example_23_batch_all_timeframes_training,
    }
    
    # 如果有命令行参数，直接运行指定示例
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        
        if example_num == 0:
            print("\n👋 再见！")
            return
        
        if example_num in examples:
            try:
                examples[example_num]()
                print("\n✅ 示例运行完成！")
            except Exception as e:
                logger.error(f"示例运行失败: {e}", exc_info=True)
                print(f"\n❌ 错误: {e}")
        else:
            print("❌ 无效的示例编号")
    else:
        # 交互式菜单
        while True:
            print_menu()
            try:
                example_num = int(input("\n请输入示例编号: "))
            except ValueError:
                print("❌ 请输入有效的数字")
                continue
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            
            if example_num == 0:
                print("\n👋 再见！")
                break
            
            if example_num in examples:
                try:
                    examples[example_num]()
                    print("\n✅ 示例运行完成！")
                except Exception as e:
                    logger.error(f"示例运行失败: {e}", exc_info=True)
                    print(f"\n❌ 错误: {e}")
                
                # 继续显示菜单
                try:
                    input("\n按回车键继续...")
                except KeyboardInterrupt:
                    print("\n\n👋 再见！")
                    break
            else:
                print("❌ 无效的示例编号")
                try:
                    input("\n按回车键继续...")
                except KeyboardInterrupt:
                    print("\n\n👋 再见！")
                    break


if __name__ == "__main__":
    main()
