"""
快速开始示例脚本
"""
import sys
import logging
from config import TrainingConfig, setup_logging
from training_pipeline import TrainingPipeline
from realtime_predictor import RealtimeRegimePredictor, MultiSymbolRegimeTracker

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        print(f"测试集准确率: {result['test_accuracy']:.2%}")
        print(f"测试集损失: {result['test_loss']:.4f}")
        if 'val_accuracy' in result:
            print(f"验证集准确率: {result['val_accuracy']:.2%}")
        
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
    """示例 3: 实时市场状态预测"""
    print("\n" + "="*80)
    print("示例 3: 实时市场状态预测")
    print("="*80 + "\n")
    
    try:
        # 创建预测器
        predictor = RealtimeRegimePredictor("BTCUSDT", TrainingConfig)
        
        # 获取当前市场状态
        current = predictor.get_current_regime()
        
        print(f"\n{current['symbol']} 当前市场状态:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"状态: {current['regime_name']}")
        print(f"置信度: {current['confidence']:.2%}")
        print(f"时间: {current['timestamp']}")
        
        print(f"\n所有状态概率分布:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for regime, prob in sorted(
            current['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            bar = "█" * int(prob * 50)
            print(f"{regime:20s} {prob:6.2%} {bar}")
        
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


def print_menu():
    """打印菜单"""
    print("\n" + "="*80)
    print("加密货币市场状态分类器 - 示例脚本")
    print("="*80)
    print("\n选择要运行的示例：")
    print("\n训练相关:")
    print("  1. 训练单个交易对 (BTCUSDT)")
    print("  2. 批量训练多个交易对")
    print("  6. 增量训练（需要先运行示例 1）")
    print("\n推理相关:")
    print("  3. 实时市场状态预测")
    print("  4. 查看历史市场状态变化")
    print("  5. 多交易对市场状态跟踪")
    print("\n其他:")
    print("  0. 退出")
    print("\n" + "="*80)


def main():
    """主函数"""
    # 确保目录存在
    TrainingConfig.ensure_dirs()
    
    # 运行选定的示例
    examples = {
        1: example_1_single_symbol_training,
        2: example_2_multiple_symbols_training,
        3: example_3_realtime_prediction,
        4: example_4_regime_history,
        5: example_5_multi_symbol_tracking,
        6: example_6_incremental_training,
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
