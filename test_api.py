"""
测试 Model API 功能
"""
import sys
import logging
from model_api import ModelAPI, predict_regime, get_regime_probability
from config import TrainingConfig, setup_logging

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_list_models():
    """测试列出可用模型"""
    print("\n" + "="*70)
    print("测试 1: 列出可用的模型")
    print("="*70)
    
    api = ModelAPI()
    available = api.list_available_models()
    
    if not available:
        print("⚠️  没有可用的模型")
        print("   请先运行训练（python examples.py 1）")
        return False
    
    print(f"✓ 找到 {len(available)} 个可用模型:")
    for symbol in available:
        print(f"  - {symbol}")
    
    return True


def test_predict_regime(symbol: str = None):
    """测试预测market regime"""
    print("\n" + "="*70)
    print("测试 2: 预测下一根K线的market regime")
    print("="*70)
    
    api = ModelAPI()
    available = api.list_available_models()
    
    if not available:
        print("⚠️  没有可用的模型，跳过测试")
        return False
    
    if symbol is None:
        symbol = available[0]
    
    if symbol not in available:
        print(f"⚠️  {symbol} 的模型不存在，使用 {available[0]}")
        symbol = available[0]
    
    try:
        result = api.predict_next_regime(symbol, "15m")
        
        print(f"\n✓ 预测成功:")
        print(f"  交易对: {result['symbol']}")
        print(f"  时间框架: {result['timeframe']}")
        print(f"  预测时间: {result['timestamp']}")
        print(f"  使用历史K线数: {result['model_info']['sequence_length']}")
        print(f"\n  最可能的状态: {result['most_likely_regime']['name']}")
        print(f"  概率: {result['most_likely_regime']['probability']:.2%}")
        print(f"  置信度: {result['confidence']:.2%}")
        
        print(f"\n  所有状态概率分布:")
        for regime_name, prob in sorted(
            result['regime_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar = "█" * int(prob * 50)
            print(f"    {regime_name:25s} {prob:6.2%} {bar}")
        
        return True
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        logger.exception("预测失败")
        return False


def test_get_regime_probability(symbol: str = None):
    """测试获取特定状态的概率"""
    print("\n" + "="*70)
    print("测试 3: 获取特定状态的概率")
    print("="*70)
    
    api = ModelAPI()
    available = api.list_available_models()
    
    if not available:
        print("⚠️  没有可用的模型，跳过测试")
        return False
    
    if symbol is None:
        symbol = available[0]
    
    if symbol not in available:
        symbol = available[0]
    
    try:
        # 测试获取几个状态的概率
        regime_names = ["Strong_Trend", "Range", "Volatility_Spike"]
        
        print(f"\n✓ 获取 {symbol} 下一根K线的状态概率:")
        for regime_name in regime_names:
            prob = api.get_regime_probability(symbol, regime_name)
            print(f"  {regime_name:25s} {prob:6.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 获取概率失败: {e}")
        logger.exception("获取概率失败")
        return False


def test_get_metadata(symbol: str = None):
    """测试获取模型元数据"""
    print("\n" + "="*70)
    print("测试 4: 获取模型元数据")
    print("="*70)
    
    api = ModelAPI()
    available = api.list_available_models()
    
    if not available:
        print("⚠️  没有可用的模型，跳过测试")
        return False
    
    if symbol is None:
        symbol = available[0]
    
    if symbol not in available:
        symbol = available[0]
    
    try:
        metadata = api.get_model_metadata(symbol)
        
        print(f"\n✓ 模型元数据:")
        print(f"  交易对: {metadata['symbol']}")
        print(f"  主时间框架: {metadata['primary_timeframe']}")
        print(f"  状态数量: {metadata['n_states']}")
        print(f"  状态映射: {metadata['regime_mapping']}")
        print(f"  状态名称列表: {metadata['regime_names']}")
        print(f"  序列长度: {metadata['training_info']['sequence_length']}")
        if metadata['training_info']['feature_count']:
            print(f"  特征数量: {metadata['training_info']['feature_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 获取元数据失败: {e}")
        logger.exception("获取元数据失败")
        return False


def test_batch_predict():
    """测试批量预测"""
    print("\n" + "="*70)
    print("测试 5: 批量预测多个交易对")
    print("="*70)
    
    api = ModelAPI()
    available = api.list_available_models()
    
    if not available:
        print("⚠️  没有可用的模型，跳过测试")
        return False
    
    # 只测试前2个交易对（如果有的话）
    symbols = available[:2]
    
    try:
        results = api.batch_predict(symbols, "15m", 6)
        
        print(f"\n✓ 批量预测结果:")
        for symbol, result in results.items():
            if 'error' in result:
                print(f"  {symbol}: ❌ {result['error']}")
            else:
                regime = result['most_likely_regime']
                print(f"  {symbol}: {regime['name']} (概率: {regime['probability']:.2%})")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量预测失败: {e}")
        logger.exception("批量预测失败")
        return False


def test_convenience_functions(symbol: str = None):
    """测试便捷函数"""
    print("\n" + "="*70)
    print("测试 6: 便捷函数")
    print("="*70)
    
    api = ModelAPI()
    available = api.list_available_models()
    
    if not available:
        print("⚠️  没有可用的模型，跳过测试")
        return False
    
    if symbol is None:
        symbol = available[0]
    
    if symbol not in available:
        symbol = available[0]
    
    try:
        # 测试便捷函数 predict_regime
        print(f"\n✓ 测试便捷函数 predict_regime():")
        result = predict_regime(symbol, "15m")
        print(f"  最可能状态: {result['most_likely_regime']['name']}")
        
        # 测试便捷函数 get_regime_probability
        print(f"\n✓ 测试便捷函数 get_regime_probability():")
        prob = get_regime_probability(symbol, "Strong_Trend")
        print(f"  Strong_Trend 概率: {prob:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 便捷函数测试失败: {e}")
        logger.exception("便捷函数测试失败")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Model API 功能测试")
    print("="*70)
    
    # 检查是否有可用的模型
    api = ModelAPI()
    available = api.list_available_models()
    
    if not available:
        print("\n⚠️  没有可用的模型！")
        print("   请先运行训练:")
        print("   python examples.py 1")
        print("\n   或者:")
        print("   python training_pipeline.py")
        return
    
    # 获取第一个可用的交易对
    test_symbol = available[0]
    
    # 运行所有测试
    tests = [
        ("列出可用模型", lambda: test_list_models()),
        ("预测market regime", lambda: test_predict_regime(test_symbol)),
        ("获取特定状态概率", lambda: test_get_regime_probability(test_symbol)),
        ("获取模型元数据", lambda: test_get_metadata(test_symbol)),
        ("批量预测", lambda: test_batch_predict()),
        ("便捷函数", lambda: test_convenience_functions(test_symbol)),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} 测试异常: {e}")
            logger.exception(f"{test_name} 测试异常")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "❌ 失败"
        print(f"  {test_name:30s} {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✅ 所有测试通过！")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.exception("测试失败")
        sys.exit(1)

