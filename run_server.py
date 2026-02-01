"""
一键启动服务器 - 同时运行 API 服务器和训练调度器

运行方式:
    python run_server.py

这将自动：
1. 启动 HTTP API 服务器（端口 5000）
2. 在后台启动训练调度器（自动执行增量训练）

API 端点:
    GET  /api/health
    GET  /api/predict/<symbol>?timeframe=15m
    GET  /api/predict_regimes/<symbol>?timeframe=15m
    GET  /api/metadata/<symbol>?timeframe=15m
    GET  /api/models/available
    GET  /api/models/by_timeframe
    POST /api/batch_predict
"""
import logging
import threading
import sys

from config import TrainingConfig, setup_logging
from model_api import ModelAPI, create_app
from scheduler import TrainingScheduler
from forward_testing import ForwardTestCronManager
from config_registry import list_config_versions, init_from_config_file
import os

# 配置日志
setup_logging(log_file='server.log', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """主函数 - 启动 API 服务器和训练调度器"""
    # 确保目录存在
    TrainingConfig.ensure_dirs()
    
    # Check for config initialization
    configs = list_config_versions()
    auto_init = os.getenv('AUTO_INIT_CONFIG', 'false').lower() == 'true'
    
    if len(configs) == 0:
        if auto_init:
            logger.info("No configs found, auto-initializing from TrainingConfig...")
            try:
                config_version_id = init_from_config_file('Auto-initialized on server startup')
                logger.info(f"✅ Config initialized: {config_version_id}")
            except Exception as e:
                logger.warning(f"Failed to auto-initialize config: {e}")
        else:
            logger.info("No configs found in database. Using TrainingConfig defaults.")
            logger.info("To initialize configs, use POST /api/configs/init or set AUTO_INIT_CONFIG=true")
    else:
        logger.info(f"Using database configs ({len(configs)} versions available)")
        logger.info("Note: POST /api/configs/init will always create a new version from TrainingConfig")
    
    # 初始化 API
    api = ModelAPI()
    
    # 启动训练调度器（后台线程）
    logger.info("正在启动训练调度器...")
    scheduler = TrainingScheduler(TrainingConfig)
    scheduler_thread = threading.Thread(target=scheduler.run, daemon=True)
    scheduler_thread.start()
    logger.info("✅ 训练调度器已启动（后台运行）")
    
    # 启动 forward test cron manager（每个 campaign 独立的 cron job）
    cron_manager = ForwardTestCronManager(TrainingConfig)
    cron_manager.sync_jobs_from_db()  # Sync existing campaigns from DB
    cron_manager.start()  # Start background thread running schedule.run_pending()
    logger.info("✅ Forward test cron manager 已启动（后台运行，每个 campaign 独立 cron job）")
    
    # 创建 Flask 应用
    try:
        app = create_app(api)
    except ImportError as e:
        logger.error(f"❌ 无法启动 HTTP 服务器: {e}")
        logger.error("请安装 Flask 和 flask-cors: pip install flask flask-cors")
        sys.exit(1)
    
    # 配置服务器
    host = '0.0.0.0'
    port = 5000
    
    # 显示启动信息
    logger.info("="*80)
    logger.info("🚀 API 服务器启动")
    logger.info("="*80)
    logger.info(f"📡 监听地址: http://{host}:{port}")
    logger.info(f"📊 监控交易对: {TrainingConfig.SYMBOLS}")
    logger.info("")
    logger.info("API 端点:")
    logger.info("  GET  /api/health                          - 健康检查")
    logger.info("  GET  /api/predict/<symbol>?timeframe=15m  - 预测下一根K线")
    logger.info("  GET  /api/predict_regimes/<symbol>        - 多步预测（推荐）")
    logger.info("  GET  /api/history/<symbol>                - 获取历史regime序列（支持日期范围）")
    logger.info("  GET  /api/metadata/<symbol>               - 获取模型元数据")
    logger.info("  GET  /api/models/available                - 列出可用模型")
    logger.info("  GET  /api/models/by_timeframe             - 按时间框架列出模型")
    logger.info("  GET  /api/models/prod                     - 获取 PROD 指针 (query: symbol, timeframe)")
    logger.info("  POST /api/models/prod                     - 设置 PROD 指针 (body: symbol, timeframe, version_id)")
    logger.info("  GET  /api/models/versions                 - 列出版本及 symbol/timeframe")
    logger.info("  POST /api/forward_test/trigger_all        - 触发所有待执行的 forward test")
    logger.info("  GET  /api/forward_test/status             - 获取 forward test cron 状态")
    logger.info("  GET  /api/forward_test/accuracy           - 获取 campaign 准确率 (query: version_id, symbol, timeframe)")
    logger.info("  POST /api/batch_predict                   - 批量预测")
    logger.info("  GET  /api/docs                             - Swagger API 文档")
    logger.info("")
    logger.info("训练调度:")
    logger.info(f"  ✅ 15m 模型: 每 {getattr(TrainingConfig, 'INCREMENTAL_TRAIN_INTERVAL_15M', 3)} 小时增量训练")
    logger.info(f"  ✅ 5m 模型: 每 {getattr(TrainingConfig, 'INCREMENTAL_TRAIN_INTERVAL_5M', 60)} 分钟增量训练")
    logger.info("Forward test: 每个 campaign 独立 cron job（5m/15m/1h 按时间框架），5 次后 qualified")
    logger.info("")
    logger.info("按 Ctrl+C 停止服务器")
    logger.info("="*80)
    
    # 运行 Flask 应用
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("\n正在关闭服务器...")
        if scheduler:
            scheduler.stop()
        if cron_manager:
            cron_manager.stop()
        logger.info("服务器已停止")


if __name__ == "__main__":
    main()

