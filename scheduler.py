"""
定时任务调度器 - 自动化增量训练调度

支持按时间框架分别调度增量训练：
- 15m级别模型：每隔指定时间触发一次
- 5m级别模型：每隔指定时间触发一次

训练在后台线程执行，不阻塞API调用。

注意：完整重训（full retrain）需要手动执行，不在此调度器中。
"""
import schedule
import time
import threading
import logging
from datetime import datetime
from config import TrainingConfig, setup_logging
from training_pipeline import TrainingPipeline

setup_logging(log_file='scheduler.log', level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingScheduler:
    """训练调度器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.pipeline = TrainingPipeline(config)
        self.is_running = False
        self.training_lock = threading.Lock()  # 防止多个训练任务同时运行
        self.current_training = None  # 当前正在运行的训练线程
    
    def _run_in_background(self, func, *args, **kwargs):
        """在后台线程运行训练任务，避免阻塞主线程"""
        def wrapper():
            with self.training_lock:
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"后台训练任务异常: {e}", exc_info=True)
                finally:
                    self.current_training = None
        
        # 如果已有训练任务在运行，跳过本次任务
        if self.current_training is not None and self.current_training.is_alive():
            logger.warning("上一个训练任务仍在运行，跳过本次任务")
            return
        
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        self.current_training = thread
        logger.info(f"训练任务已在后台线程启动: {thread.name}")
    
    def incremental_training_job_15m(self):
        """15m级别模型的增量训练任务"""
        logger.info("="*80)
        logger.info("触发15m级别模型增量训练任务")
        logger.info("="*80)
        
        def train():
            try:
                # 训练所有交易对的15m模型
                results = {}
                for symbol in self.config.SYMBOLS:
                    try:
                        result = self.pipeline.incremental_train(
                            symbol=symbol, 
                            primary_timeframe="15m"
                        )
                        results[symbol] = result
                        logger.info(f"✅ {symbol} 15m 增量训练完成")
                    except Exception as e:
                        logger.error(f"❌ {symbol} 15m 增量训练失败: {e}", exc_info=True)
                        results[symbol] = {'error': str(e)}
                
                logger.info(f"15m级别模型增量训练完成，结果: {results}")
            except Exception as e:
                logger.error(f"15m级别模型增量训练失败: {e}", exc_info=True)
        
        self._run_in_background(train)
    
    def incremental_training_job_5m(self):
        """5m级别模型的增量训练任务"""
        logger.info("="*80)
        logger.info("触发5m级别模型增量训练任务")
        logger.info("="*80)
        
        def train():
            try:
                # 训练所有交易对的5m模型
                results = {}
                for symbol in self.config.SYMBOLS:
                    try:
                        result = self.pipeline.incremental_train(
                            symbol=symbol, 
                            primary_timeframe="5m"
                        )
                        results[symbol] = result
                        logger.info(f"✅ {symbol} 5m 增量训练完成")
                    except Exception as e:
                        logger.error(f"❌ {symbol} 5m 增量训练失败: {e}", exc_info=True)
                        results[symbol] = {'error': str(e)}
                
                logger.info(f"5m级别模型增量训练完成，结果: {results}")
            except Exception as e:
                logger.error(f"5m级别模型增量训练失败: {e}", exc_info=True)
        
        self._run_in_background(train)
    
    def incremental_training_job_1h(self):
        """1h级别模型的增量训练任务"""
        logger.info("="*80)
        logger.info("触发1h级别模型增量训练任务")
        logger.info("="*80)
        
        def train():
            try:
                # 训练所有交易对的1h模型
                results = {}
                for symbol in self.config.SYMBOLS:
                    try:
                        result = self.pipeline.incremental_train(
                            symbol=symbol, 
                            primary_timeframe="1h"
                        )
                        results[symbol] = result
                        logger.info(f"✅ {symbol} 1h 增量训练完成")
                    except Exception as e:
                        logger.error(f"❌ {symbol} 1h 增量训练失败: {e}", exc_info=True)
                        results[symbol] = {'error': str(e)}
                
                logger.info(f"1h级别模型增量训练完成，结果: {results}")
            except Exception as e:
                logger.error(f"1h级别模型增量训练失败: {e}", exc_info=True)
        
        self._run_in_background(train)
    
    def setup_schedule(self):
        """设置调度任务"""
        # 15m级别模型：每隔3小时触发一次增量训练
        schedule.every(3).hours.do(self.incremental_training_job_15m)
        logger.info("已设置15m级别模型增量训练任务: 每隔3小时")
        
        # 5m级别模型：每隔指定时间触发一次增量训练
        schedule.every(60).minutes.do(self.incremental_training_job_5m)
        logger.info("已设置5m级别模型增量训练任务: 每隔60分钟")
        
        # 1h级别模型：每隔6小时触发一次增量训练（比15m频率低）
        schedule.every(6).hours.do(self.incremental_training_job_1h)
        logger.info("已设置1h级别模型增量训练任务: 每隔6小时")
    
    def run(self):
        """运行调度器"""
        logger.info("训练调度器启动")
        logger.info(f"监控交易对: {self.config.SYMBOLS}")
        
        self.setup_schedule()
        self.is_running = True
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            logger.info("调度器被用户中断")
        except Exception as e:
            logger.error(f"调度器异常: {e}", exc_info=True)
        finally:
            self.is_running = False
            logger.info("调度器已停止")
    
    def stop(self):
        """停止调度器"""
        self.is_running = False


def main():
    """主函数"""
    # 确保目录存在
    TrainingConfig.ensure_dirs()
    
    # 创建调度器
    scheduler = TrainingScheduler(TrainingConfig)
    
    # 启动调度器
    scheduler.run()


if __name__ == "__main__":
    main()
