"""
定时任务调度器 - 自动化训练调度
"""
import schedule
import time
import logging
from datetime import datetime
from config import TrainingConfig
from training_pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingScheduler:
    """训练调度器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.pipeline = TrainingPipeline(config)
        self.is_running = False
    
    def incremental_training_job(self):
        """增量训练任务"""
        logger.info("="*80)
        logger.info("触发增量训练任务")
        logger.info("="*80)
        
        try:
            results = self.pipeline.train_all_symbols(training_type='incremental')
            logger.info(f"增量训练完成，结果: {results}")
        except Exception as e:
            logger.error(f"增量训练失败: {e}", exc_info=True)
    
    def full_retrain_job(self):
        """完整重训任务"""
        logger.info("="*80)
        logger.info("触发完整重训任务")
        logger.info("="*80)
        
        try:
            results = self.pipeline.train_all_symbols(training_type='full')
            logger.info(f"完整重训完成，结果: {results}")
        except Exception as e:
            logger.error(f"完整重训失败: {e}", exc_info=True)
    
    def setup_schedule(self):
        """设置调度任务"""
        # 增量训练：每天 8am 和 8pm HKT
        for train_time in self.config.INCREMENTAL_TRAIN_TIMES:
            schedule.every().day.at(train_time).do(self.incremental_training_job)
            logger.info(f"已设置增量训练任务: 每天 {train_time}")
        
        # 完整重训：每周日 3am HKT
        day_name = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'][
            self.config.FULL_RETRAIN_DAY
        ]
        getattr(schedule.every(), day_name).at(self.config.FULL_RETRAIN_TIME).do(
            self.full_retrain_job
        )
        logger.info(f"已设置完整重训任务: 每周{day_name} {self.config.FULL_RETRAIN_TIME}")
    
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
    
    # 可选：立即执行一次完整重训（首次运行时）
    import sys
    if '--init' in sys.argv:
        logger.info("首次运行，执行完整重训...")
        scheduler.full_retrain_job()
    
    # 启动调度器
    scheduler.run()


if __name__ == "__main__":
    main()
