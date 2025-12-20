"""
数据获取模块 - 从 Binance 获取 K线数据
"""
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
from typing import List, Dict
import logging
from config import TrainingConfig

logger = logging.getLogger(__name__)

class BinanceDataFetcher:
    """Binance 数据获取器"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        初始化
        
        Args:
            api_key: Binance API Key (可选，读取公开数据不需要)
            api_secret: Binance API Secret (可选)
        """
        self.client = Client(api_key, api_secret)
    
    def get_klines(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            interval: 时间间隔，如 '5m', '15m', '1h'
            start_date: 开始日期
            end_date: 结束日期（默认为当前时间）
            
        Returns:
            包含 OHLCV 数据的 DataFrame
        """
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"获取 {symbol} {interval} 数据，从 {start_date} 到 {end_date}")
        
        # Binance API 限制：每次最多1000条
        all_klines = []
        current_start = start_date
        
        while current_start < end_date:
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(current_start.timestamp() * 1000),
                    endTime=int(end_date.timestamp() * 1000),
                    limit=1000
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # 更新下一次查询的起始时间
                last_timestamp = klines[-1][0]
                current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
                
                logger.debug(f"已获取 {len(klines)} 条数据，总计 {len(all_klines)} 条")
                
            except Exception as e:
                logger.error(f"获取数据失败: {e}")
                raise
        
        # 转换为 DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # 数据类型转换
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # 设置索引
        df.set_index('timestamp', inplace=True)
        
        # 只保留必要的列
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"成功获取 {len(df)} 条 {symbol} {interval} 数据")
        return df
    
    def get_multi_timeframe_data(
        self, 
        symbol: str, 
        timeframes: List[str], 
        start_date: datetime,
        end_date: datetime = None
    ) -> Dict[str, pd.DataFrame]:
        """
        获取多时间框架数据
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表，如 ['5m', '15m', '1h']
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            时间框架到 DataFrame 的字典
        """
        data = {}
        for tf in timeframes:
            df = self.get_klines(symbol, tf, start_date, end_date)
            data[tf] = df
        
        return data
    
    def fetch_latest_data(
        self, 
        symbol: str, 
        timeframes: List[str], 
        days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        获取最近N天的数据（用于增量训练）
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表
            days: 天数
            
        Returns:
            时间框架到 DataFrame 的字典
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_multi_timeframe_data(symbol, timeframes, start_date, end_date)
    
    def fetch_full_training_data(
        self, 
        symbol: str, 
        timeframes: List[str], 
        days: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """
        获取完整训练数据（用于完整重训）
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表
            days: 天数
            
        Returns:
            时间框架到 DataFrame 的字典
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_multi_timeframe_data(symbol, timeframes, start_date, end_date)


def save_data(data: Dict[str, pd.DataFrame], symbol: str, config: TrainingConfig):
    """保存数据到本地"""
    import os
    import pickle
    
    save_dir = os.path.join(config.DATA_DIR, symbol)
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"data_{timestamp}.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"数据已保存到 {save_path}")
    return save_path


def load_latest_data(symbol: str, config: TrainingConfig) -> Dict[str, pd.DataFrame]:
    """加载最新保存的数据"""
    import os
    import pickle
    
    save_dir = os.path.join(config.DATA_DIR, symbol)
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"数据目录不存在: {save_dir}")
    
    # 获取最新的数据文件
    files = [f for f in os.listdir(save_dir) if f.startswith("data_") and f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError(f"未找到数据文件: {save_dir}")
    
    latest_file = sorted(files)[-1]
    load_path = os.path.join(save_dir, latest_file)
    
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"已加载数据: {load_path}")
    return data
