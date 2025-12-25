"""
数据获取模块 - 从 Binance 获取 K线数据
遵循 Binance API 使用规范，实现速率限制和错误处理
集成数据缓存系统，减少 API 请求
"""
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import time
from config import TrainingConfig
from data_cache import DataCacheManager

logger = logging.getLogger(__name__)

class BinanceDataFetcher:
    """
    Binance 数据获取器
    
    遵循 Binance API 使用规范：
    - K线接口权重：1
    - 速率限制：每分钟 1200 权重（IP级别）
    - 建议：每3次请求后延迟1秒，避免触发限制
    """
    
    # Binance API 速率限制配置
    REQUESTS_PER_MINUTE = 1200  # 每分钟最大权重
    KLINES_WEIGHT = 1  # 每次 klines 请求的权重
    SAFE_REQUESTS_PER_BATCH = 3  # 每批安全请求数
    DELAY_BETWEEN_BATCHES = 1.0  # 批次间延迟（秒）
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY = 5  # 重试延迟（秒）
    
    def __init__(self, api_key: str = None, api_secret: str = None, cache_enabled: bool = True):
        """
        初始化
        
        Args:
            api_key: Binance API Key (可选，读取公开数据不需要)
            api_secret: Binance API Secret (可选)
            cache_enabled: 是否启用缓存
        """
        self.client = Client(api_key, api_secret)
        self.request_count = 0  # 请求计数器
        self.last_request_time = 0  # 上次请求时间
        
        # 初始化缓存管理器
        self.cache_enabled = cache_enabled and TrainingConfig.CACHE_ENABLED
        if self.cache_enabled:
            self.cache_manager = DataCacheManager(
                db_path=TrainingConfig.CACHE_DB_PATH,
                compression=TrainingConfig.CACHE_COMPRESSION
            )
            logger.info(f"数据缓存已启用: {TrainingConfig.CACHE_DB_PATH}")
        else:
            self.cache_manager = None
            logger.info("数据缓存已禁用")
        
        # API 请求统计
        self.api_request_count = 0  # API 请求总数
        self.cache_hit_count = 0  # 缓存命中次数
        self.cache_miss_count = 0  # 缓存未命中次数
        self.rate_limit_429_count = 0  # 429 错误次数
        self.total_delay_time = 0.0  # 总延迟时间
    
    def _rate_limit_check(self):
        """
        速率限制检查
        遵循 Binance API 规范：每3次请求后延迟1秒
        """
        self.request_count += 1
        
        # 每 SAFE_REQUESTS_PER_BATCH 次请求后延迟
        if self.request_count % self.SAFE_REQUESTS_PER_BATCH == 0:
            sleep_time = self.DELAY_BETWEEN_BATCHES
            logger.debug(f"速率限制：已发送 {self.request_count} 次请求，延迟 {sleep_time} 秒")
            self.total_delay_time += sleep_time
            time.sleep(sleep_time)
        
        # 检查是否接近每分钟限制
        current_time = time.time()
        if current_time - self.last_request_time < 60:
            # 如果在一分钟内请求过多，增加延迟
            if self.request_count > self.REQUESTS_PER_MINUTE * 0.8:
                extra_delay = 2.0
                logger.warning(f"接近速率限制，额外延迟 {extra_delay} 秒")
                self.total_delay_time += extra_delay
                time.sleep(extra_delay)
        else:
            # 重置计数器（新的一分钟）
            self.request_count = 1
        
        self.last_request_time = current_time
    
    def _handle_api_error(self, error: Exception, retry_count: int) -> bool:
        """
        处理 API 错误
        
        Args:
            error: 异常对象
            retry_count: 当前重试次数
            
        Returns:
            True 如果应该重试，False 如果不应该重试
        """
        if isinstance(error, BinanceAPIException):
            status_code = error.status_code
            
            # HTTP 429: 请求频率过高
            if status_code == 429:
                self.rate_limit_429_count += 1
                if retry_count < self.MAX_RETRIES:
                    wait_time = self.RETRY_DELAY * (2 ** retry_count)  # 指数退避
                    logger.warning(
                        f"收到 429 错误（速率限制），等待 {wait_time} 秒后重试 ({retry_count}/{self.MAX_RETRIES})",
                        exc_info=True
                    )
                    self.total_delay_time += wait_time
                    time.sleep(wait_time)
                    return True
                else:
                    logger.error(
                        f"达到最大重试次数，停止请求以避免封号。最后错误: {error}",
                        exc_info=True
                    )
                    return False
            
            # HTTP 418: IP 被封禁
            elif status_code == 418:
                logger.error(
                    f"收到 418 错误：IP 已被 Binance 封禁！请停止请求并联系 Binance 支持。错误详情: {error}",
                    exc_info=True
                )
                return False
            
            # 其他 API 错误
            else:
                logger.error(
                    f"Binance API 错误 {status_code}: {error.message}",
                    exc_info=True
                )
                if retry_count < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)
                    return True
                return False
        
        # 其他类型的错误
        logger.error(
            f"请求失败: {error}",
            exc_info=True
        )
        if retry_count < self.MAX_RETRIES:
            time.sleep(self.RETRY_DELAY)
            return True
        return False
    
    def get_klines(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        获取K线数据
        
        遵循 Binance API 使用规范：
        - 每次请求最多 1000 条数据
        - 实现速率限制和错误处理
        - 自动重试机制
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            interval: 时间间隔，如 '5m', '15m', '1h'
            start_date: 开始日期
            end_date: 结束日期（默认为当前时间）
            
        Returns:
            包含 OHLCV 数据的 DataFrame
            
        Raises:
            BinanceAPIException: Binance API 错误
            Exception: 其他错误
        """
        if end_date is None:
            end_date = datetime.now()
        
        # 计算需要的数据量
        interval_minutes = self._interval_to_minutes(interval)
        total_minutes = (end_date - start_date).total_seconds() / 60
        estimated_requests = int(total_minutes / interval_minutes / 1000) + 1
        
        logger.info(
            f"获取 {symbol} {interval} 数据，从 {start_date} 到 {end_date} "
            f"(预计需要约 {estimated_requests} 次请求)"
        )
        
        # Binance API 限制：每次最多1000条
        all_klines = []
        current_start = start_date
        request_count = 0
        no_more_data = False  # 标志：是否已经没有更多数据了
        
        while current_start < end_date and not no_more_data:
            retry_count = 0
            success = False
            last_exception = None  # 保存最后一次异常
            
            while retry_count <= self.MAX_RETRIES and not success:
                try:
                    # 速率限制检查
                    self._rate_limit_check()
                    
                    # 发送请求
                    self.api_request_count += 1
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=int(current_start.timestamp() * 1000),
                        endTime=int(end_date.timestamp() * 1000),
                        limit=1000
                    )
                    
                    # 空数组是正常情况（没有更多数据了），不是错误
                    if not klines:
                        success = True  # 标记为成功，因为这是正常的结束
                        no_more_data = True  # 标记没有更多数据，退出外层循环
                        break
                    
                    all_klines.extend(klines)
                    request_count += 1
                    success = True
                    
                    # 更新下一次查询的起始时间
                    last_timestamp = klines[-1][0]
                    current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
                    
                    logger.info(
                        f"已获取 {len(klines)} 条数据，总计 {len(all_klines)} 条 "
                        f"(第 {request_count} 次请求)"
                    )
                    
                except BinanceAPIException as e:
                    last_exception = e
                    should_retry = self._handle_api_error(e, retry_count)
                    if not should_retry:
                        raise
                    retry_count += 1
                
                except Exception as e:
                    last_exception = e
                    logger.error(f"获取数据失败: {e}", exc_info=True)
                    if retry_count < self.MAX_RETRIES:
                        retry_count += 1
                        time.sleep(self.RETRY_DELAY)
                    else:
                        raise
            
            if not success:
                # 只有在真正发生错误时才记录错误
                # 空数组是正常情况，不会到达这里
                if last_exception:
                    logger.error(
                        f"无法获取数据，停止请求。最后异常: {type(last_exception).__name__}: {last_exception}",
                        exc_info=True
                    )
                else:
                    logger.error(
                        "无法获取数据，停止请求（未捕获到异常，可能是重试次数用尽）",
                        extra={"symbol": symbol, "interval": interval, "retry_count": retry_count}
                    )
                break
        
        if not all_klines:
            logger.warning(f"未获取到任何数据: {symbol} {interval}")
            return pd.DataFrame()
        
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
        
        logger.info(
            f"成功获取 {len(df)} 条 {symbol} {interval} 数据 "
            f"(共 {request_count} 次请求)"
        )
        return df
    
    def _interval_to_minutes(self, interval: str) -> int:
        """将时间间隔转换为分钟数"""
        interval_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return interval_map.get(interval, 15)
    
    def get_api_stats(self) -> Dict:
        """
        获取 API 请求统计信息
        
        Returns:
            统计信息字典
        """
        total_requests = self.cache_hit_count + self.cache_miss_count
        cache_hit_rate = (
            (self.cache_hit_count / total_requests * 100) 
            if total_requests > 0 else 0
        )
        
        return {
            'api_request_count': self.api_request_count,
            'cache_hit_count': self.cache_hit_count,
            'cache_miss_count': self.cache_miss_count,
            'cache_hit_rate': round(cache_hit_rate, 2),
            'rate_limit_429_count': self.rate_limit_429_count,
            'total_delay_time': round(self.total_delay_time, 2),
            'average_delay_per_request': (
                round(self.total_delay_time / self.api_request_count, 2)
                if self.api_request_count > 0 else 0
            )
        }
    
    def reset_stats(self):
        """重置统计计数器"""
        self.api_request_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.rate_limit_429_count = 0
        self.total_delay_time = 0.0
        logger.info("统计计数器已重置")
    
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
        智能使用缓存，只请求新数据
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表
            days: 天数
            
        Returns:
            时间框架到 DataFrame 的字典
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = {}
        
        for tf in timeframes:
            logger.info(f"获取增量数据: {symbol} {tf} ({days} 天)")
            
            # 检查缓存
            if self.cache_enabled and self.cache_manager:
                # 获取缓存中最新数据的时间戳（精确到时间）
                latest_cached_datetime = self.cache_manager.get_latest_datetime(symbol, tf)
                
                if latest_cached_datetime:
                    # 计算时间框架的间隔（分钟）
                    interval_minutes = self._interval_to_minutes(tf)
                    
                    # 如果缓存数据足够新，只请求最新数据
                    if latest_cached_datetime >= start_date:
                        # 从缓存获取历史数据（从 start_date 到 latest_cached_datetime 的日期）
                        # 注意：如果缓存中没有 start_date 的数据，get_cached_data 可能返回空
                        # 所以我们需要获取缓存中所有可用的数据，然后过滤
                        cached_df = self.cache_manager.get_cached_data(
                            symbol, tf, start_date.date(), latest_cached_datetime.date()
                        )
                        
                        # 如果缓存数据为空，尝试获取缓存中所有可用的数据
                        if cached_df.empty:
                            # 获取缓存中最早的数据日期
                            earliest_cached_date = self.cache_manager.get_earliest_date(symbol, tf)
                            if earliest_cached_date:
                                # 获取从最早日期到最新日期的所有缓存数据
                                cached_df = self.cache_manager.get_cached_data(
                                    symbol, tf, earliest_cached_date, latest_cached_datetime.date()
                                )
                                logger.debug(
                                    f"缓存数据为空，尝试获取所有可用缓存数据: {symbol} {tf} "
                                    f"({earliest_cached_date} 至 {latest_cached_datetime.date()})"
                                )
                        
                        # 计算需要请求的新数据起始时间
                        # 从最新缓存数据的下一个时间间隔开始
                        # 例如：如果缓存最新是10:00，时间框架是5m，那么应该从10:05开始请求
                        new_start_date = latest_cached_datetime + timedelta(minutes=interval_minutes)
                        
                        # 如果新起始时间小于结束时间，说明有需要请求的新数据
                        if new_start_date < end_date:
                            logger.info(
                                f"缓存命中: {symbol} {tf} - "
                                f"缓存覆盖至 {latest_cached_datetime.strftime('%Y-%m-%d %H:%M:%S')}, "
                                f"仅请求 {new_start_date.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_date.strftime('%Y-%m-%d %H:%M:%S')} 的数据"
                            )
                            self.cache_hit_count += 1
                            
                            # 请求新数据
                            new_df = self.get_klines(symbol, tf, new_start_date, end_date)
                            
                            # 合并数据
                            if not cached_df.empty and not new_df.empty:
                                # 有新数据，合并缓存和新数据
                                df = pd.concat([cached_df, new_df], axis=0)
                                df = df.sort_index()
                                df = df[~df.index.duplicated(keep='last')]
                                # 过滤到请求的时间范围
                                df = df[(df.index >= start_date) & (df.index <= end_date)]
                                
                                # 只保存新获取的数据到缓存
                                self.cache_manager.save_data_range(symbol, tf, new_df)
                            elif not cached_df.empty:
                                # 有新数据请求但返回空数组（正常情况，可能数据还未生成），使用缓存数据
                                df = cached_df
                                # 确保数据在请求的时间范围内
                                df = df[(df.index >= start_date) & (df.index <= end_date)]
                                # 没有新数据，无需保存
                                logger.debug(
                                    f"请求新数据返回空数组（可能是数据还未生成），使用缓存数据: {symbol} {tf}"
                                )
                            else:
                                # 缓存和新数据都为空，使用新数据（虽然为空）
                                df = new_df
                                # 新数据为空，无需保存
                        else:
                            # 缓存已是最新，直接使用
                            logger.info(
                                f"缓存完全命中: {symbol} {tf} - "
                                f"缓存数据已是最新（至 {latest_cached_datetime.strftime('%Y-%m-%d %H:%M:%S')}）"
                            )
                            self.cache_hit_count += 1
                            df = cached_df
                            # 确保数据在请求的时间范围内
                            df = df[(df.index >= start_date) & (df.index <= end_date)]
                            # 缓存已是最新，无需再次保存
                    else:
                        # 缓存数据太旧，需要请求
                        logger.info(
                            f"缓存未命中: {symbol} {tf} - "
                            f"缓存数据至 {latest_cached_datetime.strftime('%Y-%m-%d %H:%M:%S')}，需要请求 {start_date.strftime('%Y-%m-%d %H:%M:%S')} 起的数据"
                        )
                        self.cache_miss_count += 1
                        df = self.get_klines(symbol, tf, start_date, end_date)
                        
                        # 保存新数据到缓存
                        if not df.empty:
                            self.cache_manager.save_data_range(symbol, tf, df)
                else:
                    # 没有缓存，直接请求
                    logger.info(f"无缓存数据: {symbol} {tf}，从 API 获取")
                    self.cache_miss_count += 1
                    df = self.get_klines(symbol, tf, start_date, end_date)
                    
                    # 保存新数据到缓存
                    if not df.empty:
                        self.cache_manager.save_data_range(symbol, tf, df)
            else:
                # 缓存未启用，直接请求
                df = self.get_klines(symbol, tf, start_date, end_date)
            
            data[tf] = df
        
        return data
    
    def fetch_full_training_data(
        self, 
        symbol: str, 
        timeframes: List[str], 
        days: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """
        获取完整训练数据（用于完整重训）
        智能使用缓存，只请求缺失的数据
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表
            days: 天数
            
        Returns:
            时间框架到 DataFrame 的字典
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = {}
        
        for tf in timeframes:
            logger.info(f"获取完整训练数据: {symbol} {tf} ({days} 天)")
            
            # 检查缓存
            if self.cache_enabled and self.cache_manager:
                # 获取缓存数据和缺失时间段
                cached_df, missing_ranges = self.cache_manager.get_cached_data_range(
                    symbol, tf, start_date, end_date
                )
                
                # 计算缓存覆盖率
                total_days = (end_date - start_date).days + 1
                cached_days = len(cached_df.groupby(cached_df.index.date)) if not cached_df.empty else 0
                cache_coverage = (cached_days / total_days * 100) if total_days > 0 else 0
                
                logger.info(
                    f"缓存检查: {symbol} {tf} - "
                    f"缓存覆盖 {cached_days}/{total_days} 天 ({cache_coverage:.1f}%)"
                )
                
                if missing_ranges:
                    logger.info(
                        f"缺失时间段: {len(missing_ranges)} 个时间段，"
                        f"共约 {sum((r[1] - r[0]).days + 1 for r in missing_ranges)} 天"
                    )
                    
                    # 请求缺失的数据
                    missing_dfs = []
                    for miss_start, miss_end in missing_ranges:
                        logger.info(
                            f"请求缺失数据: {miss_start.date()} 至 {miss_end.date()}"
                        )
                        missing_df = self.get_klines(symbol, tf, miss_start, miss_end)
                        if not missing_df.empty:
                            missing_dfs.append(missing_df)
                    
                    # 合并所有数据
                    all_dfs = [cached_df] + missing_dfs if not cached_df.empty else missing_dfs
                    if all_dfs:
                        df = pd.concat(all_dfs, axis=0)
                        df = df.sort_index()
                        df = df[~df.index.duplicated(keep='last')]
                    else:
                        df = cached_df
                    
                    # 保存新数据到缓存
                    for missing_df in missing_dfs:
                        if not missing_df.empty:
                            self.cache_manager.save_data_range(symbol, tf, missing_df)
                    
                    self.cache_hit_count += 1 if cached_days > 0 else 0
                    self.cache_miss_count += len(missing_ranges)
                else:
                    # 缓存完全覆盖
                    logger.info(
                        f"缓存完全命中: {symbol} {tf} - "
                        f"所有数据已缓存，无需 API 请求"
                    )
                    self.cache_hit_count += 1
                    df = cached_df
            else:
                # 缓存未启用，直接请求
                logger.info(f"缓存未启用: {symbol} {tf}，从 API 获取所有数据")
                self.cache_miss_count += 1
                df = self.get_klines(symbol, tf, start_date, end_date)
                
                # 如果缓存可用，保存数据
                if self.cache_enabled and self.cache_manager and not df.empty:
                    self.cache_manager.save_data_range(symbol, tf, df)
            
            data[tf] = df
        
        return data


def save_data(data: Dict[str, pd.DataFrame], symbol: str, config: TrainingConfig):
    """
    保存数据到本地（已弃用）
    
    注意：此函数已弃用。数据现在自动保存到 SQLite 缓存中。
    保留此函数仅用于向后兼容。
    
    Args:
        data: 数据字典
        symbol: 交易对
        config: 配置对象
        
    Returns:
        保存路径
    """
    import warnings
    warnings.warn(
        "save_data() 已弃用。数据现在自动保存到 SQLite 缓存中。"
        "请使用 DataCacheManager 来管理数据。",
        DeprecationWarning,
        stacklevel=2
    )
    
    import os
    import pickle
    
    save_dir = os.path.join(config.DATA_DIR, symbol)
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"data_{timestamp}.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.warning(f"数据已保存到 {save_path} (已弃用，建议使用 SQLite 缓存)")
    return save_path


def load_latest_data(symbol: str, config: TrainingConfig) -> Dict[str, pd.DataFrame]:
    """
    加载最新保存的数据（已弃用）
    
    注意：此函数已弃用。请使用 DataCacheManager.get_cached_data() 来获取数据。
    保留此函数仅用于向后兼容。
    
    Args:
        symbol: 交易对
        config: 配置对象
        
    Returns:
        数据字典
    """
    import warnings
    warnings.warn(
        "load_latest_data() 已弃用。请使用 DataCacheManager.get_cached_data() 来获取数据。",
        DeprecationWarning,
        stacklevel=2
    )
    
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
    
    logger.warning(f"已加载数据: {load_path} (已弃用，建议使用 SQLite 缓存)")
    return data
