"""
数据缓存管理器 - 使用 SQLite 存储历史 K线数据
实现智能缓存策略，避免重复请求 Binance API
"""
import sqlite3
import pickle
import gzip
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCacheManager:
    """
    数据缓存管理器
    
    使用 SQLite 数据库存储历史 K线数据，按天存储
    支持快速查询、增量更新和数据完整性检查
    """
    
    def __init__(self, db_path: str, compression: bool = True):
        """
        初始化缓存管理器
        
        Args:
            db_path: SQLite 数据库路径
            compression: 是否压缩存储数据
        """
        self.db_path = db_path
        self.compression = compression
        
        # 确保数据库目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS klines_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    date DATE NOT NULL,
                    data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, date)
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_date 
                ON klines_cache(symbol, timeframe, date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_date 
                ON klines_cache(date)
            """)
            
            # 创建特征缓存表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    date DATE NOT NULL,
                    data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, date)
                )
            """)
            
            # 创建特征缓存索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_symbol_timeframe_date 
                ON features_cache(symbol, timeframe, date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_date 
                ON features_cache(date)
            """)
            
            conn.commit()
            logger.debug(f"数据库初始化完成: {self.db_path}")
    
    def _serialize_data(self, df: pd.DataFrame) -> bytes:
        """序列化 DataFrame 为字节"""
        data_bytes = pickle.dumps(df)
        if self.compression:
            data_bytes = gzip.compress(data_bytes)
        return data_bytes
    
    def _deserialize_data(self, data_bytes: bytes) -> pd.DataFrame:
        """反序列化字节为 DataFrame"""
        if self.compression:
            data_bytes = gzip.decompress(data_bytes)
        return pickle.loads(data_bytes)
    
    def save_data(self, symbol: str, timeframe: str, date_val: date, df: pd.DataFrame):
        """
        保存数据到缓存（按天）
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            date_val: 日期
            df: DataFrame 数据
        """
        if df.empty:
            logger.warning(f"尝试保存空数据: {symbol} {timeframe} {date_val}")
            return
        
        data_bytes = self._serialize_data(df)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO klines_cache 
                (symbol, timeframe, date, data, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (symbol, timeframe, date_val.isoformat(), data_bytes))
            conn.commit()
        
        logger.debug(f"已保存缓存: {symbol} {timeframe} {date_val} ({len(df)} 条)")
    
    def save_data_range(
        self, 
        symbol: str, 
        timeframe: str, 
        df: pd.DataFrame
    ):
        """
        保存时间范围内的所有数据（按天拆分）
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            df: DataFrame 数据（必须包含 timestamp 索引）
        """
        if df.empty:
            return
        
        # 按天分组保存
        # 使用 copy() 避免 SettingWithCopyWarning
        df_with_date = df.copy()
        df_with_date['date'] = df_with_date.index.date
        
        # 统计保存的天数
        unique_dates = df_with_date['date'].unique()
        days_count = len(unique_dates)
        
        for date_val, group_df in df_with_date.groupby('date'):
            # 移除临时列
            group_df_clean = group_df.drop(columns=['date'])
            self.save_data(symbol, timeframe, date_val, group_df_clean)
        
        logger.info(f"已保存 {days_count} 天的数据到缓存: {symbol} {timeframe}")
    
    def get_cached_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        从缓存获取指定时间范围的数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            合并后的 DataFrame
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, data 
                FROM klines_cache
                WHERE symbol = ? AND timeframe = ? 
                AND date >= ? AND date <= ?
                ORDER BY date
            """, (symbol, timeframe, start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        # 合并所有数据
        dfs = []
        for date_str, data_bytes in rows:
            try:
                df = self._deserialize_data(data_bytes)
                dfs.append(df)
            except Exception as e:
                logger.error(f"反序列化数据失败 {symbol} {timeframe} {date_str}: {e}")
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, axis=0)
        result = result.sort_index()
        result = result[~result.index.duplicated(keep='last')]  # 去重
        
        logger.debug(
            f"从缓存获取: {symbol} {timeframe} "
            f"{start_date} 至 {end_date} ({len(result)} 条)"
        )
        return result
    
    def get_cached_data_range(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Tuple[pd.DataFrame, List[Tuple[datetime, datetime]]]:
        """
        获取缓存数据并识别缺失的时间段
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            (缓存数据DataFrame, 缺失时间段列表)
        """
        start_date_only = start_date.date()
        end_date_only = end_date.date()
        
        # 获取缓存数据
        cached_df = self.get_cached_data(symbol, timeframe, start_date_only, end_date_only)
        
        # 识别缺失的时间段
        missing_ranges = self.check_data_gaps(symbol, timeframe, start_date, end_date)
        
        return cached_df, missing_ranges
    
    def get_latest_date(self, symbol: str, timeframe: str) -> Optional[date]:
        """
        获取缓存中最新的数据日期
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            最新日期，如果没有数据则返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(date) 
                FROM klines_cache
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
        
        if result and result[0]:
            return datetime.fromisoformat(result[0]).date()
        return None
    
    def get_earliest_date(self, symbol: str, timeframe: str) -> Optional[date]:
        """
        获取缓存中最早的数据日期
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            最早日期，如果没有数据则返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MIN(date) 
                FROM klines_cache
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
        
        if result and result[0]:
            return datetime.fromisoformat(result[0]).date()
        return None
    
    def get_latest_datetime(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        获取缓存中最新的数据时间戳（精确到时间，而非仅日期）
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            最新数据的时间戳，如果没有数据则返回 None
        """
        # 获取最新日期
        latest_date = self.get_latest_date(symbol, timeframe)
        if not latest_date:
            return None
        
        # 获取该日期的所有数据
        cached_df = self.get_cached_data(symbol, timeframe, latest_date, latest_date)
        
        if cached_df.empty:
            return None
        
        # 返回最新数据的时间戳
        return cached_df.index.max()
    
    def check_data_gaps(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        检查数据缺失的时间段
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            缺失时间段列表 [(start, end), ...]
        """
        start_date_only = start_date.date()
        end_date_only = end_date.date()
        
        # 获取缓存中已有的日期
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT date 
                FROM klines_cache
                WHERE symbol = ? AND timeframe = ? 
                AND date >= ? AND date <= ?
                ORDER BY date
            """, (symbol, timeframe, start_date_only.isoformat(), end_date_only.isoformat()))
            
            cached_dates = {datetime.fromisoformat(row[0]).date() for row in cursor.fetchall()}
        
        # 生成期望的所有日期
        expected_dates = set()
        current_date = start_date_only
        while current_date <= end_date_only:
            expected_dates.add(current_date)
            current_date += timedelta(days=1)
        
        # 找出缺失的日期
        missing_dates = sorted(expected_dates - cached_dates)
        
        if not missing_dates:
            return []
        
        # 将连续的缺失日期合并为时间段
        missing_ranges = []
        if missing_dates:
            range_start = datetime.combine(missing_dates[0], datetime.min.time())
            range_end = datetime.combine(missing_dates[0], datetime.max.time())
            
            for i in range(1, len(missing_dates)):
                current_date = datetime.combine(missing_dates[i], datetime.min.time())
                prev_date = datetime.combine(missing_dates[i-1], datetime.min.time())
                
                # 如果日期连续，扩展范围
                if (current_date - prev_date).days == 1:
                    range_end = datetime.combine(missing_dates[i], datetime.max.time())
                else:
                    # 保存当前范围，开始新范围
                    missing_ranges.append((range_start, range_end))
                    range_start = datetime.combine(missing_dates[i], datetime.min.time())
                    range_end = datetime.combine(missing_dates[i], datetime.max.time())
            
            # 添加最后一个范围
            missing_ranges.append((range_start, range_end))
        
        return missing_ranges
    
    def get_cache_stats(self, symbol: str, timeframe: str) -> Dict:
        """
        获取缓存统计信息
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            统计信息字典
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 获取数据范围
            cursor.execute("""
                SELECT MIN(date), MAX(date), COUNT(DISTINCT date)
                FROM klines_cache
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
            
            # 获取总数据量
            cursor.execute("""
                SELECT SUM(LENGTH(data))
                FROM klines_cache
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            total_size = cursor.fetchone()[0] or 0
        
        if not result or not result[0]:
            return {
                'min_date': None,
                'max_date': None,
                'days_count': 0,
                'total_size_mb': 0
            }
        
        return {
            'min_date': datetime.fromisoformat(result[0]).date() if result[0] else None,
            'max_date': datetime.fromisoformat(result[1]).date() if result[1] else None,
            'days_count': result[2] or 0,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
    
    def clear_cache(self, symbol: str = None, timeframe: str = None):
        """
        清除缓存数据
        
        Args:
            symbol: 如果指定，只清除该交易对的缓存
            timeframe: 如果指定，只清除该时间框架的缓存
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if symbol and timeframe:
                cursor.execute("""
                    DELETE FROM klines_cache
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                logger.info(f"已清除缓存: {symbol} {timeframe}")
            elif symbol:
                cursor.execute("""
                    DELETE FROM klines_cache
                    WHERE symbol = ?
                """, (symbol,))
                logger.info(f"已清除缓存: {symbol}")
            else:
                cursor.execute("DELETE FROM klines_cache")
                logger.info("已清除所有缓存")
            
            conn.commit()
    
    # ========== 特征缓存方法 ==========
    
    def save_features(self, symbol: str, timeframe: str, date_val: date, features_df: pd.DataFrame):
        """
        保存特征到缓存（按天）
        
        Args:
            symbol: 交易对
            timeframe: 时间框架（'5m', '15m', '1h', 'combined_15m' 等）
            date_val: 日期
            features_df: 特征 DataFrame
        """
        if features_df.empty:
            logger.warning(f"尝试保存空特征: {symbol} {timeframe} {date_val}")
            return
        
        data_bytes = self._serialize_data(features_df)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO features_cache 
                (symbol, timeframe, date, data, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (symbol, timeframe, date_val.isoformat(), data_bytes))
            conn.commit()
        
        logger.debug(f"已保存特征缓存: {symbol} {timeframe} {date_val} ({len(features_df)} 行, {len(features_df.columns)} 列)")
    
    def save_features_range(
        self, 
        symbol: str, 
        timeframe: str, 
        features_df: pd.DataFrame
    ):
        """
        保存时间范围内的所有特征（按天拆分）
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            features_df: 特征 DataFrame（必须包含 timestamp 索引）
        """
        if features_df.empty:
            return
        
        # 按天分组保存
        features_with_date = features_df.copy()
        features_with_date['date'] = features_with_date.index.date
        
        unique_dates = features_with_date['date'].unique()
        days_count = len(unique_dates)
        
        for date_val, group_df in features_with_date.groupby('date'):
            # 移除临时列
            group_df_clean = group_df.drop(columns=['date'])
            self.save_features(symbol, timeframe, date_val, group_df_clean)
        
        logger.info(f"已保存 {days_count} 天的特征到缓存: {symbol} {timeframe}")
    
    def get_cached_features(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        从缓存获取指定时间范围的特征
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            合并后的特征 DataFrame
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, data 
                FROM features_cache
                WHERE symbol = ? AND timeframe = ? 
                AND date >= ? AND date <= ?
                ORDER BY date
            """, (symbol, timeframe, start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        # 合并所有数据
        dfs = []
        for date_str, data_bytes in rows:
            try:
                df = self._deserialize_data(data_bytes)
                dfs.append(df)
            except Exception as e:
                logger.error(f"反序列化特征失败 {symbol} {timeframe} {date_str}: {e}")
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, axis=0)
        result = result.sort_index()
        result = result[~result.index.duplicated(keep='last')]  # 去重
        
        logger.debug(
            f"从缓存获取特征: {symbol} {timeframe} "
            f"{start_date} 至 {end_date} ({len(result)} 行, {len(result.columns)} 列)"
        )
        return result
    
    def get_latest_features_date(self, symbol: str, timeframe: str) -> Optional[date]:
        """
        获取缓存中最新的特征日期
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            最新日期，如果没有数据则返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(date) 
                FROM features_cache
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
        
        if result and result[0]:
            return datetime.fromisoformat(result[0]).date()
        return None
    
    def get_latest_features_datetime(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        获取缓存中最新的特征时间戳（精确到时间，而非仅日期）
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            最新特征的时间戳，如果没有数据则返回 None
        """
        # 获取最新日期
        latest_date = self.get_latest_features_date(symbol, timeframe)
        if not latest_date:
            return None
        
        # 获取该日期的所有特征
        cached_features = self.get_cached_features(symbol, timeframe, latest_date, latest_date)
        
        if cached_features.empty:
            return None
        
        # 返回最新特征的时间戳
        return cached_features.index.max()
    
    def get_earliest_features_date(self, symbol: str, timeframe: str) -> Optional[date]:
        """
        获取缓存中最早的特征日期
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            最早日期，如果没有数据则返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MIN(date) 
                FROM features_cache
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
        
        if result and result[0]:
            return datetime.fromisoformat(result[0]).date()
        return None
    
    def check_features_gaps(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        检查特征缺失的时间段
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            缺失时间段列表
        """
        start_date_only = start_date.date()
        end_date_only = end_date.date()
        
        # 获取缓存中已有的日期
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT date
                FROM features_cache
                WHERE symbol = ? AND timeframe = ?
                AND date >= ? AND date <= ?
                ORDER BY date
            """, (symbol, timeframe, start_date_only.isoformat(), end_date_only.isoformat()))
            
            cached_dates = {datetime.fromisoformat(row[0]).date() for row in cursor.fetchall()}
        
        # 生成期望的所有日期
        expected_dates = set()
        current_date = start_date_only
        while current_date <= end_date_only:
            expected_dates.add(current_date)
            current_date += timedelta(days=1)
        
        # 找出缺失的日期
        missing_dates = sorted(expected_dates - cached_dates)
        
        if not missing_dates:
            return []
        
        # 将连续的缺失日期合并为时间段
        missing_ranges = []
        if missing_dates:
            current_range_start = missing_dates[0]
            current_range_end = missing_dates[0]
            
            for i in range(1, len(missing_dates)):
                if missing_dates[i] == current_range_end + timedelta(days=1):
                    current_range_end = missing_dates[i]
                else:
                    missing_ranges.append((
                        datetime.combine(current_range_start, datetime.min.time()),
                        datetime.combine(current_range_end, datetime.max.time())
                    ))
                    current_range_start = missing_dates[i]
                    current_range_end = missing_dates[i]
            
            missing_ranges.append((
                datetime.combine(current_range_start, datetime.min.time()),
                datetime.combine(current_range_end, datetime.max.time())
            ))
        return missing_ranges
    
    def clear_features_cache(self, symbol: str = None, timeframe: str = None):
        """
        清除特征缓存数据
        
        Args:
            symbol: 如果指定，只清除该交易对的特征缓存
            timeframe: 如果指定，只清除该时间框架的特征缓存
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if symbol and timeframe:
                cursor.execute("""
                    DELETE FROM features_cache
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                logger.info(f"已清除特征缓存: {symbol} {timeframe}")
            elif symbol:
                cursor.execute("""
                    DELETE FROM features_cache
                    WHERE symbol = ?
                """, (symbol,))
                logger.info(f"已清除特征缓存: {symbol}")
            else:
                cursor.execute("DELETE FROM features_cache")
                logger.info("已清除所有特征缓存")
            
            conn.commit()

