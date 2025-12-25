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
    
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """将时间框架转换为分钟数"""
        timeframe_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return timeframe_map.get(timeframe, 15)
    
    def check_data_gaps(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        检查数据缺失的时间段（精确到时间戳级别）
        
        修复：不再只检查日期级别的缺失，而是精确检查时间戳级别的缺失。
        
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
        
        missing_ranges = []
        
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
        
        # 找出完全缺失的日期（整天都没有数据）
        missing_dates = sorted(expected_dates - cached_dates)
        
        # 将连续的缺失日期合并为时间段
        if missing_dates:
            range_start = datetime.combine(missing_dates[0], datetime.min.time())
            range_end = datetime.combine(missing_dates[0], datetime.max.time())
            
            for i in range(1, len(missing_dates)):
                current_dt = datetime.combine(missing_dates[i], datetime.min.time())
                prev_dt = datetime.combine(missing_dates[i-1], datetime.min.time())
                
                # 如果日期连续，扩展范围
                if (current_dt - prev_dt).days == 1:
                    range_end = datetime.combine(missing_dates[i], datetime.max.time())
                else:
                    # 保存当前范围，开始新范围
                    missing_ranges.append((range_start, range_end))
                    range_start = datetime.combine(missing_dates[i], datetime.min.time())
                    range_end = datetime.combine(missing_dates[i], datetime.max.time())
            
            # 添加最后一个范围
            missing_ranges.append((range_start, range_end))
        
        # ===== 关键修复：检查时间戳级别的缺口 =====
        # 获取缓存数据的实际时间范围
        cached_df = self.get_cached_data(symbol, timeframe, start_date_only, end_date_only)
        
        if cached_df.empty:
            # 如果没有任何缓存数据，整个时间范围都是缺失的
            if not missing_ranges:
                missing_ranges.append((start_date, end_date))
            return missing_ranges
        
        actual_start = cached_df.index.min()
        actual_end = cached_df.index.max()
        
        interval_minutes = self._get_timeframe_minutes(timeframe)
        tolerance = timedelta(minutes=interval_minutes * 2)  # 允许2个时间间隔的容差
        
        # 检查头部缺口：请求的 start_date 到缓存数据的实际开始时间
        if actual_start > start_date + tolerance:
            head_gap = (start_date, actual_start - timedelta(minutes=1))
            # 检查是否与已有的缺失范围重叠，如果不重叠则添加
            if not any(r[0] <= head_gap[0] <= r[1] or r[0] <= head_gap[1] <= r[1] for r in missing_ranges):
                missing_ranges.append(head_gap)
                logger.debug(f"发现头部缺口: {head_gap[0]} 至 {head_gap[1]}")
        
        # 检查尾部缺口：缓存数据的实际结束时间到请求的 end_date
        if actual_end + tolerance < end_date:
            tail_gap_start = actual_end + timedelta(minutes=interval_minutes)
            tail_gap = (tail_gap_start, end_date)
            # 检查是否与已有的缺失范围重叠
            if not any(r[0] <= tail_gap[0] <= r[1] or r[0] <= tail_gap[1] <= r[1] for r in missing_ranges):
                missing_ranges.append(tail_gap)
                logger.debug(f"发现尾部缺口: {tail_gap[0]} 至 {tail_gap[1]}")
        
        # 按开始时间排序
        missing_ranges.sort(key=lambda x: x[0])
        
        # 输出调试信息
        if missing_ranges:
            logger.info(
                f"数据缺口检查 ({symbol} {timeframe}): "
                f"请求范围 {start_date.strftime('%Y-%m-%d %H:%M')} 至 {end_date.strftime('%Y-%m-%d %H:%M')}, "
                f"缓存范围 {actual_start.strftime('%Y-%m-%d %H:%M')} 至 {actual_end.strftime('%Y-%m-%d %H:%M')}, "
                f"发现 {len(missing_ranges)} 个缺口"
            )
        
        return missing_ranges
    
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

