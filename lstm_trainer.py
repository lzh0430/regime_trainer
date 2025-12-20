"""
LSTM 训练模块 - 用于市场状态预测
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
from typing import Tuple, Dict
import os

logger = logging.getLogger(__name__)

class LSTMRegimeClassifier:
    """LSTM 市场状态分类器"""
    
    def __init__(
        self,
        n_states: int = 6,
        sequence_length: int = 64,
        lstm_units: list = [128, 64],
        dropout_rate: float = 0.2
    ):
        """
        初始化
        
        Args:
            n_states: 状态数量
            sequence_length: 输入序列长度
            lstm_units: LSTM 层单元数列表
            dropout_rate: Dropout 比率
        """
        self.n_states = n_states
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = None
        self.history = None
    
    def build_model(self, n_features: int):
        """
        构建 LSTM 模型
        
        Args:
            n_features: 输入特征数量
        """
        model = keras.Sequential()
        
        # 第一层 LSTM
        model.add(layers.LSTM(
            self.lstm_units[0],
            input_shape=(self.sequence_length, n_features),
            return_sequences=len(self.lstm_units) > 1
        ))
        model.add(layers.Dropout(self.dropout_rate))
        
        # 中间 LSTM 层
        for i in range(1, len(self.lstm_units) - 1):
            model.add(layers.LSTM(self.lstm_units[i], return_sequences=True))
            model.add(layers.Dropout(self.dropout_rate))
        
        # 最后一层 LSTM（如果有多层）
        if len(self.lstm_units) > 1:
            model.add(layers.LSTM(self.lstm_units[-1]))
            model.add(layers.Dropout(self.dropout_rate))
        
        # 全连接层
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        
        # 输出层
        model.add(layers.Dense(self.n_states, activation='softmax'))
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"LSTM 模型已构建")
        logger.info(f"\n{model.summary()}")
    
    def prepare_data(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            features: 特征 DataFrame
            labels: 标签数组（来自 HMM）
            test_size: 测试集比例
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        # 标准化特征
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # 创建序列数据
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i+self.sequence_length])
            y.append(labels[i+self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"序列数据形状: X={X.shape}, y={y.shape}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping: bool = True,
        model_path: str = None
    ) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            early_stopping: 是否使用早停
            model_path: 模型保存路径
            
        Returns:
            训练历史
        """
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        # 回调函数
        callback_list = []
        
        # 早停
        if early_stopping:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callback_list.append(early_stop)
        
        # 学习率衰减
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        callback_list.append(lr_scheduler)
        
        # 模型检查点
        if model_path:
            checkpoint = callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            callback_list.append(checkpoint)
        
        # 训练
        logger.info("开始训练 LSTM 模型...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        return self.history.history
    
    def incremental_train(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32
    ):
        """
        增量训练（在现有模型基础上继续训练）
        
        Args:
            X_new: 新的训练数据
            y_new: 新的标签
            epochs: 训练轮数
            batch_size: 批次大小
        """
        if self.model is None:
            raise ValueError("模型尚未初始化，请先进行完整训练")
        
        logger.info("开始增量训练...")
        
        # 使用较小的学习率
        self.model.optimizer.learning_rate = 1e-5
        
        self.model.fit(
            X_new, y_new,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        logger.info("增量训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测市场状态
        
        Args:
            X: 输入序列
            
        Returns:
            预测的状态标签
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测市场状态的概率分布
        
        Args:
            X: 输入序列
            
        Returns:
            状态概率矩阵
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            评估指标
        """
        loss, accuracy = self.model.evaluate(X_test, y_test)
        
        # 详细评估
        y_pred = self.predict(X_test)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"测试集 Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        logger.info(f"\n混淆矩阵:\n{cm}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save(self, model_path: str, scaler_path: str):
        """保存模型和标准化器"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 保存模型
        self.model.save(model_path)
        logger.info(f"LSTM 模型已保存到 {model_path}")
        
        # 保存标准化器
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler 已保存到 {scaler_path}")
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> 'LSTMRegimeClassifier':
        """加载模型和标准化器"""
        # 加载模型
        model = keras.models.load_model(model_path)
        
        # 加载标准化器
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # 创建实例
        classifier = cls()
        classifier.model = model
        classifier.scaler = scaler
        
        logger.info(f"LSTM 模型已从 {model_path} 加载")
        logger.info(f"Scaler 已从 {scaler_path} 加载")
        
        return classifier
    
    def prepare_live_data(
        self,
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        准备实时数据（用于推理）
        
        Args:
            features: 特征 DataFrame（至少包含 sequence_length 行）
            
        Returns:
            准备好的序列数据
        """
        if len(features) < self.sequence_length:
            raise ValueError(
                f"数据长度不足，需要至少 {self.sequence_length} 行，当前只有 {len(features)} 行"
            )
        
        # 只取最后 sequence_length 行
        features = features.iloc[-self.sequence_length:]
        
        # 标准化
        features_scaled = self.scaler.transform(features)
        
        # 添加 batch 维度
        X = np.array([features_scaled])
        
        return X
