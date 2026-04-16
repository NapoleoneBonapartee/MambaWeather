"""
MCSTWeather: Multi-Channel Spatio-Temporal Mamba with Weather Integration
基于 MCSTMamba_optimized，融合天气数据进行交通预测
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

from libcity.model.traffic_speed_prediction.MambaWeather import SimpleMambaWeatherBlock


class WeatherProcessor:
    """天气数据处理器：处理缺失值、构建衍生特征"""
    
    def __init__(self, weather_file_path):
        self.weather_file_path = weather_file_path
        self.weather_df = None
        self.weather_features = None
        self.feature_names = None
        
    def load_data(self):
        """加载天气数据"""
        if not os.path.exists(self.weather_file_path):
            raise FileNotFoundError(f"Weather file not found: {self.weather_file_path}")
        
        self.weather_df = pd.read_csv(self.weather_file_path, index_col=0, parse_dates=True)
        self._process_missing_values()
        self._build_derived_features()
        return self
    
    def _process_missing_values(self):
        """处理缺失值：前向填充 + 后向填充"""
        # 数值列：先线性插值，再前向填充，最后后向填充
        numeric_cols = self.weather_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 线性插值
            self.weather_df[col] = self.weather_df[col].interpolate(method='linear', limit_direction='both')
            # 前向填充（处理开头的缺失值）
            self.weather_df[col] = self.weather_df[col].fillna(method='ffill')
            # 后向填充（处理末尾的缺失值）
            self.weather_df[col] = self.weather_df[col].fillna(method='bfill')
            # 剩余缺失值用均值填充
            self.weather_df[col] = self.weather_df[col].fillna(self.weather_df[col].mean())
        
        # 类别列：用众数填充
        categorical_cols = self.weather_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.weather_df[col] = self.weather_df[col].fillna(self.weather_df[col].mode()[0] if not self.weather_df[col].mode().empty else 0)
    
    def _build_derived_features(self):
        """构建天气衍生特征"""
        df = self.weather_df.copy()
        derived_features = pd.DataFrame(index=df.index)
        
        # 1. 温度相关特征（如果存在）
        if 'TMAX' in df.columns and 'TMIN' in df.columns:
            # 日温差
            derived_features['temp_range'] = df['TMAX'] - df['TMIN']
            # 平均温度
            derived_features['temp_avg'] = (df['TMAX'] + df['TMIN']) / 2
        
        if 'AWBT' in df.columns and 'TMAX' in df.columns:
            # 体感温度差
            derived_features['apparent_temp_diff'] = df['AWBT'] - df['TMAX']
        
        # 2. 湿度相关特征
        if 'RHAV' in df.columns:
            # 湿度等级（低、中、高）
            derived_features['humidity_level'] = pd.cut(df['RHAV'], bins=[0, 30, 60, 100], labels=[0, 1, 2]).astype(float)
            # 湿度变化率（一阶差分）
            derived_features['humidity_change'] = df['RHAV'].diff().fillna(0)
        
        if 'RHMX' in df.columns and 'RHMN' in df.columns:
            # 湿度范围
            derived_features['humidity_range'] = df['RHMX'] - df['RHMN']
        
        # 3. 风速相关特征
        if 'WSF2' in df.columns and 'WSF5' in df.columns:
            # 风速比（短时 gust 与平均风速比）
            derived_features['wind_gust_ratio'] = df['WSF5'] / (df['WSF2'] + 1e-6)
            # 平均风速
            derived_features['wind_avg'] = (df['WSF2'] + df['WSF5']) / 2
        
        if 'AWND' in df.columns:
            derived_features['wind_speed'] = df['AWND']
        
        # 4. 气压相关特征
        if 'ADPT' in df.columns and 'ASTP' in df.columns:
            # 站压与海平面气压差
            derived_features['pressure_diff'] = df['ASTP'] - df['ADPT']
        
        if 'ASLP' in df.columns:
            # 气压变化率
            derived_features['pressure_change'] = df['ASLP'].diff().fillna(0)
            # 气压等级（用于判断天气稳定性）
            derived_features['pressure_level'] = pd.cut(df['ASLP'], bins=[0, 1010, 1020, 1100], labels=[0, 1, 2]).astype(float)
        
        # 5. 降水相关特征
        if 'PRCP' in df.columns:
            # 是否有降水
            derived_features['has_precipitation'] = (df['PRCP'] > 0).astype(float)
            # 降水强度等级
            derived_features['precipitation_level'] = pd.cut(df['PRCP'], bins=[-0.1, 0, 2.5, 10, 1000], labels=[0, 1, 2, 3]).astype(float)
        
        # 6. 风向特征（转换为正弦/余弦编码）
        if 'WDF2' in df.columns:
            derived_features['wind_dir_sin'] = np.sin(np.radians(df['WDF2']))
            derived_features['wind_dir_cos'] = np.cos(np.radians(df['WDF2']))
        
        # 7. 天气现象特征（WT系列）
        wt_cols = [col for col in df.columns if col.startswith('WT')]
        if wt_cols:
            # 统计同时发生的天气现象数量
            derived_features['weather_phenomena_count'] = df[wt_cols].notna().sum(axis=1).astype(float)
            # 是否有恶劣天气（雾、雨、雪等）
            severe_weather_cols = [c for c in wt_cols if any(x in c for x in ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06'])]
            if severe_weather_cols:
                derived_features['has_severe_weather'] = df[severe_weather_cols].notna().any(axis=1).astype(float)
        
        # 8. 时间特征
        derived_features['hour'] = df.index.hour
        derived_features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        derived_features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # 填充可能的缺失值
        derived_features = derived_features.fillna(0)
        
        self.weather_features = derived_features
        self.feature_names = list(derived_features.columns)
        self.num_weather_features = len(self.feature_names)
        
        return self
    
    def get_features_at_timestamps(self, timestamps):
        """
        获取指定时间戳的天气特征
        处理时间戳可能超出天气数据范围的情况
        Args:
            timestamps: list or array of timestamps
        Returns:
            numpy array of shape (len(timestamps), num_weather_features)
        """
        if self.weather_features is None:
            raise ValueError("Weather features not built. Call load_data() first.")
        
        # 获取天气数据的时间范围
        min_time = self.weather_features.index.min()
        max_time = self.weather_features.index.max()
        
        # 将输入时间戳转换为 pandas DatetimeIndex
        ts_index = pd.DatetimeIndex(timestamps)
        
        # 处理超出范围的时间戳：截断到有效范围
        ts_index = ts_index.map(lambda x: min_time if x < min_time else (max_time if x > max_time else x))
        
        # 使用 get_indexer 方法找到最近的时间索引（支持 method='nearest'）
        indices = self.weather_features.index.get_indexer(ts_index, method='nearest')
        
        # 处理 -1（未找到）的情况
        indices = [idx if idx >= 0 else len(self.weather_features) - 1 for idx in indices]
        
        return self.weather_features.iloc[indices].values
    
    def normalize_features(self, scaler=None):
        """标准化天气特征"""
        if scaler is None:
            # 使用 Z-score 标准化
            self.feature_mean = self.weather_features.mean()
            self.feature_std = self.weather_features.std().replace(0, 1)  # 避免除零
            self.weather_features = (self.weather_features - self.feature_mean) / self.feature_std
        else:
            self.weather_features = scaler.transform(self.weather_features)
        return self


class AdaptiveFusion(nn.Module):
    """
    输入条件化的时-空特征融合模块。
    将投影后的原始输入 x、时序输出 temporal、空间输出 spatial 拼接后通过 MLP 学习融合表示。
    不使用门控或归一化权重，避免此消彼长；同时通过残差连接保留原始输入信息。
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.input_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x_input, x_temporal, x_spatial):
        # x_input / x_temporal / x_spatial: (B, L, N, d_model)
        combined = torch.cat([x_input, x_temporal, x_spatial], dim=-1)
        out = self.fusion_mlp(combined)
        out = out + self.input_proj(x_input)
        return out


class MCSTWeather(AbstractTrafficStateModel):
    """
    MCST-Mamba with Weather Integration
    融合天气数据的时空交通预测模型
    """
    
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        # 保存 config 供后续使用
        self.config = config
        
        # 基础数据特征
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        
        # 时间相关特征（从dataset传入）
        self.start_time = self.data_feature.get('start_time', None)
        self.total_time_steps = self.data_feature.get('total_time_steps', None)
        self.time_intervals = self.config.get('time_intervals', 300)
        self.has_time_column = self.start_time is not None and self.total_time_steps is not None
        if self.has_time_column:
            # 输入特征中最后一列为时间步索引，需要去掉后再投影
            self.feature_dim = self.feature_dim - 1
        
        # 模型配置
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self._logger = getLogger()
        
        # 时间特征配置
        self.add_time_in_day = config.get("add_time_in_day", False)
        self.add_day_in_week = config.get("add_day_in_week", False)
        self.steps_per_day = config.get("steps_per_day", 288)
        
        # 嵌入维度配置
        self.input_embedding_dim = config.get('input_embedding_dim', 24)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 24) if self.add_time_in_day else 0
        self.dow_embedding_dim = config.get('dow_embedding_dim', 24) if self.add_day_in_week else 0
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 16)
        self.adaptive_embedding_dim = config.get('adaptive_embedding_dim', 80)
        
        # 天气相关配置
        self.weather_embed_dim = config.get('weather_embed_dim', 64)
        self.weather_file = config.get('weather_file', None)  # 天气文件路径
        self.use_weather = config.get('use_weather', True)
        
        # 计算模型总维度（不含天气，天气单独处理）
        self.model_dim = (
            self.input_embedding_dim +
            self.tod_embedding_dim +
            self.dow_embedding_dim +
            self.spatial_embedding_dim +
            self.adaptive_embedding_dim
        )
        
        # 创建嵌入层
        self.input_proj = nn.Linear(self.feature_dim, self.input_embedding_dim)
        
        if self.add_time_in_day:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.add_day_in_week:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        
        # 空间嵌入
        self.spatial_embedding = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
        nn.init.xavier_uniform_(self.spatial_embedding)
        
        # 自适应嵌入
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(self.input_window, self.num_nodes, self.adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)
        
        # Mamba 参数
        self.d_model = config.get('d_model', 96)
        self.d_state = config.get('d_state', 32)
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # 输入投影到 Mamba 维度
        self.mamba_input_proj = nn.Linear(self.model_dim, self.d_model)
        
        # 天气数据处理器
        self.weather_processor = None
        if self.use_weather:
            self._setup_weather_processor()
        
        # 天气特征嵌入层
        if self.use_weather and self.weather_processor is not None:
            self.weather_feature_proj = nn.Sequential(
                nn.Linear(self.weather_processor.num_weather_features, self.weather_embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.weather_embed_dim * 2, self.weather_embed_dim)
            )
            # 为每个节点复制天气嵌入
            self.weather_spatial_expand = nn.Linear(self.weather_embed_dim, self.weather_embed_dim)
        
        # 时空 MambaWeather 块
        self._logger.info("Building MCSTWeather model with weather integration")
        
        # 时序处理块
        self.temporal_block = SimpleMambaWeatherBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            weather_embed_dim=self.weather_embed_dim,
            dropout=self.dropout
        )
        
        # 空间处理块
        self.spatial_block = SimpleMambaWeatherBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            weather_embed_dim=self.weather_embed_dim,
            dropout=self.dropout
        )
        
        # 融合模式配置
        self.fusion_mode = config.get('fusion_mode', 'weighted')
        
        # 组合权重（用于 weighted 模式）
        self.combine_weights = nn.Parameter(torch.randn(2, self.d_model))
        
        # 自适应融合模块（用于 adaptive 模式）
        if self.fusion_mode == 'adaptive':
            self.adaptive_fusion = AdaptiveFusion(self.d_model, self.dropout)
        
        # 时序投影：将 input_window 映射到 output_window
        self.temporal_proj = nn.Linear(self.input_window, self.output_window)
        
        # 输出投影
        self.output_proj = nn.Linear(self.d_model, self.output_dim)
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(self.d_model)
        
        # 预计算时间嵌入
        self._precompute_time_embeddings()
        
        # 移动到设备
        self.to(self.device)
        
        self._logger.info(f"MCSTWeather model initialized with weather_embed_dim={self.weather_embed_dim}")
    
    def _setup_weather_processor(self):
        """设置天气数据处理器"""
        # 尝试自动查找天气文件
        if self.weather_file is None:
            # 尝试从数据集路径推断
            dataset_name = self.config.get('dataset', 'METR_LA')
            possible_paths = [
                f'./raw_data/{dataset_name}/weather.csv',
                f'./raw_data/weather/{dataset_name.lower()}_weather.csv',
                f'./raw_data/weather/noaa_weather_5min.csv',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.weather_file = path
                    break
        
        if self.weather_file and os.path.exists(self.weather_file):
            try:
                self.weather_processor = WeatherProcessor(self.weather_file).load_data()
                self.weather_processor.normalize_features()
                self._logger.info(f"Weather data loaded from {self.weather_file}")
                self._logger.info(f"Weather features: {self.weather_processor.feature_names}")
            except Exception as e:
                self._logger.warning(f"Failed to load weather data: {e}")
                self.weather_processor = None
        else:
            self._logger.warning(f"Weather file not found: {self.weather_file}")
            self.weather_processor = None
    
    def _extract_timestamps_from_x(self, x):
        """
        从模型输入x中取出时间步对应数字，生成实际时间戳
        Args:
            x: (B, input_window, num_nodes, feature_dim) 包含时间步列
        Returns:
            list of pd.Timestamp or None
        """
        if not self.has_time_column:
            return None
        
        # 取出最后一个特征列（时间步索引），取第一个节点即可（所有节点相同）
        time_indices = x[:, :, 0, -1].detach().cpu().numpy()  # (B, input_window)
        
        # 如果有外部归一化器，对时间步索引进行反归一化
        ext_scaler = self.data_feature.get('ext_scaler', None)
        if ext_scaler is not None and hasattr(ext_scaler, 'inverse_transform'):
            time_indices_flat = time_indices.reshape(-1, 1)
            time_indices_flat = ext_scaler.inverse_transform(time_indices_flat)
            time_indices = time_indices_flat.reshape(time_indices.shape)
        
        time_indices = np.round(time_indices).astype(np.int64)
        
        base_time = pd.Timestamp(self.start_time)
        interval_minutes = self.time_intervals / 60.0
        
        timestamps = []
        for b in range(time_indices.shape[0]):
            for idx in time_indices[b]:
                timestamps.append(base_time + pd.Timedelta(minutes=int(idx * interval_minutes)))
        
        return timestamps
    
    def _precompute_time_embeddings(self):
        """预计算时间嵌入索引"""
        if self.add_time_in_day:
            tod = torch.linspace(0, 0.99, self.input_window, device=self.device)
            self.register_buffer('tod_indices', (tod * self.steps_per_day).long().clamp_(0, self.steps_per_day - 1))
        else:
            self.tod_indices = None
        
        if self.add_day_in_week:
            dow = torch.arange(0, self.input_window, device=self.device) % 7
            self.register_buffer('dow_indices', dow.long().clamp_(0, 6))
        else:
            self.dow_indices = None
    
    def _get_weather_embedding(self, batch_size, timestamps=None):
        """
        获取天气嵌入
        Args:
            batch_size: batch大小
            timestamps: 可选，具体的时间戳列表
        Returns:
            weather_embed: (batch_size, input_window, num_nodes, weather_embed_dim)
        """
        if not self.use_weather or self.weather_processor is None:
            # 返回零向量
            return torch.zeros(batch_size, self.input_window, self.num_nodes, self.weather_embed_dim, device=self.device)
        
        # 如果没有提供timestamps，生成默认时间序列（基于input_window）
        if timestamps is None:
            # 生成虚拟时间戳（这里简化处理，实际应从batch中获取）
            # 使用时间索引作为替代
            time_indices = torch.arange(self.input_window, device=self.device)
            # 将索引转换为实际时间戳（假设5分钟间隔）
            base_time = pd.Timestamp('2012-03-01')  # METR_LA默认起始时间
            timestamps = [base_time + pd.Timedelta(minutes=int(i * 5)) for i in time_indices.cpu().numpy()]
        
        # 获取天气特征
        weather_features = self.weather_processor.get_features_at_timestamps(timestamps)
        weather_tensor = torch.tensor(weather_features, dtype=torch.float32, device=self.device)
        
        # 投影到嵌入维度
        weather_embed = self.weather_feature_proj(weather_tensor)  # (num_timestamps, weather_embed_dim)
        
        # 根据timestamps维度reshape并扩展到batch和nodes维度
        if weather_embed.shape[0] == batch_size * self.input_window:
            weather_embed = weather_embed.reshape(batch_size, self.input_window, self.weather_embed_dim)
        elif weather_embed.shape[0] == self.input_window:
            weather_embed = weather_embed.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            raise ValueError(f"Unexpected weather_embed shape after projection: {weather_embed.shape}, "
                             f"expected ({batch_size * self.input_window}, {self.weather_embed_dim}) or "
                             f"({self.input_window}, {self.weather_embed_dim})")
        weather_embed = weather_embed.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)  # (B, L, N, D_w)
        
        # 空间扩展（为每个节点学习不同的天气影响）
        B, L, N, D = weather_embed.shape
        weather_embed = weather_embed.reshape(B * L * N, D)
        
        #不一定有用，可以通过实验验证效果。
        weather_embed = self.weather_spatial_expand(weather_embed)
        weather_embed = weather_embed.reshape(B, L, N, -1)
        
        return weather_embed
    
    def forward(self, batch):
        """
        前向传播
        Args:
            batch: Batch对象，包含 'X' 和其他特征
        Returns:
            预测结果: (batch_size, output_window, num_nodes, output_dim)
        """
        # 获取输入
        x = batch['X'].to(self.device)  # (B, input_window, num_nodes, feature_dim)
        batch_size = x.shape[0]
        
        # 从x中提取时间戳（最后一列为时间步索引）
        timestamps = self._extract_timestamps_from_x(x)
        
        # 去掉时间步列，恢复原始特征维度
        if self.has_time_column:
            x = x[..., :-1]
        
        # 特征提取
        features = []
        
        # 主特征投影
        x_main = self.input_proj(x)  # (B, L, N, input_embedding_dim)
        features.append(x_main)
        
        # 时间嵌入
        if self.add_time_in_day and self.tod_indices is not None:
            tod_indices = self.tod_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_nodes)
            tod_emb = self.tod_embedding(tod_indices)
            features.append(tod_emb)
        
        if self.add_day_in_week and self.dow_indices is not None:
            dow_indices = self.dow_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_nodes)
            dow_emb = self.dow_embedding(dow_indices)
            features.append(dow_emb)
        
        # 空间嵌入
        spatial_emb = self.spatial_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, self.input_window, -1, -1)
        features.append(spatial_emb)
        
        # 自适应嵌入
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1)
            features.append(adp_emb)
        
        # 拼接所有特征
        x = torch.cat(features, dim=-1)  # (B, L, N, model_dim)
        
        # 投影到 Mamba 维度
        x = self.mamba_input_proj(x)  # (B, L, N, d_model)
        
        # 获取天气嵌入 (B, L, N, weather_embed_dim)
        weather_embed = self._get_weather_embedding(batch_size, timestamps)
        
        # 时序处理：每个节点独立处理
        # 重塑为 (N, B*L, d_model) 以便高效处理
        x_temporal = x.permute(2, 0, 1, 3).reshape(self.num_nodes, batch_size * self.input_window, self.d_model)
        
        # 天气嵌入也需要相应重塑
        w_temporal = weather_embed.permute(2, 0, 1, 3).reshape(self.num_nodes, batch_size * self.input_window, self.weather_embed_dim)
        
        # 通过时序块
        x_temporal = self.temporal_block(x_temporal, w_temporal)
        
        # 重塑回 (B, L, N, d_model)
        x_temporal = x_temporal.reshape(self.num_nodes, batch_size, self.input_window, self.d_model).permute(1, 2, 0, 3)
        
        # 空间处理
        x_spatial_input = x_temporal.reshape(batch_size * self.input_window, self.num_nodes, self.d_model)
        w_spatial = weather_embed.reshape(batch_size * self.input_window, self.num_nodes, self.weather_embed_dim)
        
        x_spatial = self.spatial_block(x_spatial_input, w_spatial)
        
        # 重塑回 (B, L, N, d_model)
        x_spatial = x_spatial.reshape(batch_size, self.input_window, self.num_nodes, self.d_model)
        
        # 组合时序和空间输出
        if self.fusion_mode == 'adaptive':
            x_combined = self.adaptive_fusion(x, x_temporal, x_spatial)
        else:
            x_combined = x_temporal * self.combine_weights[0] + x_spatial * self.combine_weights[1]
        
        # 最终处理和输出投影
        x_out = self.final_layer_norm(x_combined)  # (B, input_window, N, d_model)
        x_out = self.output_proj(x_out)  # (B, input_window, N, output_dim)
        
        # 时序投影：将 input_window 映射到 output_window
        # 转置为 (B, N, output_dim, input_window) -> 投影 -> 转置回 (B, output_window, N, output_dim)
        x_out = x_out.permute(0, 2, 3, 1)  # (B, N, output_dim, input_window)
        x_out = self.temporal_proj(x_out)  # (B, N, output_dim, output_window)
        x_out = x_out.permute(0, 3, 1, 2)  # (B, output_window, N, output_dim)
        
        return x_out  # 直接预测未来 output_window 步
    
    def calculate_loss(self, batch):
        """计算训练损失"""
        y_true = batch['y']
        if hasattr(y_true, 'to'):
            y_true = y_true.to(self.device)
        y_predicted = self.predict(batch)
        
        # 反归一化
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        
        # 计算 masked MAE 损失
        return loss.masked_mae_torch(y_predicted, y_true, 0)
    
    def predict(self, batch):
        """
        直接预测未来 output_window 步
        Args:
            batch: 包含 'X' 的 Batch 对象
        Returns:
            预测结果: (batch_size, output_window, num_nodes, output_dim)
        """
        return self.forward(batch)  # forward 直接输出未来预测
