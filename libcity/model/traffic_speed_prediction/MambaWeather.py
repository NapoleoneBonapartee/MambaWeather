import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from mamba_ssm import Mamba as MambaSSM



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
        
        # 3. 风速相关特征
        if 'WSF2' in df.columns and 'WSF5' in df.columns:
            # 风速比（短时 gust 与平均风速比）
            derived_features['wind_gust_ratio'] = df['WSF5'] / (df['WSF2'] + 1e-6)
            # 平均风速
            derived_features['wind_avg'] = (df['WSF2'] + df['WSF5']) / 2
        
        # 4. 气压相关特征
        if 'ADPT' in df.columns and 'ASTP' in df.columns:
            # 站压与海平面气压差
            derived_features['pressure_diff'] = df['ASTP'] - df['ADPT']
        
        if 'ASLP' in df.columns:
            # 气压等级（用于判断天气稳定性）
            derived_features['pressure_level'] = pd.cut(df['ASLP'], bins=[0, 1010, 1020, 1100], labels=[0, 1, 2]).astype(float)
        
        # 5. 降水相关特征
        if 'PRCP' in df.columns:
            # 降水强度等级
            derived_features['precipitation_level'] = pd.cut(df['PRCP'], bins=[-0.1, 0, 2.5, 10, 1000], labels=[0, 1, 2, 3]).astype(float)
        
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
            timestamps: 时间戳列表或数组
        Returns:
            形状为 (len(timestamps), num_weather_features) 的numpy数组
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



class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mamba = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 带残差连接的Mamba块
        residual = x
        x = self.layer_norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual
        
        # 带残差连接的前馈网络
        residual = x
        x = self.ff_norm(x)
        x = self.feed_forward(x)
        x = self.ff_dropout(x)
        x = x + residual
        
        return x


class MambaWeather(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # 保存 config 供后续使用
        self.config = config

        # 首先获取数据特征以确保num_nodes已定义
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # 获取模型配置
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self._logger = getLogger()
        
        # 添加时间处理参数
        self.add_time_in_day = config.get("add_time_in_day", False)
        self.add_day_in_week = config.get("add_day_in_week", False)
        self.steps_per_day = config.get("steps_per_day", 288)
        
        # 从配置中获取嵌入维度
        self.input_embedding_dim = config.get('input_embedding_dim', 24)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 24) if self.add_time_in_day else 0
        self.dow_embedding_dim = config.get('dow_embedding_dim', 24) if self.add_day_in_week else 0
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 16)
        self.adaptive_embedding_dim = config.get('adaptive_embedding_dim', 80)

        # 从配置中获取Mamba特定参数
        self.d_model = config.get('d_model', 96)
        self.d_state = config.get('d_state', 32)
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)
        self.dropout = config.get('dropout', 0.1)

        # 时间相关特征（从dataset传入）
        self.start_time = self.data_feature.get('start_time', None)
        self.total_time_steps = self.data_feature.get('total_time_steps', None)
        self.time_intervals = self.config.get('time_intervals', 300)
        self.has_time_column = self.start_time is not None and self.total_time_steps is not None
        if self.has_time_column:
            # 输入特征中包含节点索引、时间步索引、时间嵌入和星期嵌入，需要去掉后再投影
            self.feature_dim = self.feature_dim - 4
        else:
            # 输入特征中包含节点索引、时间嵌入和星期嵌入，需要去掉后再投影
            self.feature_dim = self.feature_dim - 3

        # 天气相关配置
        self.weather_embed_dim = config.get('weather_embed_dim', 64)
        self.weather_file = config.get('weather_file', None)  # 天气文件路径
        self.use_weather = config.get('use_weather', False)

        # 创建天气处理器
        self.weather_processor = None
        if self.use_weather:
            self._setup_weather_processor()
        
        # 天气特征嵌入层
        if self.use_weather and self.weather_processor is not None:
            self.weather_feature_proj = nn.Sequential(
                nn.Linear(self.weather_processor.num_weather_features, self.weather_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
        
        # 计算模型维度（总嵌入大小）
        self.model_dim = (
            self.input_embedding_dim +
            self.tod_embedding_dim +
            self.dow_embedding_dim +
            self.adaptive_embedding_dim
        )

        if self.use_weather:
            self.model_dim += self.weather_embed_dim
        
        # 创建嵌入层
        self.input_proj = nn.Linear(self.feature_dim, self.input_embedding_dim)
        
        if self.add_time_in_day:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.add_day_in_week:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        
        # # 初始化节点身份嵌入（可学习）
        # self.node_identity_embedding = nn.Embedding(self.num_nodes, self.spatial_embedding_dim)
        
        # 初始化自适应嵌入
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(self.input_window, self.num_nodes, self.adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)
        
        # 输入投影到Mamba维度
        self.mamba_input_proj = nn.Linear(self.model_dim, self.d_model)
        
        # 仅使用两个Mamba块 - 一个用于时间处理，一个用于空间处理
        self._logger.info("构建简化的MCSTMamba模型，仅包含两个Mamba块")

        # 时间处理块
        self.temporal_block = SimpleMambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout
        )

        # 时间双向融合模块（将前向+后向Mamba输出融合回d_model维度）
        self.temporal_fusion = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 空间处理块
        self.spatial_block = SimpleMambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout
        )
        
        # 组合权重（动态门控或静态权重）
        if self.use_weather:
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.Sigmoid()
            )
        else:
            # 保留原版静态权重，供不使用天气时回退
            self.combine_weights = nn.Parameter(torch.randn(2, self.d_model))

        self.combine_weights = nn.Parameter(torch.randn(2, self.d_model))        
        
        # 输出投影层
        self.output_proj = nn.Linear(self.d_model, self.output_dim)
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(self.d_model)
        
        # Step 2 改进：条件天气偏置层
        if self.use_weather:
            self.x_to_weather_coeff_t = nn.Linear(self.d_model, self.weather_embed_dim)
            self.x_to_weather_coeff_s = nn.Linear(self.d_model, self.weather_embed_dim)
            self.weather_bias_proj_t = nn.Linear(self.weather_embed_dim, self.d_model)
            self.weather_bias_proj_s = nn.Linear(self.weather_embed_dim, self.d_model)
            self.add_gate_t = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.Sigmoid()
            )
            self.add_gate_s = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.Sigmoid()
            )

        # 记录正在使用的设备
        self._logger.info(f"简化的MCSTMamba模型配置用于设备: {self.device}")

        # 将模型移动到设备
        self.to(self.device)


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
        time_indices = x[:, :, 0, -3].detach().cpu().numpy()  # (B, input_window)
        
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
        
        if timestamps is None:
            print("缺少时间序列timestamps")
        
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
        
        return weather_embed


    def forward(self, batch):
        # 如有必要，将输入移动到设备
        x = batch['X'].to(self.device)  # [batch_size, input_window, num_nodes, feature_dim]
        batch_size = x.shape[0]
        
        # 特征提取
        features = []

        # 从x中提取时间戳（最后一列为时间步索引）
        if self.use_weather:
            timestamps = self._extract_timestamps_from_x(x)
        
        if self.has_time_column:
            temp = x[...,-4:]  # [node_index, time_index, tod, dow]
            x = x[...,:-4]
        else:
            temp = x[...,-3:]  # [node_index, tod, dow]
            x = x[...,:-3]
            
        # 处理主要特征
        x_main = self.input_proj(x)  # [batch_size, input_window, num_nodes, input_embedding_dim]
        features.append(x_main)
        
        # 如需要，添加时间嵌入
        if self.add_time_in_day:
            tod_indices = temp[..., -2].long()
            tod_emb = self.tod_embedding(tod_indices)  # [batch_size, input_window, num_nodes, tod_embedding_dim]
            features.append(tod_emb)
            
        if self.add_day_in_week:
            # 基于序列位置创建一周中的天（0-6）
            dow_indices = temp[..., -1].long()
            dow_emb = self.dow_embedding(dow_indices)  # [batch_size, input_window, num_nodes, dow_embedding_dim]
            features.append(dow_emb)
        
        # 如启用，添加自适应嵌入
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.unsqueeze(0)  # [1, input_window, num_nodes, adaptive_dim]
            adp_emb = adp_emb.expand(batch_size, -1, -1, -1)
            features.append(adp_emb)

        weather_embed = None
        if self.use_weather:
            weather_embed = self._get_weather_embedding(batch_size, timestamps)
            features.append(weather_embed)
        
        # 连接所有特征
        x = torch.cat(features, dim=-1)  # [batch_size, input_window, num_nodes, model_dim]
        
        # 投影到Mamba维度
        x = self.mamba_input_proj(x)  # [batch_size, input_window, num_nodes, d_model]

        x_for_temporal = x
        x_for_spatial = x
        
        # 时间处理（独立处理每个节点）- 双向Mamba
        # Forward Mamba
        x_temporal_fw = x_for_temporal.permute(2, 0, 1, 3)  # [num_nodes, batch_size, input_window, d_model]
        x_temporal_fw = x_temporal_fw.reshape(self.num_nodes, batch_size * self.input_window, self.d_model)
        x_temporal_fw = self.temporal_block(x_temporal_fw)
        x_temporal_fw = x_temporal_fw.reshape(self.num_nodes, batch_size, self.input_window, self.d_model)
        
        # Backward Mamba（时间反向）
        x_temporal_bw = x_for_temporal.flip(dims=[1])  # [batch_size, input_window, num_nodes, d_model] 反向时间
        x_temporal_bw = x_temporal_bw.permute(2, 0, 1, 3)  # [num_nodes, batch_size, input_window, d_model]
        x_temporal_bw = x_temporal_bw.reshape(self.num_nodes, batch_size * self.input_window, self.d_model)
        x_temporal_bw = self.temporal_block(x_temporal_bw)
        x_temporal_bw = x_temporal_bw.reshape(self.num_nodes, batch_size, self.input_window, self.d_model)
        x_temporal_bw = x_temporal_bw.flip(dims=[2])  # 将时间维度翻转回原始顺序
        
        # 拼接前向和后向，融合回原始维度
        x_temporal = torch.cat([x_temporal_fw, x_temporal_bw], dim=-1)  # [num_nodes, batch_size, input_window, 2*d_model]
        x_temporal = self.temporal_fusion(x_temporal)  # [num_nodes, batch_size, input_window, d_model]
        
        # 空间处理 
        is_large_dataset = self.num_nodes > 300
        
        if is_large_dataset and batch_size > 1:
            # 对于大型数据集，使用GPU高效分块优化
            # 直接在GPU上初始化输出张量
            x_spatial = torch.zeros(self.input_window, batch_size, self.num_nodes, self.d_model, 
                                    device=self.device)
            
            # 处理每个时间步
            for t in range(self.input_window):
                # 对于每个时间步，将节点视为序列: [batch_size, num_nodes, d_model]
                nodes_seq = x_for_spatial[:, t, :, :]
                
                # 计算有效批次大小
                effective_batch_size = 1
                
                # if t == 0:  # 仅记录一次
                #     self._logger.info(f"空间处理使用有效批次大小 {effective_batch_size} "
                #                    f"(数据集有 {self.num_nodes} 个节点)")
                
                # 处理批次
                all_results = []
                for b_idx in range(0, batch_size, effective_batch_size):
                    end_idx = min(b_idx + effective_batch_size, batch_size)
                    # 提取批次切片: [small_batch, num_nodes, d_model]
                    batch_slice = nodes_seq[b_idx:end_idx]
                    
                    # 通过空间块处理
                    spatial_hidden = self.spatial_block(batch_slice)
                    
                    all_results.append(spatial_hidden)
                
                # 合并结果
                x_spatial[t] = torch.cat(all_results, dim=0)
        else:
            # 对于较小的数据集，一次性处理所有数据
            x_spatial = x_for_spatial.permute(1, 0, 2, 3)  # [input_window, batch_size, num_nodes, d_model]
            x_spatial = x_spatial.reshape(self.input_window, batch_size * self.num_nodes, -1)
            
            # 通过空间块处理
            x_spatial = self.spatial_block(x_spatial)
                
            # 重塑回原形状
            x_spatial = x_spatial.reshape(self.input_window, batch_size, self.num_nodes, self.d_model)
        
        # 组合时间和空间输出
        x_t = x_temporal.permute(1, 2, 0, 3)   # (B, L, N, d_model)
        x_s = x_spatial.permute(1, 0, 2, 3)    # (B, L, N, d_model)
        
        #门控融合
        if self.use_weather and weather_embed is not None:
            # 拼接时序输出、空间输出、天气上下文
            gate_input = torch.cat([x, weather_embed], dim=-1)
            gate = self.fusion_gate(gate_input)  # (B, L, N, d_model), 每个维度独立门控
            
            # gate -> 1 偏好时序，gate -> 0 偏好空间
            x_combined = gate * x_t + (1.0 - gate) * x_s
        else:
            # 回退到静态加权（与原版MCST一致）
            x_combined = x_t * self.combine_weights[0] + x_s * self.combine_weights[1]

        # 最终处理和输出投影
        x_out = self.final_layer_norm(x_combined)
        x_out = self.output_proj(x_out)
        
        return x_out[:, -self.output_window:]  # 返回最后output_window步


    def calculate_loss(self, batch):
        """
        计算一批数据的训练损失
        :param batch: 输入数据字典
        :return: 训练损失（张量）
        """
        y_true = batch['y'].to(self.device)
        y_predicted = self.predict(batch)
        
        # 应用逆归一化
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        
        # 计算掩码MAE损失
        return loss.masked_mae_torch(y_predicted, y_true, 0)


    def predict(self, batch):
        """
        对一批数据进行预测
        :param batch: 输入数据字典
        :return: 预测结果，形状为 [batch_size, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)

