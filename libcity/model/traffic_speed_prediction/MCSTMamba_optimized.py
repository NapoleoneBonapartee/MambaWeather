import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

from mamba_ssm import Mamba as MambaSSM


class SimpleMambaBlock(nn.Module):
    """Simple Mamba block with layer normalization and residual connections"""
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
        # Mamba block with residual
        residual = x
        x = self.layer_norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward with residual
        residual = x
        x = self.ff_norm(x)
        x = self.feed_forward(x)
        x = self.ff_dropout(x)
        x = x + residual
        
        return x


class MCSTMamba_optimized(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # Get data features first to ensure num_nodes is defined
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # Get model config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self._logger = getLogger()
        
        # Add time handling parameters
        self.add_time_in_day = config.get("add_time_in_day", False)
        self.add_day_in_week = config.get("add_day_in_week", False)
        self.steps_per_day = config.get("steps_per_day", 288)
        
        # Get embedding dimensions from config
        self.input_embedding_dim = config.get('input_embedding_dim', 24)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 24) if self.add_time_in_day else 0
        self.dow_embedding_dim = config.get('dow_embedding_dim', 24) if self.add_day_in_week else 0
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 16)
        self.adaptive_embedding_dim = config.get('adaptive_embedding_dim', 80)
        
        # Calculate model dimension (total embedding size)
        self.model_dim = (
            self.input_embedding_dim +
            self.tod_embedding_dim +
            self.dow_embedding_dim +
            self.spatial_embedding_dim +
            self.adaptive_embedding_dim
        )
        
        # Create embeddings
        self.input_proj = nn.Linear(self.feature_dim, self.input_embedding_dim)
        
        if self.add_time_in_day:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.add_day_in_week:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        
        # Initialize spatial embedding
        self.spatial_embedding = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
        nn.init.xavier_uniform_(self.spatial_embedding)
        
        # Initialize adaptive embedding
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(self.input_window, self.num_nodes, self.adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)
        
        # Get Mamba-specific parameters from config
        self.d_model = config.get('d_model', 96)
        self.d_state = config.get('d_state', 32)
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # Input projection to Mamba dimension
        self.mamba_input_proj = nn.Linear(self.model_dim, self.d_model)
        
        # Just two Mamba blocks - one for temporal and one for spatial
        self._logger.info("Building simplified MCSTMamba model with just two Mamba blocks")

        # Temporal processing block
        self.temporal_block = SimpleMambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout
        )
        
        # Spatial processing block
        self.spatial_block = SimpleMambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout
        )
        
        # Combination weights
        self.combine_weights = nn.Parameter(torch.randn(2, self.d_model))
        
        # Output projection layer
        self.output_proj = nn.Linear(self.d_model, self.output_dim)
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(self.d_model)
        
        # Pre-compute time embeddings to avoid repeated computation
        self._precompute_time_embeddings()
        
        # Log the device being used
        self._logger.info(f"Simplified MCSTMamba model configured for device: {self.device}")

        # Move model to device
        self.to(self.device)

    def _precompute_time_embeddings(self):
        """Pre-compute time embeddings to avoid repeated computation in forward"""
        if self.add_time_in_day:
            # Pre-compute time of day indices
            tod = torch.linspace(0, 0.99, self.input_window, device=self.device)
            self.register_buffer('tod_indices', (tod * self.steps_per_day).long().clamp_(0, self.steps_per_day - 1))
        else:
            self.tod_indices = None
            
        if self.add_day_in_week:
            # Pre-compute day of week indices
            dow = torch.arange(0, self.input_window, device=self.device) % 7
            self.register_buffer('dow_indices', dow.long().clamp_(0, 6))
        else:
            self.dow_indices = None

    def forward(self, batch):
        # Move input to device if needed
        x = batch['X'].to(self.device)  # [batch_size, input_window, num_nodes, feature_dim]
        batch_size = x.shape[0]
        
        # Feature extraction
        features = []
        
        # Process main features
        x_main = self.input_proj(x)  # [batch_size, input_window, num_nodes, input_embedding_dim]
        features.append(x_main)
        
        # Add time embeddings using pre-computed indices
        if self.add_time_in_day and self.tod_indices is not None:
            # Expand pre-computed indices to batch size
            tod_indices = self.tod_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_nodes)
            tod_emb = self.tod_embedding(tod_indices)  # [batch_size, input_window, num_nodes, tod_embedding_dim]
            features.append(tod_emb)
            
        if self.add_day_in_week and self.dow_indices is not None:
            # Expand pre-computed indices to batch size
            dow_indices = self.dow_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_nodes)
            dow_emb = self.dow_embedding(dow_indices)  # [batch_size, input_window, num_nodes, dow_embedding_dim]
            features.append(dow_emb)
        
        # Add spatial embeddings - expand once
        spatial_emb = self.spatial_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, self.input_window, -1, -1)
        features.append(spatial_emb)
        
        # Add adaptive embeddings if enabled
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1)
            features.append(adp_emb)
        
        # Concatenate all features
        x = torch.cat(features, dim=-1)  # [batch_size, input_window, num_nodes, model_dim]
        
        # Project to Mamba dimension
        x = self.mamba_input_proj(x)  # [batch_size, input_window, num_nodes, d_model]
        
        # Temporal processing (process each node independently)
        # Reshape to [num_nodes, batch_size * input_window, d_model] for efficient processing
        x_temporal = x.permute(2, 0, 1, 3).reshape(self.num_nodes, batch_size * self.input_window, self.d_model)
        
        # Process through temporal block
        x_temporal = self.temporal_block(x_temporal)
            
        # Reshape back to [batch_size, input_window, num_nodes, d_model]
        x_temporal = x_temporal.reshape(self.num_nodes, batch_size, self.input_window, self.d_model).permute(1, 2, 0, 3)
        
        # Spatial processing - optimized version
        # Reshape to process all timesteps and batches together: [batch_size * input_window, num_nodes, d_model]
        x_spatial_input = x_temporal.reshape(batch_size * self.input_window, self.num_nodes, self.d_model)
        
        # Process through spatial block in one go
        x_spatial = self.spatial_block(x_spatial_input)
            
        # Reshape back to [batch_size, input_window, num_nodes, d_model]
        x_spatial = x_spatial.reshape(batch_size, self.input_window, self.num_nodes, self.d_model)
        
        # Combine temporal and spatial outputs
        x_combined = x_temporal * self.combine_weights[0] + x_spatial * self.combine_weights[1]
        
        # Final processing and output projection
        x_out = self.final_layer_norm(x_combined)
        x_out = self.output_proj(x_out)
        
        return x_out[:, -self.output_window:]  # Return last output_window steps

    def calculate_loss(self, batch):
        """
        Calculate the training loss for a batch of data
        :param batch: Input data dictionary
        :return: Training loss (tensor)
        """
        y_true = batch['y'].to(self.device)
        y_predicted = self.predict(batch)
        
        # Apply inverse normalization
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        
        # Calculate masked MAE loss
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        """
        Make predictions for a batch of data
        :param batch: Input data dictionary
        :return: Predictions with shape [batch_size, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)
