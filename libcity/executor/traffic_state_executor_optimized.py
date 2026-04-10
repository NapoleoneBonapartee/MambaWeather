import os
import time
import numpy as np
import random
import torch
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.utils import get_evaluator, ensure_dir
from libcity.model import loss
from functools import partial


def get_autocast_context(fp16_enabled=True):
    """Get autocast context manager for mixed precision training"""
    if fp16_enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    else:
        # Return a dummy context manager that does nothing
        from contextlib import nullcontext
        return nullcontext()


class TrafficStateExecutorOptimized(TrafficStateExecutor):
    """Optimized TrafficStateExecutor with gradient accumulation and mixed precision"""
    
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        
        # Gradient accumulation steps
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self._logger.info(f'Gradient accumulation steps: {self.gradient_accumulation_steps}')
        
        # Mixed precision training
        self.fp16 = config.get('fp16', False) and torch.cuda.is_available()
        self._logger.info(f'FP16 mixed precision training: {self.fp16}')
        
        # Initialize gradient scaler for FP16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
            self._logger.info('Using torch.cuda.amp.GradScaler for FP16 training')
        else:
            self.scaler = None
        
        # Enable cuDNN benchmarking for better performance with fixed input sizes
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self._logger.info('Enabled cuDNN benchmark mode')
            
            # Enable TF32 for Ampere GPUs (RTX 30 series and above)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self._logger.info('Enabled TF32 for faster training on Ampere GPUs')
        
        # ========== PEMSD4 小数据集 GPU 预加载缓存 ==========
        self.gpu_train_data = None  # 预加载的训练数据列表
        self.gpu_valid_data = None  # 预加载的验证数据列表
        self._logger.info("Executor 已初始化 GPU 数据预加载功能（PEMSD4 专用）")


    def preload_to_gpu(self, dataloader, data_type='train'):
        """
        将 DataLoader 数据全量预加载到 GPU 显存
        适用于 PEMSD4 等小数据集（307节点，11K样本，约120MB显存）
        """
        if data_type == 'train' and self.gpu_train_data is not None:
            return  # 已预加载，跳过
        if data_type == 'valid' and self.gpu_valid_data is not None:
            return
            
        self._logger.info(f"PEMSD4 数据集较小，正在预加载 {data_type} 数据到 GPU 显存...")
        start_time = time.time()
        
        gpu_data = []
        for batch_idx, batch in enumerate(dataloader):
            batch.to_tensor(self.device)
            gpu_data.append(batch)
            if (batch_idx + 1) % 10 == 0:
                self._logger.info(f"  已预加载 {batch_idx + 1} batches...")
        
        elapsed = time.time() - start_time
        self._logger.info(f"预加载完成: {len(gpu_data)} batches ({elapsed:.2f}s)")
        
        if data_type == 'train':
            self.gpu_train_data = gpu_data
        else:
            self.gpu_valid_data = gpu_data


    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """
        Optimized training epoch with gradient accumulation and mixed precision
        针对 PEMSD4：首次调用时预加载全量数据到 GPU，后续 epoch 零延迟读取
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        
        # ========== PEMSD4 懒加载逻辑 ==========
        if self.gpu_train_data is None:
            self.preload_to_gpu(train_dataloader, 'train')
        
        # 使用预加载的 GPU 数据（零拷贝，无需 DataLoader）
        data_source = self.gpu_train_data if self.gpu_train_data is not None else train_dataloader
        
        # Zero gradients at the beginning
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(data_source):
            # 预加载数据已在 GPU，跳过 to_tensor；否则需要传输
            if self.gpu_train_data is None:
                with get_autocast_context(self.fp16):
                    batch.to_tensor(self.device)
            
            # Determine if we should update weights
            is_update_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
            
            # Use autocast for mixed precision
            with get_autocast_context(self.fp16):
                # batch.to_tensor(self.device)  # 已注释：预加载数据已在 GPU
                loss = loss_func(batch)
                
                # Normalize loss by accumulation steps
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling if FP16 is enabled
            if self.fp16 and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Record the original loss value (before normalization)
            loss_item = loss.item() * self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else loss.item()
            losses.append(loss_item)
            
            # Update weights only after accumulating gradients
            if is_update_step or (batch_idx + 1) == len(train_dataloader):
                if self.fp16 and self.scaler is not None:
                    # Unscale gradients for gradient clipping
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular gradient clipping and optimizer step
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Zero gradients with set_to_none for better memory efficiency
                self.optimizer.zero_grad(set_to_none=True)
            
            self._logger.debug(loss_item)
        
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        """
        Optimized validation epoch with mixed precision inference
        针对 PEMSD4：预加载验证数据到 GPU
        """
        # 预加载验证数据（首次调用）
        if self.gpu_valid_data is None:
            self.preload_to_gpu(eval_dataloader, 'valid')
            
        data_source = self.gpu_valid_data if self.gpu_valid_data is not None else eval_dataloader
        
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []
            
            for batch in data_source:
                # 预加载数据已在 GPU，跳过 to_tensor；否则需要传输
                if self.gpu_valid_data is None:
                    with get_autocast_context(self.fp16):
                        batch.to_tensor(self.device)
                
                # Use autocast for mixed precision inference
                with get_autocast_context(self.fp16):
                    loss = loss_func(batch)
                
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss

    def save_model_with_epoch(self, epoch):
        """
        Save model with additional scaler state if using FP16
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch

        # Store training state
        if self.lr_scheduler is not None:
            config['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        if hasattr(self, 'best_epoch'):
            config['best_epoch'] = self.best_epoch
        if hasattr(self, 'min_val_loss'):
            config['min_val_loss'] = self.min_val_loss
        if hasattr(self, 'wait'):
            config['wait'] = self.wait
        
        # Store gradient scaler state if using FP16
        fp16 = self.config.get('fp16', False)
        if fp16 and hasattr(self, 'scaler') and self.scaler is not None:
            config['scaler_state_dict'] = self.scaler.state_dict()
        
        # Store random state
        config['torch_rng_state'] = torch.get_rng_state()
        if torch.cuda.is_available():
            config['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        config['numpy_rng_state'] = np.random.get_state()
        config['random_rng_state'] = random.getstate()

        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        Load model with gradient scaler state if using FP16
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')

        # Load model and optimizer
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            # 处理 _orig_mod. 前缀问题（通常由 torch.compile() 导致）
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                # 去除 _orig_mod. 前缀
                if k.startswith('_orig_mod.'):
                    new_key = k[10:]  # len('_orig_mod.') == 10
                else:
                    new_key = k
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict)
            self._logger.info("已自动转换带 _orig_mod. 前缀的 state_dict")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore gradient scaler state if using FP16
        fp16 = self.config.get('fp16', False)
        if fp16 and hasattr(self, 'scaler') and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self._logger.info("已恢复梯度缩放器状态")

        # Restore learning rate
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self._logger.info("已恢复学习率调度器状态 (当前学习率: {:.6f})".format(
                self.optimizer.param_groups[0]['lr']))
        # Restore training state
        if 'best_epoch' in checkpoint:
            self.best_epoch = checkpoint['best_epoch']
            self._logger.info("已恢复最佳epoch记录: {}".format(self.best_epoch))
        if 'min_val_loss' in checkpoint:
            self.min_val_loss = checkpoint['min_val_loss']
            self._logger.info("已恢复最小验证损失: {:.4f}".format(self.min_val_loss))
        if 'wait' in checkpoint:
            self.wait = checkpoint['wait']
            self._logger.info("已恢复early stopping等待计数: {}".format(self.wait))
        # Restore random state
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        
        self._logger.info("Loaded model at {}".format(epoch))
