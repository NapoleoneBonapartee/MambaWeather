import os
import time
import numpy as np
import random
import torch
from ray import tune
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator, ensure_dir
from libcity.model import loss
from functools import partial
import matplotlib.pyplot as plt


class TrafficStateExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.data_feature = data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self.exp_id = self.config.get('exp_id', None)
        
        # Check GPU usage if the model has this capability
        if hasattr(self.model, 'check_gpu_usage'):
            self._logger = getLogger()
            self._logger.info("Checking GPU usage...")
            gpu_usage_check = self.model.check_gpu_usage()
            self._logger.info(f"All model components on correct device: {gpu_usage_check}")

        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './libcity/cache/{}/'.format(self.exp_id)
        self.visualization_dir = './libcity/cache/{}/visualization'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)
        ensure_dir(self.visualization_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.epochs = self.config.get('max_epoch', 100)
        self.train_loss = self.config.get('train_loss', 'none')
        self.learner = self.config.get('learner', 'adam')
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.weight_decay = self.config.get('weight_decay', 0)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.lr_decay = self.config.get('lr_decay', False)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        self.hyper_tune = self.config.get('hyper_tune', False)

        self.output_dim = self.config.get('output_dim', 1)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)
        self.loss_func = self._build_train_loss()

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model_with_epoch(self, epoch):
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch

        # 存储训练状态
        if self.lr_scheduler is not None:
            config['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        if hasattr(self, 'best_epoch'):
            config['best_epoch'] = self.best_epoch
        if hasattr(self, 'min_val_loss'):
            config['min_val_loss'] = self.min_val_loss
        if hasattr(self, 'wait'):
            config['wait'] = self.wait
        
        # 存储随机状态
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
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')

        # 加载模型和优化器
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

        # 恢复学习率
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self._logger.info("已恢复学习率调度器状态 (当前学习率: {:.6f})".format(
                self.optimizer.param_groups[0]['lr']))
        # 恢复训练状态
        if 'best_epoch' in checkpoint:
            self.best_epoch = checkpoint['best_epoch']
            self._logger.info("已恢复最佳epoch记录: {}".format(self.best_epoch))
        if 'min_val_loss' in checkpoint:
            self.min_val_loss = checkpoint['min_val_loss']
            self._logger.info("已恢复最小验证损失: {:.4f}".format(self.min_val_loss))
        if 'wait' in checkpoint:
            self.wait = checkpoint['wait']
            self._logger.info("已恢复early stopping等待计数: {}".format(self.wait))
        # 恢复随机状态
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def _build_train_loss(self):
        """
        根据全局参数`train_loss`选择训练过程的loss函数
        如果该参数为none，则需要使用模型自定义的loss函数
        注意，loss函数应该接收`Batch`对象作为输入，返回对应的loss(torch.tensor)
        """
        if self.train_loss.lower() == 'none':
            self._logger.warning('Received none train loss func and will use the loss func defined in the model.')
            return None
        if self.train_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        else:
            self._logger.info('You select `{}` as train loss function.'.format(self.train_loss.lower()))

        def func(batch):
            y_true = batch['y']
            y_predicted = self.model.predict(batch)
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
            if self.train_loss.lower() == 'mae':
                lf = loss.masked_mae_torch
            elif self.train_loss.lower() == 'mse':
                lf = loss.masked_mse_torch
            elif self.train_loss.lower() == 'rmse':
                lf = loss.masked_rmse_torch
            elif self.train_loss.lower() == 'mape':
                lf = loss.masked_mape_torch
            elif self.train_loss.lower() == 'logcosh':
                lf = loss.log_cosh_loss
            elif self.train_loss.lower() == 'huber':
                lf = loss.huber_loss
            elif self.train_loss.lower() == 'quantile':
                lf = loss.quantile_loss
            elif self.train_loss.lower() == 'masked_mae':
                lf = partial(loss.masked_mae_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mse':
                lf = partial(loss.masked_mse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_rmse':
                lf = partial(loss.masked_rmse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mape':
                lf = partial(loss.masked_mape_torch, null_val=0)
            elif self.train_loss.lower() == 'r2':
                lf = loss.r2_score_torch
            elif self.train_loss.lower() == 'evar':
                lf = loss.explained_variance_score_torch
            else:
                lf = loss.masked_mae_torch
            return lf(y_predicted, y_true)
        return func

    def evaluate(self, test_dataloader):
        """
        use model to test data, calculate metrics, and save visualizations & predictions.
        Increases robustness by handling errors in sub-steps like visualization or file saving.

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        y_truths, y_preds = [], []
        prediction_failed_batches = 0
        visualization_failed_batches = 0
        total_batches = len(test_dataloader)

        with torch.no_grad():
            self.model.eval()
            batch_mae_list = []
            batch_rmse_list = []
            batch_mape_list = []
            for batch_idx, batch in enumerate(test_dataloader):
                try:
                    # 1. Predict and Scale - 使用predict()保持与验证一致
                    batch.to_tensor(self.device)
                    output = self.model.predict(batch)  # 修改这里：使用predict而不是forward
                    # Ensure slicing uses self.output_dim consistently
                    y_true_scaled = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                    y_pred_scaled = self._scaler.inverse_transform(output[..., :self.output_dim])
                    
                    # 计算多种损失（不将0视作无效值，使用默认参数）
                    mae_loss = loss.masked_mae_torch(y_pred_scaled, y_true_scaled.clone(), 0)
                    rmse_loss = loss.masked_rmse_torch(y_pred_scaled, y_true_scaled.clone(), 0)
                    mape_loss = loss.masked_mape_torch(y_pred_scaled, y_true_scaled.clone(), 0, eps=1e-1)
                    
                    batch_mae_list.append(mae_loss.item())
                    batch_rmse_list.append(rmse_loss.item())
                    batch_mape_list.append(mape_loss.item())
                    
                    if (batch_idx % self.log_every) == 0:
                        self._logger.info(
                            f"Batch [{batch_idx}/{total_batches}], "
                            f"MAE: {mae_loss.item():.6f}, RMSE: {rmse_loss.item():.6f}, MAPE: {mape_loss.item():.6f}"
                        )

                    # Ensure outputs are numpy for visualization and aggregation
                    y_true_np = y_true_scaled.cpu().numpy()
                    y_pred_np = y_pred_scaled.cpu().numpy()

                except Exception as e:
                    self._logger.error(f"Error during prediction/scaling for batch {batch_idx}: {e}")
                    prediction_failed_batches += 1
                    continue # Skip this batch if prediction/scaling fails

                # 2. Visualize (already has internal try-except from previous edit)
                try:
                    self.visualize_predictions(y_true_np, y_pred_np, batch_idx)
                except Exception as e:
                    # Catch potential unexpected errors during the call itself.
                    self._logger.error(f"Unexpected error calling visualize_predictions for batch {batch_idx}: {e}")
                    visualization_failed_batches += 1
                    # Continue evaluation even if visualization fails

                # 3. Aggregate Results
                y_truths.append(y_true_np)
                y_preds.append(y_pred_np)


        # Log batch-level failures
        if prediction_failed_batches > 0:
            self._logger.warning(f"Prediction/scaling failed for {prediction_failed_batches}/{total_batches} batches.")
        if visualization_failed_batches > 0:
            self._logger.warning(f"Visualization failed or errored for {visualization_failed_batches}/{total_batches} batches.")
        
        # 输出各类型loss的均值
        if batch_mae_list:
            mean_mae = np.mean(batch_mae_list)
            mean_rmse = np.mean(batch_rmse_list)
            mean_mape = np.mean(batch_mape_list)
            self._logger.info(
                f"Evaluation mean loss -> MAE: {mean_mae:.6f}, "
                f"RMSE: {mean_rmse:.6f}, MAPE: {mean_mape:.6f}"
            )

        # Check if any results were collected
        if not y_truths or not y_preds:
            self._logger.error("Evaluation failed: No results collected, possibly due to errors in all batches.")
            return None # Indicate failure

        # 4. Concatenate Results
        try:
            y_preds_concat = np.concatenate(y_preds, axis=0)
            y_truths_concat = np.concatenate(y_truths, axis=0)
        except Exception as e:
            self._logger.error(f"Failed to concatenate evaluation results: {e}")
            return None # Cannot proceed without concatenated results

        # 5. Save Predictions (.npz)
        try:
            outputs = {'prediction': y_preds_concat, 'truth': y_truths_concat}
            filename = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                       + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            filepath = os.path.join(self.evaluate_res_dir, filename)
            np.savez_compressed(filepath, **outputs)
            self._logger.info(f"Saved predictions to {filepath}")
        except Exception as e:
            self._logger.error(f"Failed to save predictions npz file at {filepath}: {e}")
            # Continue to metric calculation even if saving predictions fails

        # 6. Calculate and Save Metrics using Evaluator
        test_result = None
        try:
            self.evaluator.clear()
            # Ensure data passed to evaluator is torch tensor with correct dtype
            y_true_tensor = torch.tensor(y_truths_concat, dtype=torch.float32)
            y_pred_tensor = torch.tensor(y_preds_concat, dtype=torch.float32)
            self.evaluator.collect({'y_true': y_true_tensor, 'y_pred': y_pred_tensor})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            self._logger.info(f"Evaluation metrics calculated and saved to {self.evaluate_res_dir}")
        except Exception as e:
            self._logger.error(f"Failed to calculate or save evaluation metrics: {e}")
            # Return None as the primary goal (metrics) failed.
            return None 

        return test_result

    def visualize_predictions(self, y_true, y_pred, batch_idx):
        """
        Create and save visualizations for predictions vs ground truth for one random sample per batch

        Args:
            y_true (numpy.ndarray): Ground truth values shape (batch_size, timesteps, nodes, metrics)
            y_pred (numpy.ndarray): Predicted values shape (batch_size, timesteps, nodes, metrics)
            batch_idx (int): Batch index for filename
        """
        # Ensure y_true and y_pred are numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        if y_true.shape[0] == 0 or y_true.shape[2] == 0:
             self._logger.warning(f"Skipping visualization for batch {batch_idx}: Empty batch or node dimension.")
             return

        # Select one random sample from the batch
        sample_idx = np.random.randint(0, y_true.shape[0])
        # Select one random node
        node_idx = np.random.randint(0, y_true.shape[2])
        
        num_metrics = y_true.shape[-1] # Get the actual number of metrics
        if num_metrics == 0:
            self._logger.warning(f"Skipping visualization for batch {batch_idx}, sample {sample_idx}, node {node_idx}: No metrics found (output_dim is likely 0).")
            return

        # Create one plot with subplots for each metric
        fig_width = 5 * num_metrics
        plt.figure(figsize=(fig_width, 5))
        
        # Get the data for the selected sample and node
        true_sample = y_true[sample_idx, :, node_idx, :]  # shape (timesteps, metrics)
        pred_sample = y_pred[sample_idx, :, node_idx, :]  # shape (timesteps, metrics)
        
        # Plot each metric
        for metric_idx in range(num_metrics):
            plt.subplot(1, num_metrics, metric_idx + 1)
            # Check if true_sample or pred_sample have the expected dimension
            if true_sample.ndim < 2 or pred_sample.ndim < 2:
                 self._logger.warning(f"Skipping visualization plot for metric {metric_idx} due to unexpected data dimensions. True shape: {true_sample.shape}, Pred shape: {pred_sample.shape}")
                 continue
            if true_sample.shape[1] <= metric_idx or pred_sample.shape[1] <= metric_idx:
                self._logger.warning(f"Skipping visualization plot for metric {metric_idx} due to insufficient columns. True shape: {true_sample.shape}, Pred shape: {pred_sample.shape}")
                continue

            plt.plot(true_sample[:, metric_idx], label='Ground Truth', marker='o')
            plt.plot(pred_sample[:, metric_idx], label='Prediction', marker='x')
            plt.title(f'Metric {metric_idx + 1}\\nNode {node_idx}') # Use generic metric name
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        # Save the figure
        filename = f'batch_{batch_idx}_sample_{sample_idx}_node_{node_idx}.png'
        filepath = os.path.join(self.visualization_dir, filename)
        try:
             plt.savefig(filepath)
        except Exception as e:
             self._logger.error(f"Failed to save visualization figure {filepath}: {e}")
        plt.close() # Close the figure to free memory
        
        if batch_idx % 10 == 0:  # Log every 10 batches
            self._logger.info(f'Saved visualizations for batch {batch_idx}')

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        
        # 修改：使用实例变量而非局部变量，以便在保存checkpoint时访问和恢复
        # 如果是resume加载的模型，会恢复之前保存的best_epoch和min_val_loss
        self.wait = getattr(self, 'wait', 0)
        self.best_epoch = getattr(self, 'best_epoch', 0)
        self.min_val_loss = getattr(self, 'min_val_loss', float('inf'))
        
        # 记录加载时的恢复状态（如果是resume）
        if self._epoch_num > 0:
            self._logger.info(f'Resumed from epoch {self._epoch_num}, '
                              f'best_epoch={self.best_epoch}, min_val_loss={self.min_val_loss:.4f}')

        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.epochs, np.mean(losses), val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune:
                # use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # ray tune use loss to determine which params are best
                tune.report(loss=val_loss)

            if val_loss < self.min_val_loss:
                self.wait = 0
                if self.saved:
                    # 先记录旧的min_val_loss用于日志打印
                    old_min_val_loss = self.min_val_loss
                    # 更新best_epoch和min_val_loss后再保存，确保checkpoint包含正确的状态
                    self.best_epoch = epoch_idx
                    self.min_val_loss = val_loss
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(old_min_val_loss, val_loss, model_file_name))
                else:
                    # 即使不保存模型，也要更新最佳记录
                    self.best_epoch = epoch_idx
                    self.min_val_loss = val_loss
            else:
                self.wait += 1
                if self.wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        
        # 训练结束日志：记录最佳epoch和对应的验证损失
        if self.best_epoch >= 0:
            self._logger.info('Training ended. Best epoch: {}, min val loss: {:.4f}'.
                              format(self.best_epoch, self.min_val_loss))
        
        if self.load_best_epoch:
            # 检查最佳epoch的模型文件是否存在
            best_model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % self.best_epoch
            if os.path.exists(best_model_path):
                self.load_model_with_epoch(self.best_epoch)
            else:
                self._logger.warning(
                    'Best epoch {} model file not found at: {}. '
                    'Current model state remains loaded.'.format(self.best_epoch, best_model_path)
                )
        return self.min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            list: 每个batch的损失的数组
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            float: 评估数据的平均损失值
        """
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss
