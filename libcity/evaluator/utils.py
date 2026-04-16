import json
from heapq import nlargest
import pandas as pd
from libcity.model.loss import *


def output(method, value, field):
    """
    Args:
        method: 评估方法
        value: 对应评估方法的评估结果值
        field: 评估的范围, 对一条轨迹或是整个模型
    """
    if method == 'ACC':
        if field == 'model':
            print('---- 该模型在 {} 评估方法下 avg_acc={:.3f} ----'.format(method,
                                                                  value))
        else:
            print('{} avg_acc={:.3f}'.format(method, value))
    elif method in ['MSE', 'RMSE', 'MAE', 'MAPE', 'MARE', 'SMAPE']:
        if field == 'model':
            print('---- 该模型在 {} 评估方法下 avg_loss={:.3f} ----'.format(method,
                                                                   value))
        else:
            print('{} avg_loss={:.3f}'.format(method, value))
    else:
        if field == 'model':
            print('---- 该模型在 {} 评估方法下 avg_acc={:.3f} ----'.format(method,
                                                                  value))
        else:
            print('{} avg_acc={:.3f}'.format(method, value))


def transfer_data(data, model, maxk):
    """
    Here we transform specific data types to standard input type
    """
    if type(data) == str:
        data = json.loads(data)
    assert type(data) == dict, "待评估数据的类型/格式不合法"
    if model == 'DeepMove':
        user_idx = data.keys()
        for user_id in user_idx:
            trace_idx = data[user_id].keys()
            for trace_id in trace_idx:
                trace = data[user_id][trace_id]
                loc_pred = trace['loc_pred']
                new_loc_pred = []
                for t_list in loc_pred:
                    new_loc_pred.append(sort_confidence_ids(t_list, maxk))
                data[user_id][trace_id]['loc_pred'] = new_loc_pred
    return data


def sort_confidence_ids(confidence_list, threshold):
    """
    Here we convert the prediction results of the DeepMove model
    DeepMove model output: confidence of all locations
    Evaluate model input: location ids based on confidence
    :param threshold: maxK
    :param confidence_list:
    :return: ids_list
    """
    """sorted_list = sorted(confidence_list, reverse=True)
    mark_list = [0 for i in confidence_list]
    ids_list = []
    for item in sorted_list:
        for i in range(len(confidence_list)):
            if confidence_list[i] == item and mark_list[i] == 0:
                mark_list[i] = 1
                ids_list.append(i)
                break
        if len(ids_list) == threshold:
            break
    return ids_list"""
    max_score_with_id = nlargest(
        threshold, enumerate(confidence_list), lambda x: x[1])
    return list(map(lambda x: x[0], max_score_with_id))


def evaluate_model(y_pred, y_true, metrics, mode='single', path='metrics.csv'):
    """
    交通状态预测评估函数
    :param y_pred: (num_samples/batch_size, timeslots, ..., feature_dim)
    :param y_true: (num_samples/batch_size, timeslots, ..., feature_dim)
    :param metrics: 评估指标
    :param mode: 单步or多步平均
    :param path: 保存结果
    :return:
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true.shape is not equal to y_pred.shape")
    len_timeslots = y_true.shape[1]
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.FloatTensor(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.FloatTensor(y_true)
    assert isinstance(y_pred, torch.Tensor)
    assert isinstance(y_true, torch.Tensor)

    # 定义指标函数映射表
    masked_metric_funcs = {
        'masked_MAE': lambda pred, true: masked_mae_torch(pred, true, 0),
        'masked_MSE': lambda pred, true: masked_mse_torch(pred, true, 0),
        'masked_RMSE': lambda pred, true: masked_rmse_torch(pred, true, 0),
        'masked_MAPE': lambda pred, true: masked_mape_torch(pred, true, 0),
    }
    
    normal_metric_funcs = {
        'MAE': lambda pred, true: masked_mae_torch(pred, true),
        'MSE': lambda pred, true: masked_mse_torch(pred, true),
        'RMSE': lambda pred, true: masked_rmse_torch(pred, true),
        'MAPE': lambda pred, true: masked_mape_torch(pred, true),
        'R2': lambda pred, true: r2_score_torch(pred, true),
        'EVAR': lambda pred, true: explained_variance_score_torch(pred, true),
    }
    
    all_metric_funcs = {**masked_metric_funcs, **normal_metric_funcs}

    df = []
    for i in range(1, len_timeslots + 1):
        line = {}
        for metric in metrics:
            if metric not in all_metric_funcs:
                raise ValueError('Error parameter metric={}!'.format(metric))
            
            # 根据模式确定数据切片
            if mode.lower() == 'single':
                pred_slice = y_pred[:, i - 1]
                true_slice = y_true[:, i - 1]
            elif mode.lower() == 'average':
                pred_slice = y_pred[:, :i]
                true_slice = y_true[:, :i]
            else:
                raise ValueError('Error parameter mode={}, please set `single` or `average`.'.format(mode))
            
            # 计算指标值
            result = all_metric_funcs[metric](pred_slice, true_slice)
            line[metric] = float(result)
        df.append(line)
    df = pd.DataFrame(df, columns=metrics)
    df.loc['average'] = df.mean()
    print(df)
    df.to_csv(path)
    return df
