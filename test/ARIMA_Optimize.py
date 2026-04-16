import json
import warnings
import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from multiprocessing import Pool, cpu_count

# 抑制 statsmodels 的警告信息
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from libcity.evaluator.utils import evaluate_model
from libcity.utils import preprocess_data


config = {
    'model': 'ARIMA',
    'p_range': [0, 4],
    'd_range': [0, 3],
    'q_range': [0, 4],
    'dataset': 'METR_LA',
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'input_window': 12,
    'output_window': 12,
    'metrics': ['masked_MAE']
    # 'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE',
    #             'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']
}


def get_data(dataset):
    # path
    path = 'raw_data/' + dataset + '/'
    config_path = path + 'config.json'
    dyna_path = path + dataset + '.dyna'
    geo_path = path + dataset + '.geo'

    # read config
    with open(config_path, 'r') as f:
        json_obj = json.load(f)
        for key in json_obj:
            if key not in config:
                config[key] = json_obj[key]

    # read geo
    geo_file = pd.read_csv(geo_path)
    geo_ids = list(geo_file['geo_id'])

    # read dyna
    dyna_file = pd.read_csv(dyna_path)
    data_col = config.get('data_col', '')
    if data_col != '':  # 根据指定的列加载数据集
        if isinstance(data_col, list):
            data_col = data_col.copy()
        else:  # str
            data_col = [data_col].copy()
        data_col.insert(0, 'time')
        data_col.insert(1, 'entity_id')
        dyna_file = dyna_file[data_col]
    else:  # 不指定则加载所有列
        dyna_file = dyna_file[dyna_file.columns[2:]]  # 从time列开始所有列

    # 求时间序列
    time_slots = list(dyna_file['time'][:int(dyna_file.shape[0] / len(geo_ids))])

    # 转3-d数组
    feature_dim = len(dyna_file.columns) - 2
    df = dyna_file[dyna_file.columns[-feature_dim:]]
    len_time = len(time_slots)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i + len_time].values)
    data = np.array(data, dtype=float)  # (N, T, F)
    data = data.swapaxes(0, 1)  # (T, N, F)
    return data


# ============== 多进程 Worker 函数 ==============

def _order_select_worker(args):
    """在训练序列上搜索最佳 (p,d,q)"""
    seq_train, p_range, d_range, q_range = args
    res = ARIMA(seq_train, order=(0, 0, 0)).fit()
    bic = res.bic
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        warnings.simplefilter("error", category=RuntimeWarning)
        for p in range(p_range[0], p_range[1]):
            for d in range(d_range[0], d_range[1]):
                for q in range(q_range[0], q_range[1]):
                    if p + d + q > 6:  # 限制复杂度，避免高阶模型拟合过慢
                        continue
                    try:
                        cur_res = ARIMA(seq_train, order=(p, d, q)).fit()
                    except:
                        continue
                    if cur_res.bic < bic:
                        bic = cur_res.bic
                        res = cur_res
    return res.specification['order']


def _forecast_worker(args):
    """对单个 test sample 进行所有节点/特征的 ARIMA 预测"""
    test_seq, best_orders, output_window = args
    n, f = best_orders.shape
    y_pred_sample = np.zeros((output_window, n, f))
    for node in range(n):
        for feat in range(f):
            seq = test_seq[:, node, feat]
            order = best_orders[node, feat]
            try:
                model = ARIMA(seq, order=order).fit()
                pred = model.forecast(steps=output_window)
            except:
                try:
                    model = ARIMA(seq, order=(0, 0, 0)).fit()
                    pred = model.forecast(steps=output_window)
                except:
                    pred = np.ones(output_window) * np.mean(seq)
            y_pred_sample[:, node, feat] = pred
    return y_pred_sample


# ============== 主预测流程 ==============

def arima_parallel(train_data, testx, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    output_window = config.get('output_window', 3)
    t_train, n, f = train_data.shape
    p_range = config.get('p_range', [0, 4])
    d_range = config.get('d_range', [0, 3])
    q_range = config.get('q_range', [0, 4])

    # Step 1: 并行定阶（每个节点/特征一次）
    order_tasks = [
        (train_data[:, node, feat], p_range, d_range, q_range)
        for node in range(n) for feat in range(f)
    ]

    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_order_select_worker, order_tasks),
            total=len(order_tasks),
            desc='Order selection'
        ))

    best_orders = np.array(results, dtype=object).reshape(n, f)

    # Step 2: 并行预测（每个 test sample 一次）
    forecast_tasks = [
        (testx[i], best_orders, output_window)
        for i in range(testx.shape[0])
    ]

    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_forecast_worker, forecast_tasks),
            total=len(forecast_tasks),
            desc='Forecasting'
        ))

    y_pred = np.stack(results, axis=0)  # (test_size, output_window, n, f)
    return y_pred


def main():
    print(config)
    data = get_data(config.get('dataset', ''))
    train_rate = config.get('train_rate', 0.7)
    eval_rate = config.get('eval_rate', 0.1)
    train_size = int(len(data) * (train_rate + eval_rate))
    train_data = data[:train_size]  # (T_train, N, F)

    _, _, testx, testy = preprocess_data(data, config)
    y_pred = arima_parallel(train_data, testx, n_jobs=None)
    evaluate_model(y_pred=y_pred, y_true=testy, metrics=config['metrics'],
                   path=config['model'] + '_' + config['dataset'] + '_metrics.csv')


if __name__ == '__main__':
    main()