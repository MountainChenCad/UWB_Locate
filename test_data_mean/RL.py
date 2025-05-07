# -*- coding: utf-8 -*-
"""
Created on Tue May  6 12:28:12 2025

@author: 61413
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "24"
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import time
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simpson
from ast import literal_eval

component = 3
# k_means 聚类拿到初始参数
def k_means(kmeans_data):
    # K-means 聚类
    kmeans = KMeans(n_clusters=component, n_init=10, random_state=42)
    kmeans.fit(kmeans_data)
    labels = kmeans.labels_

    # 计算比例
    alphas = np.bincount(labels) / len(labels)

    # 均值（直接从聚类中心获取）
    mus = kmeans.cluster_centers_
    # 计算协方差矩阵
    sigmas = np.stack([np.cov(kmeans_data[labels == i], rowvar=False) for i in range(component)])

    return alphas, mus, sigmas

def f_probability(alphas, mus, sigmas, data):
    n_components = len(alphas)
    n_samples = data.shape[0]
    f_pdf = np.zeros((n_components, n_samples))
    for j in range(n_components):
        f_pdf[j] = multivariate_normal(
            mean=mus[j],
            cov=sigmas[j],
            allow_singular=True
        ).pdf(data) * alphas[j]
    p_f = f_pdf.sum(axis=0)
    p_f = np.maximum(p_f, 1e-300)
    return p_f


def EM(alphas, mus, sigmas, data, delta):
    th = 0
    N = data.shape[0]

    # 初始化存储空间
    alphas_new = np.zeros_like(alphas)
    mus_new = np.zeros_like(mus)
    sigmas_new = np.zeros_like(sigmas)

    for j in range(len(alphas)):
        p_gaussian = multivariate_normal(
            mean=mus[j],cov=sigmas[j],allow_singular=True
        ).pdf(data)

        f = f_probability(alphas, mus, sigmas, data)
        alpha_new = ((alphas[j] / N) *
                     ((p_gaussian / f).sum()))

        mu_new = ((alphas[j] / (N * alpha_new)) *
                  ((data * (p_gaussian / f).reshape(-1, 1)).sum(axis=0)))


        sigma_new = ((alphas[j] / (N * alpha_new)) *
                     (((data[:, :, np.newaxis] * data[:, np.newaxis, :]) * (
                                 p_gaussian / f).reshape(-1, 1, 1)).sum(axis=0) -
                      np.outer(mu_new, mu_new)))

        if abs(alpha_new - alphas[j]) < delta:
            th += 1

        # 存储
        alphas_new[j] = alpha_new
        mus_new[j] = mu_new
        sigmas_new[j] = sigma_new

    return alphas_new, mus_new, sigmas_new, th

def parse_pos(pos_str, h=None):
    arr = np.array([float(v) for v in pos_str.split('_')])
    if h is not None:
        arr[2] = h  # 强制第三维度为指定值
    return arr


def gmm_log_pdf(alphas, mus, sigmas, data):
    return np.log(f_probability(alphas, mus, sigmas, data))

def log_likelihood(x_2d, anchors, y_obs, alphas, mus, sigmas, h):
    x = np.array([x_2d[0], x_2d[1], h])
    # 计算所有锚点的预测距离
    deltas = x - anchors  # 向量化计算，得到形状 (n_anchors, 3)
    d_pred = np.linalg.norm(deltas, axis=1)
    d_pred = np.maximum(d_pred, 1e-8)  # 避免零


    # 这里假设y_obs的第一列是测量距离（range），后面是其他特征，构造的特征顺序应与训练数据一致
    samples = np.hstack([d_pred.reshape(-1, 1), y_obs])

    # 计算所有样本的对数概率
    log_probs = gmm_log_pdf(alphas, mus, sigmas, samples)  # 形状 (n_anchors,)
    total_log_likelihood = log_probs.sum()

    # 计算距离惩罚项（向量化）
    d_meas = y_obs[:, 0]  # 假设y_obs的第一列是测量距离
    lamda = 0
    distance_penalty = lamda * np.sum((d_pred - d_meas) ** 2)

    return -(total_log_likelihood - distance_penalty)

# 数据处理
def process_row(row):
    # 采样时间
    symbol_time = 508 / 499.2 * 1e-6  # us
    acc_sample_time = symbol_time / 1016
    # 时间序列
    t = np.arange(1016) * acc_sample_time
    # 转换字符串复数到模值数组

    arr = np.abs([complex(x.strip('()')) for x in row.cir]) / row.rxpacc
    arr = arr.astype(np.float32)

    # 接收信号最大幅值
    r_max = arr.max()

    # # 噪声方差
    # sub_arr = arr[:300]
    # noise_var = sub_arr.var(ddof=0)

    # # 接收信号上升时间
    # t_RT = ((arr > 0.6 * r_max).argmax() - (arr > 6 * noise_var).argmax()) * acc_sample_time

    # 接收信号能量
    r_energy = simpson(arr ** 2, t)

    # 平均延迟差
    t_MDS = simpson(t * (arr ** 2) / r_energy, t)

    # # 均方根延迟差
    # t_RMS = simpson(((t - t_MDS) ** 2) * (arr ** 2) / r_energy, t)
    #
    # # 峰度
    # k = simpson((arr - arr.mean()) ** 4, t) / ((t[-1] - t[0]) * (arr.var(ddof=0) ** 2))
    return r_max, r_energy, t_MDS
    # return r_max, t_RT, r_energy, t_MDS, t_RMS, k

n = 1
# ------------------------- 数据准备 -------------------------
dateset_dir = '../data_set/'
environments = ['environment0', 'environment1', 'environment2', 'environment3']
h_env = [1.2, 0.95, 1.32, 1.2]
run_time= [0, 0, 0, 0]
all_error = [0, 0, 0, 0]
error_df = pd.DataFrame(columns=['env', 'error'], dtype=object)
for i in range(len(environments)):
    train_env = environments[i]
    # current_h = h_env[i]

    train_df = pd.read_csv(dateset_dir + '/train_mean/' + train_env + '.csv', encoding='utf-8-sig')
    # train_df = train.sample(n=500)

    feature_lst = ['true_range', 'range', 'rss', 'fp_point1', 'cir_power', 'max_noise', 'r_max', 'r_energy', 't_MDS']

    pos_information = ['agent_pos', 'anchor', 'anchor_pos']

    # 训练数据和测试数据
    train_data = train_df[feature_lst].values

    # ------------------------- RL -------------------------
    alphas, mus, sigmas = k_means(train_data)
    # EM算法，迭代，alpha全部变化小于delta认为收敛
    iter = 0
    while True:
        alphas, mus, sigmas, dere = EM(alphas, mus, sigmas, train_data, delta=1e-5)
        iter += 1
        if dere == component:
            break

    # for j in range(len(environments)):
    for nn in range(n):
        start = time.perf_counter()
        test_env = environments[i]
        current_h = h_env[i]

        # 读取测试数据集
        test_data = pd.read_csv(dateset_dir + '/test_data_mean/' + test_env + '.csv', encoding='utf-8-sig')
        # 路径和anchor的集合
        agents_pos = test_data['agent_pos'].unique()

        # 得到三维坐标和误差
        error_lst = []
        for agent_pos in agents_pos:
            pos = test_data[test_data['agent_pos'] == agent_pos].copy()

            # 处理测试数据
            pos["cir"] = pos["cir"].apply(literal_eval)
            pos[['r_max', 'r_energy', 't_MDS']] = pos.apply(
                process_row,
                axis=1,
                result_type='expand'
            )
            pos = pos[pos_information + feature_lst[1:]].copy()
            # 'r_max', 't_RT', 'r_energy', 't_MDS', 't_RMS', 'k'

            # 标准化
            col_scale = pos.iloc[:, 4:].columns
            col_scale = col_scale[col_scale != 'nlos']
            scaler = StandardScaler()
            pos[col_scale] = scaler.fit_transform(pos[col_scale])

            anchor_positions = np.vstack(pos['anchor_pos'].apply(lambda s: parse_pos(s, current_h)))
            y_observations = pos.iloc[:, 3:].values  # 观测特征
            x_true = parse_pos(agent_pos)  # 真实坐标（仅用于验证）

            # 优化执行
            initial_guess = np.mean(anchor_positions[:, :2], axis=0)   # 使用锚点均值作为初始值

            result = minimize(
                log_likelihood,
                x0=initial_guess,
                args=(anchor_positions, y_observations, alphas, mus, sigmas, current_h),
                method='L-BFGS-B',
                bounds=[(0, np.inf), (0, np.inf)],
                options={'maxiter': 500, 'gtol': 1e-4, 'ftol': 1e-4}
            )

            estimated_position = np.append(result.x, current_h)
            true_position = parse_pos(agent_pos)
            error = np.linalg.norm(estimated_position - true_position)
            error_lst.append(abs(error))

            # 输出结果
            print(f" [{test_env}]True Position: {true_position}, Estimated Position: {estimated_position},Error:{error}")

        end = time.perf_counter()
        error_mean = np.mean(error_lst)
        print(f"\n[{test_env}]平均定位误差: {error_mean} 米")
        print(f"程序运行时间: {end - start:.6f} 秒")
        run_time[i] = run_time[i] + (end - start)
        all_error[i] = all_error[i] + error_mean

    error_df.loc[len(error_df)] = [test_env, error_lst]
    error_df.to_csv('../result/12/' + f'SRI.csv', index=False, encoding='utf-8-sig')
run_time = [x/n for x in run_time]
all_error = [x/n for x in all_error]
print(run_time)
print(all_error)
