# -*- coding: utf-8 -*-
"""
完整的基于测距似然（Ranging Likelihood, RL）的无线定位代码实现
Created on Tue May  6 12:28:12 2025
@author: 61413
"""

import os  # 导入操作系统接口模块，用于配置环境变量
import time  # 导入时间模块，用于计算程序运行时间
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理和分析
from sklearn.cluster import KMeans  # 从sklearn.cluster导入KMeans，用于聚类初始化
from scipy.stats import multivariate_normal  # 从scipy.stats导入multivariate_normal，用于多元正态分布计算
from scipy.optimize import minimize  # 从scipy.optimize导入minimize，用于优化算法
from sklearn.preprocessing import StandardScaler  # 从sklearn.preprocessing导入StandardScaler，用于数据标准化
from scipy.integrate import simpson  # 从scipy.integrate导入simpson，用于积分计算
from ast import literal_eval  # 从ast模块导入literal_eval，用于安全地评估字符串表达式

# 配置环境变量，设置LOKY和OMP的CPU使用数量，避免多线程冲突
os.environ["LOKY_MAX_CPU_COUNT"] = "24"
os.environ["OMP_NUM_THREADS"] = '1'

# 定义常量
COMPONENT = 3  # 设置GMM组件数量为3
DELTA = 1e-5   # 设置EM算法收敛阈值为1e-5
LAMBDA = 0     # 设置距离惩罚项权重为0（暂时不使用距离惩罚）
RUNS = 1       # 设置测试运行次数为1

class RangingLikelihoodLocalization:
    """
    基于测距似然（Ranging Likelihood, RL）的无线定位类
    该类封装了RL方法的核心功能，包括模型初始化、EM算法优化和对数似然计算
    """
    def __init__(self, component=COMPONENT):
        """
        初始化方法
        :param component: GMM组件数量，默认值为COMPONENT常量
        """
        self.component = component  # 设置组件数量
        self.alphas = None  # 初始化各组件权重为None
        self.mus = None  # 初始化各组件均值为None
        self.sigmas = None  # 初始化各组件协方差矩阵为None

    def initialize_with_kmeans(self, data):
        """
        使用K-Means算法初始化生成模型参数
        :param data: 训练数据
        """
        # 创建KMeans聚类模型，设置聚类数量为component，初始化次数为10，随机种子为42保证结果可复现
        kmeans = KMeans(n_clusters=self.component, n_init=10, random_state=42)
        kmeans.fit(data)  # 对数据进行聚类
        
        # 计算各组件权重，权重为各聚类的样本数量占比
        self.alphas = np.bincount(kmeans.labels_) / len(data)
        
        # 获取各聚类中心作为组件均值
        self.mus = kmeans.cluster_centers_
        
        # 计算各聚类的协方差矩阵，作为组件协方差矩阵
        self.sigmas = np.stack([
            np.cov(data[kmeans.labels_ == i], rowvar=False)
            for i in range(self.component)
        ])

    def expectation_maximization(self, data, max_iter=100):
        """
        执行EM算法优化生成模型参数
        :param data: 训练数据
        :param max_iter: 最大迭代次数，默认为100
        """
        for _ in range(max_iter):  # 迭代优化，最大迭代次数为max_iter
            # E步：计算后验概率
            posterior = self._calculate_posterior(data)
            
            # M步：更新参数
            new_alphas = posterior.mean(axis=0)  # 更新各组件权重
            new_mus = np.array([  # 更新各组件均值
                (posterior[:, i].reshape(-1, 1) * data).mean(axis=0)
                for i in range(self.component)
            ])
            new_sigmas = np.array([  # 更新各组件协方差矩阵
                np.cov(data.T, a=posterior[:, i], rowvar=False)
                for i in range(self.component)
            ])
            
            # 检查收敛条件：如果新旧参数差异小于DELTA，则认为收敛，跳出循环
            if np.all(np.abs(new_alphas - self.alphas) < DELTA) and \
               np.all(np.abs(new_mus - self.mus) < DELTA) and \
               np.all(np.abs(new_sigmas - self.sigmas) < DELTA):
                break
                
            # 更新模型参数
            self.alphas = new_alphas
            self.mus = new_mus
            self.sigmas = new_sigmas

    def _calculate_posterior(self, data):
        """
        计算后验概率
        :param data: 输入数据
        :return: 后验概率矩阵
        """
        # 计算每个组件的对数概率
        log_probs = np.array([
            multivariate_normal(mean=self.mus[i], cov=self.sigmas[i], allow_singular=True)
                   .logpdf(data) + np.log(self.alphas[i])
            for i in range(self.component)
        ])
        # 将对数概率转换为概率，并进行归一化，避免数值不稳定
        return np.exp(log_probs - log_probs.max(axis=0, keepdims=True)).T

    def log_likelihood(self, x_2d, anchors, observations, height):
        """
        计算对数似然函数（测距似然）
        :param x_2d: 当前估计的二维位置
        :param anchors: 锚节点位置
        :param observations: 观测数据
        :param height: 高度信息
        :return: 负的对数似然值（因为优化算法默认最小化）
        """
        x = np.array([x_2d[0], x_2d[1], height])  # 构造三维位置向量
        deltas = x - anchors  # 计算到各锚节点的向量差
        pred_distances = np.linalg.norm(deltas, axis=1)  # 计算到各锚节点的预测距离
        
        # 构造样本特征，将预测距离与观测数据结合
        samples = np.hstack([pred_distances.reshape(-1, 1), observations])
        
        # 计算对数概率并求和
        log_probs = np.log(self._calculate_posterior(samples).sum(axis=1))
        total_log_likelihood = log_probs.sum()
        
        # 计算距离惩罚项（暂时未使用，权重LAMBDA设为0）
        measured_distances = observations[:, 0]
        distance_penalty = LAMBDA * np.sum((pred_distances - measured_distances) ** 2)
        
        # 返回负的对数似然值（因为优化算法需要最小化目标函数）
        return -(total_log_likelihood - distance_penalty)

def parse_pos(pos_str, h=None):
    """
    解析位置字符串为numpy数组
    :param pos_str: 位置字符串，格式为"X_Y_Z"
    :param h: 高度信息，如果提供，则强制设置Z坐标为h
    :return: 位置的numpy数组
    """
    pos = np.array([float(v) for v in pos_str.split('_')])  # 将字符串拆分为浮点数数组
    if h is not None:
        pos[2] = h  # 如果提供了高度信息，则更新Z坐标
    return pos

def extract_waveform_features(row):
    """
    从原始波形数据提取特征
    :param row: 数据行，包含波形数据和其他信息
    :return: 提取的特征
    """
    symbol_time = 508 / 499.2 * 1e-6  # 计算符号时间（单位：秒）
    acc_sample_time = symbol_time / 1016  # 计算每个样本的时间间隔
    t = np.arange(1016) * acc_sample_time  # 构造时间序列
    
    # 将复数形式的波形数据转换为模值，并进行归一化
    waveform = np.abs([complex(x.strip('()')) for x in row['cir']]) / row['rxpacc']
    
    # 计算信号能量
    energy = np.trapz(waveform ** 2, t)  # 通过积分计算信号能量
    max_amp = waveform.max()  # 获取最大振幅
    
    # 计算上升时间
    noise_std = row['max_noise']  # 获取噪声标准差
    t_low = np.where(waveform >= 6 * noise_std)[0][0]  # 找到波形首次超过6倍噪声标准差的位置
    t_high = np.where(waveform >= 0.6 * max_amp)[0][0]  # 找到波形首次达到最大振幅60%的位置
    rise_time = t[t_high] - t[t_low]  # 计算上升时间
    
    # 计算平均延迟扩散
    delay_spread = np.trapz(t * (waveform ** 2) / energy, t)  # 计算平均延迟扩散
    
    # 计算均方根延迟扩散
    rms_delay = np.sqrt(np.trapz((t - delay_spread) ** 2 * (waveform ** 2) / energy, t))
    
    # 计算峰度
    mean = np.mean(waveform)  # 计算波形均值
    var = np.var(waveform)  # 计算波形方差
    kurtosis = np.trapz((waveform - mean) ** 4, t) / (var ** 2)  # 计算峰度
    
    # 返回提取的特征
    return pd.Series({
        'energy': energy,
        'max_amp': max_amp,
        'rise_time': rise_time,
        'delay_spread': delay_spread,
        'rms_delay': rms_delay,
        'kurtosis': kurtosis
    })

def main():
    """
    主函数，执行整个定位流程
    """
    # 数据路径配置
    dataset_dir = '../data_set/'  # 数据集根目录
    environments = ['environment0', 'environment1', 'environment2', 'environment3']  # 定义环境名称列表
    heights = [1.2, 0.95, 1.32, 1.2]  # 定义各环境的高度信息
    
    # 初始化结果存储变量
    run_times = [0.0] * len(environments)  # 存储每个环境的运行时间
    average_errors = [0.0] * len(environments)  # 存储每个环境的平均定位误差
    error_df = pd.DataFrame(columns=['env', 'error'])  # 创建DataFrame存储误差数据
    
    # 遍历每个环境
    for env_idx in range(len(environments)):
        train_env = environments[env_idx]  # 获取当前环境的训练数据名称
        current_h = heights[env_idx]  # 获取当前环境的高度信息
        
        # 读取训练数据
        train_df = pd.read_csv(f'{dataset_dir}/train_mean/{train_env}.csv', encoding='utf-8-sig')
        
        # 特征提取：对训练数据的每一行应用extract_waveform_features函数
        features = train_df.apply(extract_waveform_features, axis=1)
        train_data = features.values  # 将特征转换为NumPy数组
        
        # 初始化并训练RL模型
        rl_model = RangingLikelihoodLocalization()  # 创建RL模型实例
        rl_model.initialize_with_kmeans(train_data)  # 使用K-Means初始化模型参数
        rl_model.expectation_maximization(train_data)  # 使用EM算法优化模型参数
        
        # 测试模型，重复RUNS次取平均结果
        for _ in range(RUNS):
            start = time.perf_counter()  # 记录开始时间
            test_env = environments[env_idx]  # 获取当前环境的测试数据名称
            test_df = pd.read_csv(f'{dataset_dir}/test_data_mean/{test_env}.csv', encoding='utf-8-sig')
            
            agents_pos = test_df['agent_pos'].unique()  # 获取所有代理位置
            error_lst = []  # 初始化误差列表
            
            # 遍历每个代理位置进行定位
            for agent_pos in agents_pos:
                pos_data = test_df[test_df['agent_pos'] == agent_pos].copy()  # 获取当前代理位置的数据
                
                # 特征提取：对测试数据的每一行应用extract_waveform_features函数
                pos_features = pos_data.apply(extract_waveform_features, axis=1)
                pos_data = pd.concat([pos_data, pos_features], axis=1)  # 将特征合并到数据中
                
                # 数据标准化
                col_scale = pos_features.columns
                scaler = StandardScaler()  # 创建标准化转换器
                pos_data[col_scale] = scaler.fit_transform(pos_data[col_scale])  # 应用标准化
                
                # 解析锚节点位置和观测数据
                anchor_positions = np.vstack(pos_data['anchor_pos'].apply(lambda s: parse_pos(s, current_h)))
                y_observations = pos_data[col_scale].values  # 获取观测数据
                x_true = parse_pos(agent_pos)  # 获取真实位置
                
                # 设置初始猜测位置为锚节点位置的平均值
                initial_guess = np.mean(anchor_positions[:, :2], axis=0)
                
                # 执行优化算法，估计当前位置
                result = minimize(
                    rl_model.log_likelihood,  # 目标函数（负对数似然）
                    x0=initial_guess,  # 初始猜测位置
                    args=(anchor_positions, y_observations, current_h),  # 目标函数的额外参数
                    method='L-BFGS-B',  # 优化算法方法
                    bounds=[(0, np.inf), (0, np.inf)],  # 位置坐标的边界约束
                    options={'maxiter': 500, 'gtol': 1e-4, 'ftol': 1e-4}  # 优化选项
                )
                
                # 构造估计的三维位置
                estimated_position = np.append(result.x, current_h)
                
                # 计算定位误差
                error = np.linalg.norm(estimated_position - x_true)
                error_lst.append(error)  # 将误差添加到列表中
                
                # 打印定位结果
                print(f"[{test_env}] True Position: {x_true}, Estimated Position: {estimated_position}, Error: {error} meters")
            
            end = time.perf_counter()  # 记录结束时间
            error_mean = np.mean(error_lst)  # 计算平均误差
            run_times[env_idx] += (end - start)  # 累加运行时间
            average_errors[env_idx] += error_mean  # 累加平均误差
            
            # 打印测试结果
            print(f"\n[{test_env}] Average Localization Error: {error_mean} meters")
            print(f"Program Running Time: {end - start:.6f} seconds")
        
        # 存储误差数据到DataFrame
        error_df.loc[len(error_df)] = [test_env, error_lst]
        # 将误差数据保存到CSV文件
        error_df.to_csv(f'../result/12/Ranging_Likelihood_{test_env}.csv', index=False, encoding='utf-8-sig')
    
    # 计算最终的平均运行时间和平均定位误差
    run_times = [t / RUNS for t in run_times]
    average_errors = [e / RUNS for e in average_errors]
    
    # 打印最终结果
    print("Average Running Times:", run_times)
    print("Average Localization Errors:", average_errors)

if __name__ == "__main__":
    main()  # 执行主函数