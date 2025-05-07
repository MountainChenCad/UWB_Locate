import pandas as pd  # 导入pandas库，用于数据处理和分析
import numpy as np  # 导入numpy库，用于数值计算
import json  # 导入json模块，用于读取和解析JSON文件
import time  # 导入time模块，用于计算程序运行时间
import os  # 导入os模块，用于文件和目录操作
from scipy.optimize import least_squares  # 从scipy.optimize导入least_squares，用于最小二乘法优化

def parse_pos(pos_str, h=None):
    """解析位置字符串并返回坐标数组"""
    arr = np.array([float(v) for v in pos_str.split('_')])  # 将位置字符串拆分成浮点数数组
    if h is not None:
        arr[2] = h  # 如果提供了高度信息，则更新Z坐标
    return arr  # 返回解析后的坐标数组

# 残差函数
def residual_function(x_2d, Y_points, measurements, h):
    x = np.array([x_2d[0], x_2d[1], h])  # 构建估计位置的三维坐标
    distances = np.linalg.norm(x - Y_points, axis=1)  # 计算估计位置与锚点之间的欧几里得距离
    residuals = distances - measurements  # 计算残差（估计距离与测量距离之差）
    return residuals  # 返回残差数组

# 最小二乘法优化位置
def optimize_position(Y_points, measurements, h):
    initial_guess = np.mean(Y_points[:, :2], axis=0)  # 计算初始猜测位置（锚点坐标的平均值）
    result = least_squares(
        fun=residual_function,  # 残差函数
        x0=initial_guess,  # 初始猜测
        bounds=([0, 0], [np.inf, np.inf]),  # 设置位置坐标的边界约束
        args=(Y_points, measurements, h),  # 残差函数的额外参数
        method='trf',  # 优化方法（Trust Region Reflective algorithm）
        ftol=1e-10,  # 优化的终止容差（基于目标函数值）
        xtol=1e-10,  # 优化的终止容差（基于独立变量的变动）
    )
    return np.append(result.x, h)  # 返回优化后的三维位置坐标

def position(agent_pos, anchors_df, measurements_df, h):
    """根据锚点信息和测量数据计算代理位置"""
    # 获取锚点坐标
    anchor_points = anchors_df[['x', 'y', 'z']].values
    
    # 检查measurements_df的列名，确保含有'range'列
    if 'range' not in measurements_df.columns:
        print("measurements_df列名:", measurements_df.columns)
        raise KeyError("measurements_df中没有名为'range'的列")
    
    # 获取测量值
    measurements = measurements_df[measurements_df['position'] == agent_pos]['range'].values
    
    # 如果没有测量值，返回一个默认值或跳过
    if len(measurements) == 0:
        return np.zeros(3), np.zeros(3), np.inf  # 返回默认值或处理异常
    
    # 优化位置
    estimated_position = optimize_position(anchor_points, measurements, h)
    
    # 真实位置
    true_position = parse_pos(agent_pos, h)
    
    # 计算误差
    error = np.linalg.norm(true_position - estimated_position)
    return true_position, estimated_position, error

n = 1  # 设置运行次数
# 获取当前工作目录
current_dir = os.getcwd()
# 构建数据集目录的绝对路径（修正后的路径）
dataset_dir = os.path.join(current_dir)  # 数据集应在当前目录下
environments = ['environment0', 'environment1', 'environment2', 'environment3']  # 定义环境名称列表
h_env = [1.2, 0.95, 1.32, 1.2]  # 定义各环境的高度信息
run_time = [0, 0, 0, 0]  # 初始化运行时间列表
all_error = [0, 0, 0, 0]  # 初始化平均误差列表
error_df = pd.DataFrame(columns=['env', 'error'], dtype=object)  # 创建DataFrame存储误差数据

# 确保结果目录存在
result_dir = os.path.join(current_dir, 'result', '12')
os.makedirs(result_dir, exist_ok=True)

# 检查每个环境的文件是否存在
for env in environments:
    anchors_file = os.path.join(dataset_dir, env, 'anchors.csv')
    data_file = os.path.join(dataset_dir, env, 'data.json')
    if not os.path.exists(anchors_file):
        print(f"文件不存在: {anchors_file}")
    if not os.path.exists(data_file):
        print(f"文件不存在: {data_file}")

# 遍历每个环境
for i in range(len(environments)):
    test_env = environments[i]
    current_h = h_env[i]

    for nn in range(n):
        start = time.perf_counter()  # 记录开始时间

        # 加载锚点信息
        anchors_file = os.path.join(current_dir, test_env, 'anchors.csv')
        if not os.path.exists(anchors_file):
            print(f"文件不存在: {anchors_file}")
            continue
        anchors_df = pd.read_csv(anchors_file)  # 读取锚点信息CSV文件

        # 加载测量数据
        data_file = os.path.join(current_dir, test_env, 'data.json')
        if not os.path.exists(data_file):
            print(f"文件不存在: {data_file}")
            continue
        with open(data_file, 'r') as f:
            data = json.load(f)  # 读取测量数据JSON文件

        # 获取所有位置
        positions = [pos['name'] for pos in data['path']]

        # 初始化误差列表
        error_lst = []
        for pos in positions:
            # 获取该位置的测量数据
            measurements_data = []
            for anchor in data['anchors']:
                for channel in data['channels']:
                    key = f"{pos}_{anchor}_{channel}"
                    if key in data['measurements']:
                        measurements_data.extend(data['measurements'][key])

            # 提取测量值
            measurements_df = pd.DataFrame(measurements_data)
            measurements_df['position'] = pos  # 确保包含'position'列

            # 打印调试信息
            print(f"处理位置: {pos}")
            print(f"measurements_df的列名: {measurements_df.columns}")

            # 计算位置和误差
            try:
                true_position, estimated_position, error = position(pos, anchors_df, measurements_df, current_h)
                error_lst.append(error)
            except KeyError as e:
                print(f"位置 {pos} 处理失败: {e}")
                continue

        end = time.perf_counter()  # 记录结束时间
        error_mean = np.mean(error_lst) if error_lst else np.inf  # 计算平均误差
        print(f"\n[{test_env}] 平均定位误差: {error_mean} 米")
        print(f"程序运行时间: {end - start:.6f} 秒")
        run_time[i] = run_time[i] + (end - start)  # 累加运行时间
        all_error[i] = all_error[i] + error_mean  # 累加平均误差

    # 确保error_lst已定义
    if 'error_lst' not in locals():
        error_lst = []  # 如果没有处理任何位置，初始化为空列表

    error_df.loc[len(error_df)] = [test_env, error_lst]
    error_df.to_csv(os.path.join(result_dir, 'LS.csv'), index=False, encoding='utf-8-sig')  # 保存误差数据到CSV文件

# 计算最终的平均运行时间和平均定位误差
run_time = [x / n for x in run_time]
all_error = [x / n for x in all_error]
print(run_time)
print(all_error)