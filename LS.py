import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares

def parse_pos(pos_str, h=None):
    """新增可选h参数，用于强制设置第三维度"""
    arr = np.array([float(v) for v in pos_str.split('_')])
    if h is not None:
        arr[2] = h  # 强制第三维度为指定值
    return arr

# 残差函数
def residual_function(x_2d, Y_points, measurements, h):
    # 计算X到所有Y_i的距离
    x = np.array([x_2d[0], x_2d[1], h])  # 构建三维坐标
    distances = np.linalg.norm(x - Y_points, axis=1)

    # 残差为距离与条件均值的差
    residuals = distances - measurements
    return residuals

# 最小二乘
def optimize_position(pos_df, measurements, h):
    # 解析三维坐标
    anchor_points = np.array([parse_pos(s) for s in pos_df['anchor_pos']])

    # 定义优化问题
    initial_guess = np.mean(anchor_points[:, :2], axis=0)  # 初始猜测为anchor的中心

    # 最小二乘法
    result = least_squares(
        fun=residual_function,
        x0=initial_guess,
        bounds=([0, 0], [np.inf, np.inf]),
        args=(anchor_points, measurements, h),
        method='trf',
        ftol=1e-10,
        xtol=1e-10,
    )
    return np.append(result.x, h)

def position(pos_df, agent_pos, h):
    pos = pos_df[pos_df['agent_pos'] == agent_pos].groupby('anchor').sample(n=1).reset_index(drop=True)
    measurements = pos['range']
    estimated_position = optimize_position(pos, measurements, h)
    true_position = parse_pos(agent_pos, h)
    error = np.linalg.norm(true_position - estimated_position)

    return true_position, estimated_position, error

n = 1
# ------------------------- 数据准备 -------------------------
dateset_dir = '../data_set/'
environments = ['environment0', 'environment1', 'environment2', 'environment3']
h_env = [1.2, 0.95, 1.32, 1.2]
run_time = [0, 0, 0, 0]
all_error = [0, 0, 0, 0]
error_df = pd.DataFrame(columns=['env', 'error'], dtype=object)


# 添加断点
breakpoint()
for i in range(len(environments)):
    test_env = environments[i]
    current_h = h_env[i]

    for nn in range(n):
        start = time.perf_counter()
        # 读取数据集
        test_data = pd.read_csv(dateset_dir + '/test_data_mean/' + test_env + '.csv', encoding='utf-8-sig')

        # 打印列名
        print("Columns in the DataFrame:", test_data.columns)

        # 路径和anchor的集合
        agents_pos = test_data['agent_pos'].unique()

        # 得到三维坐标和误差
        error_lst = []
        for agent_pos in agents_pos:
            pos = test_data[test_data['agent_pos'] == agent_pos].copy()

            true_position, estimated_position, error = position(pos, agent_pos, current_h)
            error_lst.append(error)
            # print(f"True Position: {true_position}, Estimated Position: {estimated_position},Error:{error}")

        end = time.perf_counter()
        error_mean = np.mean(error_lst)
        print(f"\n[{test_env}]平均定位误差: {error_mean} 米")
        print(f"程序运行时间: {end - start:.6f} 秒")
        run_time[i] = run_time[i] + (end - start)
        all_error[i] = all_error[i] + error_mean

    error_df.loc[len(error_df)] = [test_env, error_lst]
    error_df.to_csv('../result/12/' + f'LS.csv', index=False, encoding='utf-8-sig')

run_time = [x / n for x in run_time]
all_error = [x / n for x in all_error]
print(run_time)
print(all_error)