import pandas as pd
import numpy as np
import time
import os
import argparse
from scipy.optimize import least_squares # <<<< ADDED THIS IMPORT


def parse_pos(pos_str, h=None):
    arr = np.array([float(v) for v in pos_str.split('_')])
    if h is not None:
        arr[2] = h
    return arr


def residual_function(x_2d, Y_points, measurements, h):
    x = np.array([x_2d[0], x_2d[1], h])
    distances = np.linalg.norm(x - Y_points, axis=1)
    residuals = distances - measurements
    return residuals


def optimize_position(pos_df_subset, measurements, h):  # Renamed pos_df to pos_df_subset
    anchor_points = np.array([parse_pos(s) for s in pos_df_subset['anchor_pos']])
    initial_guess = np.mean(anchor_points[:, :2], axis=0)
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


def position(pos_df_agent, agent_pos_str, current_h):  # Renamed parameters for clarity
    # In LS, for each agent_pos, we usually take all available anchors once.
    # The original sample(n=1) per anchor might be for specific data versions.
    # Let's assume we use all unique anchors for a given agent_pos from the CSV.
    # If test_data_mean.csv has one row per (agent_pos, anchor), then groupby 'anchor' is not needed.
    # The original code implies test_data_mean.csv has multiple entries per anchor for an agent_pos,
    # which is unusual if it's "mean" data. We'll stick to the original logic of sampling one per anchor.
    # If your test_data_mean.csv has one entry per anchor for each agent_pos, you can simplify this.

    # Ensure we are working with a copy if sampling
    pos_for_current_agent = pos_df_agent.groupby('anchor').sample(n=1, random_state=42).reset_index(drop=True)

    measurements = pos_for_current_agent['range']
    if measurements.empty or len(measurements) < 3:  # Need at least 3 anchors for 2D solve
        # print(f"Warning: Not enough measurements for {agent_pos_str}. Found {len(measurements)}. Skipping.")
        return parse_pos(agent_pos_str, current_h), np.array([np.nan, np.nan, current_h]), np.nan

    estimated_position = optimize_position(pos_for_current_agent, measurements, current_h)
    true_position = parse_pos(agent_pos_str, current_h)
    error = np.linalg.norm(true_position - estimated_position)
    return true_position, estimated_position, error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Least Squares Localization Experiments.")
    parser.add_argument('--output_dir', type=str, default='../result/LS_output',
                        help='Directory to save the output CSV.')
    parser.add_argument('--dataset_base_dir', type=str, default='./data_set',
                        help='Base directory for the dataset (containing test_data_mean).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_runs = 1  # Number of times to run for averaging (original code had n=1)
    base_dataset_dir = args.dataset_base_dir  # From arg
    test_data_dir = os.path.join(base_dataset_dir, 'test_data_mean')

    environments = ['environment0', 'environment1', 'environment2', 'environment3']
    h_env = [1.2, 0.95, 1.32, 1.2]

    all_env_results = []

    print("Starting LS.py execution...")
    for i in range(len(environments)):
        test_env = environments[i]
        current_h = h_env[i]
        env_total_error = 0
        env_total_time = 0
        env_error_list_for_df = []

        csv_file_path = os.path.join(test_data_dir, test_env + '.csv')
        if not os.path.exists(csv_file_path):
            print(f"Warning: Test data CSV not found for {test_env} at {csv_file_path}. Skipping.")
            continue

        print(f"Processing LS for environment: {test_env}")
        test_data_full_env = pd.read_csv(csv_file_path, encoding='utf-8-sig')

        for run_iter in range(n_runs):
            start_time = time.perf_counter()

            agents_pos_unique = test_data_full_env['agent_pos'].unique()
            current_run_errors = []

            for agent_pos_str in agents_pos_unique:
                # Filter data for the current agent_pos
                pos_df_for_agent = test_data_full_env[test_data_full_env['agent_pos'] == agent_pos_str].copy()
                if pos_df_for_agent.empty:
                    continue

                true_pos, est_pos, err = position(pos_df_for_agent, agent_pos_str, current_h)
                if not np.isnan(err):
                    current_run_errors.append(err)
                    if run_iter == 0:  # Collect for DataFrame only on the first run
                        env_error_list_for_df.append({
                            'environment': test_env,
                            'agent_pos': agent_pos_str,
                            'true_x': true_pos[0], 'true_y': true_pos[1], 'true_z': true_pos[2],
                            'est_x': est_pos[0], 'est_y': est_pos[1], 'est_z': est_pos[2],
                            'error': err
                        })

            end_time = time.perf_counter()
            if current_run_errors:
                env_total_error += np.mean(current_run_errors)
            env_total_time += (end_time - start_time)

        avg_error_for_env = env_total_error / n_runs if n_runs > 0 else 0
        avg_time_for_env = env_total_time / n_runs if n_runs > 0 else 0

        print(f"[{test_env}] Average Localization Error (LS): {avg_error_for_env:.4f} meters")
        print(f"[{test_env}] Average Execution Time (LS): {avg_time_for_env:.6f} seconds")

        # Store detailed errors for this environment
        if env_error_list_for_df:  # only if there were valid errors
            all_env_results.extend(env_error_list_for_df)

    if all_env_results:
        results_df = pd.DataFrame(all_env_results)
        output_csv_path = os.path.join(args.output_dir, 'LS_detailed_errors.csv')
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"LS detailed results saved to {output_csv_path}")
    else:
        print("No results generated by LS.py.")
    print("LS.py execution finished.")
