# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
# from scipy.integrate import simpson # np.trapz is generally preferred for discrete data
from ast import literal_eval
import argparse
import itertools

# Corrected os.environ assignment
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count()) # Use available CPUs, CONVERT TO STRING
os.environ["OMP_NUM_THREADS"] = '1'

COMPONENT = 3
DELTA = 1e-5
LAMBDA_PENALTY = 0  # Keep distance penalty off unless specifically testing it
RUNS = 1  # Number of times to average results for a given setup


# --- Feature Extraction (Robust version from previous response) ---
def extract_waveform_features(row, active_features_list):
    # This function calculates all 6 features, then selects the active ones.
    # This is simpler than conditional calculation if performance is not an extreme bottleneck here.

    all_calculated_features = {}  # Store all 6 calculated features here

    symbol_time = 508 / 499.2 * 1e-6
    acc_sample_time = symbol_time / 1016
    t = np.arange(1016) * acc_sample_time

    waveform = np.zeros(1016)
    valid_cir = False
    try:
        if isinstance(row['cir'], str):
            cir_list_str = literal_eval(row['cir'])
        elif isinstance(row['cir'], list):
            cir_list_str = row['cir']
        else:
            cir_list_str = []

        if cir_list_str and isinstance(row.get('rxpacc'), (int, float)) and row['rxpacc'] > 0:
            # Ensure all elements in cir_list_str are strings before stripping
            complex_values = [complex(str(x).strip('()')) for x in cir_list_str]
            if len(complex_values) == 1016:
                waveform = np.abs(complex_values) / row['rxpacc']
                valid_cir = True
    except Exception:
        pass

    if not valid_cir:
        all_calculated_features = {'energy': 0.0, 'max_amp': 0.0, 'rise_time': 0.0,
                                   'delay_spread': 0.0, 'rms_delay': 0.0, 'kurtosis': 0.0}
    else:
        all_calculated_features['energy'] = np.trapz(waveform ** 2, t)
        all_calculated_features['max_amp'] = waveform.max()

        noise_ref_for_rise_time = row.get('max_noise', 1.0)
        if not isinstance(noise_ref_for_rise_time,
                          (int, float)) or noise_ref_for_rise_time == 0: noise_ref_for_rise_time = 1.0

        low_thresh_indices = np.where(waveform >= 0.1 * noise_ref_for_rise_time)[0]
        t_low_idx = low_thresh_indices[0] if len(low_thresh_indices) > 0 else 0

        high_thresh_indices = np.where(waveform >= 0.6 * all_calculated_features['max_amp'])[0]
        t_high_idx = high_thresh_indices[0] if len(high_thresh_indices) > 0 else t_low_idx

        rt = t[min(t_high_idx, 1015)] - t[min(t_low_idx, 1015)]
        all_calculated_features['rise_time'] = max(0, rt)

        current_energy = all_calculated_features['energy']
        if current_energy > 1e-9:
            all_calculated_features['delay_spread'] = np.trapz(t * (waveform ** 2) / current_energy, t)
            rms_delay_integrand = (t - all_calculated_features['delay_spread']) ** 2 * (waveform ** 2) / current_energy
            all_calculated_features['rms_delay'] = np.sqrt(np.abs(np.trapz(rms_delay_integrand, t)))  # abs for safety
        else:
            all_calculated_features['delay_spread'] = 0.0
            all_calculated_features['rms_delay'] = 0.0

        mean_wf = np.mean(waveform)
        var_wf = np.var(waveform)
        if var_wf > 1e-9:
            kurtosis_integrand = (waveform - mean_wf) ** 4
            kurt = np.trapz(kurtosis_integrand, t) / (var_wf ** 2)
            all_calculated_features['kurtosis'] = kurt / (t[-1] - t[0] + 1e-9)
        else:
            all_calculated_features['kurtosis'] = 0.0

    # Select only active features
    selected_features = {key: all_calculated_features[key] for key in active_features_list if
                         key in all_calculated_features}
    return pd.Series(selected_features)


class RangingLikelihoodLocalization:
    def __init__(self, component=COMPONENT):
        self.component = component
        self.alphas = None
        self.mus = None
        self.sigmas = None
        self.is_trained = False

    def initialize_with_kmeans(self, data):
        if data.shape[0] < self.component:
            # print(f"Warning: Not enough samples ({data.shape[0]}) for K-Means with {self.component} clusters. Reducing clusters or skipping.")
            # Fallback: use fewer components or a simpler initialization
            if data.shape[0] == 0:
                # print("Error: No data for K-Means initialization.")
                self.alphas = np.array([1.0 / self.component] * self.component)  # Default if no data
                self.mus = np.zeros((self.component, data.shape[1]))
                self.sigmas = np.array([np.eye(data.shape[1])] * self.component)
                return

            # Simplified initialization: random subsets or just means
            effective_components = max(1, data.shape[0])  # at least 1 component
            kmeans = KMeans(n_clusters=effective_components, n_init='auto' if pd.__version__ >= '1.3.0' else 10,
                            random_state=42)

        else:
            kmeans = KMeans(n_clusters=self.component, n_init='auto' if pd.__version__ >= '1.3.0' else 10,
                            random_state=42)

        try:
            kmeans.fit(data)
            self.alphas = np.bincount(kmeans.labels_, minlength=self.component) / len(data)  # minlength for safety
            if len(self.alphas) < self.component:  # Pad if kmeans produced fewer clusters
                self.alphas = np.pad(self.alphas, (0, self.component - len(self.alphas)), 'constant',
                                     constant_values=1e-6)
                self.alphas /= self.alphas.sum()

            self.mus = kmeans.cluster_centers_
            if self.mus.shape[0] < self.component:  # Pad means if needed
                padding_mus = np.zeros((self.component - self.mus.shape[0], data.shape[1]))
                self.mus = np.vstack([self.mus, padding_mus])

            self.sigmas = []
            for i in range(self.component):
                cluster_data = data[kmeans.labels_ == i]
                if cluster_data.shape[0] > data.shape[1]:  # Need more samples than features for covariance
                    cov_matrix = np.cov(cluster_data, rowvar=False)
                else:  # Fallback for small clusters
                    cov_matrix = np.eye(data.shape[1]) * 1e-3  # Small diagonal covariance

                # Add small diagonal term for numerical stability (regularization)
                cov_matrix += np.eye(data.shape[1]) * 1e-6
                self.sigmas.append(cov_matrix)
            self.sigmas = np.array(self.sigmas)
            if self.sigmas.shape[0] < self.component:  # Pad sigmas if needed
                padding_sigmas = np.array([np.eye(data.shape[1])] * (self.component - self.sigmas.shape[0]))
                self.sigmas = np.vstack([self.sigmas, padding_sigmas])


        except ValueError as e:  # Kmeans can fail if n_samples < n_clusters
            # print(f"KMeans initialization failed: {e}. Using default GMM parameters.")
            self.alphas = np.full(self.component, 1.0 / self.component)
            self.mus = np.random.rand(self.component, data.shape[1]) * (data.max(axis=0) - data.min(axis=0)) + data.min(
                axis=0)
            self.sigmas = np.array([np.eye(data.shape[1]) for _ in range(self.component)])

    def expectation_maximization(self, data, max_iter=100):
        if data.shape[0] == 0:
            # print("Error: No data for EM algorithm.")
            self.is_trained = False
            return
        if data.shape[1] == 0:
            # print("Error: Data has no features for EM algorithm.")
            self.is_trained = False
            return

        if self.alphas is None or self.mus is None or self.sigmas is None:
            # print("GMM not initialized. Initializing with KMeans before EM.")
            self.initialize_with_kmeans(data)
            if data.shape[0] == 0: return  # Kmeans might have failed if data is empty

        for iteration in range(max_iter):
            try:
                # E-step: Calculate responsibilities (posterior probabilities)
                responsibilities = np.zeros((data.shape[0], self.component))
                for i in range(self.component):
                    try:
                        # Ensure sigma is positive definite, add jitter if not
                        # Check for singularity before calling multivariate_normal
                        if np.linalg.det(self.sigmas[i]) <= 1e-9:  # Determinant check
                            self.sigmas[i] += np.eye(self.sigmas[i].shape[0]) * 1e-6

                        responsibilities[:, i] = self.alphas[i] * multivariate_normal.pdf(data, mean=self.mus[i],
                                                                                          cov=self.sigmas[i],
                                                                                          allow_singular=True)
                    except np.linalg.LinAlgError:  # Catch explicit linalg errors for singular cov
                        # print(f"Warning: Singular covariance matrix for component {i} in E-step. Adding jitter.")
                        self.sigmas[i] += np.eye(self.sigmas[i].shape[0]) * 1e-6  # Add jitter
                        responsibilities[:, i] = self.alphas[i] * multivariate_normal.pdf(data, mean=self.mus[i],
                                                                                          cov=self.sigmas[i],
                                                                                          allow_singular=True)
                    except Exception as e_pdf:
                        # print(f"Error in multivariate_normal.pdf for component {i}: {e_pdf}. Setting resp to 0.")
                        responsibilities[:, i] = 0

                sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
                sum_responsibilities[sum_responsibilities == 0] = 1e-9  # Avoid division by zero
                responsibilities /= sum_responsibilities

                # M-step: Update parameters
                nk = responsibilities.sum(axis=0)
                new_alphas = nk / data.shape[0]

                new_mus = np.zeros_like(self.mus)
                for i in range(self.component):
                    if nk[i] > 1e-9:  # Avoid division by zero if component has no responsibility
                        new_mus[i] = (responsibilities[:, i][:, np.newaxis] * data).sum(axis=0) / nk[i]
                    else:  # Keep old mu if component vanishes
                        new_mus[i] = self.mus[i]

                new_sigmas = np.zeros_like(self.sigmas)
                for i in range(self.component):
                    if nk[i] > 1e-9:
                        diff = data - new_mus[i]
                        new_sigmas[i] = (responsibilities[:, i][:, np.newaxis, np.newaxis] * (
                                    diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / nk[i]
                        new_sigmas[i] += np.eye(data.shape[1]) * 1e-6  # Regularization
                    else:  # Keep old sigma if component vanishes
                        new_sigmas[i] = self.sigmas[i]

                # Check for convergence
                alpha_diff = np.abs(new_alphas - self.alphas).max()
                mu_diff = np.abs(new_mus - self.mus).max()
                sigma_diff = np.abs(new_sigmas - self.sigmas).max()

                if alpha_diff < DELTA and mu_diff < DELTA and sigma_diff < DELTA:
                    # print(f"EM converged after {iteration + 1} iterations.")
                    break

                self.alphas, self.mus, self.sigmas = new_alphas, new_mus, new_sigmas
            except Exception as e:
                # print(f"Error during EM iteration {iteration}: {e}. Stopping EM.")
                self.is_trained = False
                return

        self.is_trained = True

    def log_likelihood_gmm(self, data_point):  # For a single data point or a set for GMM prob
        if not self.is_trained or data_point.shape[0] == 0:
            return -np.inf if data_point.ndim == 1 else np.full(data_point.shape[0], -np.inf)

        log_probs_sum = np.zeros(data_point.shape[0] if data_point.ndim > 1 else 1)

        weighted_probs = np.zeros((data_point.shape[0] if data_point.ndim > 1 else 1, self.component))
        for i in range(self.component):
            try:
                # Ensure sigma is positive definite
                if np.linalg.det(self.sigmas[i]) <= 1e-9:
                    self.sigmas[i] += np.eye(self.sigmas[i].shape[0]) * 1e-6
                weighted_probs[:, i] = self.alphas[i] * multivariate_normal.pdf(data_point, mean=self.mus[i],
                                                                                cov=self.sigmas[i], allow_singular=True)
            except Exception:  # Catch LinAlgError or others
                weighted_probs[:, i] = 1e-300  # Assign a very small probability

        total_prob = weighted_probs.sum(axis=1)
        total_prob = np.maximum(total_prob, 1e-300)  # Floor to avoid log(0)
        return np.log(total_prob)

    def objective_function_for_minimize(self, x_2d, anchors_np, observations_np, height_val):
        if not self.is_trained: return np.inf  # Should not happen if model is trained

        x_3d = np.array([x_2d[0], x_2d[1], height_val])

        # Calculate predicted distances from current estimate x_3d to all anchors
        pred_distances = np.linalg.norm(x_3d - anchors_np, axis=1)

        # Construct samples for GMM: [pred_distance_to_anchor_i, feature_vec_for_anchor_i]
        # observations_np should be [num_anchors, num_selected_features]
        # We need to combine pred_distances with the corresponding *observed* features.
        # The GMM was trained on (true_range, feature_vector).
        # Here, for optimization, we use (predicted_range, observed_feature_vector).

        if observations_np.shape[1] == 0:  # Only distance is used (no other features)
            samples_for_gmm = pred_distances.reshape(-1, 1)
        else:  # distance + other features
            samples_for_gmm = np.hstack([pred_distances.reshape(-1, 1), observations_np])

        log_likelihoods = self.log_likelihood_gmm(samples_for_gmm)  # This returns log P( (d_pred, obs_features) | GMM )

        total_log_likelihood = np.sum(log_likelihoods)

        # Distance penalty (optional, controlled by LAMBDA_PENALTY)
        # This penalty assumes the first column of observations_np is the *measured_range* if it's used
        # However, our GMM's first dimension is *distance*, not necessarily measured range.
        # If observations_np contains measured_range as its first column (excluding the pred_distance we just added)
        # measured_ranges = observations_np[:, 0]
        # distance_penalty = LAMBDA_PENALTY * np.sum((pred_distances - measured_ranges) ** 2)
        # return -(total_log_likelihood - distance_penalty)

        return -total_log_likelihood  # Minimize negative log-likelihood


def parse_pos_str(pos_str, h_val=None):
    pos = np.array([float(v) for v in pos_str.split('_')])
    if h_val is not None:
        pos[2] = h_val
    return pos


def main(args):
    print(f"Starting RL.py with args: {args}")
    os.makedirs(args.output_base_dir, exist_ok=True)

    base_dataset_dir = args.dataset_base_dir
    train_data_dir = os.path.join(base_dataset_dir, 'train_mean')
    test_data_dir = os.path.join(base_dataset_dir, 'test_data_mean')

    all_environments = ['environment0', 'environment1', 'environment2', 'environment3']
    env_heights_map = {'environment0': 1.2, 'environment1': 0.95, 'environment2': 1.32, 'environment3': 1.2}

    # Determine train and test environments
    if args.train_envs and args.test_env:
        train_env_names = args.train_envs.split(',')
        test_env_names = [args.test_env]
        print(f"Cross-validation mode: Training on {train_env_names}, Testing on {test_env_names[0]}")
    else:  # Standard mode: iterate through all envs, train and test on same env
        train_env_names = all_environments  # Will be filtered inside the loop
        test_env_names = all_environments
        print("Standard mode: Training and testing on each environment individually.")

    active_features = args.features_to_use.split(',')
    print(f"Using features: {active_features}")

    # --- Training Phase ---
    rl_model = RangingLikelihoodLocalization(component=COMPONENT)

    # For cross-validation, GMM is trained on combined data from all train_envs
    if args.train_envs and args.test_env:  # Cross-validation training
        print(f"Training GMM for cross-validation using: {train_env_names}")
        combined_train_features_list = []
        for train_env_name_single in train_env_names:
            train_csv_path = os.path.join(train_data_dir, f"{train_env_name_single}.csv")
            if not os.path.exists(train_csv_path):
                print(f"Warning: Training CSV {train_csv_path} not found. Skipping this train env.")
                continue
            train_df_single_env = pd.read_csv(train_csv_path, encoding='utf-8-sig')
            if train_df_single_env.empty: continue

            # Extract selected waveform features
            extracted_wf_features = train_df_single_env.apply(
                lambda row: extract_waveform_features(row, active_features), axis=1)

            # Prepare data for GMM: [true_range, selected_waveform_features...]
            # Ensure 'true_range' is present in the CSV from preprocessing
            if 'true_range' not in train_df_single_env.columns:
                print(f"ERROR: 'true_range' column missing in {train_csv_path}. Cannot train GMM.")
                return  # or skip this environment

            # Concatenate true_range with extracted waveform features
            # The order must match what's used in objective_function_for_minimize
            # GMM input: [distance_feature, waveform_feature1, waveform_feature2, ...]
            gmm_train_data_single_env = pd.concat(
                [train_df_single_env['true_range'].rename('distance'), extracted_wf_features], axis=1)

            # Drop rows with any NaN values that might mess up GMM
            gmm_train_data_single_env.dropna(inplace=True)
            if not gmm_train_data_single_env.empty:
                combined_train_features_list.append(gmm_train_data_single_env.values)

        if not combined_train_features_list:
            print("Error: No training data collected for GMM. Exiting.")
            return

        full_train_data_for_gmm = np.vstack(combined_train_features_list)
        print(
            f"Total GMM training samples (cross-validation): {full_train_data_for_gmm.shape[0]}, Features: {full_train_data_for_gmm.shape[1]}")
        if full_train_data_for_gmm.shape[0] > 0 and full_train_data_for_gmm.shape[1] > 0:
            rl_model.initialize_with_kmeans(full_train_data_for_gmm)
            rl_model.expectation_maximization(full_train_data_for_gmm)
        else:
            print("Error: Not enough data or features to train GMM.")
            return

    # --- Testing Phase ---
    all_results_for_this_run = []

    for test_env_name_single in test_env_names:
        current_h = env_heights_map.get(test_env_name_single)
        if current_h is None:
            print(f"Warning: Height not defined for {test_env_name_single}. Skipping.")
            continue

        # If not in cross-validation mode, train GMM for each environment individually
        if not (args.train_envs and args.test_env):
            print(f"Training GMM for standard mode: {test_env_name_single}")
            train_csv_path = os.path.join(train_data_dir, f"{test_env_name_single}.csv")
            if not os.path.exists(train_csv_path):
                print(
                    f"Warning: Training CSV {train_csv_path} not found for standard mode. Skipping test for {test_env_name_single}.")
                continue
            train_df_env = pd.read_csv(train_csv_path, encoding='utf-8-sig')
            if train_df_env.empty:
                print(f"Warning: Training data for {test_env_name_single} is empty. Skipping.")
                continue

            extracted_wf_features_train = train_df_env.apply(
                lambda row: extract_waveform_features(row, active_features), axis=1)
            if 'true_range' not in train_df_env.columns:
                print(f"ERROR: 'true_range' column missing in {train_csv_path} for standard mode. Cannot train GMM.")
                continue

            gmm_train_data_env = pd.concat([train_df_env['true_range'].rename('distance'), extracted_wf_features_train],
                                           axis=1)
            gmm_train_data_env.dropna(inplace=True)

            if gmm_train_data_env.shape[0] > 0 and gmm_train_data_env.shape[1] > 0:
                rl_model = RangingLikelihoodLocalization(component=COMPONENT)  # Re-init for each env in std mode
                rl_model.initialize_with_kmeans(gmm_train_data_env.values)
                rl_model.expectation_maximization(gmm_train_data_env.values)
            else:
                print(
                    f"Error: Not enough data/features to train GMM for {test_env_name_single} in standard mode. Skipping test.")
                continue

        # Proceed to test on test_env_name_single
        print(f"Testing RL on environment: {test_env_name_single}")
        test_csv_path = os.path.join(test_data_dir, f"{test_env_name_single}.csv")
        if not os.path.exists(test_csv_path):
            print(f"Warning: Test CSV {test_csv_path} not found. Skipping.")
            continue
        test_df_env = pd.read_csv(test_csv_path, encoding='utf-8-sig')
        if test_df_env.empty:
            print(f"Warning: Test data for {test_env_name_single} is empty. Skipping.")
            continue

        env_total_error = 0
        env_total_time = 0
        num_runs_for_avg = RUNS

        for run_iter in range(num_runs_for_avg):
            start_run_time = time.perf_counter()
            agent_pos_unique_test = test_df_env['agent_pos'].unique()
            current_run_errors_list = []

            for agent_pos_str_test in agent_pos_unique_test:
                pos_data_agent = test_df_env[test_df_env['agent_pos'] == agent_pos_str_test].copy()
                if pos_data_agent.empty: continue

                # Extract waveform features for test data
                # These are the "observed features" that go with predicted distances
                observed_waveform_features_df = pos_data_agent.apply(
                    lambda row: extract_waveform_features(row, active_features), axis=1)

                # The GMM expects [distance, wf_feat1, wf_feat2, ...].
                # The 'distance' part will be the *predicted* distance during optimization.
                # So, `y_observations_for_gmm` will only contain the waveform features.
                y_observations_for_gmm = observed_waveform_features_df.values

                # Ensure y_observations_for_gmm has the correct number of columns, even if all features were zero
                expected_num_features = len(active_features)
                if y_observations_for_gmm.shape[1] != expected_num_features:
                    # This might happen if a feature column was all NaN and got dropped, or if no active_features.
                    # Create a zero matrix of the correct shape.
                    # print(f"Warning: Mismatch in observed feature columns for {agent_pos_str_test}. Expected {expected_num_features}, got {y_observations_for_gmm.shape[1]}. Padding with zeros.")
                    padded_obs = np.zeros((y_observations_for_gmm.shape[0], expected_num_features))
                    # Copy data if there's any, up to the smaller of the two dimensions
                    cols_to_copy = min(y_observations_for_gmm.shape[1], expected_num_features)
                    if cols_to_copy > 0:
                        padded_obs[:, :cols_to_copy] = y_observations_for_gmm[:, :cols_to_copy]
                    y_observations_for_gmm = padded_obs

                anchor_positions_np = np.vstack(pos_data_agent['anchor_pos'].apply(
                    lambda s: parse_pos_str(s)))  # height is not forced here, GMM handles 3D
                x_true_np = parse_pos_str(agent_pos_str_test)  # True position for error calculation

                initial_guess_2d = np.mean(anchor_positions_np[:, :2], axis=0)

                if not rl_model.is_trained:
                    # print(f"Warning: RL model not trained. Skipping optimization for {agent_pos_str_test} in {test_env_name_single}.")
                    current_run_errors_list.append(np.nan)  # Or some large error
                    continue

                try:
                    result = minimize(
                        rl_model.objective_function_for_minimize,
                        x0=initial_guess_2d,
                        args=(anchor_positions_np, y_observations_for_gmm, current_h),  # Pass current_h for z-coord
                        method='L-BFGS-B',
                        bounds=[(0, 30), (0, 30)],  # Reasonable bounds for typical indoor environments
                        options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-5, 'disp': False}
                    )
                    estimated_position_3d = np.append(result.x, current_h)
                    error_val = np.linalg.norm(estimated_position_3d - x_true_np)
                except Exception as e_min:
                    # print(f"Optimization failed for {agent_pos_str_test}: {e_min}")
                    estimated_position_3d = np.array([np.nan, np.nan, current_h])
                    error_val = np.nan

                current_run_errors_list.append(error_val)

                if run_iter == 0:  # Store detailed results only on the first run iteration
                    all_results_for_this_run.append({
                        'experiment_tag': args.experiment_tag,
                        'train_envs': args.train_envs if args.train_envs else test_env_name_single,
                        'test_env': test_env_name_single,
                        'features_used': args.features_to_use,
                        'agent_pos': agent_pos_str_test,
                        'true_x': x_true_np[0], 'true_y': x_true_np[1], 'true_z': x_true_np[2],
                        'est_x': estimated_position_3d[0], 'est_y': estimated_position_3d[1],
                        'est_z': estimated_position_3d[2],
                        'error': error_val
                    })

            end_run_time = time.perf_counter()
            valid_errors_this_run = [e for e in current_run_errors_list if not np.isnan(e)]
            if valid_errors_this_run:
                env_total_error += np.mean(valid_errors_this_run)
            env_total_time += (end_run_time - start_run_time)

        avg_error = env_total_error / num_runs_for_avg if num_runs_for_avg > 0 else np.nan
        avg_time = env_total_time / num_runs_for_avg if num_runs_for_avg > 0 else np.nan
        print(
            f"[{args.experiment_tag}][Test Env: {test_env_name_single}] Avg Error: {avg_error:.4f} m, Avg Time: {avg_time:.3f} s")

    # Save all detailed results for this particular RL.py invocation
    if all_results_for_this_run:
        results_df = pd.DataFrame(all_results_for_this_run)

        # Construct a filename that reflects the experiment parameters
        if args.train_envs and args.test_env:  # Cross-validation filename
            train_envs_str = args.train_envs.replace(',', '+')
            output_filename = f"RL_{args.experiment_tag}_train_{train_envs_str}_test_{args.test_env}.csv"
        else:  # Standard mode filename (iterates, so test_env_name_single is the last one, better to just use tag)
            # If iterating all envs, the results_df contains all envs tested.
            # The filename should reflect this.
            output_filename = f"RL_{args.experiment_tag}_all_envs_individually.csv"
            if len(test_env_names) == 1:  # if only one test env was processed in this call
                output_filename = f"RL_{args.experiment_tag}_test_{test_env_names[0]}.csv"

        output_csv_path = os.path.join(args.output_base_dir, output_filename)
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"RL detailed results for this run saved to {output_csv_path}")
    else:
        print(f"No results generated by RL.py for experiment: {args.experiment_tag}, test_env(s): {test_env_names}")
    print(f"RL.py execution finished for experiment: {args.experiment_tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ranging Likelihood Localization Experiments.")
    parser.add_argument('--output_base_dir', type=str, default='../result/RL_output',
                        help='Base directory to save output CSVs.')
    parser.add_argument('--dataset_base_dir', type=str, default='./data_set', help='Base directory for the dataset.')
    parser.add_argument('--features_to_use', type=str,
                        default="energy,max_amp,rise_time,delay_spread,rms_delay,kurtosis",
                        help='Comma-separated list of features to use from the 6 available.')
    parser.add_argument('--train_envs', type=str, default=None,
                        help='Comma-separated list of environment names for training (e.g., environment0,environment1). If None, uses standard per-environment training.')
    parser.add_argument('--test_env', type=str, default=None,
                        help='Single environment name for testing. If None, uses standard per-environment testing.')
    parser.add_argument('--experiment_tag', type=str, default="default",
                        help='A tag to identify this specific experiment run in output filenames.')

    cli_args = parser.parse_args()
    main(cli_args)
