# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
import argparse
import itertools

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = '1'

COMPONENT = 3
DELTA = 1e-5
LAMBDA_PENALTY = 0
RUNS = 1  # You can increase this for averaging, but it will take longer


# --- Feature Extraction (Robust version from previous response) ---
def extract_waveform_features(row, active_features_list):
    # (Keep the robust extract_waveform_features function from the previous corrected version)
    all_calculated_features = {}
    symbol_time = 508 / 499.2 * 1e-6
    acc_sample_time = symbol_time / 1016
    t = np.arange(1016) * acc_sample_time
    waveform = np.zeros(1016)
    valid_cir = False
    try:
        cir_data = row.get('cir')
        rxpacc_val = row.get('rxpacc')
        if isinstance(cir_data, str):
            cir_list_str = literal_eval(cir_data)
        elif isinstance(cir_data, list):
            cir_list_str = cir_data
        else:
            cir_list_str = []

        if cir_list_str and isinstance(rxpacc_val, (int, float)) and rxpacc_val > 0:
            complex_values = [complex(str(x).strip('()')) for x in cir_list_str if
                              isinstance(x, (str, complex))]  # Handle if already complex
            if len(complex_values) == 1016:
                waveform = np.abs(complex_values) / rxpacc_val
                valid_cir = True
    except Exception:  # Broad exception for parsing issues
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
            all_calculated_features['rms_delay'] = np.sqrt(np.abs(np.trapz(rms_delay_integrand, t)))
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
        self.em_convergence_iterations = -1  # To store EM convergence info

    def initialize_with_kmeans(self, data):
        # (Keep the robust initialize_with_kmeans from the previous corrected version)
        if data.shape[0] == 0:
            print("    KMeans Warning: No data provided for initialization.")
            # Set to default values to prevent downstream errors if this path is taken
            self.alphas = np.full(self.component, 1.0 / self.component)
            # Determine feature_dim carefully, if data.shape[1] is 0, this will fail
            feature_dim = data.shape[1] if data.ndim > 1 and data.shape[1] > 0 else 1  # Default to 1 feature if unknown
            self.mus = np.zeros((self.component, feature_dim))
            self.sigmas = np.array([np.eye(feature_dim) for _ in range(self.component)])
            self.is_trained = False  # Mark as not properly trained
            return

        if data.shape[0] < self.component:
            # print(f"    KMeans Info: Not enough samples ({data.shape[0]}) for K-Means with {self.component} clusters. Using {data.shape[0]} clusters.")
            effective_components = max(1, data.shape[0])
        else:
            effective_components = self.component

        # Use 'auto' for n_init in newer scikit-learn versions
        try:
            kmeans = KMeans(n_clusters=effective_components, n_init='auto', random_state=42)
        except TypeError:  # Older scikit-learn
            kmeans = KMeans(n_clusters=effective_components, n_init=10, random_state=42)

        try:
            kmeans.fit(data)
            unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)

            temp_alphas = np.zeros(effective_components)
            for label, count in zip(unique_labels, counts):
                temp_alphas[label] = count
            temp_alphas = temp_alphas / len(data)

            self.alphas = np.zeros(self.component)
            self.mus = np.zeros((self.component, data.shape[1]))
            self.sigmas = np.zeros((self.component, data.shape[1], data.shape[1]))

            for i in range(effective_components):
                self.alphas[i] = temp_alphas[i]
                self.mus[i, :] = kmeans.cluster_centers_[i, :]
                cluster_data_indices = np.where(kmeans.labels_ == i)[0]
                if len(cluster_data_indices) > data.shape[1]:
                    cluster_data = data[cluster_data_indices]
                    cov_matrix = np.cov(cluster_data, rowvar=False)
                else:
                    cov_matrix = np.eye(data.shape[1]) * 1e-3
                self.sigmas[i] = cov_matrix + np.eye(data.shape[1]) * 1e-6

            # If effective_components < self.component, fill remaining with mean/default
            if effective_components < self.component:
                # print(f"    KMeans Info: Filling remaining {self.component - effective_components} GMM components with defaults.")
                remaining_alpha_mass = 1.0 - np.sum(self.alphas[:effective_components])
                if remaining_alpha_mass < 0: remaining_alpha_mass = 0  # cap at 0

                fill_alpha = remaining_alpha_mass / (self.component - effective_components) if (
                                                                                                           self.component - effective_components) > 0 else 0

                for i in range(effective_components, self.component):
                    self.alphas[i] = fill_alpha
                    self.mus[i, :] = data.mean(axis=0) if data.shape[0] > 0 else np.zeros(data.shape[1])
                    self.sigmas[i] = np.eye(data.shape[1]) * 1e-3 + np.eye(data.shape[1]) * 1e-6

            self.alphas /= np.sum(self.alphas)  # Ensure sum to 1

        except ValueError as e:
            # print(f"    KMeans initialization failed: {e}. Using default GMM parameters.")
            self.alphas = np.full(self.component, 1.0 / self.component)
            self.mus = np.random.rand(self.component, data.shape[1]) * (data.max(axis=0) - data.min(axis=0)) + data.min(
                axis=0) if data.shape[0] > 0 else np.zeros((self.component, data.shape[1]))
            self.sigmas = np.array([np.eye(data.shape[1]) for _ in range(self.component)])
        self.is_trained = True  # Mark as initialized, EM will refine

    def expectation_maximization(self, data, max_iter=100):
        # (Keep the robust expectation_maximization from the previous corrected version)
        # Add a print at the start and end/convergence
        # print("    Starting EM algorithm...")
        if data.shape[0] == 0:
            # print("    EM Error: No data provided.")
            self.is_trained = False
            self.em_convergence_iterations = -1
            return
        if data.shape[1] == 0:
            # print("    EM Error: Data has no features.")
            self.is_trained = False
            self.em_convergence_iterations = -1
            return

        if self.alphas is None or self.mus is None or self.sigmas is None or not self.is_trained:
            # print("    EM Info: GMM not initialized or previous init failed. Initializing with KMeans before EM.")
            self.initialize_with_kmeans(data)
            if not self.is_trained or data.shape[0] == 0:  # Kmeans might have failed
                self.em_convergence_iterations = -1
                return

        for iteration in range(max_iter):
            try:
                responsibilities = np.zeros((data.shape[0], self.component))
                for i in range(self.component):
                    try:
                        cov_mat = self.sigmas[i]
                        # Check condition number
                        if np.linalg.cond(cov_mat) > 1 / np.finfo(cov_mat.dtype).eps:
                            cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * 1e-6
                        responsibilities[:, i] = self.alphas[i] * multivariate_normal.pdf(data, mean=self.mus[i],
                                                                                          cov=cov_mat,
                                                                                          allow_singular=True)
                    except np.linalg.LinAlgError:
                        # print(f"    EM Warning: Singular covariance for component {i} in E-step. Adding jitter.")
                        jittered_cov = self.sigmas[i] + np.eye(self.sigmas[i].shape[0]) * 1e-6
                        responsibilities[:, i] = self.alphas[i] * multivariate_normal.pdf(data, mean=self.mus[i],
                                                                                          cov=jittered_cov,
                                                                                          allow_singular=True)
                    except Exception:  # Catch other PDF errors
                        responsibilities[:, i] = 1e-300

                sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
                sum_responsibilities[sum_responsibilities < 1e-300] = 1e-300
                responsibilities /= sum_responsibilities
                nk = responsibilities.sum(axis=0)
                new_alphas = nk / data.shape[0]
                new_mus = np.zeros_like(self.mus)
                for i in range(self.component):
                    if nk[i] > 1e-9:
                        new_mus[i] = (responsibilities[:, i][:, np.newaxis] * data).sum(axis=0) / nk[i]
                    else:
                        new_mus[i] = self.mus[i]
                new_sigmas = np.zeros_like(self.sigmas)
                for i in range(self.component):
                    if nk[i] > 1e-9:
                        diff = data - new_mus[i]
                        new_sigmas[i] = (responsibilities[:, i][:, np.newaxis, np.newaxis] * (
                                    diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / nk[i]
                        new_sigmas[i] += np.eye(data.shape[1]) * 1e-6
                    else:
                        new_sigmas[i] = self.sigmas[i]

                alpha_diff = np.abs(new_alphas - self.alphas).max() if self.alphas is not None else np.inf
                mu_diff = np.abs(new_mus - self.mus).max() if self.mus is not None else np.inf
                sigma_diff = np.abs(new_sigmas - self.sigmas).max() if self.sigmas is not None else np.inf

                self.alphas, self.mus, self.sigmas = new_alphas, new_mus, new_sigmas  # Update before checking convergence

                if alpha_diff < DELTA and mu_diff < DELTA and sigma_diff < DELTA:
                    # print(f"    EM converged after {iteration + 1} iterations.")
                    self.em_convergence_iterations = iteration + 1
                    break
                if iteration == max_iter - 1:
                    # print(f"    EM Warning: Reached max_iter ({max_iter}) without full convergence.")
                    self.em_convergence_iterations = max_iter
            except Exception as e:
                # print(f"    EM Error during iteration {iteration}: {e}. Stopping EM.")
                self.is_trained = False
                self.em_convergence_iterations = -1
                return
        self.is_trained = True

    def log_likelihood_gmm(self, data_point):
        # (Keep the robust log_likelihood_gmm from the previous corrected version)
        if not self.is_trained or self.alphas is None or self.mus is None or self.sigmas is None:
            return -np.inf if data_point.ndim == 1 else np.full(np.atleast_2d(data_point).shape[0], -np.inf)
        current_data_point = np.atleast_2d(data_point)
        if current_data_point.shape[0] == 0: return np.array([])

        weighted_probs = np.zeros((current_data_point.shape[0], self.component))
        for i in range(self.component):
            try:
                cov_mat = self.sigmas[i]
                if np.linalg.cond(cov_mat) > 1 / np.finfo(cov_mat.dtype).eps:
                    cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * 1e-6
                weighted_probs[:, i] = self.alphas[i] * multivariate_normal.pdf(current_data_point, mean=self.mus[i],
                                                                                cov=cov_mat, allow_singular=True)
            except Exception:
                weighted_probs[:, i] = 1e-300
        total_prob = weighted_probs.sum(axis=1)
        total_prob = np.maximum(total_prob, 1e-300)
        log_probs_to_return = np.log(total_prob)
        return log_probs_to_return[0] if data_point.ndim == 1 and len(log_probs_to_return) == 1 else log_probs_to_return

    def objective_function_for_minimize(self, x_2d, anchors_np, observations_np, height_val):
        # (Keep the robust objective_function_for_minimize from the previous corrected version)
        if not self.is_trained: return np.inf
        x_3d = np.array([x_2d[0], x_2d[1], height_val])
        pred_distances = np.linalg.norm(x_3d - anchors_np, axis=1)

        num_waveform_features_in_gmm = self.mus.shape[1] - 1 if self.mus is not None and self.mus.shape[1] > 0 else 0

        # observations_np comes in as [n_anchors, n_active_waveform_features]
        if observations_np.shape[1] != num_waveform_features_in_gmm:
            # This is a critical mismatch. GMM expects specific number of features after distance.
            # print(f"    Minimize Error: Mismatch in observed features for GMM. Expected {num_waveform_features_in_gmm}, Got {observations_np.shape[1]}.")
            return np.inf  # High cost

        if num_waveform_features_in_gmm == 0:  # GMM trained on distance only
            samples_for_gmm = pred_distances.reshape(-1, 1)
        else:
            samples_for_gmm = np.hstack([pred_distances.reshape(-1, 1), observations_np])

        log_likelihoods = self.log_likelihood_gmm(samples_for_gmm)
        total_log_likelihood = np.sum(log_likelihoods)
        return -total_log_likelihood


def parse_pos_str(pos_str, h_val=None):
    # (Keep parse_pos_str from the previous corrected version)
    pos = np.array([float(v) for v in pos_str.split('_')])
    if h_val is not None:
        pos[2] = h_val
    return pos


def main(args):
    # Add flush=True to print statements to ensure they appear immediately
    print(f"Starting RL.py with args: {args}", flush=True)
    os.makedirs(args.output_base_dir, exist_ok=True)

    base_dataset_dir = args.dataset_base_dir
    train_data_dir = os.path.join(base_dataset_dir, 'train_mean')
    test_data_dir = os.path.join(base_dataset_dir, 'test_data_mean')

    all_environments = ['environment0', 'environment1', 'environment2', 'environment3']
    env_heights_map = {'environment0': 1.2, 'environment1': 0.95, 'environment2': 1.32, 'environment3': 1.2}

    if args.train_envs and args.test_env:
        train_env_names = args.train_envs.split(',')
        test_env_names = [args.test_env]
        print(f"Cross-validation mode: Training on {train_env_names}, Testing on {test_env_names[0]}", flush=True)
    else:
        train_env_names = all_environments  # Placeholder, will be set to current env in loop
        test_env_names = all_environments
        print("Standard mode: Training and testing on each environment individually.", flush=True)

    active_features = [feat.strip() for feat in args.features_to_use.split(',') if feat.strip()]
    print(f"Using features: {active_features if active_features else 'None (distance only GMM)'}", flush=True)

    rl_model = RangingLikelihoodLocalization(component=COMPONENT)

    if args.train_envs and args.test_env:  # Cross-validation: Train GMM once on combined training environments
        print(f"  GMM Training (Cross-Val Mode) - Starting for train_envs: {train_env_names}", flush=True)
        start_gmm_train_time = time.perf_counter()
        combined_train_features_list = []
        for train_env_name_single in train_env_names:
            print(f"    Loading training data from: {train_env_name_single}", flush=True)
            train_csv_path = os.path.join(train_data_dir, f"{train_env_name_single}.csv")
            if not os.path.exists(train_csv_path):
                print(f"    Warning: Training CSV {train_csv_path} not found. Skipping.", flush=True)
                continue
            train_df_single_env = pd.read_csv(train_csv_path, encoding='utf-8-sig')
            if train_df_single_env.empty:
                print(f"    Warning: Training data for {train_env_name_single} is empty. Skipping.", flush=True)
                continue

            if 'true_range' not in train_df_single_env.columns:
                print(f"    ERROR: 'true_range' column missing in {train_csv_path}. Cannot train GMM.", flush=True)
                continue

            gmm_train_data_for_env = train_df_single_env[['true_range']].rename(columns={'true_range': 'distance'})
            if active_features:
                extracted_wf_features = train_df_single_env.apply(
                    lambda row: extract_waveform_features(row, active_features), axis=1)
                gmm_train_data_for_env = pd.concat([gmm_train_data_for_env, extracted_wf_features], axis=1)

            gmm_train_data_for_env.dropna(inplace=True)
            if not gmm_train_data_for_env.empty:
                combined_train_features_list.append(gmm_train_data_for_env.values)
            else:
                print(f"    Warning: No valid GMM training samples after processing for {train_env_name_single}.",
                      flush=True)

        if not combined_train_features_list:
            print("  GMM Training Error: No training data collected for GMM. Exiting RL run.", flush=True)
            return

        full_train_data_for_gmm = np.vstack(combined_train_features_list)
        print(
            f"  GMM Training (Cross-Val Mode) - Total samples: {full_train_data_for_gmm.shape[0]}, Features (incl. dist): {full_train_data_for_gmm.shape[1]}",
            flush=True)
        if full_train_data_for_gmm.shape[0] > 0 and full_train_data_for_gmm.shape[1] > 0:
            rl_model.initialize_with_kmeans(full_train_data_for_gmm)
            rl_model.expectation_maximization(full_train_data_for_gmm)
            print(
                f"  GMM Training (Cross-Val Mode) - EM finished in {rl_model.em_convergence_iterations} iterations. Model is_trained: {rl_model.is_trained}",
                flush=True)
        else:
            print("  GMM Training Error: Not enough data or features after combining. Exiting RL run.", flush=True)
            return
        end_gmm_train_time = time.perf_counter()
        print(
            f"  GMM Training (Cross-Val Mode) - Completed in {end_gmm_train_time - start_gmm_train_time:.2f} seconds.",
            flush=True)

    all_results_for_this_run = []
    for test_env_idx, test_env_name_single in enumerate(test_env_names):
        print(f"\nProcessing Test Environment: {test_env_name_single} ({test_env_idx + 1}/{len(test_env_names)})",
              flush=True)
        current_h = env_heights_map.get(test_env_name_single)
        if current_h is None:
            print(f"  Warning: Height not defined for {test_env_name_single}. Skipping.", flush=True)
            continue

        if not (args.train_envs and args.test_env):  # Standard mode: Train GMM for this specific environment
            print(f"  GMM Training (Standard Mode) - Starting for: {test_env_name_single}", flush=True)
            start_gmm_train_time = time.perf_counter()
            train_csv_path = os.path.join(train_data_dir, f"{test_env_name_single}.csv")
            if not os.path.exists(train_csv_path):
                print(
                    f"  Warning: Training CSV {train_csv_path} not found for standard mode. Skipping test for {test_env_name_single}.",
                    flush=True)
                continue
            train_df_env = pd.read_csv(train_csv_path, encoding='utf-8-sig')
            if train_df_env.empty:
                print(f"  Warning: Training data for {test_env_name_single} is empty. Skipping.", flush=True)
                continue

            if 'true_range' not in train_df_env.columns:
                print(f"  ERROR: 'true_range' column missing in {train_csv_path} for standard mode. Cannot train GMM.",
                      flush=True)
                continue

            gmm_train_data_env_for_gmm = train_df_env[['true_range']].rename(columns={'true_range': 'distance'})
            if active_features:
                extracted_wf_features_train = train_df_env.apply(
                    lambda row: extract_waveform_features(row, active_features), axis=1)
                gmm_train_data_env_for_gmm = pd.concat([gmm_train_data_env_for_gmm, extracted_wf_features_train],
                                                       axis=1)

            gmm_train_data_env_for_gmm.dropna(inplace=True)
            print(
                f"  GMM Training (Standard Mode) - Samples for {test_env_name_single}: {gmm_train_data_env_for_gmm.shape[0]}, Features (incl. dist): {gmm_train_data_env_for_gmm.shape[1]}",
                flush=True)

            if gmm_train_data_env_for_gmm.shape[0] > 0 and gmm_train_data_env_for_gmm.shape[1] > 0:
                rl_model = RangingLikelihoodLocalization(component=COMPONENT)  # Re-init for each env
                rl_model.initialize_with_kmeans(gmm_train_data_env_for_gmm.values)
                rl_model.expectation_maximization(gmm_train_data_env_for_gmm.values)
                print(
                    f"  GMM Training (Standard Mode) - EM finished in {rl_model.em_convergence_iterations} iterations for {test_env_name_single}. Model is_trained: {rl_model.is_trained}",
                    flush=True)

            else:
                print(
                    f"  GMM Training Error: Not enough data/features to train GMM for {test_env_name_single} in standard mode. Skipping test.",
                    flush=True)
                continue
            end_gmm_train_time = time.perf_counter()
            print(
                f"  GMM Training (Standard Mode) - Completed for {test_env_name_single} in {end_gmm_train_time - start_gmm_train_time:.2f} seconds.",
                flush=True)

        print(f"  Testing RL on environment: {test_env_name_single} - Starting...", flush=True)
        test_csv_path = os.path.join(test_data_dir, f"{test_env_name_single}.csv")
        if not os.path.exists(test_csv_path):
            print(f"  Warning: Test CSV {test_csv_path} not found. Skipping.", flush=True)
            continue
        test_df_env = pd.read_csv(test_csv_path, encoding='utf-8-sig')
        if test_df_env.empty:
            print(f"  Warning: Test data for {test_env_name_single} is empty. Skipping.", flush=True)
            continue

        env_total_error = 0
        env_total_time = 0
        num_runs_for_avg = RUNS  # Usually 1, unless you want to average multiple test runs with same GMM

        for run_iter in range(num_runs_for_avg):  # This loop is more for stability testing if RUNS > 1
            print(f"    Test Run Iteration: {run_iter + 1}/{num_runs_for_avg} for {test_env_name_single}", flush=True)
            start_run_time = time.perf_counter()
            agent_pos_unique_test = test_df_env['agent_pos'].unique()
            current_run_errors_list = []
            num_agent_pos = len(agent_pos_unique_test)

            for agent_idx, agent_pos_str_test in enumerate(agent_pos_unique_test):
                if (agent_idx + 1) % 20 == 0 or agent_idx == num_agent_pos - 1:  # Print progress
                    print(f"      Testing agent_pos: {agent_idx + 1}/{num_agent_pos} ('{agent_pos_str_test}')",
                          flush=True)

                pos_data_agent = test_df_env[test_df_env['agent_pos'] == agent_pos_str_test].copy()
                if pos_data_agent.empty: continue

                y_observations_for_gmm = pd.DataFrame()
                if active_features:
                    observed_waveform_features_df = pos_data_agent.apply(
                        lambda row: extract_waveform_features(row, active_features), axis=1)
                    y_observations_for_gmm = observed_waveform_features_df.values
                else:
                    y_observations_for_gmm = np.empty((pos_data_agent.shape[0], 0))  # Shape [n_anchors, 0]

                anchor_positions_np = np.vstack(pos_data_agent['anchor_pos'].apply(lambda s: parse_pos_str(s)))
                x_true_np = parse_pos_str(agent_pos_str_test)
                initial_guess_2d = np.mean(anchor_positions_np[:, :2], axis=0)

                if not rl_model.is_trained:
                    # print(f"      Warning: RL model not trained. Skipping optimization for {agent_pos_str_test}.", flush=True)
                    current_run_errors_list.append(np.nan)
                    continue

                minimize_success = False
                try:
                    result = minimize(
                        rl_model.objective_function_for_minimize,
                        x0=initial_guess_2d,
                        args=(anchor_positions_np, y_observations_for_gmm, current_h),
                        method='L-BFGS-B',
                        bounds=[(0, 30), (0, 30)],
                        options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-5, 'disp': False}
                        # Set disp=True for detailed opt output
                    )
                    minimize_success = result.success
                    estimated_position_3d = np.append(result.x, current_h)
                    error_val = np.linalg.norm(estimated_position_3d - x_true_np)
                    # if not minimize_success:
                    # print(f"      Minimize Warning for {agent_pos_str_test}: {result.message}", flush=True)

                except Exception as e_min:
                    # print(f"      Minimize Error for {agent_pos_str_test}: {e_min}", flush=True)
                    estimated_position_3d = np.array([np.nan, np.nan, current_h])
                    error_val = np.nan

                current_run_errors_list.append(error_val)

                if run_iter == 0:
                    all_results_for_this_run.append({
                        'experiment_tag': args.experiment_tag,
                        'train_envs': args.train_envs if args.train_envs else test_env_name_single,
                        'test_env': test_env_name_single,
                        'features_used': args.features_to_use if active_features else "distance_only",
                        'agent_pos': agent_pos_str_test,
                        'true_x': x_true_np[0], 'true_y': x_true_np[1], 'true_z': x_true_np[2],
                        'est_x': estimated_position_3d[0], 'est_y': estimated_position_3d[1],
                        'est_z': estimated_position_3d[2],
                        'error': error_val,
                        'minimize_success': minimize_success
                    })

            end_run_time = time.perf_counter()
            valid_errors_this_run = [e for e in current_run_errors_list if not np.isnan(e)]
            if valid_errors_this_run:
                env_total_error += np.mean(valid_errors_this_run)
            env_total_time += (end_run_time - start_run_time)
            print(
                f"    Test Run Iteration {run_iter + 1} for {test_env_name_single} took {end_run_time - start_run_time:.2f}s. Mean error: {np.mean(valid_errors_this_run) if valid_errors_this_run else 'N/A'}",
                flush=True)

        avg_error = env_total_error / num_runs_for_avg if num_runs_for_avg > 0 and env_total_error > 0 else np.nan  # only average if error was accumulated
        avg_time = env_total_time / num_runs_for_avg if num_runs_for_avg > 0 else np.nan
        print(
            f"  [{args.experiment_tag}][Test Env: {test_env_name_single}] Overall Avg Error: {avg_error:.4f} m, Overall Avg Time: {avg_time:.3f} s",
            flush=True)

    if all_results_for_this_run:
        results_df = pd.DataFrame(all_results_for_this_run)
        train_envs_str = "std"
        if args.train_envs: train_envs_str = args.train_envs.replace(',', '+')
        test_env_str = "all"
        if args.test_env:
            test_env_str = args.test_env
        elif len(test_env_names) == 1 and not (args.train_envs and args.test_env):
            test_env_str = test_env_names[0]
        features_tag = "dist_only"
        if active_features:
            features_tag = args.features_to_use.replace(',', '_').replace('energy', 'E').replace('max_amp',
                                                                                                 'MA').replace(
                'rise_time', 'RT').replace('delay_spread', 'DS').replace('rms_delay', 'RMSD').replace('kurtosis',
                                                                                                      'K')  # Shortened RMSD
        output_filename = f"RL_{args.experiment_tag}_train_{train_envs_str}_test_{test_env_str}_feats_{features_tag}.csv"
        if not (args.train_envs and args.test_env) and len(test_env_names) > 1:
            output_filename = f"RL_{args.experiment_tag}_std_allenvs_feats_{features_tag}.csv"
        output_csv_path = os.path.join(args.output_base_dir, output_filename)
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"RL detailed results for this run saved to {output_csv_path}", flush=True)
    else:
        print(f"No results generated by RL.py for experiment: {args.experiment_tag}, test_env(s): {test_env_names}",
              flush=True)
    print(f"RL.py execution finished for experiment: {args.experiment_tag}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ranging Likelihood Localization Experiments.")
    parser.add_argument('--output_base_dir', type=str, default='../result/RL_output',
                        help='Base directory to save output CSVs.')
    parser.add_argument('--dataset_base_dir', type=str, default='./data_set', help='Base directory for the dataset.')
    parser.add_argument('--features_to_use', type=str,
                        default="energy,max_amp,rise_time,delay_spread,rms_delay,kurtosis",
                        help='Comma-separated list of features to use. Empty string for distance-only GMM.')
    parser.add_argument('--train_envs', type=str, default=None,
                        help='Comma-separated list of environment names for training (e.g., environment0,environment1). If None, uses standard per-environment training.')
    parser.add_argument('--test_env', type=str, default=None,
                        help='Single environment name for testing. If None, uses standard per-environment testing.')
    parser.add_argument('--experiment_tag', type=str, default="default",
                        help='A tag to identify this specific experiment run in output filenames.')
    cli_args = parser.parse_args()
    main(cli_args)