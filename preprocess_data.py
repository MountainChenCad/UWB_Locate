import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def parse_pos_string_to_array(pos_str):
    """Converts 'x_y_z' string to numpy array [x, y, z]"""
    return np.array([float(v) for v in pos_str.split('_')])


def calculate_true_range(tag_pos_arr, anchor_pos_arr):
    """Calculates Euclidean distance between two 3D points."""
    return np.linalg.norm(tag_pos_arr - anchor_pos_arr)


def preprocess_environment_data(env_name, base_data_path, output_train_path, output_test_path, train_ratio=0.8):
    print(f"Processing environment: {env_name}...")

    anchor_file = os.path.join(base_data_path, env_name, 'anchors.csv')
    data_json_file = os.path.join(base_data_path, env_name, 'data.json')
    # walking_path_file = os.path.join(base_data_path, env_name, 'walking_path.csv') # Not directly used for this CSV structure

    if not (os.path.exists(anchor_file) and os.path.exists(data_json_file)):
        print(f"Skipping {env_name}: Missing anchors.csv or data.json")
        return

    # Load anchor positions
    anchors_df = pd.read_csv(anchor_file)
    anchor_coords_map = {}  # Store as 'anchor_id': 'x_y_z' string
    anchor_coords_np_map = {}  # Store as 'anchor_id': np.array([x,y,z])
    for _, row in anchors_df.iterrows():
        anchor_id = row['Anchor']
        pos_str = f"{row['x']}_{row['y']}_{row['z']}"
        anchor_coords_map[anchor_id] = pos_str
        anchor_coords_np_map[anchor_id] = np.array([row['x'], row['y'], row['z']])

    # Load main measurement data
    with open(data_json_file, 'r') as f:
        data = json.load(f)

    all_records = []
    agent_pos_names_in_data = []  # To keep track of agent_pos that have measurements

    # Extract tag positions that actually have measurement data
    # data['path'] contains all possible tag positions on the path
    # data['measurements'] keys are the tag_pos_names that have data

    # Use tag positions from data['path'] to maintain order for splitting
    path_positions_info = {p_info['name']: p_info for p_info in data['path']}

    # Filter path_positions to only include those present in measurements
    # and maintain their original order from data['path']
    valid_agent_pos_names = [p_info['name'] for p_info in data['path'] if p_info['name'] in data['measurements']]

    for agent_pos_name in valid_agent_pos_names:
        agent_pos_info = path_positions_info[agent_pos_name]
        agent_pos_true_np = np.array(
            [float(agent_pos_info['x']), float(agent_pos_info['y']), float(agent_pos_info['z'])])

        if agent_pos_name not in data['measurements']:
            continue

        agent_pos_measurements = data['measurements'][agent_pos_name]

        for anchor_id, anchor_data in agent_pos_measurements.items():
            if anchor_id not in anchor_coords_map:
                # print(f"Warning: Anchor {anchor_id} not in anchors.csv for {agent_pos_name}. Skipping.")
                continue
            anchor_pos_str = anchor_coords_map[anchor_id]
            anchor_pos_np = anchor_coords_np_map[anchor_id]

            true_dist = calculate_true_range(agent_pos_true_np, anchor_pos_np)

            for channel_id, measurement_list in anchor_data.items():
                # Assuming measurement_list is indeed a list of measurement dicts
                # If it's a single dict, this loop will run once.
                # Based on user's description, it seems to be a list.
                # If data.json for `channel_id` key has a dict of measurements directly,
                # then `measurement_list` should be `[anchor_data[channel_id]]`
                # Let's assume it's a list as per Bregar dataset structure for multiple readings.
                # If not, and it's a single dict: measurements_to_process = [measurement_list]

                # The provided data.json seems to have a single measurement dict per channel, not a list.
                # Example: "ch1": {"range": 7.2, "cir": [...]}
                # So, measurement_list is actually a single measurement_item_dict
                # We'll treat it as a list of one item for generality, but it's likely one.

                measurements_to_process = []
                if isinstance(measurement_list, list):  # Standard Bregar format
                    measurements_to_process = measurement_list
                elif isinstance(measurement_list, dict):  # If it's a single dict
                    measurements_to_process = [measurement_list]
                else:
                    print(
                        f"Warning: Unexpected measurement format for {agent_pos_name}, {anchor_id}, {channel_id}. Skipping.")
                    continue

                for measurement_item in measurements_to_process:
                    record = {
                        'agent_pos': agent_pos_name,
                        'anchor': anchor_id,
                        'anchor_pos': anchor_pos_str,
                        'true_range': true_dist,
                        'error': measurement_item.get('range', np.nan) - true_dist,
                        # Add all other fields from measurement_item
                        'range': measurement_item.get('range'),
                        'nlos': measurement_item.get('nlos'),
                        'rss': measurement_item.get('rss'),
                        'rss_fp': measurement_item.get('rss_fp'),
                        'fp_index': measurement_item.get('fp_index'),
                        'fp_point1': measurement_item.get('fp_point1'),
                        'fp_point2': measurement_item.get('fp_point2'),
                        'fp_point3': measurement_item.get('fp_point3'),
                        'stdev_noise': measurement_item.get('stdev_noise'),
                        'cir_power': measurement_item.get('cir_power'),
                        'max_noise': measurement_item.get('max_noise'),
                        'rxpacc': measurement_item.get('rxpacc'),
                        'channel_number': measurement_item.get('channel_number', channel_id),  # use from item or parsed
                        # 'frame_length': measurement_item.get('frame_length'), # Add if present
                        # 'preamble_length': measurement_item.get('preamble_length'), # Add if present
                        # 'bitrate': measurement_item.get('bitrate'), # Add if present
                        # 'prfr': measurement_item.get('prfr'), # Add if present
                        # 'preamble_code': measurement_item.get('preamble_code'), # Add if present
                        'cir': str(measurement_item.get('cir'))  # Store as string representation of list
                    }
                    all_records.append(record)

    if not all_records:
        print(f"No records generated for {env_name}. Skipping split and save.")
        return

    # Create a DataFrame from all records to facilitate splitting by agent_pos
    full_df = pd.DataFrame(all_records)

    # Get unique agent_pos in the order they appeared (which is path order)
    # unique_agent_pos_in_df = full_df['agent_pos'].unique() # This might lose original path order
    # Use valid_agent_pos_names which respects path order

    if not valid_agent_pos_names:
        print(f"No valid agent positions with measurements found for {env_name}. Cannot split.")
        return

    # Split unique agent_pos names for train/test
    # Ensure that even if some agent_pos had no measurements, the split is based on those that did
    train_agent_pos, test_agent_pos = train_test_split(valid_agent_pos_names, train_size=train_ratio, shuffle=False,
                                                       random_state=42)  # No shuffle to keep path order

    train_df = full_df[full_df['agent_pos'].isin(train_agent_pos)]
    test_df = full_df[full_df['agent_pos'].isin(test_agent_pos)]

    # Save to CSV
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)

    train_df.to_csv(output_train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(output_test_path, index=False, encoding='utf-8-sig')
    print(f"Finished {env_name}. Train data: {len(train_df)} rows. Test data: {len(test_df)} rows.")


if __name__ == '__main__':
    base_raw_data_dir = '.'  # Assuming environment0, environment1 etc. are in the current dir
    output_dataset_dir = os.path.join('.', 'data_set')  # Output to ./data_set/

    environments = ['environment0', 'environment1', 'environment2', 'environment3']

    for env in environments:
        train_csv_path = os.path.join(output_dataset_dir, 'train_mean', f"{env}.csv")
        test_csv_path = os.path.join(output_dataset_dir, 'test_data_mean', f"{env}.csv")
        preprocess_environment_data(env, base_raw_data_dir, train_csv_path, test_csv_path)

    print("Preprocessing complete.")
