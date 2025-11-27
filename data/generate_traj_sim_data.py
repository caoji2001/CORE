import os
import argparse
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx


def get_path_length(G, path, weight):
    length = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        length += G[u][v].get(weight)

    return length


def detour_path(G, original_path, weight, k=10):
    if len(original_path) < 2:
        return original_path
    
    source = original_path[0]
    target = original_path[-1]
    original_length = get_path_length(G, original_path, weight)

    try:
        generator = nx.shortest_simple_paths(G, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return original_path

    selected_path = None
    count = 0
    try:
        while count < k:
            path = next(generator)
            current_length = get_path_length(G, path, weight)
            if current_length > original_length:
                selected_path = path
                break
            count += 1
    except StopIteration:
        pass

    return selected_path if selected_path is not None else original_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    args = parser.parse_args()

    dataset = args.dataset
    seed = 0
    random.seed(seed)

    geo = pd.read_csv(f'./{dataset}/traj/roadmap.geo')
    rel = pd.read_csv(f'./{dataset}/traj/roadmap.rel')
    traj_test = pd.read_csv(f'./{dataset}/traj/trajectory_test.csv')
    traj_test['path'] = traj_test['path'].apply(eval)
    traj_test['timestamp'] = traj_test['timestamp'].apply(eval)

    num_roads = len(geo)

    G = nx.DiGraph()
    G.add_nodes_from(range(num_roads))
    for _, row in rel.iterrows():
        origin_id = row['origin_id']
        destination_id = row['destination_id']

        G.add_edge(origin_id, destination_id, weight=geo.at[origin_id, 'length']/2+geo.at[destination_id, 'length']/2)

    detoured_path_list = []
    detoured_timestamp_list = []
    not_detoured_path_list = []
    not_detoured_timestamp_list = []
    other_path_list = []
    other_timestamp_list = []
    for _, row in tqdm(traj_test.iterrows(), total=len(traj_test), desc='[Preparing detoured trajectories]'):
        path = row['path']
        timestamp = row['timestamp']
        path_len = len(path)
        
        mask_len = int(0.2 * path_len)
        mask_len = max(mask_len, 2)

        start_mask = random.randint(0, path_len-mask_len)

        detoured = detour_path(G, path[start_mask:start_mask+mask_len], 'weight')
        if len(detoured_path_list) == 5000 or (len(detoured) == len(path[start_mask:start_mask+mask_len]) and np.all(detoured == path[start_mask:start_mask+mask_len])):
            if len(other_path_list) < 50000:
                raw_path = np.copy(path)
                other_path_list.append(raw_path.tolist())
                raw_timestamp = np.copy(timestamp)
                other_timestamp_list.append(raw_timestamp.tolist())
        else:
            total_distance = np.sum([geo.at[rid, 'length'] for rid in path])
            total_time = timestamp[-1] - timestamp[0]
            average_speed = total_distance / total_time
            average_speed = np.clip(average_speed, 0.0, 80.0/3.6)

            raw_path = np.copy(path)
            raw_timestamp = np.copy(timestamp)
            not_detoured_path_list.append(raw_path.tolist())
            not_detoured_timestamp_list.append(raw_timestamp.tolist())

            detoured_path = np.copy(path)
            detoured_path = detoured_path[:start_mask].tolist() + detoured + detoured_path[start_mask+mask_len:].tolist()
            detoured_path = np.array(detoured_path)
            detoured_timestamp = np.array([round(geo.at[rid, 'length']/average_speed) for rid in detoured_path], dtype=np.int64)
            detoured_timestamp = np.insert(detoured_timestamp, 0, np.int64(0))
            detoured_timestamp = np.cumsum(detoured_timestamp)
            detoured_timestamp = detoured_timestamp[:-1]
            detoured_timestamp += raw_timestamp[0]

            detoured_path_list.append(detoured_path.tolist())
            detoured_timestamp_list.append(detoured_timestamp.tolist())

    assert len(detoured_path_list) == len(not_detoured_path_list) == 5000
    assert len(detoured_timestamp_list) == len(not_detoured_timestamp_list) == 5000
    assert len(other_path_list) == 50000
    assert len(other_timestamp_list) == 50000

    detoured_traj_df = pd.DataFrame({
        'path': detoured_path_list,
        'timestamp': detoured_timestamp_list,
    })
    not_detoured_traj_df = pd.DataFrame({
        'path': not_detoured_path_list,
        'timestamp': not_detoured_timestamp_list
    })
    other_traj_df = pd.DataFrame({
        'path': other_path_list,
        'timestamp': other_timestamp_list
    })

    print(f"Before detoured path length: {np.mean([get_path_length(G, row['path'], 'weight') for _, row in not_detoured_traj_df.iterrows()])}")
    print(f"After detoured path length: {np.mean([get_path_length(G, row['path'], 'weight') for _, row in detoured_traj_df.iterrows()])}")

    os.makedirs(f'./{dataset}/out', exist_ok=True)
    detoured_traj_df.to_csv(f'./{dataset}/out/detoured_traj.csv', index=False)
    not_detoured_traj_df.to_csv(f'./{dataset}/out/not_detoured_traj.csv', index=False)
    other_traj_df.to_csv(f'./{dataset}/out/other_traj.csv', index=False)
