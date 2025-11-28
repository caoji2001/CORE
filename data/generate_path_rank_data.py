import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx


class TrajGenerator:
    def __init__(self, geo, rel):
        self.G = nx.DiGraph()
        self.road_length = geo['length'].tolist()

        self.G.add_nodes_from(range(len(geo)))
        for _, row in rel.iterrows():
            origin_id = row['origin_id']
            destination_id = row['destination_id']

            self.G.add_edge(origin_id, destination_id, weight=self.road_length[origin_id]/2+self.road_length[destination_id]/2)

    def get_similarity(self, path1, path2):
        intersection_list = list(set(path1) & set(path2))
        union_list = list(set(path1) | set(path2))
        return sum([self.road_length[x] for x in intersection_list]) / sum([self.road_length[x] for x in union_list])
    
    def get_competitive_path(self, raw_path, raw_timestamp):
        origin, destination = raw_path[0], raw_path[-1]
        start_timestamp = raw_timestamp[0]

        average_speed = sum([self.road_length[x] for x in raw_path]) / (raw_timestamp[-1] - raw_timestamp[0])

        path_list = [raw_path]
        timestamp_list = [raw_timestamp]
        score_list = [1.0]
        for path_idx, another_path in enumerate(nx.shortest_simple_paths(self.G, origin, destination, weight='weight')):
            if path_idx == 9:
                break
            mx_score = 0
            for path in path_list:
                mx_score = max(mx_score, self.get_similarity(another_path, path))
            if mx_score <= 0.8:
                path_list.append(another_path)

                timestamp = np.array([self.road_length[x]/average_speed for x in another_path])
                timestamp = np.insert(timestamp, 0, np.int64(0))
                timestamp = np.cumsum(timestamp)
                timestamp = np.round(timestamp).astype(np.int64)
                timestamp += start_timestamp
                timestamp = timestamp[:-1].tolist()
                timestamp_list.append(timestamp)

                score_list.append(self.get_similarity(another_path, raw_path))

        return path_list, timestamp_list, score_list


def init_worker(geo, rel):
    global traj_generator
    traj_generator = TrajGenerator(geo, rel)


def process_row(item):
    idx, row = item
    raw_path = row['path']
    raw_timestamp = row['timestamp']
    path_list, timestamp_list, score_list = traj_generator.get_competitive_path(raw_path, raw_timestamp)
    return path_list, timestamp_list, score_list, [idx] * len(path_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    parser.add_argument('--processes', type=int, default=80)
    args = parser.parse_args()

    dataset = args.dataset
    processes = args.processes

    geo = pd.read_csv(f'./{dataset}/traj/roadmap.geo')
    rel = pd.read_csv(f'./{dataset}/traj/roadmap.rel')
    traj_train = pd.read_csv(f'./{dataset}/traj/trajectory_train.csv')
    traj_train['path'] = traj_train['path'].apply(eval)
    traj_train['timestamp'] = traj_train['timestamp'].apply(eval)
    traj_val = pd.read_csv(f'./{dataset}/traj/trajectory_val.csv')
    traj_val['path'] = traj_val['path'].apply(eval)
    traj_val['timestamp'] = traj_val['timestamp'].apply(eval)
    traj_test = pd.read_csv(f'./{dataset}/traj/trajectory_test.csv')
    traj_test['path'] = traj_test['path'].apply(eval)
    traj_test['timestamp'] = traj_test['timestamp'].apply(eval)

    path_train = []
    timestamp_train = []
    score_train = []
    group_train = []

    with mp.Pool(processes=args.processes, initializer=init_worker, initargs=(geo, rel)) as pool:
        for paths, timestamps, scores, groups in tqdm(pool.imap_unordered(process_row, traj_train.iterrows()), total=len(traj_train), desc='[Generating train data]'):
            path_train.extend(paths)
            timestamp_train.extend(timestamps)
            score_train.extend(scores)
            group_train.extend(groups)

    df_train = pd.DataFrame({
        'path': path_train,
        'timestamp': timestamp_train,
        'score': score_train,
        'group': group_train
    })

    path_val = []
    timestamp_val = []
    score_val = []
    group_val = []

    with mp.Pool(processes=args.processes, initializer=init_worker, initargs=(geo, rel)) as pool:
        for paths, timestamps, scores, groups in tqdm(pool.imap_unordered(process_row, traj_val.iterrows()), total=len(traj_val), desc='[Generating val data]'):
            path_val.extend(paths)
            timestamp_val.extend(timestamps)
            score_val.extend(scores)
            group_val.extend(groups)

    df_val = pd.DataFrame({
        'path': path_val,
        'timestamp': timestamp_val,
        'score': score_val,
        'group': group_val
    })

    path_test = []
    timestamp_test = []
    score_test = []
    group_test = []

    with mp.Pool(processes=args.processes, initializer=init_worker, initargs=(geo, rel)) as pool:
        for paths, timestamps, scores, groups in tqdm(pool.imap_unordered(process_row, traj_test.iterrows()), total=len(traj_test), desc='[Generating test data]'):
            path_test.extend(paths)
            timestamp_test.extend(timestamps)
            score_test.extend(scores)
            group_test.extend(groups)

    df_test = pd.DataFrame({
        'path': path_test,
        'timestamp': timestamp_test,
        'score': score_test,
        'group': group_test
    })

    os.makedirs(f'./{dataset}/out', exist_ok=True)
    df_train.to_csv(f'./{dataset}/out/path_rank_train.csv', index=False)
    df_val.to_csv(f'./{dataset}/out/path_rank_val.csv', index=False)
    df_test.to_csv(f'./{dataset}/out/path_rank_test.csv', index=False)
