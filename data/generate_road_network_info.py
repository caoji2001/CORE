import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def convert_highway(value):
    try:
        lst = eval(value)
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
    except Exception:
        pass
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    args = parser.parse_args()

    dataset = args.dataset

    geo = pd.read_csv(f'./{dataset}/traj/roadmap.geo')
    rel = pd.read_csv(f'./{dataset}/traj/roadmap.rel')
    traj = pd.read_csv(f'./{dataset}/traj/trajectory_train.csv')

    edge_index = rel[['origin_id', 'destination_id']].to_numpy().T
    assert np.all(edge_index[0] != edge_index[1])

    trans_counts = np.zeros((len(geo), len(geo)), dtype=np.float32)
    for _, row in tqdm(traj.iterrows(), total=len(traj)):
        path = eval(row['path'])
        for i in range(1, len(path)):
            trans_counts[path[i-1], path[i]] += 1

    row_sums = trans_counts.sum(axis=1)
    for i in range(len(geo)):
        if row_sums[i] > 0:
            trans_counts[i] = trans_counts[i] / row_sums[i]
        else:
            k = np.sum(edge_index[0] == i)
            if k > 0:
                trans_counts[i, edge_index[1][edge_index[0] == i]] = 1.0 / k
            else:
                trans_counts[i] = 0

    trans_prob = trans_counts[edge_index[0], edge_index[1]]
    trans_prob = trans_prob.astype(np.float32)

    geo['highway'] = geo['highway'].apply(convert_highway)
    type_encoder = LabelEncoder()
    geo['highway'] = type_encoder.fit_transform(geo['highway'])
    road_type = geo['highway'].to_numpy()

    road_length = geo['length'].to_numpy()
    road_length_norm = np.log1p(road_length)
    road_length_norm = (road_length_norm - np.mean(road_length_norm)) / np.std(road_length_norm)
    road_length_norm = road_length_norm.astype(np.float32)

    road_out_degree = np.bincount(edge_index[0], minlength=len(geo))
    road_in_degree = np.bincount(edge_index[1], minlength=len(geo))

    os.makedirs(f'./{dataset}/out', exist_ok=True)
    np.save(f'./{dataset}/out/edge_index.npy', edge_index)
    np.save(f'./{dataset}/out/trans_prob.npy', trans_prob)
    np.save(f'./{dataset}/out/road_type.npy', road_type)
    np.save(f'./{dataset}/out/road_length_norm.npy', road_length_norm)
    np.save(f'./{dataset}/out/road_out_degree.npy', road_out_degree)
    np.save(f'./{dataset}/out/road_in_degree.npy', road_in_degree)
