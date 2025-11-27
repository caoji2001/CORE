import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    args = parser.parse_args()

    dataset = args.dataset

    geo = pd.read_csv(f'./{dataset}/traj/roadmap.geo')
    rel = pd.read_csv(f'./{dataset}/traj/roadmap.rel')
    traj_train = pd.read_csv(f'./{dataset}/traj/trajectory_train.csv')
    traj_train['path'] = traj_train['path'].apply(eval)

    num_roads = len(geo)
    freq = np.zeros(num_roads, dtype=np.int64)

    for _, row in tqdm(traj_train.iterrows(), total=len(traj_train), desc='[Counting road frequencies]'):
        path = row['path']
        for rid in path:
            freq[rid] += 1

    os.makedirs(f'./{dataset}/out', exist_ok=True)
    np.save(f'./{dataset}/out/road_freq.npy', freq)
