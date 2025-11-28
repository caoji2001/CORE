import os
import random
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    args = parser.parse_args()

    dataset = args.dataset
    seed = 0
    random.seed(seed)

    geo = pd.read_csv(f'./{dataset}/traj/roadmap.geo')
    traj = pd.read_csv(f'./{dataset}/traj/trajectory_train.csv')

    traj_aug = traj.copy()

    for i in tqdm(range(len(traj_aug))):
        path = eval(traj_aug.at[i, 'path'])
        timestamp = eval(traj_aug.at[i, 'timestamp'])

        assert len(path) == len(timestamp)

        n = len(path)
        delete_front = random.random() < 0.5
        remove_ratio = random.uniform(0.05, 0.15)

        k = round(n * remove_ratio)
        k = max(1, min(k, n - 1))

        if delete_front:
            traj_aug.at[i, 'path'] = path[k:]
            traj_aug.at[i, 'timestamp'] = timestamp[k:]
        else:
            traj_aug.at[i, 'path'] = path[:-k]
            traj_aug.at[i, 'timestamp'] = timestamp[:-k]

    os.makedirs(f'./{dataset}/out', exist_ok=True)
    traj_aug.to_csv(f'./{dataset}/out/trajectory_trim.csv', index=False)

    num_roads = len(geo)
    travel_time_list = [[] for _ in range(num_roads)]
    for i in tqdm(range(len(traj))):
        path = eval(traj.at[i, 'path'])
        timestamp = eval(traj.at[i, 'timestamp'])

        assert len(path) == len(timestamp)

        n = len(path)

        for j in range(1, n):
            prev_road, next_road = path[j-1], path[j]
            prev_time, next_time = timestamp[j-1], timestamp[j]

            travel_time_list[prev_road].append(next_time - prev_time)

    for i in range(len(travel_time_list)):
        travel_time_list[i] = np.mean(travel_time_list[i]) if len(travel_time_list[i]) > 0 else None

    traj_aug = traj.copy()
    for i in tqdm(range(len(traj_aug))):
        path = eval(traj_aug.at[i, 'path'])
        timestamp = eval(traj_aug.at[i, 'timestamp'])

        assert len(path) == len(timestamp)

        n = len(path)

        time_interval = np.array(timestamp)[1:] - np.array(timestamp)[:-1]
        for j in range(n - 1):
            time_shift = random.random() < 0.15
            shift_ratio = random.uniform(0.15, 0.3)

            if time_shift:
                time_interval[j] = time_interval[j] - (time_interval[j] - travel_time_list[path[j]]) * shift_ratio

        time_interval = np.cumsum(time_interval)

        for j in range(n - 1):
            timestamp[j + 1] = round(timestamp[0] + time_interval[j])

        traj_aug.at[i, 'timestamp'] = timestamp

    traj_aug.to_csv(f'./{dataset}/out/trajectory_shift.csv', index=False)
