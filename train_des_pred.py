import os
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import multiprocessing as mp
from tqdm import tqdm
from shapely import LineString
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import pyproj
import torch
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from utils import set_seed, dict_to_namespace
from llm_script.grid_space import GridSpace
from model.core import Core
from model.des_pred import DesPred


_geo_dict = None
_trans_prob_mat = None
_road_id2grid_id = None
_road_id_padding_idx = None
_grid_id_padding_id = None

def init(geo_dict, trans_prob_mat, road_id2grid_id, road_id_padding_idx, grid_id_padding_id):
    global _geo_dict, _trans_prob_mat, _road_id2grid_id, _road_id_padding_idx, _grid_id_padding_id

    _geo_dict = geo_dict
    _trans_prob_mat = trans_prob_mat
    _road_id2grid_id = road_id2grid_id
    _road_id_padding_idx = road_id_padding_idx
    _grid_id_padding_id = grid_id_padding_id

def process_row(args):
    path, timestamp, label = args

    global _geo_dict, _trans_prob_mat, _road_id2grid_id, _road_id_padding_idx, _grid_id_padding_id
    geodesic = pyproj.Geod(ellps='WGS84')

    traj_road_id_data = np.array(path, dtype=np.int64)
    traj_grid_id_data = np.array([_road_id2grid_id[road_id] for road_id in path], dtype=np.int64)

    adj_road_id_data = []
    adj_grid_id_data = []
    route_choice_travel_progress_data = []
    route_choice_angle_data = []
    route_choice_trans_prob_data = []
    route_choice_selected_mask_data = []
    route_choice_unselected_mask_data = []

    des_coords = LineString(_geo_dict[path[-1]]['coordinates'])
    des_centroid = [des_coords.centroid.x, des_coords.centroid.y]

    cur_distance = 0
    total_distance = sum([_geo_dict[road_id]['length'] for road_id in path])

    for i in range(len(path)):
        cur_road_id = path[i]
        next_road_id = path[i+1] if i+1 < len(path) else -1
        cur_distance += _geo_dict[cur_road_id]['length']

        adj_road_id_list = _trans_prob_mat[cur_road_id].indices

        adj_road_id_data_tmp = []
        adj_grid_id_data_tmp = []
        route_choice_angle_data_tmp = []
        route_choice_trans_prob_data_tmp = []
        route_choice_selected_mask_data_tmp = []
        route_choice_unselected_mask_data_tmp = []

        for adj_road_id in adj_road_id_list:
            adj_road_id_data_tmp.append(adj_road_id)
            adj_grid_id_data_tmp.append(_road_id2grid_id[adj_road_id])

            adj_road_coords = _geo_dict[adj_road_id]['coordinates']
            adj_road_org = adj_road_coords[0]
            adj_road_des = adj_road_coords[-1]

            road_azimuth = geodesic.inv(adj_road_org[0], adj_road_org[1], adj_road_des[0], adj_road_des[1])[0]
            des_azimuth = geodesic.inv(adj_road_org[0], adj_road_org[1], des_centroid[0], des_centroid[1])[0]
            angle = ((des_azimuth - road_azimuth + 180) % 360 - 180) / 180

            route_choice_angle_data_tmp.append(angle)
            route_choice_trans_prob_data_tmp.append(_trans_prob_mat[cur_road_id, adj_road_id])

            if next_road_id == -1:
                route_choice_selected_mask_data_tmp.append(False)
                route_choice_unselected_mask_data_tmp.append(False)
            else:
                route_choice_selected_mask_data_tmp.append(adj_road_id == next_road_id)
                route_choice_unselected_mask_data_tmp.append(adj_road_id != next_road_id)

        adj_road_id_data.append(np.array(adj_road_id_data_tmp, dtype=np.int64))
        adj_grid_id_data.append(np.array(adj_grid_id_data_tmp, dtype=np.int64))
        route_choice_travel_progress_data.append(cur_distance / total_distance)
        route_choice_angle_data.append(np.array(route_choice_angle_data_tmp, dtype=np.float32))
        route_choice_trans_prob_data.append(np.array(route_choice_trans_prob_data_tmp, dtype=np.float32))
        route_choice_selected_mask_data.append(np.array(route_choice_selected_mask_data_tmp, dtype=np.bool_))
        route_choice_unselected_mask_data.append(np.array(route_choice_unselected_mask_data_tmp, dtype=np.bool_))

    max_adj_road_num = max([len(x) for x in adj_road_id_data])
    for i in range(len(path)):
        pad_num = max_adj_road_num-len(adj_road_id_data[i])

        adj_road_id_data[i] = np.pad(adj_road_id_data[i], (0, pad_num), 'constant', constant_values=_road_id_padding_idx)
        adj_grid_id_data[i] = np.pad(adj_grid_id_data[i], (0, pad_num), 'constant', constant_values=_grid_id_padding_id)
        route_choice_angle_data[i] = np.pad(route_choice_angle_data[i], (0, pad_num), 'constant', constant_values=np.float32(0.0))
        route_choice_trans_prob_data[i] = np.pad(route_choice_trans_prob_data[i], (0, pad_num), 'constant', constant_values=np.float32(0.0))
        route_choice_selected_mask_data[i] = np.pad(route_choice_selected_mask_data[i], (0, pad_num), 'constant', constant_values=np.bool_(False))
        route_choice_unselected_mask_data[i] = np.pad(route_choice_unselected_mask_data[i], (0, pad_num), 'constant', constant_values=np.bool_(False))

    adj_road_id_data = np.stack(adj_road_id_data, axis=0)
    adj_grid_id_data = np.stack(adj_grid_id_data, axis=0)
    route_choice_travel_progress_data = np.array(route_choice_travel_progress_data, dtype=np.float32)
    route_choice_angle_data = np.stack(route_choice_angle_data, axis=0)
    route_choice_trans_prob_data = np.stack(route_choice_trans_prob_data, axis=0)
    route_choice_selected_mask_data = np.stack(route_choice_selected_mask_data, axis=0)
    route_choice_unselected_mask_data = np.stack(route_choice_unselected_mask_data, axis=0)

    weekday_data = np.array([t.weekday() for t in timestamp], dtype=np.int64)
    time_of_day_data = np.array([t.hour*60+t.minute for t in timestamp], dtype=np.int64)
    traj_len_data = np.int64(len(path))
    label_data = np.int64(label)

    return (
        traj_road_id_data,
        traj_grid_id_data,
        adj_road_id_data,
        adj_grid_id_data,
        route_choice_travel_progress_data,
        route_choice_angle_data,
        route_choice_trans_prob_data,
        route_choice_selected_mask_data,
        route_choice_unselected_mask_data,
        weekday_data,
        time_of_day_data,
        traj_len_data,
        label_data,
    )


class TrajDataset(Dataset):
    def __init__(self, geo_dict, trans_prob_mat, road_id2grid_id, road_id_padding_idx, grid_id_padding_idx, traj):
        self.data = {
            'traj_road_id': [],
            'traj_grid_id': [],
            'adj_road_id': [],
            'adj_grid_id': [],
            'route_choice_travel_progress': [],
            'route_choice_angle': [],
            'route_choice_trans_prob': [],
            'route_choice_selected_mask': [],
            'route_choice_unselected_mask': [],
            'weekday': [],
            'time_of_day': [],
            'traj_len': [],
            'label': [],
        }

        tasks = [(traj.at[i, 'path'], traj.at[i, 'timestamp'], traj.at[i, 'label']) for i in range(len(traj))]
        with mp.Pool(processes=32, initializer=init, initargs=(geo_dict, trans_prob_mat, road_id2grid_id, road_id_padding_idx, grid_id_padding_idx)) as pool:
            results = list(tqdm(
                pool.imap(process_row, tasks),
                total=len(tasks),
                desc='[Loading data]'
            ))

        for result in results:
            self.data['traj_road_id'].append(result[0])
            self.data['traj_grid_id'].append(result[1])
            self.data['adj_road_id'].append(result[2])
            self.data['adj_grid_id'].append(result[3])
            self.data['route_choice_travel_progress'].append(result[4])
            self.data['route_choice_angle'].append(result[5])
            self.data['route_choice_trans_prob'].append(result[6])
            self.data['route_choice_selected_mask'].append(result[7])
            self.data['route_choice_unselected_mask'].append(result[8])
            self.data['weekday'].append(result[9])
            self.data['time_of_day'].append(result[10])
            self.data['traj_len'].append(result[11])
            self.data['label'].append(result[12])

    def __getitem__(self, index):
        return {
            'traj_road_id':  self.data['traj_road_id'][index],
            'traj_grid_id':  self.data['traj_grid_id'][index],
            'adj_road_id':  self.data['adj_road_id'][index],
            'adj_grid_id':  self.data['adj_grid_id'][index],
            'route_choice_travel_progress':  self.data['route_choice_travel_progress'][index],
            'route_choice_angle':  self.data['route_choice_angle'][index],
            'route_choice_trans_prob':  self.data['route_choice_trans_prob'][index],
            'route_choice_selected_mask':  self.data['route_choice_selected_mask'][index],
            'route_choice_unselected_mask':  self.data['route_choice_unselected_mask'][index],
            'weekday':  self.data['weekday'][index],
            'time_of_day':  self.data['time_of_day'][index],
            'traj_len':  self.data['traj_len'][index],
            'label': self.data['label'][index],
        }
    
    def __len__(self):
        return len(self.data['traj_road_id'])


class MyCollateFn:
    def __init__(self, road_id_padding_idx, grid_id_padding_idx, weekday_padding_idx, time_of_day_padding_idx):
        self.road_id_padding_idx = road_id_padding_idx
        self.grid_id_padding_idx = grid_id_padding_idx
        self.weekday_padding_idx = weekday_padding_idx
        self.time_of_day_padding_idx = time_of_day_padding_idx

    def __call__(self, items):
        batch = {
            'traj_road_id': [],
            'traj_grid_id': [],
            'adj_road_id': [],
            'adj_grid_id': [],
            'route_choice_travel_progress': [],
            'route_choice_angle': [],
            'route_choice_trans_prob': [],
            'route_choice_selected_mask': [],
            'route_choice_unselected_mask': [],
            'weekday': [],
            'time_of_day': [],
            'traj_len': [],
            'label': []
        }

        for item in items:
            batch['traj_road_id'].append(item['traj_road_id'])
            batch['traj_grid_id'].append(item['traj_grid_id'])
            batch['adj_road_id'].append(item['adj_road_id'])
            batch['adj_grid_id'].append(item['adj_grid_id'])
            batch['route_choice_travel_progress'].append(item['route_choice_travel_progress'])
            batch['route_choice_angle'].append(item['route_choice_angle'])
            batch['route_choice_trans_prob'].append(item['route_choice_trans_prob'])
            batch['route_choice_selected_mask'].append(item['route_choice_selected_mask'])
            batch['route_choice_unselected_mask'].append(item['route_choice_unselected_mask'])
            batch['weekday'].append(item['weekday'])
            batch['time_of_day'].append(item['time_of_day'])
            batch['traj_len'].append(item['traj_len'])
            batch['label'].append(item['label'])

        max_traj_len = max(batch['traj_len'])
        max_adj_road_num = max([x.shape[1] for x in batch['adj_road_id']])
        for i in range(len(batch['traj_road_id'])):
            traj_pad_num = max_traj_len - batch['traj_len'][i]
            adj_road_pad_num = max_adj_road_num - batch['adj_road_id'][i].shape[1]

            batch['traj_road_id'][i] = np.pad(batch['traj_road_id'][i], (0, traj_pad_num), 'constant', constant_values=self.road_id_padding_idx)
            batch['traj_grid_id'][i] = np.pad(batch['traj_grid_id'][i], (0, traj_pad_num), 'constant', constant_values=self.grid_id_padding_idx)
            batch['adj_road_id'][i] = np.pad(batch['adj_road_id'][i], ((0, traj_pad_num), (0, adj_road_pad_num)), 'constant', constant_values=self.road_id_padding_idx)
            batch['adj_grid_id'][i] = np.pad(batch['adj_grid_id'][i], ((0, traj_pad_num), (0, adj_road_pad_num)), 'constant', constant_values=self.grid_id_padding_idx)
            batch['route_choice_travel_progress'][i] = np.pad(batch['route_choice_travel_progress'][i], (0, traj_pad_num), 'constant', constant_values=np.float32(0.0))
            batch['route_choice_angle'][i] = np.pad(batch['route_choice_angle'][i], ((0, traj_pad_num), (0, adj_road_pad_num)), 'constant', constant_values=np.float32(0.0))
            batch['route_choice_trans_prob'][i] = np.pad(batch['route_choice_trans_prob'][i], ((0, traj_pad_num), (0, adj_road_pad_num)), 'constant', constant_values=np.float32(0.0))
            batch['route_choice_selected_mask'][i] = np.pad(batch['route_choice_selected_mask'][i], ((0, traj_pad_num), (0, adj_road_pad_num)), 'constant', constant_values=np.bool_(False))
            batch['route_choice_unselected_mask'][i] = np.pad(batch['route_choice_unselected_mask'][i], ((0, traj_pad_num), (0, adj_road_pad_num)), 'constant', constant_values=np.bool_(False))
            batch['weekday'][i] = np.pad(batch['weekday'][i], (0, traj_pad_num), 'constant', constant_values=self.weekday_padding_idx)
            batch['time_of_day'][i] = np.pad(batch['time_of_day'][i], (0, traj_pad_num), 'constant', constant_values=self.time_of_day_padding_idx)

        batch['traj_road_id'] = torch.from_numpy(np.stack(batch['traj_road_id'], axis=0))
        batch['traj_grid_id'] = torch.from_numpy(np.stack(batch['traj_grid_id'], axis=0))
        batch['adj_road_id'] = torch.from_numpy(np.stack(batch['adj_road_id'], axis=0))
        batch['adj_grid_id'] = torch.from_numpy(np.stack(batch['adj_grid_id'], axis=0))
        batch['route_choice_travel_progress'] = torch.from_numpy(np.stack(batch['route_choice_travel_progress'], axis=0))
        batch['route_choice_angle'] = torch.from_numpy(np.stack(batch['route_choice_angle'], axis=0))
        batch['route_choice_trans_prob'] = torch.from_numpy(np.stack(batch['route_choice_trans_prob'], axis=0))
        batch['route_choice_selected_mask'] = torch.from_numpy(np.stack(batch['route_choice_selected_mask'], axis=0))
        batch['route_choice_unselected_mask'] = torch.from_numpy(np.stack(batch['route_choice_unselected_mask'], axis=0))
        batch['weekday'] = torch.from_numpy(np.stack(batch['weekday'], axis=0))
        batch['time_of_day'] = torch.from_numpy(np.stack(batch['time_of_day'], axis=0))
        batch['traj_len'] = torch.from_numpy(np.array(batch['traj_len']))
        batch['label'] = torch.from_numpy(np.array(batch['label']))

        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='core')
    parser.add_argument('--dataset', type=str, default='Beijing')
    parser.add_argument('--grid_size', type=int, default=1000)
    parser.add_argument('--road_poi_emb_file', type=str, default='road_poi_embedding_100.pt')
    parser.add_argument('--grid_poi_emb_file', type=str, default='grid_poi_embedding_1000_top10.pt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--betas', type=str, default='(0.9, 0.999)')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_warmup_epoch', type=int, default=5)
    parser.add_argument('--lr_min', type=float, default=0.000001)
    parser.add_argument('--hidden_dim', type=int, default=128)

    parser.add_argument('--pretrain_epoch', type=int, default=50)
    parser.add_argument('--road_poi_top_ratio', type=float, default=0.2)
    parser.add_argument('--moe_n_routed_experts', type=int, default=8)
    parser.add_argument('--moe_top_k', type=int, default=2)
    parser.add_argument('--moe_update_rate', type=float, default=0.001)
    args = parser.parse_args()

    device = f'cuda:{args.cuda}'
    set_seed(args.seed)

    geo = pd.read_csv(f'./data/{args.dataset}/traj/roadmap.geo')
    rel = pd.read_csv(f'./data/{args.dataset}/traj/roadmap.rel')
    train_traj = pd.read_csv(f'./data/{args.dataset}/traj/trajectory_train.csv')
    val_traj = pd.read_csv(f'./data/{args.dataset}/traj/trajectory_val.csv')
    test_traj = pd.read_csv(f'./data/{args.dataset}/traj/trajectory_test.csv')
    poi = pd.read_csv(f'./data/{args.dataset}/poi/poi.csv')

    if args.dataset == 'Porto':
        poi_cat_list = poi['category'].unique().tolist()
    else:
        poi_cat_list = ['购物服务', '餐饮服务', '公司企业', '生活服务', '交通设施服务', '科教文化服务',\
                        '商务住宅', '政府机构及社会团体', '金融保险服务', '医疗保健服务', '体育休闲服务',\
                        '住宿服务', '汽车服务', '风景名胜']
    poi_cat2idx = {name: idx for idx, name in enumerate(poi_cat_list)}

    edge_index = np.load(f'./data/{args.dataset}/out/edge_index.npy')
    trans_prob = np.load(f'./data/{args.dataset}/out/trans_prob.npy')

    geo['coordinates'] = geo['coordinates'].apply(eval)

    train_traj['path'] = train_traj['path'].apply(eval)
    train_traj['timestamp'] = train_traj['timestamp'].apply(eval)
    train_traj['label'] = train_traj['path'].apply(lambda x: x[-1])
    train_traj['path'] = train_traj['path'].apply(lambda x: x[:len(x)//2])
    train_traj['timestamp'] = train_traj['timestamp'].apply(lambda x: x[:len(x)//2])
    train_traj['timestamp'] = train_traj['timestamp'].apply(
        lambda lst: [datetime.fromtimestamp(t, ZoneInfo('Asia/Shanghai')) for t in lst]
    )

    val_traj['path'] = val_traj['path'].apply(eval)
    val_traj['timestamp'] = val_traj['timestamp'].apply(eval)
    val_traj['label'] = val_traj['path'].apply(lambda x: x[-1])
    val_traj['path'] = val_traj['path'].apply(lambda x: x[:len(x)//2])
    val_traj['timestamp'] = val_traj['timestamp'].apply(lambda x: x[:len(x)//2])
    val_traj['timestamp'] = val_traj['timestamp'].apply(
        lambda lst: [datetime.fromtimestamp(t, ZoneInfo('Asia/Shanghai')) for t in lst]
    )

    test_traj['path'] = test_traj['path'].apply(eval)
    test_traj['timestamp'] = test_traj['timestamp'].apply(eval)
    test_traj['label'] = test_traj['path'].apply(lambda x: x[-1])
    test_traj['path'] = test_traj['path'].apply(lambda x: x[:len(x)//2])
    test_traj['timestamp'] = test_traj['timestamp'].apply(lambda x: x[:len(x)//2])
    test_traj['timestamp'] = test_traj['timestamp'].apply(
        lambda lst: [datetime.fromtimestamp(t, ZoneInfo('Asia/Shanghai')) for t in lst]
    )

    geo_dict = {
        road_id: {
            'coordinates': geo.at[road_id, 'coordinates'],
            'length': geo.at[road_id, 'length']
        }
        for road_id in geo.index
    }

    trans_prob_mat = sp.coo_matrix(
        (trans_prob, (edge_index[0], edge_index[1])),
        shape=(len(geo), len(geo))
    ).tocsr()

    road_centroid_coordinates = []
    for _, row in geo.iterrows():
        coordinates = row['coordinates']
        line = LineString(coordinates)
        road_centroid_coordinates.append((line.centroid.x, line.centroid.y))
    road_centroid_coordinates = np.array(road_centroid_coordinates)

    poi_coordinates = poi[['lon_wgs84', 'lat_wgs84']].to_numpy()

    grid_space = GridSpace(road_centroid_coordinates, poi_coordinates, args.grid_size)

    road_id2grid_id = []
    for road_id in range(len(geo)):
        road_id2grid_id.append(grid_space.coordinate2grid_id(road_centroid_coordinates[road_id]))

    road_id_padding_idx = len(geo)
    grid_id_padding_idx = grid_space.num_total_grid
    weekday_padding_idx = 7
    time_of_day_padding_idx = 1440

    train_dataset = TrajDataset(geo_dict, trans_prob_mat, road_id2grid_id, road_id_padding_idx, grid_id_padding_idx, train_traj)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=MyCollateFn(road_id_padding_idx, grid_id_padding_idx, weekday_padding_idx, time_of_day_padding_idx),
        num_workers=8,
        pin_memory=True,
    )

    val_dataset = TrajDataset(geo_dict, trans_prob_mat, road_id2grid_id, road_id_padding_idx, grid_id_padding_idx, val_traj)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MyCollateFn(road_id_padding_idx, grid_id_padding_idx, weekday_padding_idx, time_of_day_padding_idx),
        num_workers=8,
        pin_memory=True,
    )

    test_dataset = TrajDataset(geo_dict, trans_prob_mat, road_id2grid_id, road_id_padding_idx, grid_id_padding_idx, test_traj)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MyCollateFn(road_id_padding_idx, grid_id_padding_idx, weekday_padding_idx, time_of_day_padding_idx),
        num_workers=8,
        pin_memory=True,
    )

    road_type = np.load(f'./data/{args.dataset}/out/road_type.npy')
    road_length_norm = np.load(f'./data/{args.dataset}/out/road_length_norm.npy')
    road_out_degree = np.load(f'./data/{args.dataset}/out/road_out_degree.npy')
    road_in_degree = np.load(f'./data/{args.dataset}/out/road_in_degree.npy')
    road_freq = np.load(f'./data/{args.dataset}/out/road_freq.npy')

    road_poi_hidden_states = torch.load(f'./llm_cache/{args.dataset}/{args.road_poi_emb_file}')
    road_poi_hidden_states[np.argsort(road_freq)[:int(len(road_freq) * (1.0-args.road_poi_top_ratio))]] = 0.0

    grid_poi_hidden_states = torch.load(f'./llm_cache/{args.dataset}/{args.grid_poi_emb_file}', weights_only=False)

    llm_dim = 128

    road_poi_hidden_states_list = road_poi_hidden_states[:, :llm_dim]
    road_poi_hidden_states_list = (road_poi_hidden_states_list - road_poi_hidden_states_list.mean(dim=-1, keepdim=True)) / (road_poi_hidden_states_list.std(dim=-1, keepdim=True) + 1e-5)

    grid_poi_hidden_states_list = torch.zeros((len(grid_poi_hidden_states), len(poi_cat_list), llm_dim), dtype=torch.float32)
    grid_poi_hidden_states_mask = torch.zeros((len(grid_poi_hidden_states), len(poi_cat_list)), dtype=torch.bool)
    for i, d in enumerate(grid_poi_hidden_states):
        for k, v in d.items():
            grid_poi_hidden_states_list[i, poi_cat2idx[k]] = v[:llm_dim]
            grid_poi_hidden_states_mask[i, poi_cat2idx[k]] = True
    grid_poi_hidden_states_list = (grid_poi_hidden_states_list - grid_poi_hidden_states_list.mean(dim=-1, keepdim=True)) / (grid_poi_hidden_states_list.std(dim=-1, keepdim=True) + 1e-5)

    core_config = dict_to_namespace({
        'embed_dim': args.hidden_dim,

        'road_type': torch.from_numpy(road_type),
        'road_length_norm': torch.from_numpy(road_length_norm),
        'road_out_degree': torch.from_numpy(road_out_degree),
        'road_in_degree': torch.from_numpy(road_in_degree),

        'n_type_embed': len(np.unique(road_type)),
        'n_out_degree_embed': len(np.unique(road_out_degree)),
        'n_in_degree_embed': len(np.unique(road_in_degree)),

        'road_poi_hidden_states': road_poi_hidden_states_list,
        'edge_index': torch.from_numpy(edge_index),

        'grid_poi_hidden_states': grid_poi_hidden_states_list,
        'grid_poi_mask': grid_poi_hidden_states_mask,

        'road_id_padding_idx': road_id_padding_idx,
        'grid_id_padding_idx': grid_id_padding_idx,

        'n_weekday_embed': weekday_padding_idx + 1,
        'weekday_padding_idx': weekday_padding_idx,
        'n_time_of_day_embed': time_of_day_padding_idx + 1,
        'time_of_day_padding_idx': time_of_day_padding_idx,

        'fine_poi_model_config': {
            'n_layers': 3,
            'embed_dim': args.hidden_dim,
            'n_heads': 4,
        },

        'coarse_poi_model_config': {
            'embed_dim': args.hidden_dim,
            'num_grid_x': grid_space.num_grid_x,
            'num_grid_y': grid_space.num_grid_y,
            'num_poi_cats': len(poi_cat_list),
        },

        'route_choice_model_config': {
            'embed_dim': args.hidden_dim,
            'moe_n_routed_experts': args.moe_n_routed_experts,
            'moe_top_k': args.moe_top_k,
            'moe_update_rate': args.moe_update_rate,
        },

        'bert_config': {
            'n_layer': 6,
            'max_len': 500,
            'embed_dim': args.hidden_dim,
            'n_head': 4,
            'dropout': 0.1,
        }
    })
    core = Core(core_config).to(device)
    core.load_state_dict(torch.load(f'./output/{args.exp_name}/save/{args.dataset}/pretrain/seed_{args.seed}/core_epoch_{args.pretrain_epoch}.pth', map_location=device))

    des_pred_config = dict_to_namespace({
        'hidden_dim': args.hidden_dim,
        'output_dim': len(geo)
    })
    des_pred = DesPred(des_pred_config).to(device)

    os.makedirs(f'./output/{args.exp_name}/log/{args.dataset}/des_pred/seed_{args.seed}', exist_ok=True)
    logger.remove()
    logger.add(os.path.join(f'./output/{args.exp_name}/log/{args.dataset}/des_pred/seed_{args.seed}', f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'), level='INFO', format='{time:YYYY-MM-DD HH:mm:ss} | {message}')

    logger.info(f'args: {args}')
    logger.info(f'core_config: {core_config}')

    os.makedirs(f'./output/{args.exp_name}/save/{args.dataset}/des_pred/seed_{args.seed}', exist_ok=True)

    optimizer = torch.optim.AdamW(list(core.parameters())+list(des_pred.parameters()), lr=args.lr, betas=eval(args.betas), weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epoch, warmup_t=args.lr_warmup_epoch, lr_min=args.lr_min)

    best_acc1 = 0.0
    for epoch_id in range(args.epoch):
        core.train()
        des_pred.train()

        scheduler.step(epoch_id+1)

        loss_array = []
        maxvio_array = []
        for batch in tqdm(train_dataloader, desc=f'[Training] epoch {epoch_id}'):
            for k in batch.keys():
                batch[k] = batch[k].to(device, non_blocking=True)

            traj_embedding, _, maxvio = core(
                batch['traj_road_id'],
                batch['traj_grid_id'],
                batch['adj_road_id'],
                batch['adj_grid_id'],
                batch['route_choice_travel_progress'],
                batch['route_choice_angle'],
                batch['route_choice_trans_prob'],
                batch['route_choice_selected_mask'],
                batch['route_choice_unselected_mask'],
                batch['weekday'],
                batch['time_of_day'],
                batch['traj_len'],
            )
            logits = des_pred(traj_embedding)

            loss = F.cross_entropy(logits, batch['label'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(core.parameters())+list(des_pred.parameters()), max_norm=1.0)
            optimizer.step()

            loss_array.append(loss.item())
            maxvio_array.append(maxvio.item())

        core.eval()
        des_pred.eval()

        acc1_cnt, acc5_cnt, acc10_cnt, total_cnt = 0, 0, 0, 0
        for batch in tqdm(val_dataloader, desc=f'[Validating] epoch {epoch_id}'):
            for k in batch.keys():
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch.no_grad():
                traj_embedding, _, _ = core(
                    batch['traj_road_id'],
                    batch['traj_grid_id'],
                    batch['adj_road_id'],
                    batch['adj_grid_id'],
                    batch['route_choice_travel_progress'],
                    batch['route_choice_angle'],
                    batch['route_choice_trans_prob'],
                    batch['route_choice_selected_mask'],
                    batch['route_choice_unselected_mask'],
                    batch['weekday'],
                    batch['time_of_day'],
                    batch['traj_len'],
                )
                logits = des_pred(traj_embedding)

            _, pred = logits.topk(10, dim=1, largest=True, sorted=True)
            target = batch['label'].view(-1, 1).expand_as(pred)

            correct = pred.eq(target)
            acc1_cnt += correct[:, :1].sum().item()
            acc5_cnt += correct[:, :5].sum().item()
            acc10_cnt += correct[:, :10].sum().item()
            total_cnt += len(target)

        acc1 = acc1_cnt / total_cnt
        acc5 = acc5_cnt / total_cnt
        acc10 = acc10_cnt / total_cnt

        logger.info(f'[Epoch {epoch_id + 1}] loss: {np.mean(loss_array)}, maxvio: {np.mean(maxvio_array)}, lr: {optimizer.param_groups[0]["lr"]}, Acc@1: {acc1}, Acc@5: {acc5}, Acc@10: {acc10}')
        if acc1 > best_acc1:
            best_acc1 = acc1
            logger.info(f'New best Acc@1: {best_acc1:.4f}. Saving model...')
            torch.save(core.state_dict(), f'./output/{args.exp_name}/save/{args.dataset}/des_pred/seed_{args.seed}/core_best.pth')
            torch.save(des_pred.state_dict(), f'./output/{args.exp_name}/save/{args.dataset}/des_pred/seed_{args.seed}/des_pred_best.pth')

    core.load_state_dict(torch.load(f'./output/{args.exp_name}/save/{args.dataset}/des_pred/seed_{args.seed}/core_best.pth', map_location=device, weights_only=True))
    des_pred.load_state_dict(torch.load(f'./output/{args.exp_name}/save/{args.dataset}/des_pred/seed_{args.seed}/des_pred_best.pth', map_location=device, weights_only=True))

    acc1_cnt, acc5_cnt, acc10_cnt, total_cnt = 0, 0, 0, 0
    for batch in tqdm(test_dataloader, desc=f'[Testing]'):
        for k in batch.keys():
            batch[k] = batch[k].to(device, non_blocking=True)

        with torch.no_grad():
            traj_embedding, _, _ = core(
                batch['traj_road_id'],
                batch['traj_grid_id'],
                batch['adj_road_id'],
                batch['adj_grid_id'],
                batch['route_choice_travel_progress'],
                batch['route_choice_angle'],
                batch['route_choice_trans_prob'],
                batch['route_choice_selected_mask'],
                batch['route_choice_unselected_mask'],
                batch['weekday'],
                batch['time_of_day'],
                batch['traj_len'],
            )
            logits = des_pred(traj_embedding)

        _, pred = logits.topk(10, dim=1, largest=True, sorted=True)
        target = batch['label'].view(-1, 1).expand_as(pred)

        correct = pred.eq(target)
        acc1_cnt += correct[:, :1].sum().item()
        acc5_cnt += correct[:, :5].sum().item()
        acc10_cnt += correct[:, :10].sum().item()
        total_cnt += len(target)

    acc1 = acc1_cnt / total_cnt
    acc5 = acc5_cnt / total_cnt
    acc10 = acc10_cnt / total_cnt

    logger.info(f'[Testing] Acc@1: {acc1}, Acc@5: {acc5}, Acc@10: {acc10}')
