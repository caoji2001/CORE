import os
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import multiprocessing as mp
from tqdm import tqdm
from shapely import LineString
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import pyproj
from scipy.stats import kendalltau, spearmanr
import torch
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import Dataset, DataLoader, random_split
from loguru import logger

from utils import set_seed, dict_to_namespace, is_integer_lane
from llm_script.grid_space import GridSpace
from model.core import Core
from model.label_pred import LabelPred


class RoadDataset(Dataset):
    def __init__(self, geo, road_id2grid_id):
        self.data = {
            'road_id': [],
            'grid_id': [],
            'label': [],
        }

        for i, row in tqdm(geo.iterrows(), total=len(geo), desc='[Loading data]'):
            if not pd.isna(row['lanes']):
                self.data['road_id'].append(np.int64(i))
                self.data['grid_id'].append(np.int64(road_id2grid_id[i]))
                self.data['label'].append(np.int64(row['lanes']))

    def __getitem__(self, index):
        return {
            'road_id': self.data['road_id'][index],
            'grid_id': self.data['grid_id'][index],
            'label': self.data['label'][index],
        }

    def __len__(self):
        return len(self.data['road_id'])


class MyCollateFn:
    def __init__(self):
        pass

    def __call__(self, items):
        batch = {
            'road_id': [],
            'grid_id': [],
            'label': [],
        }

        for item in items:
            batch['road_id'].append(item['road_id'])
            batch['grid_id'].append(item['grid_id'])
            batch['label'].append(item['label'])

        batch['road_id'] = torch.from_numpy(np.array(batch['road_id']))
        batch['grid_id'] = torch.from_numpy(np.array(batch['grid_id']))
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
    poi = pd.read_csv(f'./data/{args.dataset}/poi/poi.csv')

    if args.dataset == 'Porto':
        poi_cat_list = poi['category'].unique().tolist()
    else:
        poi_cat_list = ['购物服务', '餐饮服务', '公司企业', '生活服务', '交通设施服务', '科教文化服务',\
                        '商务住宅', '政府机构及社会团体', '金融保险服务', '医疗保健服务', '体育休闲服务',\
                        '住宿服务', '汽车服务', '风景名胜']
    poi_cat2idx = {name: idx for idx, name in enumerate(poi_cat_list)}

    edge_index = np.load(f'./data/{args.dataset}/out/edge_index.npy')

    geo['coordinates'] = geo['coordinates'].apply(eval)
    geo['lanes'] = geo['lanes'].apply(lambda x: x if is_integer_lane(x) else np.nan)
    lanes_encoder = LabelEncoder()
    non_nan_mask = geo['lanes'].notna()
    geo.loc[non_nan_mask, 'lanes'] = lanes_encoder.fit_transform(geo.loc[non_nan_mask, 'lanes'])

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

    all_dataset = RoadDataset(geo, road_id2grid_id)

    total_size = len(all_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        all_dataset,
        [train_size, val_size, test_size],
        torch.Generator().manual_seed(args.seed)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=MyCollateFn(),
        num_workers=8,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MyCollateFn(),
        num_workers=8,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MyCollateFn(),
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

    label_pred_config = dict_to_namespace({
        'hidden_dim': args.hidden_dim,
        'output_dim': len(lanes_encoder.classes_),
    })
    label_pred = LabelPred(label_pred_config).to(device)

    os.makedirs(f'./output/{args.exp_name}/log/{args.dataset}/label_pred/seed_{args.seed}', exist_ok=True)
    logger.remove()
    logger.add(os.path.join(f'./output/{args.exp_name}/log/{args.dataset}/label_pred/seed_{args.seed}', f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'), level='INFO', format='{time:YYYY-MM-DD HH:mm:ss} | {message}')

    logger.info(f'args: {args}')
    logger.info(f'core_config: {core_config}')

    os.makedirs(f'./output/{args.exp_name}/save/{args.dataset}/label_pred/seed_{args.seed}', exist_ok=True)

    optimizer = torch.optim.AdamW(list(core.parameters())+list(label_pred.parameters()), lr=args.lr, betas=eval(args.betas), weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epoch, warmup_t=args.lr_warmup_epoch, lr_min=args.lr_min)

    best_macro_f1 = 0.0
    for epoch_id in range(args.epoch):
        core.train()
        label_pred.train()

        scheduler.step(epoch_id+1)

        loss_array = []
        for batch in tqdm(train_dataloader, desc=f'[Training] epoch {epoch_id}'):
            for k in batch.keys():
                batch[k] = batch[k].to(device, non_blocking=True)

            road_emb = core.get_road_emb(batch['road_id'], batch['grid_id'])
            logits = label_pred(road_emb)

            loss = F.cross_entropy(logits, batch['label'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(core.parameters())+list(label_pred.parameters()), max_norm=1.0)
            optimizer.step()

            loss_array.append(loss.item())

        core.eval()
        label_pred.eval()

        probs, labels = [], []
        for batch in tqdm(train_dataloader, desc=f'[Validating] epoch {epoch_id}'):
            for k in batch.keys():
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch.no_grad():
                road_emb = core.get_road_emb(batch['road_id'], batch['grid_id'])
                logits = label_pred(road_emb)
                prob = F.softmax(logits, dim=-1)

            probs.append(prob.cpu().numpy())
            labels.append(batch['label'].cpu().numpy())

        probs = np.vstack(probs)
        labels = np.concatenate(labels)

        preds = np.argmax(probs, axis=1)
        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro')
        micro_f1 = f1_score(labels, preds, average='micro')

        logger.info(f'[Epoch {epoch_id + 1}] loss: {np.mean(loss_array)}, lr: {optimizer.param_groups[0]["lr"]}, Accuracy: {accuracy}, Macro-F1: {macro_f1}, Micro-F1: {micro_f1}')

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            logger.info(f'New best Macro-F1: {macro_f1:.4f}. Saving model...')
            torch.save(core.state_dict(), f'./output/{args.exp_name}/save/{args.dataset}/label_pred/seed_{args.seed}/core_best.pth')
            torch.save(label_pred.state_dict(), f'./output/{args.exp_name}/save/{args.dataset}/label_pred/seed_{args.seed}/label_pred_best.pth')

    core.load_state_dict(torch.load(f'./output/{args.exp_name}/save/{args.dataset}/label_pred/seed_{args.seed}/core_best.pth', map_location=device, weights_only=True))
    label_pred.load_state_dict(torch.load(f'./output/{args.exp_name}/save/{args.dataset}/label_pred/seed_{args.seed}/label_pred_best.pth', map_location=device, weights_only=True))

    probs, labels = [], []
    for batch in tqdm(train_dataloader, desc=f'[Testing]'):
        for k in batch.keys():
            batch[k] = batch[k].to(device, non_blocking=True)

        with torch.no_grad():
            road_emb = core.get_road_emb(batch['road_id'], batch['grid_id'])
            logits = label_pred(road_emb)
            prob = F.softmax(logits, dim=-1)

        probs.append(prob.cpu().numpy())
        labels.append(batch['label'].cpu().numpy())

    probs = np.vstack(probs)
    labels = np.concatenate(labels)

    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')

    logger.info(f'[Testing] Accuracy: {accuracy}, Macro-F1: {macro_f1}, Micro-F1: {micro_f1}')
