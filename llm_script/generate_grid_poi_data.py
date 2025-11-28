import os
import argparse
import pickle
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from grid_space import GridSpace


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['Beijing', 'Chengdu', 'Xian', 'Porto'])
    parser.add_argument('--grid_size', type=int, default=1000)
    parser.add_argument('--top_ratio', type=float, default=0.1)
    args = parser.parse_args()

    dataset = args.dataset
    grid_size = args.grid_size
    top_ratio = args.top_ratio

    geo = pd.read_csv(f'../data/{dataset}/traj/roadmap.geo')
    poi = pd.read_csv(f'../data/{dataset}/poi/poi.csv')

    road_centroid_coordinates = []
    for _, row in geo.iterrows():
        coordinates = eval(row['coordinates'])
        line = LineString(coordinates)
        road_centroid_coordinates.append((line.centroid.x, line.centroid.y))
    road_centroid_coordinates = np.array(road_centroid_coordinates)

    poi_coordinates = poi[['lon_wgs84', 'lat_wgs84']].to_numpy()

    grid_space = GridSpace(road_centroid_coordinates, poi_coordinates, grid_size)

    grid_x_list, grid_y_list, grid_id_list = [], [], []
    for _, row in poi.iterrows():
        lon = row['lon_wgs84']
        lat = row['lat_wgs84']
        grid_x, grid_y = grid_space.coordinate2grid((lon, lat))
        grid_id = grid_space.grid2grid_id(grid_x, grid_y)
        grid_x_list.append(grid_x)
        grid_y_list.append(grid_y)
        grid_id_list.append(grid_id)

    poi['grid_x'] = grid_x_list
    poi['grid_y'] = grid_y_list
    poi['grid_id'] = grid_id_list

    num_total_grid = grid_space.num_total_grid
    grid_poi = [{} for _ in range(num_total_grid)]

    if dataset == 'Porto':
        poi_cat_list = poi['category'].unique()
    else:
        poi_cat_list = ['购物服务', '餐饮服务', '公司企业', '生活服务', '交通设施服务', '科教文化服务',\
                    '商务住宅', '政府机构及社会团体', '金融保险服务', '医疗保健服务', '体育休闲服务',\
                    '住宿服务', '汽车服务', '风景名胜']
    
    for poi_cat in poi_cat_list:
        if dataset == 'Porto':
            count = np.bincount(poi[poi['category'] == poi_cat]['grid_id'].to_numpy(), minlength=num_total_grid)
        else:
            count = np.bincount(poi[poi['大类'] == poi_cat]['grid_id'].to_numpy(), minlength=num_total_grid)

        top_n = int(num_total_grid * top_ratio)

        sorted_indices = np.argsort(count)
        top_indices = sorted_indices[-top_n:]

        for grid_id in top_indices:
            if dataset == 'Porto':
                poi_index = poi[(poi['category'] == poi_cat) & (poi['grid_id'] == grid_id)].index.to_numpy()
            else:
                poi_index = poi[(poi['大类'] == poi_cat) & (poi['grid_id'] == grid_id)].index.to_numpy()
            grid_poi[grid_id][poi_cat] = poi_index

    os.makedirs(f'../llm_cache/{dataset}', exist_ok=True)
    with open(f'../llm_cache/{dataset}/grid_poi_data_{grid_size}_top{int(top_ratio*100)}.pkl', 'wb') as file:
        pickle.dump(grid_poi, file)
