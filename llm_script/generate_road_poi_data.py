import os
import argparse
import pickle
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point


def get_utm_epsg_name(coordinate):
    utm_zone = int((coordinate[0] + 180) / 6) + 1
    utm_band = 'N' if coordinate[1] >= 0 else 'S'
    epsg_name = f'EPSG:326{utm_zone}' if utm_band == 'N' else f'EPSG:327{utm_zone}'

    return epsg_name

def init_points(points_, max_dis_):
    global points, max_dis
    points = points_
    max_dis = max_dis_

def process_item(geo_coord):
    polyline = LineString(geo_coord)
    distances = polyline.distance(points)
    return np.where(np.array(distances) <= max_dis)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing', choices=['Beijing', 'Chengdu', 'Xian', 'Porto'])
    parser.add_argument('--max_dis', type=int, default=50)
    parser.add_argument('--processes', type=int, default=mp.cpu_count())
    args = parser.parse_args()

    dataset = args.dataset
    max_dis = args.max_dis
    processes = args.processes

    geo = pd.read_csv(f'../data/{dataset}/traj/roadmap.geo')
    poi = pd.read_csv(f'../data/{dataset}/poi/poi.csv')

    geo['coordinates'] = geo['coordinates'].apply(eval)

    geo_coordinates = geo['coordinates'].tolist()
    geo_coordinates_utm = []

    epsg_name = get_utm_epsg_name(geo_coordinates[0][0])
    wgs84_to_utm = Transformer.from_crs(
        crs_from='EPSG:4326',
        crs_to=epsg_name,
        always_xy=True
    )

    for coordinates in geo_coordinates:
        assert epsg_name == get_utm_epsg_name(coordinates[0])
        geo_coordinates_utm.append([wgs84_to_utm.transform(coordinate[0], coordinate[1]) for coordinate in coordinates])

    poi_coordinates = [(lon, lat) for lon, lat in zip(poi['lon_wgs84'].tolist(), poi['lat_wgs84'].tolist())]
    poi_coordinates_utm = []

    for coordinate in poi_coordinates:
        assert epsg_name == get_utm_epsg_name(coordinate)
        poi_coordinates_utm.append(wgs84_to_utm.transform(coordinate[0], coordinate[1]))

    with mp.Pool(processes=processes, initializer=init_points, initargs=([Point(p) for p in poi_coordinates_utm], max_dis)) as pool:
        result = list(tqdm(
            pool.imap(process_item, geo_coordinates_utm),
            total=len(geo_coordinates_utm)
        ))

    os.makedirs(f'../llm_cache/{dataset}', exist_ok=True)
    with open(f'../llm_cache/{dataset}/road_poi_data_{max_dis}.pkl', 'wb') as file:
        pickle.dump(result, file)
