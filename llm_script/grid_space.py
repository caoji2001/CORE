import math
import numpy as np
from pyproj import Transformer


class GridSpace:
    def __init__(self, road_centroid_coordinates, poi_coordinates, grid_size):
        self.epsg_name = self._get_utm_epsg_name(road_centroid_coordinates[0])
        self.wgs84_to_utm = Transformer.from_crs(
            crs_from='EPSG:4326',
            crs_to=self.epsg_name,
            always_xy=True
        )
        self.utm_to_wgs84 = Transformer.from_crs(
            crs_from=self.epsg_name,
            crs_to='EPSG:4326',
            always_xy=True
        )

        road_centroid_coordinates_utm = []
        for i in range(len(road_centroid_coordinates)):
            assert self._get_utm_epsg_name(road_centroid_coordinates[i]) == self.epsg_name
            road_centroid_coordinates_utm.append(self.wgs84_to_utm.transform(road_centroid_coordinates[i][0], road_centroid_coordinates[i][1]))
        road_centroid_coordinates_utm = np.array(road_centroid_coordinates_utm)

        poi_coordinates_utm = []
        for i in range(len(poi_coordinates)):
            assert self._get_utm_epsg_name(poi_coordinates[i]) == self.epsg_name
            poi_coordinates_utm.append(self.wgs84_to_utm.transform(poi_coordinates[i][0], poi_coordinates[i][1]))
        poi_coordinates_utm = np.array(poi_coordinates_utm)

        self.x_min = min(np.min(road_centroid_coordinates_utm[:, 0]), np.min(poi_coordinates_utm[:, 0]))
        self.y_min = min(np.min(road_centroid_coordinates_utm[:, 1]), np.min(poi_coordinates_utm[:, 1]))
        self.x_max = max(np.max(road_centroid_coordinates_utm[:, 0]), np.max(poi_coordinates_utm[:, 0]))
        self.y_max = max(np.max(road_centroid_coordinates_utm[:, 1]), np.max(poi_coordinates_utm[:, 1]))

        self.grid_size = grid_size
        self.num_grid_x = math.floor((self.x_max - self.x_min) / self.grid_size) + 1
        self.num_grid_y = math.floor((self.y_max - self.y_min) / self.grid_size) + 1
        self.num_total_grid = self.num_grid_x * self.num_grid_y

    @staticmethod
    def _get_utm_epsg_name(coordinate):
        lon, lat = coordinate[0], coordinate[1]
        utm_zone = int((lon + 180) // 6) + 1
        utm_band = 'N' if lat >= 0 else 'S'
        epsg_name = f'EPSG:326{utm_zone:02d}' if utm_band == 'N' else f'EPSG:327{utm_zone:02d}'

        return epsg_name

    def wgs84_to_utm_coordinate(self, coordinate):
        assert self._get_utm_epsg_name(coordinate) == self.epsg_name
        x, y = self.wgs84_to_utm.transform(coordinate[0], coordinate[1])
        assert self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
        return x, y

    def grid2grid_id(self, grid_x, grid_y):
        return grid_x + grid_y * self.num_grid_x
    
    def grid_id2grid(self, grid_id):
        return grid_id % self.num_grid_x, grid_id // self.num_grid_x

    def coordinate2grid(self, coordinate):
        x, y = self.wgs84_to_utm_coordinate(coordinate)

        grid_x = math.floor((x - self.x_min) / self.grid_size)
        grid_y = math.floor((y - self.y_min) / self.grid_size)

        assert 0 <= grid_x < self.num_grid_x
        assert 0 <= grid_y < self.num_grid_y

        return grid_x, grid_y

    def coordinate2grid_id(self, coordinate):
        grid_x, grid_y = self.coordinate2grid(coordinate)

        grid_id = self.grid2grid_id(grid_x, grid_y)
        return grid_id
