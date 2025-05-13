#!/usr/bin/python3

"""
Module to extract data from OpenStreetMap using Overpass API.
This script uses OSMnx to extract data for a given city, processes it, and saves the results as a CSV file.
"""

import osmnx as ox
import geopandas as gpd
from shapely.geometry import box, Point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class OSMExtractor:
    def __init__(self, place_name, cell_size=100, output_file="data/osm_grid_centroids.csv"):
        self.place_name = place_name
        self.cell_size = cell_size
        self.output_file = output_file
        self.crs_wgs84 = "EPSG:4326"
        self.utm_crs = None
        self.polygon_gdf = None
        self.line_gdf = None
        self.point_gdf = None
        self.grid_gdf = None

    def download_osm_data(self):
        """
        Download OSM data for the specified place name.
        """
        print(f"Downloading OSM data for {self.place_name}...")
        landuse_tags = {"landuse": ["commercial", "industrial", "residential"]}
        highway_tags = {"highway": ["motorway", "primary", "secondary", "tertiary", "residential"]}
        point_tags = {"highway": "traffic_signals"}

        self.polygon_gdf = ox.features_from_place(self.place_name, tags=landuse_tags)
        self.line_gdf = ox.features_from_place(self.place_name, tags=highway_tags)
        self.point_gdf = ox.features_from_place(self.place_name, tags=point_tags)

    def reproject_to_utm(self, gdf):
        """
        Reproject a GeoDataFrame to the specified UTM CRS.
        """
        return gdf.to_crs(self.utm_crs)

    def generate_grid(self):
        """
        Generate a grid of cells over the bounding box of the polygon GeoDataFrame.
        """
        print("Generating grid...")
        bbox = self.polygon_gdf.total_bounds
        xmin, ymin, xmax, ymax = bbox

        grid_cells = [
            box(x0, y0, x0 + self.cell_size, y0 + self.cell_size)
            for x0 in np.arange(xmin, xmax, self.cell_size)
            for y0 in np.arange(ymin, ymax, self.cell_size)
        ]

        self.grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs=self.utm_crs)
        self.grid_gdf["grid_id"] = range(1, len(self.grid_gdf) + 1)
        self.grid_gdf["centroid"] = self.grid_gdf.geometry.centroid

    def query_osm_polygon(self, lon_query, lat_query, radii, key, value):
        """
        Calculate the area of intersecting polygons within buffers of varying distances.
        """
        point = Point(lon_query, lat_query)
        filtered_polygons = self.polygon_gdf[self.polygon_gdf[key] == value].to_crs(self.utm_crs)
        point = gpd.GeoSeries([point], crs=self.crs_wgs84).to_crs(self.utm_crs)[0]

        results = {}
        for radius in radii:
            buffer = point.buffer(radius)
            intersecting_polygons = filtered_polygons[filtered_polygons.intersects(buffer)]
            total_area = intersecting_polygons.geometry.area.sum()
            results[f"{value}_{radius}m"] = total_area if not intersecting_polygons.empty else 0
        return results

    def create_features(self, lon, lat):
        """
        Create features for a given point (lon, lat).
        """
        features = {}
        try:
            features = {
                **self.query_osm_polygon(lon, lat, list(range(50, 3050, 50)), "landuse", "commercial"),
                **self.query_osm_polygon(lon, lat, list(range(50, 3050, 50)), "landuse", "industrial"),
                **self.query_osm_polygon(lon, lat, list(range(50, 3050, 50)), "landuse", "residential"),
            }
        except Exception as e:
            print(f"Error at point ({lat}, {lon}): {e}")
            return {}
        return features

    def extract_features(self):
        """
        Extract features for each grid centroid.
        """
        print("Extracting features for each grid centroid...")
        results = []
        centroids_gdf = self.grid_gdf[["grid_id", "centroid"]].set_geometry("centroid").to_crs(self.crs_wgs84)

        for _, row in centroids_gdf.iterrows():
            centroid = row["centroid"]
            grid_id = row["grid_id"]
            lon, lat = centroid.x, centroid.y

            features = self.create_features(lon, lat)
            features.update({"grid_id": grid_id, "latitude": lat, "longitude": lon})
            results.append(features)

        return pd.DataFrame(results)

    def run(self):
        """
        Main method to execute the OSM data extraction and feature generation.
        """
        # Step 1: Download OSM data
        self.download_osm_data()

        # Step 2: Get city boundary and UTM CRS
        city_boundary = ox.geocoder.geocode_to_gdf(self.place_name)
        city_boundary_polygon = city_boundary["geometry"].iloc[0]
        self.utm_crs = ox.projection.project_geometry(city_boundary_polygon)[1]

        # Step 3: Reproject data to UTM
        self.polygon_gdf = self.reproject_to_utm(self.polygon_gdf)
        self.line_gdf = self.reproject_to_utm(self.line_gdf)
        self.point_gdf = self.reproject_to_utm(self.point_gdf)

        # Step 4: Generate grid
        self.generate_grid()

        # Step 5: Extract features
        results_df = self.extract_features()

        # Step 6: Save results
        results_df.to_csv(self.output_file, index=False)
        print(f"Feature extraction complete! Results saved to {self.output_file}")


if __name__ == "__main__":
    extractor = OSMExtractor("Dublin, Ireland")
    extractor.run()