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
import datetime
import logging
import os
import warnings
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

# Generate a timestamp for filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

log_filename = f"osm_extraction_{timestamp}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(message)s",
)
print("Script started.")
logging.info("Script started.")

# Check if the data directory exists, if not create it
if not os.path.exists("data"):
    os.makedirs("data")

data_path = os.path.join(os.path.dirname(__file__), "data")

driver = "GeoJSON"  # Saving output file as a GeoJSON file
crs = "EPSG:4326"  # WGS84 coordinate reference system

scats_file = "/data/traffic_sites.csv"


class OSMExtractor:
    def __init__(
        self,
        place_name,
        cell_size=100,
        output_file=data_path + "/osm_features.csv",
    ):
        self.place_name = place_name
        self.cell_size = cell_size
        self.output_file = output_file
        self.crs_wgs84 = "EPSG:4326"
        self.utm_crs = None
        self.polygon_gdf = None
        self.line_gdf = None
        self.point_gdf = None
        self.grid_gdf = None
        self.scats_file = scats_file

    def download_osm_data(self):
        """
        Download OSM data for the specified place name.
        """
        print(f"Downloading OSM data for {self.place_name}...")
        landuse_tags = {"landuse": ["commercial", "industrial", "residential"]}
        highway_tags = {
            "highway": ["motorway", "primary", "secondary", "tertiary", "residential"]
        }
        point_tags = {"highway": "traffic_signals"}

        print("Downloading polygon data...")
        self.polygon_gdf = ox.features_from_place(self.place_name, tags=landuse_tags)
        # Save the polygon data to a file
        self.polygon_gdf.to_file(data_path + "polygon_data.geojson", driver=driver)

        print("Downloading line data...")
        self.line_gdf = ox.features_from_place(self.place_name, tags=highway_tags)

        # Save the line data to a file
        self.line_gdf.to_file(data_path + "line_data.geojson", driver=driver)

        print("Downloading point data...")
        self.point_gdf = ox.features_from_place(self.place_name, tags=point_tags)
        # Save the point data to a file
        self.point_gdf.to_file(data_path + "point_data.geojson", driver=driver)

        print("OSM data downloaded and saved to files.")

    def reproject_to_utm(self, gdf):
        """
        Reproject a GeoDataFrame to the specified UTM CRS.
        """
        return gdf.to_crs(self.utm_crs)

    def generate_grid(self):
        """
        Generate a grid of 100m by 100m over the bounding box of the
        polygon GeoDataFrame.
        """
        print("Generating 100m by 100 m Grid.")
        print("We extract centroids of each grid cell to extract features.")
        print("Generating grid...")
        bbox = self.polygon_gdf.total_bounds
        xmin, ymin, xmax, ymax = bbox

        grid_cells = []

        for x0 in np.arange(xmin, xmax, self.cell_size):
            for y0 in np.arange(ymin, ymax, self.cell_size):
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                # Create a box for each grid cell
                grid_cells.append(box(x0, y0, x1, y1))

        # Create a GeoDataFrame from the grid cells and set the CRS to UTM
        self.grid_gdf = gpd.GeoDataFrame(
            grid_cells, columns=["geometry"], crs=self.utm_crs
        )

        # Add grid_id and centroid columns
        self.grid_gdf["grid_id"] = range(1, len(self.grid_gdf) + 1)
        print(f" {len(self.grid_gdf)} grid cells generated before filtering.")

        # Ensure line_gdf is in the same CRS as the grid

        self.line_utm = self.line_gdf.to_crs(self.utm_crs)

        # Filter the grid cells to those that intersect with the line_gdf

        all_lines = unary_union(self.line_utm.geometry)

        mask = self.grid_gdf.intersects(all_lines)
        self.grid_gdf = self.grid_gdf[mask].reset_index(drop=True)

        # Save the grid data to a file
        print(f"Grid generated with {len(self.grid_gdf)} cells.")
        # Save the grid data to a file
        self.grid_gdf.to_file(data_path + "grid.geojson", driver=driver)

        """
        Filter the grids to those that interset with the line_gdf
        """

        # Calculate centroids of each grid cell
        self.grid_gdf["centroid"] = self.grid_gdf.geometry.centroid
        print("Grid and centroids generated successfully.")

        # Reproject the centroids to WGS84 (lat/long)

        self.grid_gdf["centroid"] = self.grid_gdf["centroid"].to_crs(self.crs_wgs84)

        print("Grid and centroids generated successfully.")

    def _get_utm_point(self, lon_query, lat_query):
        """Helper method to create a point and convert it to UTM CRS."""
        return (
            gpd.GeoSeries([Point(lon_query, lat_query)], crs=self.crs_wgs84)
            .to_crs(self.utm_crs)
            .iloc[0]
        )

    def query_osm_polygon(self, lon_query, lat_query, radii, key, value):
        """
        Calculate the total area of polygons matching (key == value)
        that intersect with buffers of varying distances around a point.
        Works on in-memory GeoDataFrame (no PostGIS).
        """

        point = self._get_utm_point(lon_query, lat_query)

        # Step 2: Filter polygon_gdf by the key-value pair
        filtered_polygons = self.polygon_gdf[(self.polygon_gdf[key] == value)].to_crs(
            self.utm_crs
        )

        results = {}
        for radius in radii:
            # Step 3: Create buffer and intersect
            buffer = point.buffer(radius)
            intersecting_polygons = filtered_polygons[
                filtered_polygons.intersects(buffer)
            ]
            # Step 4: Calculate total area of intersecting polygons
            total_area = intersecting_polygons.geometry.area.sum()
            results[f"{value}_{radius}m"] = (
                total_area if not intersecting_polygons.empty else 0
            )

        return results

    def _query_osm_roads(self, lon_query, lat_query, radii, road_types):
        """Core reusable method for road length queries"""
        point = self._get_utm_point(lon_query, lat_query)
        relevant_roads = self.line_gdf[
            self.line_gdf["highway"].isin(road_types)
        ].to_crs(self.utm_crs)

        results = {}
        for r in radii:
            buffer = point.buffer(r)
            for road_type in road_types:
                mask = (relevant_roads["highway"] == road_type) & (
                    relevant_roads.intersects(buffer)
                )
                results[f"{road_type}_{r}m"] = relevant_roads[
                    mask
                ].geometry.length.sum()
        return results

    def query_osm_highway_length(self, lon_query, lat_query, radii):
        """For major highways (motorway/trunk/primary/secondary)"""
        return self._query_osm_roads(
            lon_query, lat_query, radii, {"motorway", "trunk", "primary", "secondary"}
        )

    def query_osm_local_road(self, lon_query, lat_query, radii):
        """For local roads (tertiary/residential)"""
        return self._query_osm_roads(
            lon_query, lat_query, radii, {"tertiary", "residential"}
        )

    def query_osm_line_distance(self, lon_query, lat_query, key, value):
        print(f"Querying line distance for {key} {value}")
        """
        Calculate the distance from a centroid to a nearest line feature.
        In this method, we are extracting distance to a motorway and primary road
        as we will specify in the final query string (key == value) pair."""

        # point = Point(lon_query, lat_query)
        # point = gpd.GeoSeries([point], crs=self.crs_wgs84).to_crs(self.utm_crs)[0]
        point = self._get_utm_point(lon_query, lat_query)

        filtered_lines = self.line_gdf[self.line_gdf[key] == value].to_crs(self.utm_crs)

        # Calculate the minimum distance to the nearest line
        min_distance = filtered_lines.geometry.distance(point).min()

        # Return the result as a dictionary
        return {value: min_distance if min_distance is not None else 0}

    def query_osm_polygon_distance(self, lon_query, lat_query, key, value):
        """
        Calculate the distance from a centroid to the nearest polygon ferature.
        In this method, we are extracting distance to an industrial areas
        as we will specifcy in the final query string (key == value) pair.
        """
        print(f"Querying polygon distance for {key} {value}")

        # Step 1: Create the point and reproject to UTM for accurate distance measurement
        # point = Point(lon_query, lat_query)
        # point = gpd.GeoSeries([point], crs=self.crs_wgs84).to_crs(self.utm_crs)[0]
        point = self._get_utm_point(lon_query, lat_query)

        # Step 2: Filter the GeoDataFrame based on the specified key and value
        filtered_polygons = self.polygon_gdf[self.polygon_gdf[key] == value].to_crs(
            self.utm_crs
        )

        # Step 3: Calculate the minimum distance to the nearest polygon
        min_distance = filtered_polygons.geometry.distance(point).min()

        # Step 4: Return the result as a dictionary
        return {value: min_distance if min_distance is not None else 0}

    def query_osm_point_distance(self, lon_query, lat_query, key, value):
        print(f"Querying point distance for {key} {value}")
        """
        Calculate the distance from a centroid to the nearest point feature.
        In this method, we are extracting distance to a traffic signal
        as we will specify in the final query string (key == value) pair.

        TO DO: Maybe I need to connect this with the SCATS data. 
        Traffic Signal location data here is more accurate than what I am 
        having in the SCATS data. A key challenge of with SCATS data .. maybe??
        """
        # print(
        #     f"Checking query_osm_point_distance: point_gdf.empty: {self.point_gdf.empty}"
        # )

        # point = Point(lon_query, lat_query)
        # point = gpd.GeoSeries([point], crs=self.crs_wgs84).to_crs(self.utm_crs)[0]
        point = self._get_utm_point(lon_query, lat_query)

        filtered_points = self.point_gdf[self.point_gdf[key] == value].to_crs(
            self.utm_crs
        )

        # Calculate the minimum distance to the nearest point
        min_distance = filtered_points.geometry.distance(point).min()

        # Step 4: Return the result as a dictionary
        return {value: min_distance if min_distance is not None else 0}

    def query_scats_distance(self, lon_query, lat_query):
        """
        Calculate the distance from a centroid to the nearest SCATS point.
        scats_file: path to CSV or GeoJSON with columns 'longitude' and 'latitude' or geometry.
        """
        # Try to read as GeoDataFrame, else as CSV
        try:
            scats_gdf = gpd.read_file(self.scats_file)
            if scats_gdf.crs is None:
                scats_gdf.set_crs(self.crs_wgs84, inplace=True)
        except Exception:
            scats_df = pd.read_csv(self.scats_file)
            scats_gdf = gpd.GeoDataFrame(
                scats_df,
                geometry=gpd.points_from_xy(scats_df.longitude, scats_df.latitude),
                crs=self.crs_wgs84,
            )

        # Reproject to UTM for distance calculation
        scats_gdf = scats_gdf.to_crs(self.utm_crs)
        point = self._get_utm_point(lon_query, lat_query)

        min_distance = scats_gdf.geometry.distance(point).min()
        return {"scats_distance": min_distance if min_distance is not None else 0}

    def create_features(self, lon, lat):
        """
        Create features for a given point (lon, lat).
        """
        features = {}
        try:
            features = {
                **self.query_osm_polygon(
                    lon, lat, list(range(50, 3050, 50)), "landuse", "commercial"
                ),
                **self.query_osm_polygon(
                    lon, lat, list(range(50, 3050, 50)), "landuse", "industrial"
                ),
                **self.query_osm_polygon(
                    lon, lat, list(range(50, 3050, 50)), "landuse", "residential"
                ),
                **self.query_osm_highway_length(lon, lat, list(range(50, 1050, 50))),
                **self.query_osm_local_road(lon, lat, list(range(50, 1050, 50))),
                **self.query_osm_point_distance(lon, lat, "highway", "traffic_signals"),
                **self.query_osm_line_distance(lon, lat, "highway", "motorway"),
                **self.query_osm_polygon_distance(lon, lat, "landuse", "industrial"),
                **self.query_scats_distance(lon, lat),
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
        centroids_gdf = (
            self.grid_gdf[["grid_id", "centroid"]]
            .set_geometry("centroid")
            .to_crs(self.crs_wgs84)
        )

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

        final_csv_file = os.path.join(
            data_path,
            f"osm_extraction_{self.place_name.replace(' ', '_')}_{timestamp}.csv",
        )

        results_df.to_csv(final_csv_file, index=False)
        # results_df.to_csv(self.output_file , index=False)
        print(f"Feature extraction complete! Results saved to {self.output_file}")


if __name__ == "__main__":
    extractor = OSMExtractor("Dublin, Ireland")

    extractor.run()
