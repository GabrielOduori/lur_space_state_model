#!/usr/bin/python3

"""
Main script to run the OSMExtractor program.
Allows specifying a city name as a command-line argument.
"""

import argparse
from data_acquisition.osm_extractor import OSMExtractor


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract OSM data for a specified city.")
    parser.add_argument(
        "city_name",
        type=str,
        help="The name of the city for which to extract OSM data (e.g., 'Dublin, Ireland').",
    )
    parser.add_argument(
        "--cell_size",
        type=int,
        default=100,
        help="The size of the grid cells in meters (default: 100).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="osm_features.csv",
        help="The output CSV file to save the extracted features (default: osm_features.csv).",
    )
    args = parser.parse_args()

    # Run the OSMExtractor
    extractor = OSMExtractor(args.city_name, cell_size=args.cell_size, output_file=args.output_file)
    extractor.run()


if __name__ == "__main__":
    main()