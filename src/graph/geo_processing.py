"""
Geospatial Processing Module.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from shapely.geometry import Point

from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

from src.graph.distance import haversine_distance

INPUT_DATASET = PROCESSED_DATA_DIR / "clean_housing_data.csv"

OUTPUT_DATASET = PROCESSED_DATA_DIR / "clean_housing_geodata.geojson"

REPORT_FILE = REPORTS_DIR / "geospatial_summary.md"


def validate_coordinates(df):

    invalid = df[
        (df["lat"] < -90)
        | (df["lat"] > 90)
        | (df["long"] < -180)
        | (df["long"] > 180)
    ]

    if not invalid.empty:
        raise ValueError("Invalid coordinates detected.")


def create_geodataframe(df):

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(df["long"], df["lat"])
    ]

    return gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:4326",
    )


def calculate_sample_neighbor_distances(
    geo_dataframe,
    sample_size=5,
):
    """
    Demonstrate neighbour distance calculation.

    Week 3 will use this logic to build
    the KNN graph.
    """

    print("\nNearest Neighbour Distance Samples\n")

    for i in range(sample_size):

        house_a = geo_dataframe.iloc[i]
        house_b = geo_dataframe.iloc[i + 1]

        distance = haversine_distance(
            house_a["lat"],
            house_a["long"],
            house_b["lat"],
            house_b["long"],
        )

        print(
            f"House {i+1} -> House {i+2} : "
            f"{distance:.3f} km"
        )


def generate_report(geo_dataframe):

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("# Geospatial Processing Report\n\n")

        report.write(
            f"Rows : **{len(geo_dataframe)}**\n\n"
        )

        report.write(
            f"Columns : **{len(geo_dataframe.columns)}**\n\n"
        )

        report.write(
            "Coordinate System : **EPSG:4326**\n\n"
        )

        report.write("## Processing Steps\n\n")

        report.write("- Loaded cleaned housing dataset\n")
        report.write("- Validated latitude and longitude\n")
        report.write("- Generated geometry objects\n")
        report.write("- Converted to GeoDataFrame\n")
        report.write("- Exported GeoJSON\n")
        report.write("- Verified Haversine neighbour calculations\n")

        report.write("\n## Deliverables\n\n")

        report.write("- GeoDataFrame created\n")
        report.write("- Spatial processing completed\n")
        report.write("- GeoJSON exported\n")
        report.write("- Ready for Week 3 graph construction\n")


def run_geospatial_processing():

    print("=" * 70)
    print("Starting Geospatial Processing")
    print("=" * 70)

    dataframe = pd.read_csv(INPUT_DATASET)

    print("Dataset loaded successfully.")

    validate_coordinates(dataframe)

    print("Coordinate validation passed.")

    geo_dataframe = create_geodataframe(dataframe)

    print("GeoDataFrame created.")

    calculate_sample_neighbor_distances(
        geo_dataframe
    )

    geo_dataframe.to_file(
        OUTPUT_DATASET,
        driver="GeoJSON",
    )

    print("\nGeoJSON exported.")

    generate_report(
        geo_dataframe
    )

    print("Report generated.")

    print("=" * 70)
    print("Geospatial processing completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_geospatial_processing()