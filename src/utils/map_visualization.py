"""
Interactive Geospatial Visualization

Creates an interactive Folium map for the King County Housing Dataset.

Author:
    Aman Shukla
"""

from __future__ import annotations

import folium
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

INPUT_DATASET = PROCESSED_DATA_DIR / "clean_housing_data.csv"

OUTPUT_MAP = REPORTS_DIR / "housing_price_map.html"


def create_price_map():

    dataframe = pd.read_csv(INPUT_DATASET)

    center_lat = dataframe["lat"].mean()
    center_long = dataframe["long"].mean()

    housing_map = folium.Map(
        location=[center_lat, center_long],
        zoom_start=10,
        tiles="OpenStreetMap",
    )

    for _, row in dataframe.iterrows():

        folium.CircleMarker(

            location=[
                row["lat"],
                row["long"],
            ],

            radius=3,

            color="blue",

            fill=True,

            fill_opacity=0.6,

            popup=(
                f"Price : ${row['price']:,.0f}<br>"
                f"Bedrooms : {row['bedrooms']}<br>"
                f"Bathrooms : {row['bathrooms']}<br>"
                f"Sqft : {row['sqft_living']}"
            ),

        ).add_to(housing_map)

    housing_map.save(OUTPUT_MAP)

    print("=" * 70)
    print("Interactive map generated successfully.")
    print(OUTPUT_MAP)
    print("=" * 70)


if __name__ == "__main__":
    create_price_map()