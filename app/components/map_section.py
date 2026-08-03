"""
Interactive Housing Map Component.

Displays the King County housing locations on an interactive Folium map.

Author:
    Aman Shukla
"""

from __future__ import annotations

import folium
import streamlit as st

from streamlit_folium import st_folium

from app.services.map_service import (
    load_map_dataset,
)


def render_map_section():

    st.header("🗺 Interactive Housing Map")

    dataframe = load_map_dataset()

    st.caption(
        "Explore the geographic distribution of properties in the King County Housing Dataset."
    )

    # ==========================================================
    # Controls
    # ==========================================================

    left_column, right_column = st.columns(2)

    with left_column:

        number_of_properties = st.slider(
            "Number of Properties",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
        )

    with right_column:

        marker_radius = st.slider(
            "Marker Radius",
            min_value=2,
            max_value=8,
            value=4,
        )

    # ==========================================================
    # Dataset Summary
    # ==========================================================

    summary1, summary2, summary3 = st.columns(3)

    summary1.metric(
        "Total Properties",
        f"{len(dataframe):,}",
    )

    summary2.metric(
        "Displayed",
        f"{number_of_properties:,}",
    )

    summary3.metric(
        "Average Price",
        f"${dataframe['price'].mean():,.0f}",
    )

    # ==========================================================
    # Create Base Map
    # ==========================================================

    housing_map = folium.Map(

        location=[
            dataframe["lat"].mean(),
            dataframe["long"].mean(),
        ],

        zoom_start=10,

        control_scale=True,

        tiles="OpenStreetMap",
    )

    folium.TileLayer(
        "CartoDB positron"
    ).add_to(housing_map)

    # folium.TileLayer(
    #     "CartoDB dark_matter"
    # ).add_to(housing_map)

    # ==========================================================
    # Sample Data
    # ==========================================================

    sample = dataframe.sample(
        n=number_of_properties,
        random_state=42,
    )

    average_price = dataframe["price"].median()

    # ==========================================================
    # Plot Markers
    # ==========================================================

    for _, row in sample.iterrows():

        if row["price"] >= average_price:

            color = "red"

        else:

            color = "blue"

        popup = f"""
        <b>Price</b>: ${row['price']:,.0f}<br>
        <b>Bedrooms</b>: {row['bedrooms']}<br>
        <b>Bathrooms</b>: {row['bathrooms']}<br>
        <b>Living Area</b>: {row['sqft_living']} sqft<br>
        <b>House Age</b>: {row['house_age']} years<br>
        <b>Distance to City Center</b>: {row['distance_to_city_center_km']:.2f} km
        """

        folium.CircleMarker(

            location=[
                row["lat"],
                row["long"],
            ],

            radius=marker_radius,

            color=color,

            fill=True,

            fill_color=color,

            fill_opacity=0.80,

            popup=popup,

        ).add_to(housing_map)

    # ==========================================================
    # Layer Control
    # ==========================================================

    folium.LayerControl().add_to(
        housing_map
    )

    # ==========================================================
    # Legend
    # ==========================================================

    st.markdown(
        """
**Legend**

🔴 Higher than median property price

🔵 Lower than median property price
"""
    )

    # ==========================================================
    # Render Map
    # ==========================================================

    st_folium(

        housing_map,

        width=None,

        height=700,

        returned_objects=[],
    )