"""
Prediction Visualization Component.

Displays prediction-related statistics and charts.

Author:
    Aman Shukla
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.services.prediction_service import (
    load_prediction_dataset,
    get_prediction_summary,
)


def render_prediction_section():

    st.header("🏠 Property Price Visualization")

    dataframe = load_prediction_dataset()

    summary = get_prediction_summary()

    # ==========================================================
    # Summary Cards
    # ==========================================================

    column1, column2, column3, column4 = st.columns(4)

    column1.metric(
        "Average Price",
        f"${summary['Average Price']:,.0f}"
    )

    column2.metric(
        "Median Price",
        f"${summary['Median Price']:,.0f}"
    )

    column3.metric(
        "Minimum Price",
        f"${summary['Minimum Price']:,.0f}"
    )

    column4.metric(
        "Maximum Price",
        f"${summary['Maximum Price']:,.0f}"
    )

    st.markdown("---")

    # ==========================================================
    # Filters
    # ==========================================================

    minimum_price = int(dataframe["price"].min())
    maximum_price = int(dataframe["price"].max())

    selected_price_range = st.slider(
        "Price Range ($)",
        minimum_price,
        maximum_price,
        (
            minimum_price,
            maximum_price,
        ),
    )

    number_of_records = st.slider(
        "Number of Records",
        min_value=10,
        max_value=200,
        value=25,
        step=5,
    )

    filtered_dataframe = dataframe[
        (
            dataframe["price"] >= selected_price_range[0]
        )
        &
        (
            dataframe["price"] <= selected_price_range[1]
        )
    ]

    # ==========================================================
    # Histogram
    # ==========================================================

    st.subheader("Price Distribution")

    histogram = px.histogram(
        filtered_dataframe,
        x="price",
        nbins=50,
        title="Housing Price Distribution",
    )

    st.plotly_chart(
        histogram,
        use_container_width=True,
    )

    # ==========================================================
    # Scatter Plot
    # ==========================================================

    st.subheader(
        "Living Area vs Property Price"
    )

    scatter = px.scatter(
        filtered_dataframe,
        x="sqft_living",
        y="price",
        color="bedrooms",
        hover_data=[
            "bathrooms",
            "grade",
            "house_age",
        ],
        title="Property Price vs Living Area",
    )

    st.plotly_chart(
        scatter,
        use_container_width=True,
    )

    # ==========================================================
    # Prediction Insights
    # ==========================================================

    st.subheader("Prediction Insights")

    insight1, insight2, insight3, insight4 = st.columns(4)

    insight1.metric(
        "Matching Properties",
        f"{len(filtered_dataframe):,}"
    )

    insight2.metric(
        "Average Living Area",
        f"{filtered_dataframe['sqft_living'].mean():,.0f} sqft"
    )

    insight3.metric(
        "Average House Age",
        f"{filtered_dataframe['house_age'].mean():.1f} yrs"
    )

    insight4.metric(
        "Average Bedrooms",
        f"{filtered_dataframe['bedrooms'].mean():.1f}"
    )

    st.markdown("---")

    # ==========================================================
    # Property Table
    # ==========================================================

    st.subheader("Filtered Property Records")

    st.dataframe(
        filtered_dataframe[
            [
                "price",
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "grade",
                "house_age",
                "distance_to_city_center_km",
            ]
        ].head(number_of_records),
        use_container_width=True,
        hide_index=True,
    )