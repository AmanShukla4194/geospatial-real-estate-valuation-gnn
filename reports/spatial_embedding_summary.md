# Spatial Embedding Summary

## Overview

This report summarizes the standardized spatial feature representation prepared for Graph Neural Network training.

The embedding matrix combines structural housing attributes with geospatial information and serves as the initial node representation for graph-based learning.

## Embedding Statistics

- Number of Nodes: **21,613**
- Embedding Dimension: **17**
- Mean: **-0.000000**
- Standard Deviation: **1.000000**

## Node Features

- bedrooms
- bathrooms
- sqft_living
- sqft_lot
- floors
- waterfront
- view
- condition
- grade
- sqft_above
- sqft_basement
- house_age
- is_renovated
- years_since_renovation
- distance_to_city_center_km
- lat
- long

## Validation

- Embedding matrix generated successfully.
- No missing values detected.
- No infinite values detected.
- Feature standardization completed.
- Ready for Graph Neural Network training.

## Next Stage

The Graph Neural Network will consume this node feature matrix together with the K-Nearest Neighbor graph to learn spatial relationships between properties. Model performance will then be compared against the Week 2 XGBoost baseline using MAE, RMSE, MAPE and R².
