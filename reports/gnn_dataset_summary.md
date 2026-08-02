# Graph Neural Network Dataset Summary

## Overview

This report summarizes the prepared PyTorch Geometric dataset used for Graph Neural Network training.

## Graph Statistics

- Nodes: **21,613**
- Edges: **216,130**
- Node Features: **15**
- Target Column: **price**

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

## Tensor Shapes

- Node Feature Matrix: **(21613, 15)**
- Edge Index: **(2, 216130)**
- Target Tensor: **(21613,)**

## Validation

- Dataset validation passed.
- Graph connectivity validated.
- Node feature dimensions verified.
- Target tensor verified.
- Dataset ready for Graph Neural Network training.

## Next Stage

The prepared PyTorch Geometric Data object will be used to train a Graph Neural Network regression model. The resulting model will be evaluated using MAE, RMSE, MAPE and R² before being compared against the Week 2 XGBoost baseline.
