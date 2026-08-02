# Baseline XGBoost Model Report

## Overview

This report summarizes the baseline XGBoost regression model developed during Week 2 of the project. The model serves as the benchmark for comparison with the future Graph Neural Network (GNN) model.

## Evaluation Metrics

- **MAE:** 76,713.21
- **RMSE:** 105,325.82
- **MAPE:** 17.19%
- **R² Score:** 0.8271

## Interpretation

- The baseline model explains approximately **82.71%** of the variance in housing prices.
- The current benchmark MAPE is **17.19%**.
- The objective of the spatial graph model is to reduce this MAPE by incorporating neighborhood relationships and spatial embeddings.

## Next Phase

Week 3 will construct a K-Nearest Neighbor (KNN) graph and prepare spatial embeddings for Graph Neural Network training. The GNN will be evaluated against this baseline using the same regression metrics.
