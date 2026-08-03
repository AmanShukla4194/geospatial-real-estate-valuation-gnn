# Model Comparison Summary

## Evaluation Metrics

| Model                |      MAE |   RMSE |    MAPE |       R2 |
|:---------------------|---------:|-------:|--------:|---------:|
| XGBoost              |  76713.2 | 105326 | 17.19   |  0.8271  |
| Graph Neural Network | 399650   | 467054 | 73.5149 | -2.59721 |

## Observations

- XGBoost currently outperforms the Graph Neural Network on the selected evaluation metrics.
- The Graph Neural Network successfully completed training and inference.
- Additional hyperparameter tuning, graph refinement, and feature engineering may improve GNN performance.
- The complete graph-based pipeline has been implemented successfully.
