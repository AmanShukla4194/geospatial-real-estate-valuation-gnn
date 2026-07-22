# Feature Engineering Summary

## Overview

This report documents the baseline tabular feature engineering stage for the Geospatial Real Estate Valuation project.

These features establish the input foundation for the traditional machine-learning baseline that will later be compared against spatial and graph-based valuation models.

## Dataset Summary

- Total observations: **21,613**
- Total columns after feature engineering: **27**
- Engineered features added: **6**

## Engineered Features

### `sale_year`

Calendar year in which the property sale occurred.

### `sale_month`

Calendar month in which the property sale occurred.

### `house_age`

Age of the property at the time of sale, calculated from sale year and construction year.

### `is_renovated`

Binary indicator showing whether a property has a recorded renovation year.

### `years_since_renovation`

Number of years between the recorded renovation and sale. For properties without a recorded renovation, house age is used to represent time since original construction.

### `distance_to_city_center_km`

Haversine distance in kilometers from the property to the configured Seattle city-center reference coordinate.

## Feature Summary Statistics

|                            |   count |      mean |     std |       min |       25% |     50% |       75% |       max |
|:---------------------------|--------:|----------:|--------:|----------:|----------:|--------:|----------:|----------:|
| sale_year                  |   21613 | 2014.32   |  0.4676 | 2014      | 2014      | 2014    | 2015      | 2015      |
| sale_month                 |   21613 |    6.5744 |  3.1153 |    1      |    4      |    6    |    9      |   12      |
| house_age                  |   21613 |   43.3184 | 29.3747 |    0      |   18      |   40    |   63      |  115      |
| is_renovated               |   21613 |    0.0423 |  0.2013 |    0      |    0      |    0    |    0      |    1      |
| years_since_renovation     |   21613 |   40.9378 | 28.8124 |    0      |   15      |   37    |   60      |  115      |
| distance_to_city_center_km |   21613 |   18.4585 | 10.6419 |    0.9831 |    9.7836 |   16.51 |   25.2661 |   77.0939 |

## Data Quality Decisions

- **12 property records** contained a construction year later than the recorded sale year. The original `yr_built` values were preserved, while the derived `house_age` values were normalized to `0` to prevent invalid negative ages.

- **6 property records** contained a renovation year later than the recorded sale year. The original `yr_renovated` values were preserved, while the derived `years_since_renovation` values were normalized to `0`.

- These corrections apply only to derived modeling features and preserve the original source columns for traceability.

## Renovation Profile

- Properties with recorded renovations: **914**
- Percentage with recorded renovations: **4.23%**

## Validation Results

- Missing engineered-feature values: **0**
- Infinite engineered-feature values: **0**
- Remaining negative house-age values: **0**
- Invalid sale-month values: **0**
- Negative city-center distances: **0**

All engineered features passed the required validation checks.

## Modeling Readiness

The feature-engineered dataset is ready for the baseline machine-learning stage. The next modeling phase will select appropriate predictors, separate the target variable (`price`), create reproducible training and holdout partitions, and train an XGBoost regression baseline.

Baseline performance will be evaluated primarily using Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE). These results will provide the benchmark against which the later spatial embedding and Graph Neural Network models will be compared.
