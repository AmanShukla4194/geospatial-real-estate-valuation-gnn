# Data Preprocessing Report

## Duplicate Records

- Duplicate rows removed: **0**

## Missing Values (Before Cleaning)

```
               Missing Values  Percentage
id                          0         0.0
date                        0         0.0
price                       0         0.0
bedrooms                    0         0.0
bathrooms                   0         0.0
sqft_living                 0         0.0
sqft_lot                    0         0.0
floors                      0         0.0
waterfront                  0         0.0
view                        0         0.0
condition                   0         0.0
grade                       0         0.0
sqft_above                  0         0.0
sqft_basement               0         0.0
yr_built                    0         0.0
yr_renovated                0         0.0
zipcode                     0         0.0
lat                         0         0.0
long                        0         0.0
sqft_living15               0         0.0
sqft_lot15                  0         0.0
```

## Processing Steps

- Loaded raw housing dataset
- Removed duplicate records
- Filled missing numerical values using median
- Filled missing categorical values using mode
- Validated required numerical columns
- Normalized numerical outliers using the IQR method
- Exported cleaned dataset

## Deliverables

- Clean dataset generated successfully
- Missing value summary generated
- Duplicate removal completed
- Numeric validation completed
- Outlier normalization completed
- Dataset ready for feature engineering
