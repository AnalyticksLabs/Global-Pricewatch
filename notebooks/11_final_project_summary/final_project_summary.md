# GlobalPriceWatch — Final Project Summary

## Project
GlobalPriceWatch — Global Inflation Pattern Analysis and Forecasting

## Dataset
- Countries: 193
- Historical period: 1960–2025
- Main data source: World Bank
- Target: Inflation

## Main Pipeline

1. Data Collection
2. Data Preprocessing
3. Missing-Value Analysis
4. DTW-Based Clustering
5. Feature Engineering
6. Leakage-Safe Sequence Preparation
7. GRU + Multi-Head Temporal Attention
8. Model Training
9. Prediction
10. Model Evaluation
11. Model Improvement
12. SHAP Explainability

## Clustering

- Number of clusters selected: 2
- Final countries: 193

## Primary Leakage-Safe Model

Model: Leakage-Safe GRU + Multi-Head Temporal Attention

Test period: 2023–2025

MAE: 5.0807
RMSE: 12.7841
R²: 0.7968
Explained Variance: 0.7968

## Model Comparison

                        Model    MAE    RMSE     R2  Explained_Variance
     Original GRU + Attention 7.0178 20.7851 0.4628              0.4714
 Leakage-Safe GRU + Attention 5.0807 12.7841 0.7968              0.7968
Time-Aware GRU + Attention V1 4.6966 13.1223 0.7859              0.7859
    Feature-Weighted Rich GRU 5.7189 19.6237 0.7062                 NaN

## SHAP Analysis

The baseline SHAP analysis showed that recent time-series sequence information was the most important feature group.

Approximate sampled SHAP importance:

- Recent Time-Series Sequence: 62.129%
- Historical Difference Features: 18.941%
- Engineered Historical Features: 18.929%

## Current Model Improvement

The best experimental model so far is:

Feature-Weighted Rich GRU

Validation R²: 0.7062
Validation RMSE: 19.6237
Validation MAE: 5.7189

The 2023–2025 test set remains locked until final model selection.

## Important Evaluation Principle

Model selection should be based on the validation period.

The final test period (2023–2025) should only be used after selecting the final model.

