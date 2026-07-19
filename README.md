# README — Chapter V: Linear Models and Regularization

## About the project
This project was completed as part of the School 21 curriculum and is dedicated to solving a regression task based on apartment rental data. The work focuses on implementing linear models, feature engineering, normalization, overfitting analysis, and comparison with library-based solutions.

The project demonstrates a full machine learning workflow: from data preparation and feature extraction to model evaluation, regularization, and final comparison of results.

## What was done
The following steps were completed in the project:
- imported the required libraries;
- loaded train and test datasets;
- preprocessing;
- analyzed and transformed the `features` column;
- extracted the most frequent apartment features;
- created 20 binary features from the most common values in `features`;
- extended the final feature set with `bathrooms` and `bedrooms`;
- built a final feature list of 22 variables for training.

## Models implemented
Several regression models were implemented and tested:
- custom Linear Regression;
- custom Ridge;
- custom Lasso;
- custom ElasticNet;
- corresponding models from `sklearn` for comparison.

The project also included:
- prediction on train and test sets;
- evaluation with MAE, RMSE, and R2;
- comparison between custom implementations and `sklearn` models;
- storing results in summary tables.

## Feature scaling
Feature normalization was implemented and tested in two ways:
- custom MinMaxScaler and `sklearn` MinMaxScaler;
- custom StandardScaler and `sklearn` StandardScaler.

After scaling, all main models were retrained on:
- MinMax-scaled data;
- Standard-scaled data.

The results were added to the final comparison tables.

## Overfitting analysis
To demonstrate overfitting in practice, polynomial features of degree 10 were created using only `bathrooms` and `bedrooms`.

The following models were trained on these polynomial features:
- Linear Regression;
- Ridge;
- Lasso;
- ElasticNet.

This part of the project included:
- comparison of train and test metrics;
- analysis of overfitting;
- evaluation of how regularization affects model stability;
- testing different regularization parameters.

## Baselines and comparison
Baseline models from the previous stage were also included:
- mean prediction;
- median prediction.

All metrics were collected into final tables for:
- MAE;
- RMSE;
- R2.

Based on these results, the models were compared by quality and stability.

## Tools and libraries
The project uses:
- Python;
- Jupyter Notebook;
- pandas;
- numpy;
- scikit-learn;
- collections.

## Repository contents
The repository includes:
- Jupyter Notebook with the full solution;
- custom implementations of regression models;
- custom implementations of scalers;
- metric calculation functions;

## Result
This project shows practical work with linear regression models, feature generation, normalization, regularization, and overfitting detection. It also demonstrates the ability to reproduce classical machine learning algorithms manually and compare them with standard implementations from `scikit-learn`.