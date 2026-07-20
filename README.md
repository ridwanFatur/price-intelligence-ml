# Product Price Prediction using Machine Learning

An end-to-end machine learning project for predicting e-commerce product prices from historical marketplace data. This repository covers the complete workflow from exploratory data analysis (EDA), data cleaning, feature engineering, baseline regression modeling, and categorical feature benchmarking using gradient boosting algorithms.

---

## Overview

Product price prediction plays an important role in price monitoring, recommendation systems, and market intelligence. This project analyzes historical product information collected from an e-commerce platform and develops machine learning models to accurately estimate product prices.

The project focuses on:

- Exploratory Data Analysis (EDA)
- Data Quality Assessment
- Data Cleaning
- Feature Engineering
- Baseline Machine Learning Models
- Categorical Feature Benchmarking

---

## Dataset

The dataset consists of:

- **306,226 observations**
- **26 features**
- **81 calendar days** (January 1 – March 22, 2025)
- **59 available observation days**
- **22 missing collection dates**

The available features include:

- Product information
- Shop information
- Historical pricing
- Promotions
- Customer ratings
- Inventory information
- Collection timestamp

---

# Repository Structure

```
.
├── 01. EDA - Data Overview.ipynb
├── 02. EDA – Daily Record Distribution.ipynb
├── 03. EDA – Price Distribution and Log Transformation.ipynb
├── 04. EDA – Product Attribute Relationships.ipynb
├── 05. EDA - Promotion and Discount Integrity Analysis.ipynb
├── 06. EDA - Product & Identifier Relationships.ipynb
├── 07. EDA - Shop Status & Verification Analysis.ipynb
├── 08. EDA - Product Price Stability and Variation Analysis.ipynb
├── 09. EDA - Price Consistency and Range Validation.ipynb
├── 10. EDA – Item Price Consistency Analysis.ipynb
├── 11. EDA - Model-Level Price Variation Analysis.ipynb
├── 12. Data Cleaning & Quality Assurance.ipynb
├── 13. Data Preprocessing & Feature Engineering.ipynb
├── 14. Baseline Regression Models for Price Prediction.ipynb
├── 15. Categorical Feature Benchmarking.ipynb
└── README.md
```

---

# Exploratory Data Analysis

The exploratory analysis is divided into eleven notebooks covering different aspects of the dataset.

### 1. Data Overview

- Dataset summary
- Missing value analysis
- Duplicate detection
- Constant feature identification

### 2. Daily Record Distribution

- Daily observation counts
- Missing collection dates
- Temporal coverage analysis

### 3. Price Distribution

- Raw price distribution
- Log transformation
- Outlier detection using IQR

### 4. Product Attribute Relationships

- Correlation analysis
- Price-related feature relationships
- Customer engagement analysis

### 5. Promotion & Discount Integrity

- Promotion consistency
- Discount validation
- Promotional anomaly detection

### 6. Product & Identifier Relationships

- Identifier cardinality
- One-to-many relationships
- Many-to-many relationships

### 7. Shop Status & Verification

- Official shop distribution
- Verification status changes
- Temporal verification analysis

### 8. Product Price Stability

- Stable vs dynamic pricing
- Promotion effects
- Product consistency analysis

### 9. Price Range Validation

- Historical minimum and maximum validation
- Boundary violations
- Pricing anomalies

### 10. Item Price Consistency

- Item-level price variation
- Statistical analysis
- IQR-based anomaly detection

### 11. Model-Level Price Variation

- Price variation within product models
- Feature consistency
- Category consistency analysis

---

# Data Cleaning

The data cleaning pipeline performs several quality assurance steps before model training.

Cleaning includes:

- Removing duplicate records
- Handling missing values
- Removing constant features
- Standardizing categorical values
- Validating promotion information
- Validating discount calculations
- Removing invalid price observations
- Removing item-specific outliers

After cleaning:

| Dataset | Records |
|----------|---------|
| Original | 303,982 |
| Cleaned | 281,147 |

---

# Feature Engineering

The preprocessing pipeline prepares the cleaned dataset for machine learning.

### Temporal Features

- Sequential day index
- Weekend indicator
- Cyclical day encoding

### Categorical Features

- Ordinal Encoding

### Numerical Features

- Min-Max Scaling

### Target Transformation

- Log1p transformation of product price

---

# Baseline Regression Models

Several regression algorithms are benchmarked to establish baseline performance.

Models include:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- Extra Trees
- K-Nearest Neighbors

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score
- Training Time

The baseline experiments demonstrate that tree-based ensemble models significantly outperform traditional linear regression models.

---

# Categorical Feature Benchmarking

This experiment evaluates the predictive contribution of different categorical features while keeping all numerical features unchanged.

The evaluated categorical variables are:

- itemId
- modelId
- cat_id
- brand
- promotionId

Each experiment trains the model using **only one categorical feature at a time** together with the remaining numerical features.

Three gradient boosting frameworks are evaluated:

- XGBoost
- LightGBM
- CatBoost

## XGBoost Results

| Feature | MAE | RMSE | MAPE | R² |
|---------|----:|----:|----:|----:|
| **modelId** | **1.92** | **21.05** | **0.41%** | **0.9995** |
| itemId | 25.98 | 99.77 | 7.72% | 0.9885 |
| cat_id | 28.91 |118.31 |7.82%|0.9838|
| brand |28.69|126.11|7.84%|0.9816|
| promotionId |31.26|146.71|7.93%|0.9752|

---

## LightGBM Results

| Feature | MAE | RMSE | MAPE | R² |
|---------|----:|----:|----:|----:|
| **modelId** | **7.04** | **53.52** | **1.30%** | **0.9967** |
| itemId |26.49|102.22|7.56%|0.9879|
| brand |27.40|101.03|8.00%|0.9882|
| cat_id |27.68|100.87|8.02%|0.9883|
| promotionId |27.80|100.67|8.11%|0.9883|

---

## CatBoost Results

| Feature | MAE | RMSE | MAPE | R² |
|---------|----:|----:|----:|----:|
| **modelId** | **25.83** | **91.94** | **6.65%** | **0.9902** |
| promotionId |30.49|108.21|8.47%|0.9865|
| itemId |30.55|108.18|8.47%|0.9865|
| brand |30.60|107.76|8.51%|0.9866|
| cat_id |30.62|108.32|8.46%|0.9865|

---

# Key Findings

- Historical price features (`item_price_min`, `item_price_max`, and `priceBeforeDiscount`) are the strongest numerical predictors of product price.
- Applying a **log transformation** to price significantly reduces skewness and improves regression performance.
- Most products maintain stable prices throughout the observation period, while price changes are primarily associated with promotions rather than changes in product attributes.
- `modelId` consistently provides the highest predictive power among all evaluated categorical features across XGBoost, LightGBM, and CatBoost.
- XGBoost achieves the best overall performance when using `modelId`, reaching:
  - **MAE:** 1.92
  - **RMSE:** 21.05
  - **MAPE:** 0.41%
  - **R²:** 0.9995

These results indicate that product-level identifiers contain substantially more pricing information than broader categorical attributes such as brand, category, or promotion.

---

# Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Jupyter Notebook

---

# Workflow

```
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Baseline Regression Models
      │
      ▼
Categorical Feature Benchmarking
      │
      ▼
Best Model Selection
```

---

# Future Improvements

Potential future work includes:

- Hyperparameter optimization with Optuna
- Time-aware cross-validation
- Ensemble learning
- Feature selection
- SHAP explainability
- Automated model deployment

---

# License

This repository is intended for educational and research purposes.