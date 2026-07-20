# Product Price Prediction using Machine Learning

End-to-end machine learning pipeline for predicting product prices from an e-commerce marketplace dataset. This project covers exploratory data analysis (EDA), data cleaning, feature engineering, baseline regression models, categorical feature benchmarking, and dimensionality reduction experiments.

---

## Project Overview

Accurate product price estimation is important for price monitoring, recommendation systems, and market intelligence. This project investigates historical product pricing data collected from an e-commerce platform and develops machine learning models to predict product prices.

The project includes:

- Comprehensive Exploratory Data Analysis (EDA)
- Data quality assessment
- Feature engineering
- Data preprocessing
- Baseline machine learning models
- Categorical feature benchmarking
- Dimensionality reduction experiments

The complete workflow is implemented using Jupyter Notebooks.

---

## Dataset

The dataset contains:

- **306,226 records**
- **26 features**
- Observation period:
  - **January 1, 2025**
  - **March 22, 2025**

The features include:

- Product information
- Shop information
- Historical pricing
- Promotions
- Inventory
- Customer ratings
- Timestamps

---

## Project Structure

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
├── 16. Dimensionality Reduction for Price Prediction.ipynb
└── README.md
```

---

## Exploratory Data Analysis

### 1. Data Overview

- Dataset summary
- Missing value analysis
- Duplicate detection
- Constant feature identification

### 2. Daily Record Distribution

- Daily observation counts
- Missing collection dates
- Temporal coverage

### 3. Price Distribution

- Raw price distribution
- Log transformation
- Outlier detection using IQR

### 4. Product Attribute Relationships

- Correlation analysis
- Numerical feature relationships
- Customer engagement analysis

### 5. Promotion Analysis

- Promotion consistency
- Discount validation
- Promotional anomalies

### 6. Identifier Relationships

- Cardinality analysis
- One-to-many relationships
- Many-to-many relationships

### 7. Shop Analysis

- Official shop distribution
- Verification status changes
- Temporal verification trends

### 8. Product Price Stability

- Stable vs dynamic pricing
- Promotion effects
- Feature consistency

### 9. Price Range Validation

- Historical minimum/maximum validation
- Boundary violations
- Pricing anomalies

### 10. Item Price Consistency

- Item-level price variation
- IQR analysis
- Outlier detection

### 11. Model-Level Price Variation

- Price variation within product models
- Attribute consistency
- Category changes

---

## Data Cleaning

Cleaning steps include:

- Remove duplicate records
- Handle missing values
- Remove constant features
- Validate promotions
- Validate discounts
- Remove invalid price records
- Remove item-level outliers
- Standardize categorical values

After cleaning:

| Stage | Records |
|--------|---------|
| Original | 303,982 |
| Final | 281,147 |

---

## Feature Engineering

The preprocessing pipeline includes:

### Temporal Features

- Day index
- Weekend indicator
- Cyclical day encoding

### Categorical Encoding

- Ordinal Encoding

### Numerical Processing

- Min-Max Scaling

### Target Transformation

- Log1p transformation

---

## Machine Learning Models

Baseline models include:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- Extra Trees
- K-Nearest Neighbors

Evaluation metrics:

- MAE
- RMSE
- MAPE
- R² Score
- Training Time

---

## Categorical Feature Benchmark

Boosting algorithms:

- XGBoost
- LightGBM
- CatBoost

Each categorical feature is evaluated independently.

Main finding:

> **modelId consistently provides the strongest predictive power among all categorical variables.**

---

## Dimensionality Reduction

Two dimensionality reduction techniques are evaluated:

### PCA

- 13 components
- 96.53% explained variance

### Truncated SVD

- 14 components
- 99.10% explained variance

---

## Key Findings

- Historical price features are the strongest predictors.
- Product-level identifiers (`modelId`) outperform broader categorical features.
- Log transformation significantly improves target distribution.
- Most products exhibit stable pricing.
- Promotional metadata contains several inconsistencies requiring cleaning.
- Tree-based models substantially outperform linear regression.
- Truncated SVD slightly outperforms PCA while preserving almost all variance.

---

## Technologies

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

## Repository Workflow

```
EDA
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Baseline Models
   │
   ▼
Categorical Feature Benchmark
```

---

## Future Work

- Hyperparameter optimization
- Time-aware cross-validation
- Ensemble learning
- Feature selection
- SHAP explainability
- Deep learning models
- Time-series aware price prediction

---

## License

This project is intended for educational and research purposes.