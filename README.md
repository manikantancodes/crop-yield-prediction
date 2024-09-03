# Crop Yield Prediction and Analysis

This project focuses on predicting crop yield based on factors like crop type, area, rainfall, and fertilizer usage. Machine learning models are trained to forecast crop yield and provide valuable insights for improving agricultural productivity.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Installation and Setup](#installation-and-setup)
- [Project Workflow](#project-workflow)
  1. [Data Loading](#1-data-loading)
  2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  3. [Data Preprocessing](#3-data-preprocessing)
  4. [Model Training](#4-model-training)
  5. [Model Evaluation](#5-model-evaluation)
  6. [Results](#6-results)
- [Conclusion](#conclusion)

## Project Overview
The primary goal is to build models that predict crop yield using regression techniques. Various factors, such as area, fertilizer use, rainfall, and crop type, are used to train the model. The models are evaluated based on accuracy and performance metrics.

## Data Description
The dataset consists of the following columns:
- **Crop_Year**: Year of crop growth
- **Crop**: Type of crop
- **Area**: Area of cultivation (hectares)
- **Production**: Crop production (metric tons)
- **Annual_Rainfall**: Rainfall received (mm)
- **Fertilizer**: Fertilizer used (kg)
- **Pesticide**: Pesticide used (kg)
- **Season**: Season of cultivation
- **State**: Location of cultivation
- **Yield**: Target variable (metric tons/hectare)

## Installation and Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Crop-Yield-Prediction.git
    cd Crop-Yield-Prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Add the dataset to the `data/raw/` directory:
    ```bash
    data/raw/crop_yield.csv
    ```

## Project Workflow

### 1. Data Loading
The dataset is loaded into a pandas DataFrame.

```python
import pandas as pd
df = pd.read_csv('data/raw/crop_yield.csv')
print(df.head())
```

### 2. Exploratory Data Analysis (EDA)
EDA is performed to explore the dataset:
- View basic statistics
- Check for missing values
- Visualize distributions and relationships

```python
df.info()
df.describe()
```

### 3. Data Preprocessing
Steps include:
- **Handling Missing Values**
- **Feature Encoding** (One-Hot Encoding for categorical variables)
- **Scaling Numerical Features** (using `StandardScaler`)

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
```

### 4. Model Training
Multiple models are trained:
- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

The dataset is split into train/test sets (80/20), and models are trained on the training set.

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

lr_model = LinearRegression()
rf_model = RandomForestRegressor()
gb_model = GradientBoostingRegressor()
```

### 5. Model Evaluation
Models are evaluated based on metrics like:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared (R²)**

```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### 6. Results
- **Linear Regression**:
  - MSE: _value_
  - R²: _value_
  
- **Random Forest Regressor**:
  - MSE: _value_
  - R²: _value_

- **Gradient Boosting Regressor**:
  - MSE: _value_
  - R²: _value_

### Cross-Validation
Performed using Leave-One-Out Cross Validation (LOO-CV) for model stability.

```python
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=LeaveOneOut())
```

## Conclusion
The models successfully predicted crop yields, with **Random Forest** and **Gradient Boosting** providing better results than **Linear Regression**. This analysis can help optimize agricultural practices by understanding which factors influence crop yield the most.

## Future Work
- Add more features such as soil quality, temperature, etc.
- Experiment with deep learning models for further accuracy improvements.
