import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def train_models(df):
    X = df.drop('yield', axis=1)
    y = df['yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    print(f'Linear Regression MSE: {mean_squared_error(y_test, y_pred_lin)}')
    joblib.dump(lin_reg, 'models/linear_regression.pkl')
    
    # Random Forest
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f'Random Forest MSE: {mean_squared_error(y_test, y_pred_rf)}')
    joblib.dump(rf, 'models/random_forest.pkl')

if __name__ == "__main__":
    processed_data_path = 'data/processed/crop_yield_data_processed.csv'
    df = load_data(processed_data_path)
    train_models(df)

