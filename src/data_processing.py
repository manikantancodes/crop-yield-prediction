import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load the dataset from the given file path."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data: handling categorical and numerical features."""
    numeric_features = ['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    categorical_features = ['Crop', 'Season', 'State']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X = df.drop(columns='Yield')
    y = df['Yield']
    
    X_preprocessed = preprocessor.fit_transform(X)
    
    return X_preprocessed, y

# Example usage
if __name__ == "__main__":
    file_path = "data/raw/crop_yield.csv"
    df = load_data(file_path)
    X_preprocessed, y = preprocess_data(df)
    print("Data Preprocessing Complete.")
