import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('D:/Github/Crop Yield Prediction and Analysis/climate_model.pkl')

# Streamlit app title
st.title('Climate Temperature Prediction')

# User input
st.sidebar.header('User Input')
year = st.sidebar.slider('Year', min_value=1750, max_value=2100, value=2023)

# Prepare the input data for the model
input_data = pd.DataFrame({'year': [year]})

# Predict
prediction = model.predict(input_data)[0]

# Display the result
st.write(f'Predicted Land Average Temperature for the year {year}: {prediction:.2f} °C')

# Optional: Display a message or additional information
st.info('Use the slider on the left to select the year for which you want to predict the land average temperature.')
