import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load the trained model (assuming you save your model with joblib or pickle)
import joblib

# Load your model (you'll need to save your model before this step)
model = joblib.load('model.pkl')

# Streamlit app
st.title("Enrollment Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Selecting relevant features
    features = ['CAMPUS', 'FACULTY', 'DEPARTMENT', 'PROGRAM', 'ACADEMIC YEAR', 'YEAR OF COMPLETION', 'PRINCIPAL PASSES']
    
    # Convert categorical columns to 'category' type
    for feature in features:
        df[feature] = df[feature].astype('category')

    # Make predictions
    predictions = model.predict(df[features])
    
    # Display predictions
    st.write("Predictions for REG. NUMBER:")
    st.write(predictions)

# Additional features, such as visualizations, can be added here
