import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('student_model.pkl')

# Streamlit app
st.title("Student Performance Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Assuming the CSV has the same columns as used for training
    st.write("Input Data:")
    st.write(df)

    if st.button("Predict"):
        # Prepare the features for prediction
        features = df[['High_School_Grade', 'Entry_Exam_Score']]
        
        # Make predictions
        predictions = model.predict(features)
        
        # Display predictions
        df['Predicted_Current_Marks'] = predictions
        st.write("Predictions:")
        st.write(df[['High_School_Grade', 'Entry_Exam_Score', 'Predicted_Current_Marks']])
