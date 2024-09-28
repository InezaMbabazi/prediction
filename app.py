import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# Streamlit app for student performance prediction
st.title("Student Performance Prediction")

# Try to load the model, or train a new one if not available
try:
    model = joblib.load('student_model.pkl')
except FileNotFoundError:
    st.write("Model not found. Training a new model...")

    # Sample training data
    data = {
        'High_School_Grade': [85, 90, 60, 75, 82],
        'Entry_Exam_Score': [78, 85, 65, 70, 80],
        'Current_Marks': [75, 88, 58, 72, 79],
        'Passed': [1, 1, 0, 1, 1]  # 1 for passed, 0 for not passed
    }
    df_train = pd.DataFrame(data)

    X_train = df_train[['High_School_Grade', 'Entry_Exam_Score']]
    y_train = df_train['Current_Marks']

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'student_model.pkl')
    st.write("Model trained and saved.")

# Function to determine performance status
def determine_performance_status(predicted_marks, actual_marks):
    if predicted_marks >= actual_marks:
        return "On Track"
    else:
        return "Needs Improvement"

# CSV Template
def create_template():
    template_data = {
        "Student_ID": [],
        "High_School_Grade": [],
        "Entry_Exam_Score": [],
        "Current_Marks": []
    }
    df_template = pd.DataFrame(template_data)
    return df_template

# Download the template CSV file
st.write("Download the template CSV file:")
df_template = create_template()
st.download_button("Download Template", df_template.to_csv(index=False), file_name="student_performance_template.csv", mime='text/csv')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display input data
    st.write("Input Data:")
    st.write(df)

    if st.button("Predict"):
        # Prepare the features for prediction
        features = df[['High_School_Grade', 'Entry_Exam_Score']]
        
        # Make predictions
        predictions = model.predict(features)
        
        # Determine performance status
        df['Predicted_Current_Marks'] = predictions
        df['Performance_Status'] = np.where(df['Predicted_Current_Marks'] >= df['Current_Marks'], "On Track", "Needs Improvement")
        
        # Display predictions with performance status
        st.write("Predictions and Performance Status:")
        st.write(df[['Student_ID', 'High_School_Grade', 'Entry_Exam_Score', 'Current_Marks', 'Predicted_Current_Marks', 'Performance_Status']])
