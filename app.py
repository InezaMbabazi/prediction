import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Streamlit app for student performance prediction
st.title("Student Group Performance Prediction")

# Try to load the model, or train a new one if not available
try:
    model = joblib.load('student_model.pkl')
except FileNotFoundError:
    st.write("Model not found. Training a new model...")

    # Sample training data for regression (including Principal Passes)
    data = {
        'High_School_Grade': [85, 90, 60, 75, 82],
        'Entry_Exam_Score': [78, 85, 65, 70, 80],
        'Principal_Passes': [12, 14, 10, 9, 15],
        'Current_Marks': [75, 88, 58, 72, 79],
        'Passed': [1, 1, 0, 1, 1]  # 1 for passed, 0 for not passed
    }
    df_train = pd.DataFrame(data)

    # Features and target variable
    X_train = df_train[['High_School_Grade', 'Entry_Exam_Score', 'Principal_Passes']]
    y_train = df_train['Current_Marks']

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'student_model.pkl')
    st.write("Model trained and saved.")

# Function to determine performance status based on group trends
def determine_group_performance_status(row, group_avg, prediction):
    if row['Current_Marks'] > prediction + 2:
        return "Exceeding Expectations"
    elif row['Current_Marks'] < prediction - 2:
        return "Underperforming"
    else:
        return "Meeting Expectations"

# CSV Template
def create_template():
    template_data = {
        "Student_ID": [],
        "High_School_Grade": [],
        "Entry_Exam_Score": [],
        "Principal_Passes": [],
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
        feature_columns = ['High_School_Grade', 'Entry_Exam_Score', 'Principal_Passes']
        available_features = df[feature_columns].dropna()

        # Ensure the features are in the same order as the model was trained
        available_features = available_features[feature_columns]

        # Check if there are enough features to predict
        if len(available_features) > 0:
            # Predict using available features
            predictions = model.predict(available_features)

            # Fill predictions back into the original DataFrame
            df['Predicted_Marks'] = np.nan
            df.loc[available_features.index, 'Predicted_Marks'] = predictions

            # Calculate the group average for comparison
            group_avg = df['High_School_Grade'].mean()
            
            # Determine group performance status for each student
            performance_status = []
            for i, row in df.iterrows():
                if pd.notna(row['Predicted_Marks']):
                    status = determine_group_performance_status(row, group_avg, row['Predicted_Marks'])
                else:
                    status = "Insufficient Data"
                performance_status.append(status)
            
            # Add performance status to the dataframe
            df['Performance_Status'] = performance_status
            
            # Display predictions and performance status
            st.write("Predictions and Performance Status:")
            st.write(df[['Student_ID', 'High_School_Grade', 'Entry_Exam_Score', 'Principal_Passes', 'Current_Marks', 'Predicted_Marks', 'Performance_Status']])
            
            # Plot a pie chart for performance status
            st.write("Performance Status Breakdown:")
            performance_counts = df['Performance_Status'].value_counts()

            # Plotting the pie chart
            fig, ax = plt.subplots()
            ax.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'yellow', 'red', 'blue'])
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
            plt.title("STUDENT PERFORMANCE EXPECTATIONS")
            st.pyplot(fig)
        else:
            st.write("No available features for prediction.")
