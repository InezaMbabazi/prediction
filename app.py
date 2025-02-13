import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
try:
    model = joblib.load('student_model.pkl')
except FileNotFoundError:
    st.error("Model not found. Please train and save the model first.")
    st.stop()

# Title and description
st.title("University Student's Performance Prediction")
st.write("This model predicts a student's performance using high school, entry exam, and national exam scores.")

# Function to determine performance status
def determine_group_performance_status(row, prediction):
    difference = row['Current_Marks'] - prediction
    if difference >= 5:
        return "Exceeding Expectations"
    elif 0 <= difference < 5:
        return "Meeting Expectations"
    elif difference < -5:
        return "Underperforming"
    else:
        return "Meeting Expectations"

# CSV Template for users to download
def create_template():
    template_data = {
        "Student_ID": [],
        "High_School_Grade": [],
        "Entry_Exam_Score": [],
        "National_Exam_Score": [],  # New column
        "Current_Marks": []
    }
    df_template = pd.DataFrame(template_data)
    return df_template

# Provide the template CSV file for download
st.write("Download the template CSV file:")
df_template = create_template()
st.download_button("Download Template", df_template.to_csv(index=False), file_name="student_performance_template.csv", mime='text/csv')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Display input data
    st.write("Input Data:")
    st.write(df)

    # Validate required columns
    required_columns = ['High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score', 'Current_Marks']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Uploaded CSV must contain the following columns: {required_columns}")
        st.stop()

    # Calculate and display correlations
    correlation_matrix = df[required_columns].corr()
    st.write("Correlation Matrix:")
    st.write(correlation_matrix)

    # Predict button
    if st.button("Predict"):
        # Drop rows with missing values
        df_clean = df[['Student_ID', 'High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score', 'Current_Marks']].dropna()

        # Check if there are any valid rows left
        if df_clean.empty:
            st.warning("No valid data available for prediction after dropping rows with missing values.")
        else:
            # Prepare features for prediction (now using 3 features)
            features = df_clean[['High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score']].values

            try:
                # Make predictions
                predictions = model.predict(features)

                # Determine performance status for each student
                performance_status = []
                for i, row in df_clean.iterrows():
                    prediction = predictions[i]
                    status = determine_group_performance_status(row, prediction)
                    performance_status.append(status)

                # Add predictions and performance status to the dataframe
                df_clean['Predicted_Marks'] = predictions
                df_clean['Performance_Status'] = performance_status

                # Display predictions and performance status
                st.write("Predictions and Performance Status:")
                st.write(df_clean[['Student_ID', 'High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score', 'Current_Marks', 'Predicted_Marks', 'Performance_Status']])

                # Plot a pie chart for performance status
                performance_counts = df_clean['Performance_Status'].value_counts()
                st.write("Performance Status Breakdown:")
                fig, ax = plt.subplots()
                ax.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'yellow', 'red'])
                ax.axis('equal')  # Ensure pie chart is circular
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Footer
st.write("---")
st.write("Powered by Kepler College")
