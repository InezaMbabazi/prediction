import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the trained model
try:
    model = joblib.load('student_model.pkl')
except FileNotFoundError:
    st.error("Model not found. Please train and save the model first.")
    st.stop()

# Title and description
st.title("University Student's Performance Prediction")
st.write("This model predicts a student's performance using historical data from high school and entry exams.")

# Function to determine performance status based on group trends
def determine_group_performance_status(row, prediction):
    difference = row['Current_Marks'] - prediction
    if difference >= 5:  # Significant increase
        return "Exceeding Expectations"
    elif 0 <= difference < 5:  # Minor increase or close to predicted
        return "Meeting Expectations"
    elif difference < -5:  # Significant decrease
        return "Underperforming"
    else:  # Minor decrease
        return "Meeting Expectations"

# CSV Template for users to download
def create_template():
    template_data = {
        "Student_ID": [],
        "High_School_Grade": [],
        "Entry_Exam_Score": [],
        "National_Exam_Score": [],
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

    # Calculate and display correlations
    if 'High_School_Grade' in df.columns and 'Entry_Exam_Score' in df.columns and 'National_Exam_Score' in df.columns and 'Current_Marks' in df.columns:
        correlation_matrix = df[['High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score', 'Current_Marks']].corr()
        st.write("Correlation Matrix:")
        st.write(correlation_matrix)

        # Add interpretation of the correlation matrix
        st.write("### Interpretation of the Correlation Matrix:")
        st.write("- A positive correlation between `High_School_Grade`, `Entry_Exam_Score`, and `Current_Marks` suggests that higher grades and exam scores are associated with better performance in current marks.")
        st.write("- A strong correlation between `Entry_Exam_Score` and `Current_Marks` (e.g., +0.8) indicates that performance on the entry exam is a strong predictor of current performance.")
        st.write("- A lower correlation between `National_Exam_Score` and `Current_Marks` suggests that it may have a weaker influence on current performance compared to other factors.")
        
    # Predict button
    if st.button("Predict"):
        # Drop rows with missing values in relevant columns
        df_clean = df[['Student_ID', 'High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score', 'Current_Marks']].dropna()

        # Check if there are any rows left after dropping NaNs
        if df_clean.empty:
            st.warning("No valid data available for prediction after dropping rows with missing values.")
        else:
            # Prepare the features for prediction
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
                ax.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
                st.pyplot(fig)

                # Add interpretation of the regression coefficients
                st.write("### Interpretation of the Regression Coefficients:")
                # Assuming you've already retrieved the coefficients from the model
                coefficients = model.coef_
                st.write(f"- The coefficient for `High_School_Grade` is {coefficients[0]:.2f}. This means that for each unit increase in high school grade, the predicted `Current_Marks` increases by {coefficients[0]:.2f}.")
                st.write(f"- The coefficient for `Entry_Exam_Score` is {coefficients[1]:.2f}. This means that for each unit increase in entry exam score, the predicted `Current_Marks` increases by {coefficients[1]:.2f}.")
                st.write(f"- The coefficient for `National_Exam_Score` is {coefficients[2]:.2f}. This means that for each unit increase in national exam score, the predicted `Current_Marks` increases by {coefficients[2]:.2f}.")
                
                # Add R-squared interpretation
                st.write("### Model Evaluation:")
                st.write(f"- The R-squared value of the model is {model.score(features, df_clean['Current_Marks']):.2f}. This indicates that {model.score(features, df_clean['Current_Marks'])*100:.2f}% of the variance in `Current_Marks` is explained by the independent variables (`High_School_Grade`, `Entry_Exam_Score`, and `National_Exam_Score`).")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Footer
st.write("---")
st.write("Powered by Kepler College")
