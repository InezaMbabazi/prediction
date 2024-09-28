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
st.title("University Student's Prediction")
st.write("This model will be used to determine a student's performance using historical data from high school and entry exams.")

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
        # Drop rows with missing values in relevant columns
        df_clean = df[['Student_ID', 'High_School_Grade', 'Entry_Exam_Score', 'Current_Marks']].dropna()

        # Check if there are any rows left after dropping NaNs
        if df_clean.empty:
            st.warning("No valid data available for prediction after dropping rows with missing values.")
        else:
            # Prepare the features for prediction
            features = df_clean[['High_School_Grade', 'Entry_Exam_Score']].values
            
            # Make predictions
            try:
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
                st.write(df_clean[['Student_ID', 'High_School_Grade', 'Entry_Exam_Score', 'Current_Marks', 'Predicted_Marks', 'Performance_Status']])
                
                # Plot a pie chart for performance status
                performance_counts = df_clean['Performance_Status'].value_counts()

                st.write("Performance Status Breakdown:")
                fig, ax = plt.subplots()
                ax.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'yellow', 'red'])
                ax.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Footer
st.write("---")
st.write("Powered by Kepler College")
