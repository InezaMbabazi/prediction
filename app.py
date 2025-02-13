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
st.write("This model predicts a student's performance using High School Grades, Entry Exams, and National Exam Scores.")

# CSV Template for users to download
def create_template():
    template_data = {
        "Student_ID": [],
        "High_School_Grade": [],
        "Entry_Exam_Score": [],
        "National_Exam_Score": [],
        "Current_Marks": []
    }
    return pd.DataFrame(template_data)

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
    if all(col in df.columns for col in ['High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score', 'Current_Marks']):
        correlation_matrix = df[['High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score', 'Current_Marks']].corr()
        st.write("Correlation Matrix:")
        st.write(correlation_matrix)

    # Predict button
    if st.button("Predict"):
        # Drop rows with missing values
        df_clean = df.dropna(subset=['High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score', 'Current_Marks'])

        if df_clean.empty:
            st.warning("No valid data available for prediction after removing missing values.")
        else:
            # Prepare the features for prediction
            features = df_clean[['High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score']].values

            try:
                # Make predictions
                predictions = model.predict(features)

                # Add predictions to the dataframe
                df_clean['Predicted_Marks'] = predictions

                # Display predictions
                st.write("Predictions:")
                st.write(df_clean[['Student_ID', 'High_School_Grade', 'Entry_Exam_Score', 'National_Exam_Score', 'Current_Marks', 'Predicted_Marks']])

                # Plot actual vs. predicted marks
                st.write("Actual vs. Predicted Marks:")
                fig, ax = plt.subplots()
                ax.scatter(df_clean['Current_Marks'], df_clean['Predicted_Marks'], alpha=0.5, label="Predicted vs Actual")
                ax.plot([df_clean['Current_Marks'].min(), df_clean['Current_Marks'].max()],
                        [df_clean['Current_Marks'].min(), df_clean['Current_Marks'].max()],
                        'r--', label="Perfect Prediction")
                ax.set_xlabel("Actual Marks")
                ax.set_ylabel("Predicted Marks")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Footer
st.write("---")
st.write("Powered by Kepler College")
