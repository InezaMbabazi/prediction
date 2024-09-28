import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Try to load the model, or train it if not available
try:
    model = joblib.load('student_model.pkl')
except FileNotFoundError:
    st.write("Model not found. Training a new model...")
    
    # Sample training data (you should replace this with your own dataset)
    data = {
        'High_School_Grade': [85, 90, 60, 75, 82],
        'Entry_Exam_Score': [78, 85, 65, 70, 80],
        'Current_Marks': [75, 88, 58, 72, 79]
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

# Streamlit app for prediction
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
