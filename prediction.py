import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Function to train and save the model if it does not exist
def train_model():
    if not os.path.exists('student_model.pkl'):
        # Sample dataset (you can replace this with your own data)
        data = {
            'High_School_Grade': [85, 90, 60, 75, 82],
            'Entry_Exam_Score': [78, 85, 65, 70, 80],
            'Current_Marks': [75, 88, 58, 72, 79]
        }
        df = pd.DataFrame(data)

        X = df[['High_School_Grade', 'Entry_Exam_Score']]
        y = df['Current_Marks']

        model = LinearRegression()
        model.fit(X, y)

        joblib.dump(model, 'student_model.pkl')

# Train the model if it doesn't exist
train_model()

# Load the trained model
model = joblib.load('student_model.pkl')

# Rest of your Streamlit app
