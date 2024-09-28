import os
import joblib

# Set the path to your model
model_path = os.path.join(os.path.dirname(__file__), 'student_model.pkl')

# Load the model
model = joblib.load(model_path)
