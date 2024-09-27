import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

st.title("Student Enrollment Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset", type="csv")

if uploaded_file:
    # Read dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    # Feature selection (modify based on your dataset)
    features = ['Principle Passes', 'Option']
    df = df.dropna(subset=features)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # K-Means Clustering
    kmeans = KMeans(n_clusters=5)  # Change number of clusters as needed
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    # Group by secondary school
    school_group = df.groupby('Secondary School Name')['Cluster'].count()

    # Display predictions
    st.write("Predicted number of students per school", school_group)
