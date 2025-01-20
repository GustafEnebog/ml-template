import streamlit as st
import pandas as pd

# Define the show_page function for this page
def show_page():
    # Page Title
    st.title("Data Overview")

    # Dataset Overview Section with a heading
    st.subheader("Dataset Overview")
    st.markdown("""
    - Brief description of the data (size, features, etc.)
    """)

    # Sample Data (replace with your actual dataset)
    df = pd.DataFrame({
        'Feature 1': [1, 2, 3, 4],
        'Feature 2': [10, 20, 30, 40]
    })

    # Add a heading for the table
    st.subheader("Sample Data")
    st.write(df)

    # Basic Statistics Section
    st.subheader("Basic Statistics")
    st.write(df.describe())
