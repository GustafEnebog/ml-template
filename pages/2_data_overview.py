import streamlit as st
import pandas as pd

# Page Title
st.title("Data Overview")

# Basic content for the page
st.markdown("""
## Dataset Description:
- Brief description of the data (size, features, etc.)
""")

# Sample Data (replace with your actual dataset)
df = pd.DataFrame({
    'Feature 1': [1, 2, 3, 4],
    'Feature 2': [10, 20, 30, 40]
})
st.write(df)

# Visualizations or summaries
st.subheader("Basic Statistics")
st.write(df.describe())
