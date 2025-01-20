import streamlit as st

# Define the show_page function for this page
def show_page():
    # Page Title
    st.title("Classification")

    # Overview of classification models used
    st.markdown("""
    ## Classification Models:
    - Logistic Regression
    - Random Forest
    - Support Vector Machine
    """)

    # Model Performance Visualization
    st.subheader("Model Performance")
    st.write("Classification Model Performance: Accuracy, Precision, Recall, etc.")
