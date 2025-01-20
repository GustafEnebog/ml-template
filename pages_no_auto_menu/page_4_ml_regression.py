import streamlit as st

# Define the show_page function for this page
def show_page():
    # Page Title
    st.title("Regression")

    # Overview of regression models used
    st.markdown("""
    ## Regression Models:
    - Linear Regression
    - Decision Trees
    - Random Forest
    """)

    # Model Performance Visualization
    st.subheader("Model Performance")
    st.write("Regression Model Performance: RMSE, MAE, etc.")
