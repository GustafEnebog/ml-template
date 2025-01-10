import streamlit as st

# Title of the App
st.set_page_config(page_title="Data Driven Design Dashboard", page_icon="📊", layout="wide")

# Adding the logo to the sidebar (left menu)
st.sidebar.image("static/images-app/data_driven_design_logo_300.png", use_column_width=True)

# Sidebar navigation menu
menu = ["Summary and Results", "Data Overview", "Problem Formulation and Hypothesis", 
        "Regression", "Classification", "Clustering", "Conclusions and Outlook"]
choice = st.sidebar.radio("Select a Page", menu)

# Routing to the respective pages
if choice == "Summary and Results":
    import pages.1_summary_results
elif choice == "Data Overview":
    import pages.2_data_overview
elif choice == "Problem Formulation and Hypothesis":
    import pages.3_problem_formulation_and_hypothesis
elif choice == "Regression":
    import pages.4_ml_regression
elif choice == "Classification":
    import pages.5_ml_classification
elif choice == "Clustering":
    import pages.6_ml_clustering
elif choice == "Conclusions and Outlook":
    import pages.7_conclusions_and_outlook
