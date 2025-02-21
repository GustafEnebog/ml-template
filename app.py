import streamlit as st
from app_pages.multipage import MultiPage

# Set up the page config (optional, but useful for customizing)
st.set_page_config(
    page_title="Machine Learning Project App",  # Title of the app
    page_icon="📊",  # Icon shown in the browser tab
    layout="wide",  # Layout option, can be 'centered' or 'wide'
    menu_items={
        'Get Help': None,  # Optional menu item
        'Report a Bug': None,  # Optional menu item
        'About': "This is a machine learning project app that demonstrates various ML models."  # Customize the About section
    }
)

# load pages scripts
from app_pages.page_1_summary_and_results import page_1_summary_and_results_body
from app_pages.page_2_data_overview import page_2_data_overview_body
from app_pages.page_3_problem_formulation_and_hypothesis import page_3_problem_formulation_and_hypothesis_body
from app_pages.page_4_ml_regression import page_4_ml_regression_body
from app_pages.page_5_ml_classification import page_5_ml_classification_body
from app_pages.page_6_ml_clustering import page_6_ml_clustering_body
from app_pages.page_7_conclusions_and_outlook import page_7_conclusions_and_outlook_body

app = MultiPage(app_name="Machine Learning Project App")  # Create an instance of the app 

# Add pages to the MultiPage app
app.add_page("Summary and Results", page_1_summary_and_results_body)
app.add_page("Data Overview", page_2_data_overview_body)
app.add_page("Problem Formulation and Hypothesis", page_3_problem_formulation_and_hypothesis_body)
app.add_page("ML Regression", page_4_ml_regression_body)
app.add_page("ML Classification", page_5_ml_classification_body)
app.add_page("ML Clustering", page_6_ml_clustering_body)
app.add_page("Conclusions and Outlook", page_7_conclusions_and_outlook_body)

app.run()  # Run the app
