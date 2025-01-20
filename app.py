import streamlit as st

# Title of the App
st.set_page_config(
    page_title="Data Driven Design Dashboard", 
    page_icon="📊", 
    layout="wide",
    menu_items={'Get Help': None, 'Report a Bug': None, 'About': None}
)

# Sidebar logo and manual navigation
st.sidebar.image("static/images-app/data_driven_design_logo_300.png", use_container_width=True)

# Manually coded sidebar navigation
menu = ["Summary and Results", "Data Overview", "Problem Formulation and Hypothesis", 
        "Regression", "Classification", "Clustering", "Conclusions and Outlook"]
choice = st.sidebar.radio("Select a Page", menu)

# Routing to the respective pages
if choice == "Summary and Results":
    import pages_no_auto_menu.page_1_summary_and_results
    pages_no_auto_menu.page_1_summary_and_results.show_page()
elif choice == "Data Overview":
    import pages_no_auto_menu.page_2_data_overview
    pages_no_auto_menu.page_2_data_overview.show_page()
elif choice == "Problem Formulation and Hypothesis":
    import pages_no_auto_menu.page_3_problem_formulation_and_hypothesis
    pages_no_auto_menu.page_3_problem_formulation_and_hypothesis.show_page()
elif choice == "Regression":
    import pages_no_auto_menu.page_4_ml_regression
    pages_no_auto_menu.page_4_ml_regression.show_page()
elif choice == "Classification":
    import pages_no_auto_menu.page_5_ml_classification
    pages_no_auto_menu.page_5_ml_classification.show_page()
elif choice == "Clustering":
    import pages_no_auto_menu.page_6_ml_clustering
    pages_no_auto_menu.page_6_ml_clustering.show_page()
elif choice == "Conclusions and Outlook":
    import pages_no_auto_menu.page_7_conclusions_and_outlook
    pages_no_auto_menu.page_7_conclusions_and_outlook.show_page()
