
import streamlit as st
import pickle
import numpy as np

# Load the models
try:
    with open("placement_model.pkl", "rb") as f:
        placement_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: placement_model.pkl not found. Please ensure the model file exists.")
    st.stop()

try:
    with open("salary_model.pkl", "rb") as f:
        salary_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: salary_model.pkl not found. Please ensure the model file exists.")
    st.stop()

def predict_placement_and_salary(cgpa, projects, workshops, mini_projects, skills, communication_skills, internship, hackathon, tw_percentage, te_percentage, backlogs):
    """Predicts placement status and salary based on user input."""

    s = len(skills.split(',')) if skills else 0

    # Placement prediction
    placement_input = np.array([
        cgpa, projects, workshops, mini_projects, s, communication_skills,
        internship, hackathon, tw_percentage, te_percentage, backlogs
    ]).astype(float)

    placement_prediction = placement_model.predict([placement_input])[0]

    # Salary prediction
    p = 1 if placement_prediction == "Placed" else 0
    salary_input = np.array([
        cgpa, projects, workshops, mini_projects, s, communication_skills,
        internship, hackathon, tw_percentage, te_percentage, backlogs, p
    ]).astype(float)

    salary_prediction = salary_model.predict([salary_input])[0]

    return placement_prediction, salary_prediction

def main():
    """Streamlit app for placement and salary prediction."""

    st.set_page_config(page_title="Placement and Salary Predictor", layout="wide")

    st.title("🎓 Placement and Salary Predictor")
    st.write("Enter your details to predict your placement status and expected salary.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input("Full Name")
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
            projects = st.number_input("Number of Projects", min_value=0, step=1)
            workshops = st.number_input("Number of Workshops", min_value=0, step=1)

        with col2:
            mini_projects = st.number_input("Number of Mini Projects", min_value=0, step=1)
            skills = st.text_input("Skills (comma-separated)")
            communication_skills = st.slider("Communication Skills (1-5)", 1, 5, 3)
            internship = st.selectbox("Internship?", ("Yes", "No"))

        with col3:
            hackathon = st.selectbox("Participated in Hackathon?", ("Yes", "No"))
            tw_percentage = st.number_input("12th Percentage", min_value=0.0, max_value=100.0, step=0.1)
            te_percentage = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, step=0.1)
            backlogs = st.number_input("Number of Backlogs", min_value=0, step=1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        internship_val = 1 if internship == "Yes" else 0
        hackathon_val = 1 if hackathon == "Yes" else 0

        placement_status, salary = predict_placement_and_salary(
            cgpa, projects, workshops, mini_projects, skills, communication_skills,
            internship_val, hackathon_val, tw_percentage, te_percentage, backlogs
        )

        st.subheader("Prediction Results")

        if placement_status == "Placed":
            st.success(f"Congratulations {name}! You have a high chance of getting placed.")
            st.info(f"Expected Salary: ₹{salary:,.2f} per annum")
        else:
            st.error(f"Sorry {name}, you have a low chance of getting placed.")
            st.warning("Focus on improving your skills and experience.")

        st.subheader("What-If Analysis")
        st.write("Edit the values below to see how they affect the predictions.")

        col1, col2, col3 = st.columns(3)

        with col1:
            new_cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=cgpa, step=0.1)
            new_projects = st.slider("Number of Projects", min_value=0, max_value=20, value=projects, step=1)
            new_workshops = st.slider("Number of Workshops", min_value=0, max_value=20, value=workshops, step=1)

        with col2:
            new_mini_projects = st.slider("Number of Mini Projects", min_value=0, max_value=20, value=mini_projects, step=1)
            new_skills = st.text_input("Skills (comma-separated)", value=skills)
            new_communication_skills = st.slider("Communication Skills (1-5)", 1, 5, value=communication_skills)
            new_internship = st.selectbox("Internship?", ("Yes", "No"), index=0 if internship == "Yes" else 1)

        with col3:
            new_hackathon = st.selectbox("Participated in Hackathon?", ("Yes", "No"), index=0 if hackathon == "Yes" else 1)
            new_tw_percentage = st.slider("12th Percentage", min_value=0.0, max_value=100.0, value=tw_percentage, step=0.1)
            new_te_percentage = st.slider("10th Percentage", min_value=0.0, max_value=100.0, value=te_percentage, step=0.1)
            new_backlogs = st.slider("Number of Backlogs", min_value=0, max_value=10, value=backlogs, step=1)

        new_internship_val = 1 if new_internship == "Yes" else 0
        new_hackathon_val = 1 if new_hackathon == "Yes" else 0

        new_placement_status, new_salary = predict_placement_and_salary(
            new_cgpa, new_projects, new_workshops, new_mini_projects, new_skills, new_communication_skills,
            new_internship_val, new_hackathon_val, new_tw_percentage, new_te_percentage, new_backlogs
        )

        st.subheader("New Prediction Results")

        if new_placement_status == "Placed":
            st.success("This student would likely be placed.")
            st.info(f"New Expected Salary: ₹{new_salary:,.2f} per annum")
        else:
            st.error("This student would likely not be placed.")

if __name__ == "__main__":
    main()
