import pickle

import pandas as pd
import streamlit as st


 
def load_model():
    with open("best_model.pkl", "rb") as file:
        return pickle.load(file)


pipe = load_model()

st.set_page_config(page_title="Credit Scoring Model", page_icon=":credit_card:", layout="wide")

st.title("💳Credit Scoring Model")
st.markdown(
    "Predict loan approval status based on applicant details. "
    "Fill in the form below and click **Check Loan Status**."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a machine learning model trained on historical loan data "
        "to predict approval/rejection."
    )
    st.write("**Model Accuracy:** ~78%")
    st.write(
        "**Features Used:** Gender, Marital Status, Education, Loan Amount, "
        "Credit History, Total Income, Loan Term."
    )
    st.markdown("---")
    st.write("**Units:**")
    st.write("- Loan Amount: In thousands, e.g. 100 = 100,000")
    st.write("- Total Income: ApplicantIncome + CoapplicantIncome")
    st.write("- Loan Term: In months")
    if st.button("🔄Reset App"):
        st.rerun()


st.header("📝Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Select", "Male", "Female"])
    married = st.selectbox("Marital Status", ["Select", "Yes", "No"])
    education = st.selectbox("Education", ["Select", "Graduate", "Not Graduate"])

with col2:
    loan_amount = st.number_input(
        "Loan Amount (in thousands)",
        min_value=0,
        step=50,
        help="Example: enter 160 for 160,000.",
    )
    credit_history_selection = st.selectbox(
        "Credit History",
        ["Select", "Good", "Bad"],
        help="Good = 1.0, Bad = 0.0",
    )
    total_income = st.number_input(
        "Total Income",
        min_value=0,
        step=1000,
        help="ApplicantIncome + CoapplicantIncome from the dataset.",
    )

with col3:
    loan_amount_term = st.number_input(
        "Loan Term (months)",
        min_value=0,
        step=12,
        help="Example: 360 = 30 years.",
    )


st.header("🔍Prediction Result")

if st.button("🚀Check Loan Status", type="primary"):
    missing_selection = (
        gender == "Select"
        or married == "Select"
        or education == "Select"
        or credit_history_selection == "Select"
    )

    if missing_selection:
        st.error("Please select all required fields.")
    elif loan_amount == 0 or total_income == 0 or loan_amount_term == 0:
        st.error("Please enter valid loan amount, total income, and loan term.")
    else:
        credit_history = 1.0 if credit_history_selection == "Good" else 0.0
        input_data = pd.DataFrame(
            {
                "Gender": [gender],
                "Married": [married],
                "Education": [education],
                "LoanAmount": [loan_amount],
                "Loan_Amount_Term": [loan_amount_term],
                "Credit_History": [credit_history],
                "TotalIncome": [total_income],
            }
        )

        prediction = pipe.predict(input_data)[0]
        prediction_proba = pipe.predict_proba(input_data)[0]

        if prediction == 1:
            st.success("Loan Approved!")
        else:
            st.error("Loan Rejected.")

        st.subheader("📊Prediction Details")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Approval Probability", f"{prediction_proba[1] * 100:.2f}%")
        with col_b:
            st.metric("Rejection Probability", f"{prediction_proba[0] * 100:.2f}%")

        st.write("**Model Prediction:**", "Approved" if prediction == 1 else "Rejected")

st.markdown("---")
st.caption("Built with Streamlit | Model trained on historical loan data")
