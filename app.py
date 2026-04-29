import streamlit as st
import pandas as pd
import pickle


with open('best_model.pkl', 'rb') as file:
    pipe = pickle.load(file)


print("Model Loaded Successfully")

st.set_page_config(page_title="Credit Scoring Model", page_icon="💳", layout="wide")
    

st.title("💳 Credit Scoring Model")
st.markdown("Predict loan approval status based on applicant details. Fill in the form below and click **Check Loan Status**.")

# Sidebar for additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a machine learning model trained on historical loan data to predict approval/rejection.")
    st.write("**Model Accuracy:** ~65%")
    st.write("**Features Used:** Gender, Marital Status, Education, Loan Amount, Credit History, Total Income, Loan Term.")
    st.markdown("---")
    st.write("**Units:**")
    st.write("- Loan Amount: In thousands (e.g., 100 = 100,000)")
    st.write("- Total Income: Same currency as loan amount")
    st.write("- Loan Term: In months")
    if st.button("🔄 Reset App"):
        st.rerun()


st.header("📝 Applicant Details")

col1, col2, col3 = st.columns(3)
with col1:
    Gender = st.selectbox("Gender", ["Select", "Male", "Female"], help="Applicant's gender")
    Married = st.selectbox("Marital Status", ["Select", "Yes", "No"], help="Is the applicant married?")
    Education = st.selectbox("Education", ["Select", "Graduate", "Not Graduate"], help="Education level")

with col2:
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0, help="Loan amount requested, e.g., 100 = 100,000")
    Credit_History = st.selectbox("Credit History", ["Select", 1.0, 0.0], help="1.0 = Good credit history, 0.0 = Bad credit history")
    TotalIncome = st.number_input("Total Income", min_value=0, help="Applicant's total income (applicant + co-applicant)")

with col3:
    Loan_Amount_Term = st.number_input("Loan Term (months)", min_value=1, help="Repayment period in months, e.g., 360 = 30 years")
    
    # Auto-calculate EMI
    if LoanAmount > 0 and Loan_Amount_Term > 0:
        EMI = LoanAmount / Loan_Amount_Term
        st.metric("Estimated Monthly EMI", f"{EMI:.2f}", help="Simple calculation: Loan Amount / Term")
    else:
        st.write("Enter loan amount and term to see EMI")



st.header("🔍 Prediction Result")

if st.button("🚀 Check Loan Status", type="primary"):
    # Validate that all fields are properly selected
    if Gender == "Select" or Married == "Select" or Education == "Select" or Credit_History == "Select":
        st.error("❌ Please select all required fields!")
    elif LoanAmount == 0 or TotalIncome == 0:
        st.error("❌ Please enter valid loan amount and total income!")
    else:
        with st.spinner("Analyzing your application..."):
            input_data = pd.DataFrame({
                "Gender": [Gender],
                "Married": [Married],
                "Education": [Education],
                "LoanAmount": [LoanAmount],
                "Credit_History": [Credit_History],
                "TotalIncome": [TotalIncome],
                "Loan_Amount_Term": [Loan_Amount_Term]
            })

            prediction = pipe.predict(input_data)
            prediction_proba = pipe.predict_proba(input_data)
            
            # Display result
            if prediction[0] == 1:
                st.success("✅ **Loan Approved!** Congratulations!")
                st.balloons()
            else:
                st.error("❌ **Loan Rejected** Sorry, your application does not meet the criteria.")
            
            st.subheader("📊 Prediction Details")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Approval Probability", f"{prediction_proba[0][1]*100:.2f}%")
            with col_b:
                st.metric("Rejection Probability", f"{prediction_proba[0][0]*100:.2f}%")
            
            st.write("**Model Prediction:**", "Approved" if prediction[0] == 1 else "Rejected")
            st.info("💡 Tip: A good credit history and higher income improve approval chances!")

st.markdown("---")
st.caption("Built with Streamlit | Model trained on historical loan data | For educational purposes only")