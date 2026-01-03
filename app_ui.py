import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost

# 1. Page Styling (Must be the first Streamlit command)
st.set_page_config(page_title="Liver Diagnostic AI", page_icon="🩺", layout="centered")

# 2. Load Resources with Caching
@st.cache_resource
def load_resources():
    try:
        # Load model and mapping
        loaded_model = joblib.load('liver_disease_pipeline.pkl')
        loaded_map = joblib.load('class_mapping.pkl')
        return loaded_model, loaded_map
    except Exception as e:
        # Return None and the error message if something fails
        return None, str(e)

# UI Header
st.title("🩺 Liver Disease Prediction Dashboard")
st.markdown("Enter patient clinical laboratory values to get an AI-driven diagnosis.")

# Load the model
with st.spinner("Loading AI Model..."):
    # Load the resources into a temporary variable
    resources = load_resources()

# Check if loading failed
if resources[0] is None:
    st.error("🚨 Error loading model files!")
    st.code(f"Details: {resources[1]}")
    st.warning("Please ensure 'liver_disease_pipeline.pkl' and 'class_mapping.pkl' are in the GitHub repository.")
    st.stop()

# Assign the loaded resources to the correct variables
model = resources[0]
class_map = resources[1]  # <--- This fixes the NameError you are seeing

# Create Inverse Mapping (Number -> Label)
inv_map = {v: k for k, v in class_map.items()}

# 3. Layout: Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Basics")
        age = st.number_input("Age", min_value=1, max_value=110, value=45)
        # Note: Notebook uses 1=Male, 0=Female
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        
        st.subheader("Liver Enzymes")
        alt = st.number_input("ALT (Alanine Aminotransferase)", value=20.0)
        ast = st.number_input("AST (Aspartate Aminotransferase)", value=25.0)
        alp = st.number_input("ALP (Alkaline Phosphatase)", value=50.0)
        ggt = st.number_input("GGT (Gamma-Glutamyl Transferase)", value=20.0)

    with col2:
        st.subheader("Proteins & Others")
        alb = st.number_input("ALB (Albumin)", value=38.0)
        prot = st.number_input("PROT (Total Protein)", value=70.0)
        bil = st.number_input("BIL (Bilirubin)", value=5.0)
        che = st.number_input("CHE (Cholinesterase)", value=9.0)
        chol = st.number_input("CHOL (Cholesterol)", value=4.5)
        crea = st.number_input("CREA (Creatinina)", value=70.0)

    # Submit Button
    submit_btn = st.form_submit_button("Run Diagnostic Analysis", use_container_width=True)

# 4. Prediction Logic
if submit_btn:
    # DataFrame construction matching training order exactly
    input_df = pd.DataFrame([[
        age, sex, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot
    ]], columns=[
        'age', 'sex', 'albumin', 'alkaline_phosphatase', 
        'alanine_aminotransferase', 'aspartate_aminotransferase', 
        'bilirubin', 'cholinesterase', 'cholesterol', 
        'creatinina', 'gamma_glutamyl_transferase', 'protein'
    ])

    try:
        # Get Prediction
        prediction_idx = model.predict(input_df)[0]
        result_text = inv_map.get(prediction_idx, "Unknown Condition")

        st.divider()
        st.subheader("Diagnostic Result:")
        
        # Logic to determine color based on result text
        safe_labels = ["Blood Donor", "no_disease", "suspect Blood Donor"]
        
        if any(label in str(result_text) for label in safe_labels):
            st.success(f"Outcome: **{result_text}** (Likely Healthy)")
        else:
            st.error(f"Outcome: **{result_text}** (Pathology Detected)")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

    st.info("Disclaimer: This is an AI assistant. Please consult a doctor for final medical decisions.")
