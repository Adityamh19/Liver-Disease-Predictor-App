import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost

# 1. Page Styling
st.set_page_config(page_title="Liver Diagnostic AI", page_icon="🩺", layout="centered")

# --- HELPER: Reference Ranges (Approximate standard medical values) ---
# Note: These are general reference ranges. Labs vary.
REF_RANGES = {
    'age': (1, 120),
    'albumin': (35, 55),       # g/L
    'alkaline_phosphatase': (40, 150), # U/L
    'alanine_aminotransferase': (7, 56), # U/L
    'aspartate_aminotransferase': (10, 40), # U/L
    'bilirubin': (1.7, 20.5),  # µmol/L (Assuming standard units for this dataset)
    'cholinesterase': (4, 12), # kU/L
    'cholesterol': (2.5, 7.8), # mmol/L
    'creatinina': (50, 110),   # µmol/L
    'gamma_glutamyl_transferase': (9, 48), # U/L
    'protein': (60, 80)        # g/L
}

def analyze_biomarkers(inputs):
    """Generates a detailed report of high/low values."""
    abnormalities = []
    report_data = []
    
    for feature, value in inputs.items():
        if feature == 'sex': continue # Skip sex
        
        low, high = REF_RANGES.get(feature, (0, 9999))
        status = "Normal"
        
        if value < low:
            status = "Low ⬇️"
            abnormalities.append(f"{feature} is Low")
        elif value > high:
            status = "High ⬆️"
            abnormalities.append(f"{feature} is High")
            
        report_data.append({
            "Biomarker": feature.replace('_', ' ').title(),
            "Patient Value": value,
            "Normal Range": f"{low} - {high}",
            "Status": status
        })
        
    return report_data, abnormalities

# 2. Load Resources
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('liver_disease_pipeline.pkl')
        mapping = joblib.load('class_mapping.pkl')
        return model, mapping
    except Exception as e:
        return None, str(e)

# UI Header
st.title("🩺 Liver Disease Prediction Dashboard")
st.markdown("Enter patient clinical laboratory values below to generate a comprehensive AI analysis.")

# Load the model
with st.spinner("Initializing AI Engine..."):
    resources = load_resources()

if resources[0] is None:
    st.error("🚨 System Error: Model files not found.")
    st.code(resources[1])
    st.stop()

model, class_map = resources
inv_map = {v: k for k, v in class_map.items()}

# 3. Layout: Input Form
with st.form("prediction_form"):
    st.markdown("### 📋 Patient Vitals & Enzymes")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=110, value=45)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        
        st.markdown("**Liver Enzymes**")
        alt = st.number_input("ALT (Alanine Aminotransferase) [U/L]", value=20.0)
        ast = st.number_input("AST (Aspartate Aminotransferase) [U/L]", value=25.0)
        alp = st.number_input("ALP (Alkaline Phosphatase) [U/L]", value=50.0)
        ggt = st.number_input("GGT (Gamma-Glutamyl Transferase) [U/L]", value=20.0)

    with col2:
        st.write("") # Spacer
        st.write("") 
        st.markdown("**Proteins & Others**")
        alb = st.number_input("ALB (Albumin) [g/L]", value=38.0)
        prot = st.number_input("PROT (Total Protein) [g/L]", value=70.0)
        bil = st.number_input("BIL (Bilirubin) [µmol/L]", value=5.0)
        che = st.number_input("CHE (Cholinesterase) [kU/L]", value=9.0)
        chol = st.number_input("CHOL (Cholesterol) [mmol/L]", value=4.5)
        crea = st.number_input("CREA (Creatinine) [µmol/L]", value=70.0)

    submit_btn = st.form_submit_button("🛡️ Run Full Diagnostic Analysis", use_container_width=True)

# 4. Prediction & Detailed Analysis Logic
if submit_btn:
    # 4.1 Prepare Data
    input_dict = {
        'age': age, 'sex': sex, 'albumin': alb, 'alkaline_phosphatase': alp,
        'alanine_aminotransferase': alt, 'aspartate_aminotransferase': ast,
        'bilirubin': bil, 'cholinesterase': che, 'cholesterol': chol,
        'creatinina': crea, 'gamma_glutamyl_transferase': ggt, 'protein': prot
    }
    
    # DataFrame must match training order
    input_df = pd.DataFrame([[
        age, sex, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot
    ]], columns=input_dict.keys())

    try:
        # 4.2 Get AI Prediction
        prediction_idx = model.predict(input_df)[0]
        result_text = inv_map.get(prediction_idx, "Unknown")
        
        # 4.3 Generate Detailed Report
        report_data, abnormalities = analyze_biomarkers(input_dict)
        report_df = pd.DataFrame(report_data)

        # --- RESULTS DISPLAY ---
        st.divider()
        st.markdown("## 🔍 Diagnostic Report")

        # A. Primary Outcome
        safe_labels = ["Blood Donor", "no_disease", "suspect Blood Donor"]
        is_healthy = any(label in str(result_text) for label in safe_labels)
        
        if is_healthy:
            st.success(f"### AI Prediction: {result_text} (Healthy)")
            st.markdown("The model detects **no significant patterns** of liver disease based on the provided parameters.")
        else:
            st.error(f"### AI Prediction: {result_text} (Pathology Detected)")
            st.markdown(f"The model has detected patterns matching **{result_text}**. Please proceed with clinical correlation.")

        # B. Detailed Analysis Tabs
        st.write("")
        tab1, tab2, tab3 = st.tabs(["👤 Patient Explanation", "🩺 Clinical Data (Doctor's View)", "📊 Biomarker Table"])

        with tab1:
            st.markdown("#### **What does this mean for you?**")
            if is_healthy:
                st.write("✅ **Good News:** Your results look generally healthy according to our AI model.")
                if abnormalities:
                    st.warning(f"However, we noticed slight variations in: **{', '.join([x.split()[0] for x in abnormalities])}**. This can sometimes happen due to diet, medication, or minor infections.")
                else:
                    st.write("All your key liver markers appear to be within the expected range.")
            else:
                st.write("⚠️ **Attention:** The AI has flagged a potential issue with your liver health.")
                st.write("This is likely due to the following indicators being outside the typical range:")
                for abn in abnormalities:
                    st.write(f"- {abn}")
                st.info("Please consult a healthcare provider for further testing.")

        with tab2:
            st.markdown("#### **Physician Notes**")
            st.write(f"**Primary Prediction:** {result_text}")
            st.write("**Key Findings:**")
            if abnormalities:
                st.write(f"Patient presents with **{len(abnormalities)} abnormal biomarkers**.")
                st.write(f"• Significant deviations: {', '.join(abnormalities)}")
            else:
                st.write("• No significant biomarker deviations detected against standard reference ranges.")
            
            # AST/ALT Ratio (De Ritis Ratio) calculation
            try:
                ratio = ast / alt
                st.write(f"**AST/ALT Ratio:** {ratio:.2f}")
                if ratio > 2.0:
                    st.caption("-> Ratio > 2.0 is often suggestive of alcoholic liver disease.")
                elif ratio < 1.0:
                    st.caption("-> Ratio < 1.0 is often seen in NAFLD or viral hepatitis.")
            except ZeroDivisionError:
                pass

        with tab3:
            st.markdown("#### **Detailed Laboratory Values**")
            
            def highlight_status(val):
                color = 'red' if 'High' in val or 'Low' in val else 'green'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                report_df.style.map(highlight_status, subset=['Status']),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"Analysis Error: {e}")

    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This tool is an AI prototype for educational and assistive purposes only. It is not a substitute for professional medical diagnosis. Standard reference ranges used here may vary by laboratory.")
