import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Liver Diagnostic AI | Professional Edition", 
    page_icon="🩺", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- HELPER: Medical Reference Ranges (Updated to match Model Feature Names) ---
REF_RANGES = {
    'age': (1, 120),
    'albumin': (35, 55),       
    'alkaline_phosphatase': (40, 150),      
    'alanine_aminotransferase': (7, 56),       
    'aspartate_aminotransferase': (10, 40),       
    'bilirubin': (1.7, 20.5),    
    'cholinesterase': (4, 12),        
    'cholesterol': (2.5, 7.8),    
    'creatinina': (50, 110),     
    'gamma_glutamyl_transferase': (9, 48),        
    'protein': (60, 80)       
}

def get_abnormalities(inputs):
    """Identifies which markers are out of range."""
    issues = []
    for feature, value in inputs.items():
        if feature == 'sex': continue
        # cleanup feature name for display
        display_name = feature.replace('_', ' ').title()
        low, high = REF_RANGES.get(feature, (0, 9999))
        
        if value < low:
            issues.append(f"Low {display_name} ({value})")
        elif value > high:
            issues.append(f"Elevated {display_name} ({value})")
    return issues

def plot_probabilities(proba_dict):
    """Creates a professional bar chart of prediction probabilities."""
    sorted_probs = dict(sorted(proba_dict.items(), key=lambda item: item[1], reverse=True))
    fig = go.Figure(go.Bar(
        x=list(sorted_probs.values()),
        y=list(sorted_probs.keys()),
        orientation='h',
        marker_color=['#ff4b4b' if 'disease' not in k.lower() and 'Donor' not in k else '#00cc96' for k in sorted_probs.keys()]
    ))
    fig.update_layout(title="AI Confidence Distribution", xaxis_title="Probability", height=300, margin=dict(l=0,r=0,t=30,b=0))
    return fig

# 2. Load Resources
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('liver_disease_pipeline.pkl')
        mapping = joblib.load('class_mapping.pkl')
        return model, mapping
    except Exception as e:
        return None, str(e)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050479.png", width=80)
    st.title("Liver AI Diagnostic")
    st.info("System Ready. Using XGBoost Architecture.")

# --- MAIN PAGE ---
st.title("🩺 Advanced Liver Disease Prediction")
st.markdown("### Clinical Interface")

# Load model
resources = load_resources()
if resources[0] is None:
    st.error("🚨 System Error: Model files missing.")
    st.stop()
model, class_map = resources
inv_map = {v: k for k, v in class_map.items()}

# INPUT FORM
with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. Demographics")
        age = st.number_input("Age", 45)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    with c2:
        st.subheader("2. Enzymes")
        alt = st.number_input("ALT (Alanine Transaminase)", value=20.0)
        ast = st.number_input("AST (Aspartate Transaminase)", value=25.0)
        alp = st.number_input("ALP (Alkaline Phosphatase)", value=50.0)
        ggt = st.number_input("GGT (Gamma-Glutamyl Transferase)", value=20.0)
    with c3:
        st.subheader("3. Proteins")
        alb = st.number_input("ALB (Albumin)", value=38.0)
        prot = st.number_input("PROT (Total Protein)", value=70.0)
        bil = st.number_input("BIL (Bilirubin)", value=5.0)
        che = st.number_input("CHE (Cholinesterase)", value=9.0)
        chol = st.number_input("CHOL (Cholesterol)", value=4.5)
        crea = st.number_input("CREA (Creatinine)", value=70.0)

    analyze = st.form_submit_button("🔍 Run Advanced Analysis", use_container_width=True)

if analyze:
    # STRICT COLUMN MATCHING: Mapping inputs to the EXACT names the model expects
    input_data = {
        'age': age, 
        'sex': sex, 
        'albumin': alb, 
        'alkaline_phosphatase': alp, 
        'alanine_aminotransferase': alt, 
        'aspartate_aminotransferase': ast, 
        'bilirubin': bil, 
        'cholinesterase': che, 
        'cholesterol': chol, 
        'creatinina': crea, 
        'gamma_glutamyl_transferase': ggt, 
        'protein': prot
    }
    
    # Ensure DataFrame columns are in the correct order
    cols_order = [
        'age', 'sex', 'albumin', 'alkaline_phosphatase', 
        'alanine_aminotransferase', 'aspartate_aminotransferase', 
        'bilirubin', 'cholinesterase', 'cholesterol', 
        'creatinina', 'gamma_glutamyl_transferase', 'protein'
    ]
    
    input_df = pd.DataFrame([input_data], columns=cols_order)

    try:
        # Prediction
        pred_idx = model.predict(input_df)[0]
        result_text = inv_map.get(pred_idx, "Unknown")
        probs = model.predict_proba(input_df)[0]
        proba_dict = {inv_map[i]: p for i, p in enumerate(probs)}
        
        # --- RESULTS DISPLAY ---
        st.divider()
        col_res, col_conf = st.columns([3, 1])
        with col_res:
            if "Donor" in result_text or "no_disease" in result_text:
                st.success(f"### Primary Diagnosis: {result_text}")
            else:
                st.error(f"### Primary Diagnosis: {result_text}")
        with col_conf:
            # Safely get confidence score
            conf_val = proba_dict.get(result_text, 0)
            st.metric("Confidence", f"{conf_val*100:.1f}%")

        # TABS
        t1, t2, t3 = st.tabs(["📊 Confidence Analysis", "🧬 Clinical Factors", "⚙️ Debug Info"])
        
        with t1:
            st.plotly_chart(plot_probabilities(proba_dict), use_container_width=True)
            
        with t2:
            st.write("#### Deviations from Normal Range:")
            abnormalities = get_abnormalities(input_data)
            if abnormalities:
                for issue in abnormalities:
                    st.warning(f"• {issue}")
            else:
                st.success("• All biomarkers within reference range.")

        with t3:
            st.write("Data sent to model:")
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error: {e}")
