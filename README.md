# 🏥 Liver Disease AI Predictor

An end-to-end Supervised Machine Learning project that predicts liver disease categories based on clinical biomarkers and blood/urine analysis.

## 🚀 Project Overview
This project addresses the critical need for early liver disease detection by classifying patients into five distinct categories: **No Disease, Suspect Disease, Hepatitis C, Fibrosis, and Cirrhosis**. By utilizing a high-performance **XGBoost Classifier** trained on a dataset of 615 clinical instances, the model provides an automated diagnostic support tool that assists healthcare professionals in identifying liver damage through 12 key physiological metrics.

## 🛠️ Tech Stack
* **Python:** Core language for data processing
* **XGBoost:** Gradient Boosted Trees for high-accuracy classification
* **Scikit-Learn:** Machine Learning pipeline and preprocessing (RobustScaler)
* **Pandas & Numpy:** Data manipulation and numerical analysis
* **Plotly:** Interactive probability visualizations
* **Streamlit:** Professional web dashboard deployment

## 📊 Key Features
* **Multi-Class Diagnosis:** Goes beyond binary "Sick/Healthy" to differentiate between specific stages like Fibrosis and Cirrhosis.
* **Clinical Guardrails:** Compares user input against international medical reference ranges (e.g., Albumin, ALT, AST, Bilirubin) to highlight abnormal levels.
* **Probability Analysis:** Provides a confidence score for each diagnosis, allowing for nuanced medical interpretation.
* **Professional UI:** A Streamlit-based interface designed for rapid data entry and visual feedback for clinical environments.

## 🗺️ Project Pipeline
<img width="1014" height="1022" alt="Liver Disease Project Flowchart" src="https://github.com/user-attachments/assets/6cf4efe0-5fdf-4996-8483-36c69d4cbca6" />

---

## 📈 Model Performance & Selection

### **1. Performance Summary**
* **Selected Model:** XGBoost Classifier (with RobustScaler)
* **Target Classes:** 5 (No Disease, Suspect Disease, Hepatitis, Fibrosis, Cirrhosis)
* **Reliability:** Integrated with a **RobustScaler** to handle outliers in clinical data, ensuring stable predictions across diverse patient profiles.

### **2. Why XGBoost?**
After evaluating various algorithms, XGBoost was selected as the champion model because:
* **Handling Imbalance:** Clinical datasets often have fewer "Cirrhosis" or "Fibrosis" cases compared to healthy donors; XGBoost handles this imbalance through internal weighting and boosting.
* **Non-Linear Medical Patterns:** The relationship between enzymes like AST and ALT is often non-linear; gradient boosting captures these complex interactions better than linear models.
* **Feature Importance:** It allows for clear identification of which biomarkers (like Cholinesterase or GGT) are driving the diagnosis.

---

## 🌐 Live Demo
Test the diagnostic system here:  
**[Liver Disease AI Predictor](https://liver-disease-predictor-app-flspwmy3igbjywmzwwxnm4.streamlit.app/)**

### 💤 Important Note on App Availability
If you are accessing the live demo and the website appears to be "sleeping":
* Please click the **"Yes, get this back up!"** button on the screen.
* This will wake up the server and restore the prediction tool within a few seconds.

---

## 🏁 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/liver-disease-predictor.git](https://github.com/YOUR-USERNAME/liver-disease-predictor.git)
   cd liver-disease-predictor
