import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/best_model.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    return model, feature_names

try:
    model, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Error loading models. Please run `train_model.py` first. Error: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Single Prediction", "Batch Prediction", "Model Performance"])

# Title
st.title("🏥 Hospital Readmission Prediction System")
st.markdown("""
Predicting 30-day readmission risk for diabetes patients to improve care outcomes.
""")

if app_mode == "Single Prediction":
    st.header("Patient Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective", "Newborn", "Trauma Center"])
        diagnosis_code = st.selectbox("Diagnosis Code", ["E11", "E10", "E13", "I10", "I20", "J44", "N18"])
    
    with col2:
        lab_results = st.number_input("HbA1c Level (%)", 4.0, 15.0, 7.0, step=0.1)
        medications = st.multiselect("Medications", ["Metformin", "Insulin", "Glipizide", "Glyburide", "Pioglitazone", "Rosiglitazone", "Sitagliptin"], default=["Metformin"])
        previous_readmissions = st.number_input("Prior Readmissions (last 12 months)", 0, 10, 0)
    
    if st.button("Predict Readmission Risk"):
        # Preprocess input
        total_medications = len(medications)
        on_insulin = 1 if "Insulin" in medications else 0
        
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Admission_Type': [admission_type],
            'Diagnosis_Code': [diagnosis_code],
            'Age': [age],
            'Lab_Results': [lab_results],
            'Previous_Readmissions': [previous_readmissions],
            'Total_Medications': [total_medications],
            'On_Insulin': [on_insulin]
        })
        
        # Predict
        prob = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        # Risk level logic
        risk_level = "High" if prob > 0.6 else "Medium" if prob > 0.4 else "Low"
        color = "#ff4b4b" if risk_level == "High" else "#ffa500" if risk_level == "Medium" else "#28a745"
        
        st.subheader("Prediction Result")
        st.markdown(f"<h2 style='color: {color};'>{risk_level} Risk (Probability: {prob:.1%})</h2>", unsafe_allow_html=True)
        
        if prediction == 1:
            st.warning("⚠️ This patient is predicted to be readmitted within 30 days.")
        else:
            st.success("✅ This patient is unlikely to be readmitted within 30 days.")
            
        # Recommendations
        st.info("### Recommendations")
        if lab_results > 8.0:
            st.write("- Poor glycemic control detected. Consider medication adjustment.")
        if previous_readmissions > 1:
            st.write("- History of readmissions. Intensive discharge planning recommended.")
        if on_insulin == 0 and lab_results > 9.0:
            st.write("- Consider initiating insulin therapy.")

elif app_mode == "Batch Prediction":
    st.header("Batch Analysis")
    uploaded_file = st.file_uploader("Upload Patient EHR CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(data.head())
        
        # Preprocess batch
        if 'Medications' in data.columns:
            data['Total_Medications'] = data['Medications'].apply(lambda x: len(str(x).split(',')))
            data['On_Insulin'] = data['Medications'].apply(lambda x: 1 if 'Insulin' in str(x) else 0)
        
        # Check for required columns
        required_cols = ['Gender', 'Admission_Type', 'Diagnosis_Code', 'Age', 'Lab_Results', 'Previous_Readmissions']
        if all(col in data.columns for col in required_cols):
            X_batch = data[['Gender', 'Admission_Type', 'Diagnosis_Code', 'Age', 'Lab_Results', 'Previous_Readmissions', 'Total_Medications', 'On_Insulin']]
            
            probs = model.predict_proba(X_batch)[:, 1]
            preds = model.predict(X_batch)
            
            data['Readmission_Probability'] = probs
            data['Predicted_Readmission'] = preds
            
            st.subheader("Results")
            st.dataframe(data)
            
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
        else:
            st.error(f"CSV missing one or more required columns: {required_cols}")

elif app_mode == "Model Performance":
    st.header("Model Evaluation Metrics")
    if os.path.exists('models/model_evaluation.csv'):
        results_df = pd.read_csv('models/model_evaluation.csv')
        st.table(results_df)
        
        # Simple plot
        fig, ax = plt.subplots()
        results_df.plot(kind='bar', x='Model', y='F1-Score', ax=ax, color='skyblue')
        plt.title('Model F1-Score Comparison')
        st.pyplot(fig)
    else:
        st.warning("Evaluation results not found.")
