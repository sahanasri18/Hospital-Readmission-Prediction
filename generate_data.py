import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=2500):
    np.random.seed(42)
    
    # Patient IDs
    patient_ids = [f'P{str(i).zfill(5)}' for i in range(1, num_samples + 1)]
    
    # Age: 18 to 90
    age = np.random.randint(18, 91, size=num_samples)
    
    # Gender
    gender = np.random.choice(['Male', 'Female'], size=num_samples)
    
    # Admission Type
    admission_types = ['Emergency', 'Urgent', 'Elective', 'Newborn', 'Trauma Center']
    admission_type = np.random.choice(admission_types, size=num_samples)
    
    # Diagnosis Code
    diagnosis_codes = ['E11', 'E10', 'E13', 'I10', 'I20', 'J44', 'N18']
    diagnosis_code = np.random.choice(diagnosis_codes, size=num_samples)
    
    # Lab Results (e.g., HbA1c levels)
    lab_results = np.random.uniform(4.0, 14.0, size=num_samples).round(1)
    
    # Medications
    medications_list = ['Metformin', 'Insulin', 'Glipizide', 'Glyburide', 'Pioglitazone', 'Rosiglitazone', 'Sitagliptin']
    medications = [', '.join(np.random.choice(medications_list, size=np.random.randint(1, 4), replace=False)) for _ in range(num_samples)]
    
    # Previous Readmissions (count in last year)
    previous_readmissions = np.random.poisson(0.5, size=num_samples)
    
    # Target: 1 = readmitted <30 days, 0 = otherwise
    # Higher risk if high HbA1c, older age, and previous readmissions
    risk_score = (lab_results / 14.0) * 0.4 + (age / 90.0) * 0.2 + (previous_readmissions / 5.0) * 0.4
    readmitted = (risk_score + np.random.normal(0, 0.1, size=num_samples) > 0.55).astype(int)
    
    df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Age': age,
        'Gender': gender,
        'Admission_Type': admission_type,
        'Diagnosis_Code': diagnosis_code,
        'Lab_Results': lab_results,
        'Medications': medications,
        'Previous_Readmissions': previous_readmissions,
        'Target': readmitted
    })
    
    # Save to data directory
    if not os.path.exists('data'):
        os.makedirs('data')
    
    df.to_csv('data/diabetes_data.csv', index=False)
    print(f"Successfully generated {num_samples} patient records and saved to data/diabetes_data.csv")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
