import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def train_and_evaluate():
    # Load data
    data_path = 'data/diabetes_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Feature Engineering
    # 1. Total medications
    df['Total_Medications'] = df['Medications'].apply(lambda x: len(x.split(',')))
    
    # 2. Medication group presence (simplified for this model)
    df['On_Insulin'] = df['Medications'].apply(lambda x: 1 if 'Insulin' in x else 0)
    
    # Define features and target
    categorical_features = ['Gender', 'Admission_Type', 'Diagnosis_Code']
    numeric_features = ['Age', 'Lab_Results', 'Previous_Readmissions', 'Total_Medications', 'On_Insulin']
    target = 'Target'
    
    X = df[categorical_features + numeric_features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Models to train
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_f1 = 0
    best_model_name = ""
    best_model_pipeline = None
    
    results = []
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUROC': auc
        })
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUROC: {auc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_pipeline = pipeline
            
    # Save best model and preprocessor artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(best_model_pipeline, f'models/best_model.joblib')
    joblib.dump(X_train.columns.tolist(), 'models/feature_names.joblib') # Save feature names for inference
    
    print(f"\nSaved Best Model ({best_model_name}) to models/best_model.joblib")
    
    # Save all results to a CSV for reference
    pd.DataFrame(results).to_csv('models/model_evaluation.csv', index=False)
    
    return results

if __name__ == "__main__":
    train_and_evaluate()
