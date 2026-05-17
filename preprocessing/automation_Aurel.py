import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import mlflow
import mlflow.sklearn

def load_data(file_path):
    print("Memuat data mentah...")
    return pd.read_csv(file_path)

def preprocess_data(df, target_col='Attrition'):
    print("Memulai pipeline preprocessing otomatis...")
    df_clean = df.copy()
    
    df_clean = df_clean.drop_duplicates()
    
    columns_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')
    
    outlier_cols = ['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole']
    for col in outlier_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            
    if 'Age' in df_clean.columns:
        df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=[0, 30, 45, 100], labels=['Young', 'Middle-Aged', 'Senior'])
    if 'YearsAtCompany' in df_clean.columns and 'TotalWorkingYears' in df_clean.columns:
        df_clean['Company_Working_Ratio'] = df_clean['YearsAtCompany'] / (df_clean['TotalWorkingYears'] + 1)
        
    target_series = None
    if target_col in df_clean.columns:
        target_series = df_clean[target_col].map({'Yes': 1, 'No': 0})
        df_clean = df_clean.drop(columns=[target_col])
        
    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df_clean[num_cols] = imputer_num.fit_transform(df_clean[num_cols])
        scaler = StandardScaler()
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
        
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_clean[cat_cols] = imputer_cat.fit_transform(df_clean[cat_cols])
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_array = encoder.fit_transform(df_clean[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols).tolist()
        df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_clean.index)
        df_clean = df_clean.drop(columns=cat_cols)
        df_clean = pd.concat([df_clean, df_encoded], axis=1)
        
    if target_series is not None:
        df_clean[target_col] = target_series.values
        
    print("Pipeline preprocessing otomatis selesai!")
    return df_clean

def train_model_local():
    mlflow.set_tracking_uri("./mlruns")
    
    experiment_name = "IBM_HR_Attrition_Eksperimen"
    mlflow.set_experiment(experiment_name)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    
    input_file = os.path.join(root_dir, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
    output_file = os.path.join(base_dir, 'ibm_attrition_preprocessing.csv')
    
    if os.path.exists(input_file):
        data_raw = load_data(input_file)
        data_ready = preprocess_data(data_raw)
        data_ready.to_csv(output_file, index=False)
        print("File data sukses diekspor!")
    else:
        print("Error: File mentah tidak ditemukan.")
        return

    df_clean = pd.read_csv(output_file)
    X = df_clean.drop(columns=['Attrition'])
    y = df_clean['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    mlflow.sklearn.autolog()
    print("MLflow Autolog aktif.")
    
    with mlflow.start_run(run_name="Random_Forest_Local_Run") as run:
        print("Memulai tracking lokal...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        print("Melatih model RandomForestClassifier...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("=== HASIL EVALUASI MODEL LOKAL ===")
        print(f"Accuracy Score: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        print("Pelatihan selesai!")

if __name__ == "__main__":
    train_model_local()