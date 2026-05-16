import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def train_baseline_model():
    # Menyamakan nama eksperimen agar tercatat di satu dashboard
    experiment_name = "IBM_HR_Attrition_Eksperimen"
    mlflow.set_experiment(experiment_name)
    
    # === 🛠️ TRIK JELI PENYESUAIAN JALUR DATASET UNTUK CI PIPELINE ===
    filename = "ibm_attrition_preprocessing.csv"
    
    if os.path.exists(os.path.join("MLProject", filename)):
        dataset_path = os.path.join("MLProject", filename)
    elif os.path.exists(filename):
        dataset_path = filename
    elif os.path.exists(os.path.join("preprocessing", filename)):
        dataset_path = os.path.join("preprocessing", filename)
    else:
        dataset_path = os.path.join("..", "preprocessing", filename)
        
    if os.path.exists(dataset_path):
        print(f"• Memuat dataset bersih dari: {dataset_path}")
        df_clean = pd.read_csv(dataset_path)
    else:
        print(f"⚠️ Error: Berkas {filename} tidak ditemukan!")
        return

    # Memisahkan Fitur (X) dan Target (y)
    target_column = 'Attrition'
    if target_column not in df_clean.columns:
        target_column = df_clean.columns[-1]
        
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Mengaktifkan MLflow Autolog sebelum proses fitting model
    mlflow.sklearn.autolog()
    print("• MLflow Autolog berhasil diaktifkan.")
    
    # --- 🚀 TRIK ANTI-TABRAKAN RUN ID ---
    # Cek apakah sudah ada run aktif dari CLI (mlflow run)
    active_run = mlflow.active_run()
    
    if active_run:
        print(f"▶ Menggunakan Run ID aktif dari CLI: {active_run.info.run_id}")
        # Melatih model baseline RandomForest
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        print("• Melatih model baseline RandomForestClassifier...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("\n=== HASIL EVALUASI MODEL BASELINE ===")
        print(f"Accuracy Score: {acc:.4f}")
        print("\n[Classification Report]")
        print(classification_report(y_test, y_pred))
    else:
        # Jika dijalankan manual tanpa 'mlflow run' (lokal biasa)
        with mlflow.start_run(run_name="Random_Forest_Basic_Run") as run:
            print(f"▶ Memulai tracking Run ID baru: {run.info.run_id}")
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            print("• Melatih model baseline RandomForestClassifier...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print("\n=== HASIL EVALUASI MODEL BASELINE ===")
            print(f"Accuracy Score: {acc:.4f}")
            print("\n[Classification Report]")
            print(classification_report(y_test, y_pred))
            
    print("\n✓ Sukses! Seluruh parameter otomatis dan model baseline berhasil dicatat.")

if __name__ == "__main__":
    train_baseline_model()