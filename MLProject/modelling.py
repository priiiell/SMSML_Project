import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def train_baseline_model():
    # Menggunakan folder lokal ./mlruns agar langsung jalan cepat
    mlflow.set_tracking_uri("./mlruns")
    
    # Menyamakan nama eksperimen agar tercatat di satu dashboard
    experiment_name = "IBM_HR_Attrition_Eksperimen"
    mlflow.set_experiment(experiment_name)
    
    # === 🛠️ TRIK JELI PENYESUAIAN JALUR DATASET UNTUK CI PIPELINE ===
    filename = "ibm_attrition_preprocessing.csv"
    
    # Cek 1: Jika dijalankan via GitHub Actions / MLflow run di root folder
    if os.path.exists(os.path.join("MLProject", filename)):
        dataset_path = os.path.join("MLProject", filename)
    # Cek 2: Jika dijalankan langsung dari dalam folder MLProject
    elif os.path.exists(filename):
        dataset_path = filename
    # Cek 3: Menggunakan struktur folder preprocessing bawaan lamamu
    elif os.path.exists(os.path.join("preprocessing", filename)):
        dataset_path = os.path.join("preprocessing", filename)
    else:
        dataset_path = os.path.join("..", "preprocessing", filename)
        
    # Validasi Akhir Keberadaan File
    if os.path.exists(dataset_path):
        print(f"• Memuat dataset bersih dari: {dataset_path}")
        df_clean = pd.read_csv(dataset_path)
    else:
        print(f"⚠️ Error: Berkas {filename} tidak ditemukan di direktori mana pun!")
        print("Pastikan posisi file CSV hasil preprocessing tidak berpindah folder ya gess.")
        return

    # Memisahkan Fitur (X) dan Target (y)
    # Menyesuaikan nama kolom target dataset kamu
    target_column = 'Attrition'
    if target_column not in df_clean.columns:
        target_column = df_clean.columns[-1] # fallback ke kolom terakhir jika namanya ter-encode
        
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # Splitting data dengan stratify agar rasio target seimbang
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Mengaktifkan MLflow Autolog sebelum proses fitting model
    mlflow.sklearn.autolog()
    print("• MLflow Autolog berhasil diaktifkan.")
    
    # Memulai pencatatan run ke MLflow
    with mlflow.start_run(run_name="Random_Forest_Basic_Run") as run:
        print(f"▶ Memulai tracking Run ID: {run.info.run_id}")
        
        # Menggunakan model baseline RandomForest
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        print("• Melatih model baseline RandomForestClassifier...")
        model.fit(X_train, y_train)
        
        # Prediksi dan Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("\n=== HASIL EVALUASI MODEL BASELINE ===")
        print(f"Accuracy Score: {acc:.4f}")
        print("\n[Classification Report]")
        print(classification_report(y_test, y_pred))
        
        print("\n✓ Sukses! Seluruh parameter otomatis dan model baseline berhasil dicatat.")

if __name__ == "__main__":
    train_baseline_model()