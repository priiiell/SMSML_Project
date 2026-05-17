import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import mlflow

def train_model_tuning():
    # 1. Menggunakan folder lokal ./mlruns agar sinkron dengan eksperimen sebelumnya
    mlflow.set_tracking_uri("./mlruns")
    
    experiment_name = "IBM_HR_Attrition_Eksperimen"
    mlflow.set_experiment(experiment_name)
    
    # Menentukan jalur file preprocessing sesuai struktur foldermu
    dataset_path = os.path.join("preprocessing", "ibm_attrition_preprocessing.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join("..", "preprocessing", "ibm_attrition_preprocessing.csv")

    if os.path.exists(dataset_path):
        print(f"• Memuat dataset bersih dari: {dataset_path}")
        df_clean = pd.read_csv(dataset_path)
    else:
        print("⚠️ Error: Berkas ibm_attrition_preprocessing.csv tidak ditemukan!")
        return

    X = df_clean.drop(columns=['Attrition'])
    y = df_clean['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # === PARAMETER GRID KILAT BIAR JALANNYA CEPAT DI LAPTOP ADVAN ===
    param_grid = {
        'n_estimators': [50],
        'max_depth': [6, 10]
    }
    
    print("• Memulai pencarian hyperparameter kilat menggunakan GridSearchCV...")
    base_rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=2, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"✓ Parameter Terbaik Ditemukan: {best_params}")
    
    # 2. Memulai Tracking Manual ke MLflow (TANPA AUTOLOG)
    with mlflow.start_run(run_name="Random_Forest_Tuning_Run") as run:
        print(f"▶ Memulai manual tracking Run ID: {run.info.run_id}")
        
        # A. Logging Parameter secara Manual
        print("• Mencatat parameter hasil tuning ke MLflow...")
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
            
        # B. Prediksi & Evaluasi
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("\n=== HASIL EVALUASI MODEL TUNING ===")
        print(f"Accuracy Score: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # C. Logging Metrik secara Manual
        print("• Mencatat metrik akurasi ke MLflow...")
        mlflow.log_metric("accuracy", acc)
        
        # D. Membuat dan Menyimpan Plot Artefak Gambar (Confusion Matrix)
        print("• Membuat plot Confusion Matrix untuk artefak...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Bertahan', 'Keluar'], yticklabels=['Bertahan', 'Keluar'])
        plt.title('Confusion Matrix - IBM HR Attrition Tuning')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        
        plot_filename = "training_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        
        # Mengunggah gambar ke menu Artifacts di MLflow
        mlflow.log_artifact(plot_filename)
        
        # Menghapus file gambar sementara di lokal
        if os.path.exists(plot_filename):
            os.remove(plot_filename)
            
        print("\n✓ Sukses! Seluruh parameter tuning, metrik, dan plot artefak berhasil disimpan.")

if __name__ == "__main__":
    train_model_tuning()