import os
import time
import pickle
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Histogram
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- 📊 DEKLARASI METRIK PROMETHEUS ---
REQUEST_COUNT = Counter(
    'model_prediction_requests_total', 
    'Total nomor request prediksi ke model IBM Attrition gess'
)
PREDICTION_COUNT = Counter(
    'model_predictions_total', 
    'Total hasil prediksi model berdasarkan label',
    ['prediction_label']
)
REQUEST_LATENCY = Histogram(
    'model_prediction_latency_seconds', 
    'Waktu yang dibutuhkan model untuk memproses prediksi gess'
)

# --- 🤖 MEMBUAT MODEL BASELINE MANDIRI (ANTI-ERROR MLFLOW) ---
print("• Menginisialisasi model simulasi RandomForest untuk monitoring lokal...")
# Kita buat model tiruan sederhana langsung di memori gess agar tidak butuh file eksternal
dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
# Umpan data mainan sekadar agar modelnya ter-fit dan siap memprediksi gess
X_dummy = [[35, 1, 1373, 2, 8], [28, 2, 410, 1, 24]]
y_dummy = [0, 1]
dummy_model.fit(X_dummy, y_dummy)

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.time()
def predict():
    REQUEST_COUNT.inc() # Tambah hitungan total request (+1)
        
    try:
        # Menerima data JSON dari kiriman file inference gess
        data = request.get_json()
        
        # Simulasi pengambilan keputusan model secara dinamis
        # Jika nilai Age ganjil atau DistanceFromHome > 10, kita set prediksi Attrition
        age_val = data.get("Age", 30)
        dist_val = data.get("DistanceFromHome", 5)
        
        if age_val % 2 != 0 or dist_val > 12:
            label_hasil = "Yes"
            prediction_code = 1
        else:
            label_hasil = "No"
            prediction_code = 0
        
        # Catat hasil prediksi ke radar metrik berdasarkan labelnya gess!
        PREDICTION_COUNT.labels(prediction_label=label_hasil).inc()
        
        return jsonify({
            "status": "Sukses",
            "prediction_code": prediction_code,
            "result_label": label_hasil
        })
        
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 400

if __name__ == '__main__':
    # 1. Jalankan server pengekspor metrik untuk Prometheus di port 8000 gess
    start_http_server(8000)
    print("🚀 Radar Pengekspor Metrik siap diakses di http://localhost:8000/metrics")
    
    # 2. Jalankan API Serving Flask utama di port 5000 gess
    print("🔥 API Model Serving Flask aktif di http://localhost:5000/predict")
    app.run(host='0.0.0.0', port=5000)