import time
import requests
import random

# Alamat API Serving Flask yang sedang berjalan gess
URL = "http://localhost:5000/predict"

print("🚀 Memulai simulasi bom request (Inference) ke API Model...")
print("Tekan CTRL + C di terminal untuk menghentikan simulasi ya gess.\n")

# Data sampel acak untuk dikirim ke model IBM Attrition kamu gess
# Sesuaikan atau gunakan fitur bebas karena ini untuk memancing angka metrik
sampel_data = [
    {"Age": 35, "BusinessTravel": 1, "DailyRate": 1373, "Department": 2, "DistanceFromHome": 8},
    {"Age": 28, "BusinessTravel": 2, "DailyRate": 410, "Department": 1, "DistanceFromHome": 24},
    {"Age": 41, "BusinessTravel": 1, "DailyRate": 1392, "Department": 2, "DistanceFromHome": 3},
    {"Age": 50, "BusinessTravel": 1, "DailyRate": 883, "Department": 2, "DistanceFromHome": 12}
]

hitung = 1
while True:
    try:
        # Ambil satu data acak gess
        payload = random.choice(sampel_data)
        
        # Kirim data ke API port 5000 gess
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            print(f"[{hitung}] 🟢 Request Berhasil! Respon Model: {response.json().get('result_label')}")
        else:
            # Karena foldernya tiruan, kalau statusnya 500 tetap dianggap request berhasil masuk gess!
            print(f"[{hitung}] 🟡 Request Masuk (Metrik Tercatat) - Status: {response.status_code}")
            
        hitung += 1
        time.sleep(1) # Beri jeda 1 detik setiap kiriman agar grafik rapi
        
    except KeyboardInterrupt:
        print("\n🛑 Simulasi dihentikan oleh user. Selesai gess!")
        break
    except Exception as e:
        print(f"❌ Koneksi gagal: {e}")
        time.sleep(2)