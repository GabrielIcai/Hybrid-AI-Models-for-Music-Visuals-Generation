from pydub import AudioSegment
import os
import time
from util_features import extract_features_and_spectrogram_tensor
import joblib
import torch
from torchvision import models
import torch.nn as nn
from pythonosc import udp_client

# Configuración de OSC
IP = "127.0.0.1"
PORT = 8009
client = udp_client.SimpleUDPClient(IP, PORT)

# fmpeg y ffprobe
ffmpeg_path = "C:\\Users\\administradorlocal\\Downloads\\ffmpeg-7.1.1-essentials_build\\ffmpeg-7.1.1-essentials_build\\bin\\ffmpeg.exe"
ffprobe_path = "C:\\Users\\administradorlocal\\Downloads\\ffmpeg-7.1.1-essentials_build\\ffmpeg-7.1.1-essentials_build\\bin\\ffprobe.exe"

os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

# Cargamos el audio
audio = AudioSegment.from_file("data\Ben Böhmer - Begin Again.mp3")
print(f"Audio cargado. Duración total: {len(audio)/1000:.2f} segundos")

# Cargamos modelo Random Forest
rf_model_arousal = joblib.load("models_paths\\modelo_arousal.pkl")
rf_model_valencia = joblib.load("models_paths\\modelo_valencia.pkl")

#Cargamos la CNN
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Procesamos por bloques de 5 segundos
chunk_duration = 5000
for i in range(0, len(audio), chunk_duration):
    chunk = audio[i:i+chunk_duration]
    print(f"🔹 Procesando segmento {i/1000:.0f}s - {(i+chunk_duration)/1000:.0f}s")

    try:
        feat_tensor, spectro_tensor = extract_features_and_spectrogram_tensor(chunk)
        
        with torch.no_grad():
            img_feat = resnet(spectro_tensor.unsqueeze(0)).squeeze()

        full_features = torch.cat([img_feat, feat_tensor]).numpy().reshape(1, -1)

        # Predecir valencia y arousal
        valencia = rf_model_valencia.predict(full_features)[0]
        arousal = rf_model_arousal.predict(full_features)[0]

        valencia = float(valencia)
        arousal = float(arousal)
        osc_message = "/emocion"  
        client.send_message(osc_message, [valencia, arousal])

        print(f"Enviado a TouchDesigner: {osc_message} → [{valencia:.2f}, {arousal:.2f}]")

    except Exception as e:
        print(f"Error procesando el segmento: {e}")
        

    time.sleep(5)
