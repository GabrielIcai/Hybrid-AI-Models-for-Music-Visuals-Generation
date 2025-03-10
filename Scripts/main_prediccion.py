import time
import os
import sys
import queue
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.ndimage import zoom
from torchvision import transforms
import torch
import cv2
import collections
from PIL import Image
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.models.emotions_model import ResNetCRNNEmotionModel
from src.preprocessing import (
    load_data,
    split_dataset,
    c_transform,
)

# Parámetros

SAMPLE_RATE = 44100
DURATION = 5  # Segundos
BUFFER_SIZE = SAMPLE_RATE * DURATION  # 220500 muestras
CHANNELS = 1  # Mono
buffer = collections.deque(maxlen=BUFFER_SIZE)
AUDIO_FILE = r'C:/Users/administradorlocal/OneDrive - Universidad Pontificia Comillas/TFG/TFG/data/Lane_8_red_lights.mp3'

#AUDIO
audio_stream, sr = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)
BUFFER_SIZE = SAMPLE_RATE * DURATION

plt.ion()
fig, ax = plt.subplots(figsize=(14, 6))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def mel_spectrogram(audio_fragment):
    mel_spec = librosa.feature.melspectrogram(y=audio_fragment, sr=sr, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Redimensionar a 224x224
    mel_spec_db_resized = zoom(mel_spec_db, (224 / mel_spec_db.shape[0], 224 / mel_spec_db.shape[1]))

    # Convertir a imagen RGB aplicando un colormap
    mel_spec_db_resized = cv2.applyColorMap(
        cv2.normalize(mel_spec_db_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS
    )

    mel_spec_db_resized = Image.fromarray(mel_spec_db_resized)

    image_tensor = transform(mel_spec_db_resized)
    return image_tensor


def compute_audio_features(fragment):

    rms = librosa.feature.rms(y=fragment)[0]
    zcr = librosa.feature.zero_crossing_rate(fragment)[0]
    crest_factor = np.max(fragment) / np.mean(rms)
    std_amplitude = np.std(fragment)
    spectral_centroid = librosa.feature.spectral_centroid(y=fragment, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=fragment, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=fragment, sr=sr)[0]
    spectrum = np.abs(np.fft.fft(fragment))
    spectral_flux = np.sum(np.diff(spectrum)**2)
    vad = librosa.effects.split(fragment)
    vad_length = len(vad)
    spectral_variation = np.mean(np.abs(np.diff(spectral_centroid)))
    additional_features = np.array([rms, zcr, crest_factor, std_amplitude, spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flux, vad_length, spectral_variation])
    additional_features_tensor = torch.tensor(additional_features, dtype=torch.float32)
    return additional_features_tensor

q = queue.Queue()

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
    buffer.extend(audio_chunk)
    outdata[:] = indata
    if len(buffer) >= BUFFER_SIZE:
        mel_spectogram(np.array(buffer))

model = ResNetCRNNEmotionModel()
model_path=r'C:/Users/administradorlocal/OneDrive - Universidad Pontificia Comillas/TFG/TFG/models_paths/best_CNN_LSTM_emotions.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'),weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
 

# Flujo de salida de audio
with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
    start_time = time.time()
    
    for i in range(0, len(audio_stream), BUFFER_SIZE):
        fragment = audio_stream[i:i+BUFFER_SIZE]
        if len(fragment) < BUFFER_SIZE:
            break

        q.put(fragment)
        
        # TIEMPO A 0
        process_start_time = time.time()

        # Generación de características
        image_tensor = mel_spectrogram(fragment)
        additional_features_tensor = compute_audio_features(fragment)
        input_tensor = torch.cat((image_tensor.flatten(), additional_features_tensor), dim=0)
        print(f"Dimensiones del tensor de entrada: {input_tensor.shape}")

        process_end_time = time.time()
        process_time = process_end_time - process_start_time
        print(f"Tiempo total para generación del plot y características: {process_time:.4f} segundos")
        
        # Predicción del modelo
        with torch.no_grad(): 
            output = model(input_tensor.unsqueeze(0))
        

        # salida del modelo
        predicted_valencia = output[0][0].item()
        predicted_arousal = output[0][1].item()
        
        print(f"Predicción de valencia: {predicted_valencia}, Predicción de arousal: {predicted_arousal}")

        # Sincronización con el tiempo real
        elapsed_time = time.time() - start_time
        expected_time = i / SAMPLE_RATE
        sleep_time = max(0, expected_time - elapsed_time)
        time.sleep(sleep_time)

plt.ioff()




