import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import gc

# Definición de rutas
carpeta_canciones = "/content/drive/MyDrive/TFG/data/Playlist_prediccion"
carpeta_salida = "/content/drive/MyDrive/TFG/data/Playlist_prediccion/"
os.makedirs(carpeta_salida, exist_ok=True)

duracion_fragmento = 5

# Archivo CSV
csv_file = os.path.join(carpeta_salida, "dataset_prediccion_playlist.csv")

# Leer el CSV existente asegurando que tomamos los nombres correctos
procesados = set()
if os.path.exists(csv_file):
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            ruta_procesada = os.path.basename(row[0])  # Tomamos el nombre del archivo desde la ruta
            procesados.add(ruta_procesada)  # Guardamos solo el nombre del archivo


def generar_espectrograma(archivo_audio, nombre_archivo, song_id):
    global procesados
    if song_id in procesados:
        print(f"Saltando {nombre_archivo}, ya procesado.")
        return
    
    try:
        y, sr = librosa.load(archivo_audio, sr=None)
    except Exception as e:
        print(f"Error al cargar {archivo_audio}: {e}")
        return

    duracion_muestras = duracion_fragmento * sr
    num_fragmentos = len(y) // duracion_muestras

    for i in range(num_fragmentos):
        nombre_salida = f"{nombre_archivo}_fragmento_{i}.png"
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        if os.path.exists(ruta_salida):
            print(f"Fragmento {i} de {nombre_archivo} ya existe, saltando...")
            continue

        inicio = i * duracion_muestras
        fin = inicio + duracion_muestras
        fragmento = y[inicio:fin]

        rms = np.mean(librosa.feature.rms(y=fragmento).flatten())
        zcr = librosa.feature.zero_crossing_rate(y=fragmento)[0, 0]
        mean_abs_amplitude = np.mean(np.abs(fragmento))
        crest_factor = np.max(np.abs(fragmento)) / rms if rms != 0 else 0
        std_amplitude = np.std(fragmento)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=fragmento, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=fragmento, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=fragmento, sr=sr))
        spectral_flux = librosa.onset.onset_strength(y=fragmento, sr=sr).mean()

        try:
            vad = librosa.effects.split(y=fragmento, top_db=30)
            vad_result = 1 if len(vad) > 0 else 0
        except Exception as e:
            print(f"Error en VAD para {archivo_audio} fragmento {i}: {e}")
            vad_result = None

        try:
            spectrogram = np.abs(librosa.stft(fragmento))
            spectral_variation = np.std(spectrogram, axis=1).mean()
        except Exception as e:
            print(f"Error en variacion espectral para {archivo_audio} fragmento {i}: {e}")
            spectral_variation = None

        try:
            tempo = librosa.beat.tempo(y=fragmento, sr=sr)[0]
        except Exception as e:
            print(f"Error en la estimación de tempo para {archivo_audio} fragmento {i}: {e}")
            tempo = None

        espectrograma = librosa.feature.melspectrogram(y=fragmento, sr=sr, n_mels=128, fmax=8000)
        espectrograma_db1 = librosa.power_to_db(espectrograma, ref=np.max)
        if np.max(espectrograma_db1) == np.min(espectrograma_db1):
            print("El espectrograma contiene valores constantes.")
            continue
        normalized_spectrogram = (espectrograma_db1 - np.min(espectrograma_db1)) / (np.max(espectrograma_db1) - np.min(espectrograma_db1))
        normalized_spectrogram = cv2.resize(normalized_spectrogram, (512, 512))

        plt.figure(figsize=(2, 2))
        librosa.display.specshow(normalized_spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(ruta_salida, bbox_inches='tight', pad_inches=0)
        plt.close()

        del fragmento, espectrograma, espectrograma_db1, normalized_spectrogram
        gc.collect()

        tiempo_inicio = librosa.samples_to_time(inicio, sr=sr)
        tiempo_fin = librosa.samples_to_time(fin, sr=sr)

        min_seg_inicio = f"{int(tiempo_inicio // 60)}:{int(tiempo_inicio % 60):02d}"
        min_seg_fin = f"{int(tiempo_fin // 60)}:{int(tiempo_fin % 60):02d}"

        generar_fila_csv(ruta_salida, song_id, rms, min_seg_inicio, min_seg_fin, zcr, mean_abs_amplitude, crest_factor, std_amplitude, spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flux, vad_result, spectral_variation, tempo)
        print(f"Fragmento {i} guardado en el CSV")

def generar_fila_csv(ruta_salida, song_id, rms, min_seg_inicio, min_seg_fin, zcr, mean_abs_amplitude, crest_factor, std_amplitude, spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flux, vad_result, spectral_variation, tempo):
    file_exists = os.path.isfile(csv_file)
    fila = [ruta_salida, song_id, rms, min_seg_inicio, min_seg_fin, zcr, mean_abs_amplitude, crest_factor, std_amplitude, spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flux, vad_result, spectral_variation, tempo]

    with open(csv_file, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = ["Ruta", "Song ID", "RMS", "Tiempo Inicio", "Tiempo Fin", "ZCR", "Mean Absolute Amplitude", "Crest Factor", "Standard Deviation of Amplitude", "Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off", "Spectral Flux", "VAD", "Spectral Variation", "Tempo"]
            writer.writerow(headers)
        writer.writerow(fila)


