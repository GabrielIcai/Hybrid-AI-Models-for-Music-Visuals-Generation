import os
import sys
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)
import numpy as np
from src.training import collate_fn_prediction
import torch
from src.models.genre_model import CRNN
from src.preprocessing import normalize_columns, load_data, c_transform, PredictionDatasetGenre
import pandas as pd
from torch.utils.data import DataLoader
from Scripts.set_generator import generar_espectrograma
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

# Cargar el modelo
model_path = "/content/drive/MyDrive/TFG/models/best_CRNN_genre_5_2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(num_classes=5, additional_features_dim=12, hidden_size=256)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transformaciones
mean = [0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std = [0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
class_names = ["Ambient", "Deep House", "Techno", "Trance", "Progressive House"]

base_path = "/content/drive/MyDrive/TFG/data/"
csv_path = "/content/drive/MyDrive/TFG/data/Playlist_prediccion/dataset_prediccion_playlist.csv"
output_csv_path = "/content/drive/MyDrive/TFG/predicciones_canciones_playlist.csv"

def predict_audio_genre(carpeta_canciones):
    predictions_by_song = defaultdict(list)
    probabilities_by_song = defaultdict(list)
    columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
    # Generar espectrogramas
    for archivo in os.listdir(carpeta_canciones):
        if archivo.endswith(".mp3") or archivo.endswith(".wav"):
            ruta_audio = os.path.join(carpeta_canciones, archivo)
            nombre_archivo, _ = os.path.splitext(archivo)

            song_id = nombre_archivo
            generar_espectrograma(ruta_audio, nombre_archivo, song_id)
            print(f"Espectrogramas generados para {nombre_archivo} con ID {song_id}")

    # Cargar dataset
    data = load_data(csv_path)
    data["Ruta"] = data["Ruta"].str.replace("\\", "/")

    transform = c_transform(mean, std)
    normalize_columns(data, columns)

    # Procesar cada canción
    for song_id in data["Song ID"].unique():
        song_data = data[data["Song ID"] == song_id]
        if song_data.empty:
            print(f"No hay datos para {song_id}, saltando...")
            continue

        dataset_pred = PredictionDatasetGenre(song_data, base_path, transform=transform)
        loader = DataLoader(dataset_pred, batch_size=128, collate_fn=collate_fn_prediction, shuffle=False, num_workers=2, pin_memory=True)

        with torch.no_grad():
            for images, additional_features, _ in loader:
                images = images.to(device)
                additional_features = additional_features.to(device)
                outputs = model(images, additional_features)
                
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probabilities_by_song[song_id].extend(probs)

    # **Generar predicción final por canción**
    song_results = []
    
    for song_id, probs in probabilities_by_song.items():
        probs = np.array(probs).mean(axis=0)  # Promediar probabilidades
        final_genre = class_names[np.argmax(probs)]  # Género con mayor probabilidad

        # Guardar resultado
        song_results.append([song_id, final_genre] + probs.tolist())

    # **Crear DataFrame con probabilidades**
    columns = ["Song ID", "Predicted Genre"] + class_names
    results_df = pd.DataFrame(song_results, columns=columns)

    # **Guardar CSV**
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predicciones guardadas en {output_csv_path}")

    return results_df


# MAIN
audios_path ="/content/drive/MyDrive/TFG/data/Playlist_prediccion/"
predict_audio_genre(audios_path)


#Comunicación con TOUCHDESIGNER
#from pythonosc.udp_client import SimpleUDPClient
#import pandas as pd
#import time
#
#ip = "127.0.0.1"  # IP de TouchDesigner
#port = 8000  # Puerto OSC en TouchDesigner
#
#client = SimpleUDPClient(ip, port)
#
#csv_path = "/content/drive/MyDrive/TFG/predicciones_canciones_playlist.csv"
#data = pd.read_csv(csv_path)
#
#class_names = ["Ambient", "Deep House", "Techno", "Trance", "Progressive House"]
#
#for _, row in data.iterrows():
#    fragment = row["Fragment"]
#    probabilities = eval(row["Probabilities"])  # Convertir string a lista
#
#    # Enviar las probabilidades de cada género por separado
#    for i, genre in enumerate(class_names):
#        client.send_message(f"/genre/{genre.lower().replace(' ', '_')}", probabilities[i])
#
#    print(f"Fragment {fragment} enviado con probabilidades: {probabilities}")
#
#    time.sleep(0.1)  # Evita saturar TouchDesigner

