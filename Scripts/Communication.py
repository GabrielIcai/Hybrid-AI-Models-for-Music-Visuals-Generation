import csv
import time
from pythonosc import udp_client

# Configuración de OSC
IP = "127.0.0.1"
PORT = 8001
client = udp_client.SimpleUDPClient(IP, PORT)

# Ruta del CSV
csv_path = "predicciones_canciones_playlist.csv" 

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')

    for row in reader:
        song_name = row["Song ID"]
        genres = {}
        for genre, value in row.items():
            if genre not in ["Song ID", "Predicted Genre"]:
                try:
                    genres[genre] = float(value)
                except ValueError:
                    print(f"⚠️ Error al convertir '{value}' en la canción '{song_name}'")

        sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)
        top_3_genres = sorted_genres[:3]

        for i, (genre, prob) in enumerate(top_3_genres):
            address = f"/genre{i+1}"
            client.send_message(address, [genre, prob])
        
        print(f"Enviado a TouchDesigner: {top_3_genres}")