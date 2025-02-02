import torch
from collections import defaultdict
import torch
import re
import os

def extract_song_name(image_path):
    filename = os.path.basename(image_path)  # Obtiene solo el nombre del archivo
    match = re.match(r"(.+)_fragmento_\d+\.png", filename)
    print(filename)
    print(match)
    if match:
        return match.group(1)
    else:
        return None


# Para genero necesito agrupar fragmentos de las mismas cnciones en un batch que no puedo hacer con un dataset normal. Con collate proceso y agrupo varios
# elementos individuales
def collate_fn(batch):
    grouped_by_song = defaultdict(list)

    # Agrupo los fragmentos por canción
    for img, add_feats, label, image_path in batch:
        if img is not None:
            song_name = extract_song_name(image_path)
            print(f"Extraído: {song_name} de {image_path}")
            if song_name:
                grouped_by_song[song_name].append((img, add_feats, label))
            else:
                print(f"Fragmento con nombre de canción inválido: {image_path}")
        else:
            print(f"Imagen no válida: {image_path}")

    images = []
    additional_features = []
    labels = []

    for song_name, fragments in grouped_by_song.items():
        # Padding
        while len(fragments) % 3 != 0:
            fragments.append(
                (
                    torch.zeros_like(fragments[0][0]),
                    torch.zeros_like(fragments[0][1]),
                    torch.zeros_like(fragments[0][2]),
                )
            )
        # 3 fragmentos por canción
        for i in range(0, len(fragments), 3):
            song_images = []
            song_additional_features = []
            song_labels = []

            # Cojo 3 fragmentos
            for j in range(3):
                img, add_feats, label = fragments[i+j]
                song_images.append(img)
                song_additional_features.append(add_feats)
                song_labels.append(label)
            
            song_labels = song_labels[0]
            labels.append(song_labels)

            # Convierto listas en tensores  
            images.append(torch.stack(song_images, dim=0))
            additional_features.append(torch.stack(song_additional_features, dim=0))

    images = torch.stack(images, dim=0)  #(batch_size, 3, canales, altura, anchura)
    additional_features = torch.stack(additional_features, dim=0)  # (batch_size, 3, num_features)
    labels = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    return images, additional_features, labels

from collections import defaultdict
import torch

from collections import defaultdict
import torch

def collate_fn_s(batch):
    grouped_by_song = defaultdict(list)

    for img, add_feats, label, song_id in batch:
        if img is not None:
            grouped_by_song[song_id].append((img, add_feats, label))
        else:
            print(f"Imagen no válida para canción ID: {song_id}")

    images, additional_features, labels, song_ids = [], [], [], []

    for song_id, fragments in grouped_by_song.items():
        # Asegurar que el número de fragmentos es múltiplo de 3
        while len(fragments) % 3 != 0:
            fragments.append(
                (
                    torch.zeros_like(fragments[0][0]),  # Imagen vacía
                    torch.zeros_like(fragments[0][1]),  # Features vacíos
                    fragments[0][2]  # Etiqueta real se mantiene
                )
            )

        # Agrupar en bloques de 3 (seq_len=3)
        for i in range(0, len(fragments), 3):
            song_images = torch.stack([fragments[i+j][0] for j in range(3)], dim=0)  # (3, C, H, W)
            song_additional_features = torch.stack([fragments[i+j][1] for j in range(3)], dim=0)  # (3, F)
            song_label = fragments[i][2]  # Etiqueta de la canción

            images.append(song_images)
            additional_features.append(song_additional_features)
            labels.append(song_label)
            song_ids.append(song_id)

    # Convertir a tensores 5D
    images = torch.stack(images, dim=0)  # (batch_size, 3, C, H, W)
    additional_features = torch.stack(additional_features, dim=0)  # (batch_size, 3, F)
    labels = torch.stack(labels, dim=0)  # (batch_size, num_classes)

    return images, additional_features, labels, song_ids


