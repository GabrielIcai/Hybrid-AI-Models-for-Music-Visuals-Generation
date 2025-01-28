import torch
from collections import defaultdict
import torch
import re


def extract_song_name(image_path):
    match = re.match(r"(.*)_fragmento_\d+\.png", image_path)
    if match:
        return match.group(1)
    else:
        return None


# Para genero necesito agrupar fragmentos de las mismas canciones en un batch que no puedo hacer con un dataset normal. Con collate proceso y agrupo varios
# elementos individuales
from collections import defaultdict
import torch

def collate_fn(batch):
    # Filtrar elementos inválidos
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("El batch está vacío después de filtrar elementos inválidos.")

    grouped_by_song = defaultdict(list)

    # Agrupar los fragmentos por canción
    for img, add_feats, label, image_path in batch:
        song_name = extract_song_name(image_path)
        if song_name:
            grouped_by_song[song_name].append((img, add_feats, label))

    images = []
    additional_features = []
    labels = []

    for song_name, fragments in grouped_by_song.items():
        if len(fragments) < 3:
            print(f"La canción {song_name} tiene menos de 3 fragmentos y será ignorada.")
            continue

        # Padding
        while len(fragments) % 3 != 0:
            fragments.append(
                (
                    torch.zeros_like(fragments[0][0]),
                    torch.zeros_like(fragments[0][1]),
                    torch.zeros_like(fragments[0][2]),
                )
            )

        # Crear ventanas de 3 fragmentos consecutivos
        for i in range(0, len(fragments), 3):
            song_images = []
            song_additional_features = []
            song_labels = []

            for j in range(3):
                img, add_feats, label = fragments[i + j]
                song_images.append(img)
                song_additional_features.append(add_feats)
                song_labels.append(label)

            labels.append(song_labels[0])  # Uso la etiqueta del primer fragmento
            images.append(torch.stack(song_images, dim=0))
            additional_features.append(torch.stack(song_additional_features, dim=0))

    if not images:
        raise ValueError("No se generaron imágenes válidas en el collate_fn.")

    images = torch.stack(images, dim=0)  # (batch_size, 3, canales, altura, anchura)
    additional_features = torch.stack(
        additional_features, dim=0
    )  # (batch_size, 3, num_features)
    labels = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    return images, additional_features, labels
