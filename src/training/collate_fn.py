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
def collate_fn(batch):
    grouped_by_song = defaultdict(list)

    # Agrupar los fragmentos por canción
    for img, add_feats, label, image_path in batch:
        if img is not None:
            song_name = extract_song_name(image_path)
            print(song_name)
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
        # Crear ventanas de 3 fragmentos consecutivos
        for i in range(0, len(fragments), 3):
            song_images = []
            song_additional_features = []
            song_labels = []

            # Coger 3 fragmentos
            for j in range(3):
                img, add_feats, label = fragments[i + j]
                song_images.append(img)
                song_additional_features.append(add_feats)
                song_labels.append(label)
            
            song_labels = song_labels[0]  # Uso la etiqueta del primer fragmento
            labels.append(song_labels)
            # Convertir las listas de fragmentos en tensores
            images.append(torch.stack(song_images, dim=0))
            additional_features.append(torch.stack(song_additional_features, dim=0))

    # Convertir las listas en un solo tensor
    images = torch.stack(images, dim=0)  # (batch_size, 3, canales, altura, anchura)
    additional_features = torch.stack(
        additional_features, dim=0
    )  # (batch_size, 3, num_features)
    labels = torch.stack(labels, dim=0)  # (batch_size, 3, num_labels)

    return images, additional_features, labels
