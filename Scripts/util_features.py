import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image
from torchvision import transforms

# (224x224, como espera ResNet)
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.676956295967102, 0.2529653012752533, 0.4388839304447174],
    std=[0.21755781769752502, 0.15407244861125946, 0.07557372003793716])
])

def extract_features_and_spectrogram_tensor(chunk, sr=22050):
    samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
    samples = samples / (2**15)

    if chunk.channels > 1:
        samples = samples.reshape(-1, chunk.channels)
        samples = np.mean(samples, axis=1)

    # Resamplear
    samples = librosa.resample(samples, orig_sr=chunk.frame_rate, target_sr=sr)

    # 1. RMS
    rms = librosa.feature.rms(y=samples).mean()

    # 2. ZCR
    zcr = librosa.feature.zero_crossing_rate(y=samples).mean()

    # 3. Crest Factor
    peak = np.max(np.abs(samples))
    crest_factor = peak / rms if rms != 0 else 0

    # 4. STD Amplitude
    std_amp = np.std(samples)

    # 5. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sr).mean()

    # 6. Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr).mean()

    # 7. Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr).mean()

    # 8. Flux
    stft = np.abs(librosa.stft(samples))
    flux = np.mean(librosa.onset.onset_strength(S=stft, sr=sr))

    # 9. VAD
    energy = np.mean(samples**2)
    vad = 1.0 if energy > 0.001 else 0.0

    # 10. Spectral Variation
    variation = np.var(stft)

    feature_tensor = torch.tensor([
        rms, zcr, crest_factor, std_amp,
        centroid, bandwidth, rolloff, flux, vad, variation], dtype=torch.float32)

    S = librosa.feature.melspectrogram(y=samples, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots()
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    spectro_img = Image.open(buf).convert("RGB")
    spectro_tensor = resnet_transform(spectro_img)


    return feature_tensor, spectro_tensor
