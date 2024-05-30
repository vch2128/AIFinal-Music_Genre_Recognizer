import librosa
import logging
import sys
import numpy as np
import os
import shutil
import torch
from torch import nn
from DataProcessing import MusicData


logging.getLogger("tensorflow").setLevel(logging.ERROR)
timeseries_length = 128
model_path = "決定model存的地方後幫我改一下QQ"

def load_model(path):
    model = LSTM(input_dim=33, hidden_dim=128, batch_size=1, output_dim=12, num_layers=2)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def extract_feature(song):
    data = np.zeros((1, timeseries_length, 33), dtype=np.float64)
    y, sr = librosa.load(song)
    hop_length = 512
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    data[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    data[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    data[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    data[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return data

def predict(model, song):
    features = extract_feature(song)
    features_tensor = torch.tensor(features, dtype=torch.float32).permute(1, 0, 2)
    prediction, hidden = model(features_tensor)
    predict_genre = MusicData().genre_list[torch.argmax(prediction)]  # max: model最認可的
    return predict_genre

if __name__ == "__main__":
    model = load_model(model_path)
    print("Please input the folder you want to organize: ")
    input_song = input().strip()

    for genre_name in MusicData().genre_list:
        music_folder = os.path.join(input_song, genre_name)
        if not os.path.exists(music_folder):
            os.makedirs(music_folder)

    genre_dict = {genre_name: [] for genre_name in MusicData().genre_list}
    
    for song in os.listdir(input_song):
        if song.endswith('.mp3'):
            song_path = os.path.join(input_song, song)
            prediction = predict(model, song_path)
            target_folder = os.path.join(input_song, prediction)
            shutil.move(song_path, target_folder)
            genre_dict[prediction].append(song[:-4])  # 去掉.mp3

    for genre, tracks in genre_dict.items():
        print(f"\n{genre}: ")
        for track in tracks:
            print(track)
