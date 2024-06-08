import librosa
import logging
import sys
import numpy as np
import os
import shutil
import torch
from torch import nn
from tqdm import tqdm
from DataProcessing import MusicData
from LSTM import LSTM

logging.getLogger("tensorflow").setLevel(logging.ERROR)

partition_num = 5
timeseries_length = 1200

def load_model():
    model = LSTM(input_dim=33, hidden_dim=256, batch_size=20, output_dim=9, layer_num=1, dropout=0.3)

    if os.path.exists(model.model_path):
        print("Model available. Loading model...")
        model.load_state_dict(torch.load(model.model_path))
    else:
        print("No trained model found. Loading untrained model...")
    
    model.eval()
    return model


def extract_feature(song):
    hop_length = 512
    partition_len = int(timeseries_length / partition_num)
    data = np.zeros((partition_num, partition_len, 33), dtype=np.float64)
    y, sr = librosa.load(song)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    
    for k in range(partition_num):
            data[k, 0:partition_len, 0:13] = mfcc.T[ k*partition_len:(k+1)*partition_len, :]
            data[k, 0:partition_len, 13:14] = spectral_center.T[ k*partition_len:(k+1)*partition_len, :]
            data[k, 0:partition_len, 14:26] = chroma.T[ k*partition_len:(k+1)*partition_len, :]
            data[k, 0:partition_len, 26:33] = spectral_contrast.T[ k*partition_len:(k+1)*partition_len, :]

    return data

def predict(model, song):
    features = extract_feature(song)
    features_tensor = torch.from_numpy(features).type(torch.Tensor)
    with torch.no_grad():
        outputs = model(features_tensor)
    _, prediction = torch.max(outputs, 1)

    genre_freq = np.zeros((1,len(MusicData().genre_list)))
    for index, p in enumerate(prediction):
        genre_freq[0,p] += 1
    best_predict = genre_freq.argmax()
    predict_genre = MusicData().genre_list[best_predict]  
    return predict_genre

if __name__ == "__main__":
    model = load_model()
    print("Model loading complete.")
    print("Please input the folder you want to organize: ")
    input_folder = input().strip()
    
    # for genre_name in MusicData().genre_list:
    #     music_folder = os.path.join(input_folder, genre_name)
    #     if not os.path.exists(music_folder):
    #         os.makedirs(music_folder)

    genre_dict = {genre_name: [] for genre_name in MusicData().genre_list}
    
    print("Sorting...")
    progress = tqdm(total = len(os.listdir(input_folder)))
    for song in os.listdir(input_folder):
        if song.endswith('.mp3') or song.endswith('.wav') or song.endswith('.au'):
            song_path = os.path.join(input_folder, song)
            prediction = predict(model, song_path)
            # target_folder = os.path.join(input_folder, prediction)
            # shutil.move(song_path, target_folder)
            genre_dict[prediction].append(song[:-4])  # 去掉.mp3
        progress.update(1)
    progress.close()

    for genre, tracks in genre_dict.items():
        print(f"\n{genre}: ")
        for track in tracks:
            print(track)
