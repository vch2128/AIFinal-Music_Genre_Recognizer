import librosa
import logging
import sys
import numpy as np
import os
import shutil

from keras.models import model_from_json
from DataProcessing import MusicData


logging.getLogger("tensorflow").setLevel(logging.ERROR)
timeseries_length = 128

def load_model(model_path, weight_path):
    with open(model_path, "r") as model_file:
        trained_model = model_from_json(model_file.read())

    trained_model.load_weights(weight_path)
    trained_model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return trained_model

def extract_feature(self, song):
    data = np.zeros((1, timeseries_length, 33), dtype=np.float64)
    y, sr = librosa.load(song)
    mfcc = librosa.feature.mfcc( y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13 )
    spectral_center = librosa.feature.spectral_centroid( y=y, sr=sr, hop_length=self.hop_length )
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
    spectral_contrast = librosa.feature.spectral_contrast( y=y, sr=sr, hop_length=self.hop_length )

    data[0, :, 0:13] = mfcc.T[0:self.timeseries_length, :]
    data[0, :, 13:14] = spectral_center.T[0:self.timeseries_length, :]
    data[0, :, 14:26] = chroma.T[0:self.timeseries_length, :]
    data[0, :, 26:33] = spectral_contrast.T[0:self.timeseries_length, :]
    return data

def predict(model, song):
    prediction = model.predict(extract_feature(song))
    predict_genre = MusicData().genre_list[np.argmax(prediction)]
    return predict_genre

if __name__ == "__main__":
    model = load_model("./weights/model.json", "./weights/model_weights.h5")
    print("Please input the folder you want to organize: ")
    input_song = input().strip()

    for genre_name in MusicData.genre_list:
        music_folder = os.path.join(input_song, genre_name)
        if not os.path.exists(music_folder):
            os.makedirs(music_folder)

    genre_dict = {genre_name: [] for genre_name in MusicData.genre_list}
    
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
