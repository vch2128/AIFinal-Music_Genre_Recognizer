import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm

class MusicData:
    hop_length = None
    genre_list = [
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "rock",
        "blue",
        "funk",
        "reggae",
        "EDM",
        "Jpop",
        "Kpop",
        "Cpop",
    ]

    dir_trainfolder = "./Data/train"

    def __init__(self):
        self.train_pathlist = self.music_path_list(self.dir_trainfolder)

        self.timeseries_length = ( 128 )    # length of samples
        self.hop_length = 512   # length of non-overlapping portion of window length

    @staticmethod
    def music_path_list(dir):
        path_list = []
        for file in os.listdir(dir):
            if file.endswith(".wav"):
                # print(file)
                path = "%s/%s" % (dir, file)
                path_list.append(path)
        return path_list
    
    def extract_feature(self, path_list):
        data = np.zeros( (len(path_list), self.timeseries_length, 33), dtype=np.float64 )
        genre_list = []
        print("Extracting features...")
        progress = tqdm(total = len(path_list))

        for i, file in enumerate(path_list):
            y, sr = librosa.load(file)    # load audio file
            # compute features
            mfcc = librosa.feature.mfcc( y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13 )
            spectral_center = librosa.feature.spectral_centroid( y=y, sr=sr, hop_length=self.hop_length )
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = librosa.feature.spectral_contrast( y=y, sr=sr, hop_length=self.hop_length )

            data[i, :, 0:13] = mfcc.T[0:self.timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:self.timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:self.timeseries_length, :]
            data[i, :, 26:33] = spectral_contrast.T[0:self.timeseries_length, :]

            # get true genre of the data sample
            split = re.split("[./]", file)
            genre_list.append(split[4])

            progress.update(1)
        progress.close()


    def load_feature_data(self):
        self.extract_feature(self.train_pathlist)