import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import math

class MusicData:
    hop_length = None
    genre_list = [
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "rock",
        "blues",
        "funk",
        "reggae",
        "EDM",
        "Jpop",
        "Kpop",
        "Cpop",
    ]

    dir_trainfolder = "./Data/train"

    train_X_file = "./Data/train_X.npy"
    train_Y_file = "./Data/train_Y.npy"


    def __init__(self):
        self.hop_length = 512   # length of non-overlapping portion of window length

        self.train_pathlist = self.music_path_list(self.dir_trainfolder)

        self.all_files_list = []
        self.all_files_list.extend(self.train_pathlist)

        self.timeseries_length = ( 128 )    # length of samples
        # self.get_sample_len()
        

    @staticmethod
    def music_path_list(dir):
        path_list = []
        for file in os.listdir(dir):
            if file.endswith(".wav"):
                # print(file)
                path = "%s/%s" % (dir, file)
                path_list.append(path)
        return path_list

    def get_sample_len(self):
        sr = 22050
        timestep = self.hop_length/sr
        # print('timestep:',timestep)
        samplelen = timestep*self.timeseries_length
        print("length of sample:", samplelen, "sec")

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

        genre_list = self.one_hot_encoding( np.expand_dims(np.asarray(genre_list), axis=1) )
        return data, genre_list
    
    def one_hot_encoding(self, Y_genre_list):
        one_hot_list = np.zeros(( len(Y_genre_list), len(self.genre_list) ))
        for i, genre in enumerate(Y_genre_list):
            one_hot_code = self.genre_list.index(genre)
            one_hot_list[i][one_hot_code] = 1
        return one_hot_list

    def create_feature_data(self):
        print("Creating feature data files...")
        # training
        self.train_X, self.train_Y = self.extract_feature(self.train_pathlist)
        with open(self.train_X_file, "wb") as f:
            np.save(f, self.train_X)
        with open(self.train_Y_file, "wb") as f:
            np.save(f, self.train_Y)

    #def load_feature_data(self):
