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
        "reggae",
        "EDM",
        "Jpop",
        "Kpop",
        "Cpop",
    ]

    dir_trainfolder = "./Data/train"
    dir_devfolder = "./Data/dev"

    train_X_file = "./Data/train_X.npy"
    train_Y_file = "./Data/train_Y.npy"
    dev_X_file = "./Data/dev_X.npy"
    dev_Y_file = "./Data/dev_Y.npy"


    def __init__(self):
        self.hop_length = 512   # length of non-overlapping portion of window length
        self.timeseries_length = 1200   # length of samples

        self.train_pathlist = self.music_path_list(self.dir_trainfolder)
        self.dev_pathlist = self.music_path_list(self.dir_devfolder)
        
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

    def extract_feature(self, path_list):
        partition_num = 10
        partition_len = int(self.timeseries_length / partition_num)
        data = np.zeros( (len(path_list)*partition_num, partition_len, 33), dtype=np.float64 )
        genre_list = []
        progress = tqdm(total = len(path_list))

        for i, file in enumerate(path_list):
            y, sr = librosa.load(file)    # load audio file
            # compute features
            mfcc = librosa.feature.mfcc( y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13 )
            spectral_center = librosa.feature.spectral_centroid( y=y, sr=sr, hop_length=self.hop_length )
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length )
            spectral_contrast = librosa.feature.spectral_contrast( y=y, sr=sr, hop_length=self.hop_length )

            # partition data sample into 10 parts
            for k in range(partition_num):
                data[partition_num*i+k, 0:partition_len, 0:13] = mfcc.T[ k*partition_len:(k+1)*partition_len, :]
                data[partition_num*i+k, 0:partition_len, 13:14] = spectral_center.T[ k*partition_len:(k+1)*partition_len, :]
                data[partition_num*i+k, 0:partition_len, 14:26] = chroma.T[ k*partition_len:(k+1)*partition_len, :]
                data[partition_num*i+k, 0:partition_len, 26:33] = spectral_contrast.T[ k*partition_len:(k+1)*partition_len, :]

            # get true genre of the data sample
            split = re.split("[./]", file)
            for k in range(partition_num):
                genre_list.append(split[4])

            progress.update(1)
        progress.close()

        # print("mfcc",mfcc.shape)
        # print("spec_cen",spectral_center.shape)
        # print("chroma",chroma.shape)
        # print("spec_con",spectral_contrast.shape)

        genre_list = self.one_hot_encoding( np.expand_dims(np.asarray(genre_list), axis=1) )
        return data, genre_list
    
    def one_hot_encoding(self, Y_genre_list):
        one_hot_list = np.zeros(( len(Y_genre_list), len(self.genre_list) ))
        for i, genre in enumerate(Y_genre_list):
            one_hot_code = self.genre_list.index(genre)
            one_hot_list[i][one_hot_code] = 1
        return one_hot_list

    def create_feature_data(self):
        # train
        print("Extracting training data features...")
        self.train_X, self.train_Y = self.extract_feature(self.train_pathlist)
        with open(self.train_X_file, "wb") as f:
            np.save(f, self.train_X)
        with open(self.train_Y_file, "wb") as f:
            np.save(f, self.train_Y)
        
        # dev
        print("Extracting validation data features...")
        self.dev_X, self.dev_Y = self.extract_feature(self.dev_pathlist)
        with open(self.dev_X_file, "wb") as f:
            np.save(f, self.dev_X)
        with open(self.dev_Y_file, "wb") as f:
            np.save(f, self.dev_Y)

    def load_feature_data(self):
        print("Data exist. Loading feature data files...")
        self.train_X = np.load(self.train_X_file)
        self.train_Y = np.load(self.train_Y_file)
        self.dev_X = np.load(self.dev_X_file)
        self.dev_Y = np.load(self.dev_Y_file)

