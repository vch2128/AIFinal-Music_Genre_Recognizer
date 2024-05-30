import torch as t
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim
import os
from DataProcessing import MusicData



class LSTM(nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim, hidden_layer_num, batch_size, classes_num):
        super(LSTM, self).__init__()
        self.input_dim = input_feature_dim
        self.hidden_dim = hidden_feature_dim
        self.layer_num = hidden_layer_num
        self.batch_size = batch_size 

	    # 初始化 LSTM       
        self.lstm = nn.LSTM(input_feature_dim, hidden_feature_dim, hidden_layer_num)

        # LSTM的輸出藉由單層的線性神經網路層分類
        self.linear1 = nn.Linear(hidden_feature_dim, classes_num)



def main():
    music_data = MusicData()

    file_exist = ( os.path.exists(music_data.train_X_file) and
                   os.path.exists(music_data.train_Y_file) )

    if file_exist:
        music_data.load_feature_data()
    else:
        music_data.create_feature_data()

    print(music_data.train_X.shape)
    print(music_data.train_Y.shape)
    #print(music_data.train_X[1])


if __name__ == "__main__":
    main()