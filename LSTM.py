import torch as t
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim
import os
from DataProcessing import MusicData



class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, batch_size, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.batch_size = batch_size 

	    # initialize LSTM       
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num)
        # output
        self.linear1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden=None):
        lstm_out, hidden = self.lstm(input, hidden)


def main():
    music_data = MusicData()

    file_exist = ( os.path.exists(music_data.train_X_file) and
                   os.path.exists(music_data.train_Y_file) )

    if file_exist:
        music_data.load_feature_data()
    else:
        music_data.create_feature_data()

    # print(music_data.train_X.shape)    # (audioSample, timeSlots, features)
    # print(music_data.train_Y.shape)
    # print(music_data.train_X[1])

    train_X = t.from_numpy(music_data.train_X).type(t.Tensor)
    train_Y = t.from_numpy(music_data.train_Y).type(t.Tensor)

    batch_size = 25
    input_dim = 33     # the calculated features
    hidden_dim = 128     # capture hidden features
    layer_num = 2
    output_dim = 12    # 12 genres
    epoch = 300

    print("Starting LSTM model...")
    model = LSTM(input_dim, hidden_dim, layer_num, batch_size, output_dim)





if __name__ == "__main__":
    main()