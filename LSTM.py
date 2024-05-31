import torch
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim
import os
from tqdm import tqdm
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
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden=None):
        out, _ = self.lstm(input, hidden)
        out = out[:, -1, :]      # Get the last time step's output
        out = self.fc(out)
        return out

def accuracy(outputs, Y):
    correct = 0
    _, predicted = torch.max(outputs, 1)
    _, trueclass = torch.max(Y, 1)
    print("pre ",predicted)
    print( (predicted == trueclass).sum() )
    #correct += ( predicted).sum()
    #print("cor ",correct)
    

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

    train_X = torch.from_numpy(music_data.train_X).type(torch.Tensor)
    train_Y = torch.from_numpy(music_data.train_Y).type(torch.Tensor)

    batch_size = 25
    input_dim = 33     # the calculated features
    hidden_dim = 128     # capture hidden features
    layer_num = 2
    output_dim = 12    # 12 genres
    epochs = 1
    learning_rate = 0.001

    print("Starting LSTM model...")
    model = LSTM(input_dim, hidden_dim, layer_num, batch_size, output_dim)
    lossfunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training...")
    num_batch = int(len(train_X) / batch_size)
    for epoch in range(epochs):
        model.train()
        runningloss = 0.0
        correct = 0

        for i in range(num_batch):
            model.zero_grad()
            batch_X = train_X[ i*batch_size : (i+1)*batch_size , : , : ]
            batch_Y = train_Y[ i*batch_size : (i+1)*batch_size , : ]
            # Forward pass
            outputs = model(batch_X)
            # print(outputs)
            loss = lossfunc(outputs, batch_Y)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runningloss += loss.item()
            accuracy(outputs, batch_Y)

        loss_avg = runningloss / batch_size
        
        print( "Epoch: %d / Loss: %.4f" % (epoch, loss_avg) )
        




if __name__ == "__main__":
    main()