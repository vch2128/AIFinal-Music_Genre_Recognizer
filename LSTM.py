import torch
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from DataProcessing import MusicData


class LSTM(nn.Module):
    model_path = "./lstm_model.pth"

    def __init__(self, input_dim, hidden_dim, layer_num, batch_size, output_dim, dropout):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.batch_size = batch_size 
        self.dropout = nn.Dropout(dropout)

	    # initialize LSTM       
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        # output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden=None):
        out, _ = self.lstm(input, hidden)
        out = out[:, -1, :]      # Get the last time step's output
        out = self.fc(out)
        return out


def accuracy(outputs, Y, batch_size):
    _, predicted = torch.max(outputs, 1)
    _, trueclass = torch.max(Y, 1)
    # print("pre ", predicted)
    # print("true ", trueclass)
    correct_num = (predicted == trueclass).sum().item()
    return (correct_num / batch_size) * 100


def main():
    music_data = MusicData()

    file_exist = ( os.path.exists(music_data.train_X_file) and
                   os.path.exists(music_data.train_Y_file) and
                   os.path.exists(music_data.dev_X_file) and
                   os.path.exists(music_data.dev_Y_file)       )
    if file_exist:
        music_data.load_feature_data()
    else:
        music_data.create_feature_data()
    
    # print(music_data.train_X.shape)    # (audioSample, timeSlots, features)
    # print(music_data.train_Y.shape)

    train_X = torch.from_numpy(music_data.train_X).type(torch.Tensor)
    train_Y = torch.from_numpy(music_data.train_Y).type(torch.Tensor)
    dev_X = torch.from_numpy(music_data.dev_X).type(torch.Tensor)
    dev_Y = torch.from_numpy(music_data.dev_Y).type(torch.Tensor)

    batch_size = 20
    input_dim = 33     # the calculated features
    hidden_dim = 512     # capture hidden features
    layer_num = 1
    output_dim = 12    # 12 genres
    dropout = 0.3
    epochs = 400
    learning_rate = 0.001
    val_gap = 10   # do validation after how many epochs

    print("Starting LSTM model...")
    model = LSTM(input_dim, hidden_dim, layer_num, batch_size, output_dim, dropout)
    lossfunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_train, acc_train, loss_dev, acc_dev = [],[],[],[]
    epoch_train = [i for i in range(1, epochs+1)]
    epoch_dev = [i for i in range(10, epochs+1, val_gap)]

    print("Training...")
    num_batch = int(len(train_X) / batch_size)
    val_num_batch = int(len(dev_X) / batch_size)
    model.train()
    for epoch in range(epochs):
        runningloss = 0.0
        accuracy_sum = 0.0

        for i in range(num_batch):
            model.zero_grad()
            batch_X = train_X[ i*batch_size : (i+1)*batch_size , : , : ]
            batch_Y = train_Y[ i*batch_size : (i+1)*batch_size , : ]

            # Forward pass
            outputs = model(batch_X)

            loss = lossfunc(outputs, batch_Y)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            runningloss += loss.item()
            accuracy_sum += accuracy(outputs, batch_Y, batch_size)

        loss_avg = runningloss / num_batch
        accuracy_avg = accuracy_sum / num_batch
        loss_train.append(loss_avg)
        acc_train.append(accuracy_avg)
        print( "Epoch %d:  Loss: %.4f / Accuracy: %.2f%%" % (epoch, loss_avg, accuracy_avg) )
        
        # validation
        if (epoch+1) % val_gap == 0:
            model.eval()
            val_runningloss = 0.0
            val_accuracy_sum = 0.0

            with torch.no_grad():
                for i in range(val_num_batch):
                    batch_X = dev_X[ i*batch_size : (i+1)*batch_size , : , : ]
                    batch_Y = dev_Y[ i*batch_size : (i+1)*batch_size , : ]

                    outputs = model(batch_X)
                    loss = lossfunc(outputs, batch_Y)

                    val_runningloss += loss.item()
                    val_accuracy_sum += accuracy(outputs, batch_Y, batch_size)

            model.train()
            loss_avg = val_runningloss / val_num_batch
            accuracy_avg = val_accuracy_sum / val_num_batch
            loss_dev.append(loss_avg)
            acc_dev.append(accuracy_avg)
            print("\nValidation: Loss: %.4f / Accuracy: %.2f%%\n" % (loss_avg, accuracy_avg))

    # plot the loss and accuracy on both datasets
    # # train
    # plt.plot(epoch_train, loss_train)
    # plt.xlabel("# of epochs")
    # plt.ylabel("Loss")
    # plt.title("Loss of Training Data")
    # plt.show()

    # plt.plot(epoch_train, acc_train)
    # plt.xlabel("# of epochs")
    # plt.ylabel("Accuracy(%)")
    # plt.title("Accuracy of Training Data")
    # plt.show()

    # # dev
    # plt.plot(epoch_dev, loss_dev)
    # plt.xlabel("# of epochs")
    # plt.ylabel("Loss")
    # plt.title("Loss of Validation Data")
    # plt.show()

    # plt.plot(epoch_dev, acc_dev)
    # plt.xlabel("# of epochs")
    # plt.ylabel("Accuracy(%)")
    # plt.title("Accuracy of Validation Data")
    # plt.show()

    # compare
    plt.plot(epoch_train, loss_train, label='Training')
    plt.plot(epoch_dev, loss_dev, label = 'Validation')
    plt.xlabel("# of epochs")
    plt.ylabel("Loss")
    plt.title("Comparison of Loss")
    plt.legend()
    plt.show()

    plt.plot(epoch_train, acc_train, label='Training')
    plt.plot(epoch_dev, acc_dev, label = 'Validation')
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy(%)")
    plt.title("Comparison of Accuracy")
    plt.legend()
    plt.show()

    # save the model state
    torch.save(model.state_dict(), model.model_path)


if __name__ == "__main__":
    main()