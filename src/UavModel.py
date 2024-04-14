# 定义更深的神经网络模型
import torch
import torch.nn as nn


class DnnModule1(nn.Module):
    def __init__(self):
        super(DnnModule1, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc7 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc8 = nn.Linear(128, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.fc9 = nn.Linear(64, 32)
        self.bn9 = nn.BatchNorm1d(32)
        self.fc10 = nn.Linear(32, 2)
        # Activation function can be assigned as a member variable
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.activation(self.bn4(self.fc4(x)))
        x = self.activation(self.bn5(self.fc5(x)))
        x = self.activation(self.bn6(self.fc6(x)))
        x = self.activation(self.bn7(self.fc7(x)))
        x = self.activation(self.bn8(self.fc8(x)))
        x = self.activation(self.bn9(self.fc9(x)))
        x = self.fc10(x)  # Usually no batch norm just before the final layer
        return x


class LSTMModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout_rate)
        self.final_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [BS, seq_len, hidden_dim]

        # Use the output from the last timestep
        final_output = lstm_out[:, -1, :]  # shape: [BS, hidden_dim]

        # Pass the last outputs through a final fully connected layer to get the desired output_dim
        final_output = self.final_fc(final_output)  # shape: [BS, output_dim]

        return final_output


class UavModel(nn.Module):
    def __init__(self):
        super(UavModel, self).__init__()
        self.dnn1 = DnnModule1()
        self.lstm = LSTMModule(input_dim=2, output_dim=2,
                               hidden_dim=128, num_layers=2, dropout_rate=0.2)
        # self.dnn2 = DnnModule2(dropout_rate=0.35)

    def forward(self, x):
        batch_size, seq_len, features_len = x.size()

        # reshape input to discard x temporarily for the first module
        x = x.view(-1, features_len)

        x = self.dnn1(x)

        # reshape x to original shape (restoring seq)
        x = x.view(batch_size, seq_len, 2)

        x = self.lstm(x)
        # lstm already returns the last hidden state of the sequences, no need to reshape

        # x = self.dnn2(x)

        return x
