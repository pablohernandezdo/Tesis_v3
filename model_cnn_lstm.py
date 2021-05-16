import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class ResCnnLstm(nn.Module):
