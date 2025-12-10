import torch
import torch.nn as nn
from utils.HRAMi import HRAMi
from utils.CAFM import LinAngularXCA_CA
from utils.KAN import KANLinear
from utils.ARelu import AReLU
from utils.UCDC import UCDC

class MsipNet(nn.Module):
    """
    MsipNet: Multi-modal neural network for sequence-based prediction tasks.

    This model integrates multiple input modalities (x, y, h) representing different
    features of RNA sequences. It combines 1D convolutional layers, LSTMs, custom modules,
    and fully connected layers to extract, fuse, and
    process features to output a prediction score.

    Key components:
    - Convolutional layers (conv, conv1-5, convse, convstr) extract local patterns.
    - LSTM layers capture sequential dependencies.
    - UCDC and HRAMi are custom modules for feature fusion and attention.
    - Dropout layers prevent overfitting.
    - ReLU provide non-linear activations.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(MsipNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(15840, output_size)
        self.fc3 = nn.Linear(3200, output_size)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.arelu = AReLU()

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True,
                            dropout=0.5)
        self.lstmseq = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True,
                               dropout=0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.5)

        self.conv = nn.Conv1d(128, 32, kernel_size=1, padding=0)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=9, padding=5)
        self.conv1 = nn.Conv1d(128, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 32, kernel_size=7, padding=3)

        self.convse = nn.Conv1d(640, 128, kernel_size=3, padding=1)
        self.convse1 = nn.Conv1d(256, 16, kernel_size=3, padding=1)

        self.convstr1 = nn.Conv1d(128, 32, kernel_size=1, padding=0)
        self.convstr2 = nn.Conv1d(128, 32, kernel_size=3, padding=1)
        self.convstr3 = nn.Conv1d(128, 32, kernel_size=5, padding=2)
        self.convstr4 = nn.Conv1d(128, 32, kernel_size=7, padding=3)

        self.pooling1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pooling = nn.MaxPool1d(kernel_size=5, stride=5)
        self.poolingh = nn.MaxPool1d(kernel_size=5, stride=5)
        self.convstr = nn.Conv1d(7, 128, kernel_size=3, padding=1)

        self.ucdc = UCDC(32, 32)
        self.convlast1 = nn.Conv1d(160, 32, kernel_size=5, padding=2)
        self.poollast1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.hrami = HRAMi(128)
        self.cafm = LinAngularXCA_CA()
        self.lstmlast = nn.LSTM(input_size=128*3, hidden_size=32, num_layers=2, batch_first=True,
                                dropout=0.3)
        self.fclast = nn.Linear(640, 1)
        self.kan = KANLinear(640, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, h):

        x = x.transpose(1, 2)  # 32 640 101
        x = self.convse(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout3(x) #32 128 20

        h5 = self.relu(self.conv5(h))
        h = self.poolingh(h5)
        h = self.dropout3(h)
        h1 = self.relu(self.conv(h))
        h2 = self.relu(self.conv1(h))
        h3 = self.relu(self.conv2(h))
        h4 = self.relu(self.conv3(h))
        h = torch.cat([h1, h2, h3, h4], dim=1) #32 128 20

        y = self.convstr(y)  # 32 128 101
        y = self.relu(y)
        y = self.pooling(y)
        y = self.dropout2(y)  # 32 128 20

        combined = torch.cat((x, h, y), dim=1)
        combined = combined.permute(0, 2, 1)
        combined, _ = self.lstmlast(combined)

        B, N, C = combined.size()
        H ,W = 2, 10
        combined_new = combined.reshape(B, C, H, W)
        combined = self.ucdc(combined_new)
        B, C, H, W = combined.size()
        combined = combined.view(B, N, C)

        combined = combined.view(combined.size(0), -1)
        p = self.fclast(combined)
        return p