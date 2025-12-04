import torch.nn as nn
import torch.nn.functional as F

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_classes=3):
        super(LSTMBinaryClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hidden_state, cell_state) = self.lstm(x)

        # Get the output from the last timestep
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        last_output = lstm_out[:, -1, :]

        # Pass through the linear layer
        linear_out = self.linear(last_output)

        # Apply log_softmax for classification
        return F.log_softmax(linear_out, dim=1)