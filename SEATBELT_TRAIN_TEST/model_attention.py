"""
    This is architecture for LSTM NN
    - attention - single direction
    - mean pooling
    - dense - fully connected - classification

"""

import torch
import torch.nn as nn

class ActionRecognitionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(ActionRecognitionLSTM, self).__init__()

         # attention over time steps?
        self.attn_fc = nn.Linear(hidden_size*2, 1)

        # convolution
        self.temporal_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1)

        # bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # fully connected layer - last - classification


    def forward(self, x, lengths=None):
        if lengths is not None: print(f"lengths is not None")

        # In forward (after permuting to match Conv1d):
        x = x.permute(0, 2, 1)  # from (batch, seq, feature) to (batch, feature, seq)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        # attention over time steps?
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)  # shape: (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # weighted sum over time
        output = self.fc(context)



        # final_hidden_state = lstm_out[:, -1, :]
        # final_hidden_state = torch.mean(lstm_out, dim=1)           # mean pooling          # Try this
        #final_hidden_state = torch.max(lstm_out, dim=1).values     # max pooling           # Or try this instead
        # output = self.fc(final_hidden_state)

        return output
