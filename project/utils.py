import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Conv1dSame(nn.Module):
    """
    Add PyTorch compatible support for Tensorflow/Keras padding option: padding='same'.

    Discussions regarding feature implementation:
    https://discuss.pytorch.org/t/converting-tensorflow-model-to-pytorch-issue-with-padding/84224
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-598264120

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (
            kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1
        )
        self.padding = np.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


def hard_sigmoid(x):
    return torch.clip(0.2 * x + 0.5, 0, 1)


class ActivationLSTMCell(nn.Module):
    """
    LSTM Cell using variable gating activation, by default hard sigmoid

    If gate_activation=torch.sigmoid this is the standard LSTM cell

    Uses recurrent dropout strategy from https://arxiv.org/abs/1603.05118 to match Keras implementation.
    """

    def __init__(
        self, input_size, hidden_size, gate_activation=hard_sigmoid, recurrent_dropout=0
    ):
        super(ActivationLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_activation = gate_activation
        self.recurrent_dropout = recurrent_dropout

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for param in [self.weight_hh, self.weight_ih]:
                for idx in range(4):
                    mul = param.shape[0] // 4
                    torch.nn.init.xavier_uniform_(param[idx * mul : (idx + 1) * mul])

    def forward(self, input, state):
        if state is None:
            hx = torch.zeros(
                input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype
            )
            cx = torch.zeros(
                input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype
            )
        else:
            hx, cx = state
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self.gate_activation(ingate)
        forgetgate = self.gate_activation(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = self.gate_activation(outgate)

        if self.recurrent_dropout > 0:
            cellgate = F.dropout(cellgate, p=self.recurrent_dropout)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class CustomLSTM(nn.Module):
    """
    LSTM to be used with custom cells
    """

    def __init__(self, cell, *cell_args, bidirectional=True, **cell_kwargs):
        super(CustomLSTM, self).__init__()
        self.cell_f = cell(*cell_args, **cell_kwargs)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.cell_b = cell(*cell_args, **cell_kwargs)

    def forward(self, input, state=None):
        # Forward
        state_f = state
        outputs_f = []
        for i in range(len(input)):
            out, state_f = self.cell_f(input[i], state_f)
            outputs_f += [out]

        outputs_f = torch.stack(outputs_f)

        if not self.bidirectional:
            return outputs_f, None

        # Backward
        state_b = state
        outputs_b = []
        l = input.shape[0] - 1
        for i in range(len(input)):
            out, state_b = self.cell_b(input[l - i], state_b)
            outputs_b += [out]

        outputs_b = torch.flip(torch.stack(outputs_b), dims=[0])

        output = torch.cat([outputs_f, outputs_b], dim=-1)

        # Keep second argument for consistency with PyTorch LSTM
        return output, None