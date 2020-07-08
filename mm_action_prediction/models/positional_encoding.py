"""Positional Encoding class for Transfomers.

Author(s): Satwik Kottur
Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-(math.log(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Adds positional encoding to the input.

        Args:
            x: Input of size Batch_shape x N_steps x Embed_size
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
