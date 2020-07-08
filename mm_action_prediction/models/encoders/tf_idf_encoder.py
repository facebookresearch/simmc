"""Implements TF-IDF based encoder that is history-agnostic.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn

import models.encoders as encoders


@encoders.register_encoder("tf_idf")
class TFIDFEncoder(nn.Module):
    def __init__(self, params):
        super(TFIDFEncoder, self).__init__()
        self.params = params
        self.IDF = nn.Parameter(torch.randn(params["vocab_size"]))
        self.encoder_net = nn.Sequential(
            nn.Linear(params["vocab_size"], params["vocab_size"] // 2),
            nn.ReLU(),
            nn.Linear(params["vocab_size"] // 2, params["vocab_size"] // 4),
            nn.ReLU(),
            nn.Linear(params["vocab_size"] // 4, params["hidden_size"]),
            nn.ReLU(),
            nn.Linear(params["hidden_size"], params["hidden_size"]),
        )

    def forward(self, batch):
        """Forward pass through the encoder.

        Args:
            batch: Dict of batch variables.

        Returns:
            encoder_outputs: Dict of outputs from the forward pass.
        """
        encoder_embed = self.encoder_net(batch["user_tf_idf"] * self.IDF)
        batch_size, num_rounds, feat_size = encoder_embed.shape
        encoder_embed = encoder_embed.view(1, -1, feat_size)
        return {"hidden_state": (encoder_embed, encoder_embed)}
