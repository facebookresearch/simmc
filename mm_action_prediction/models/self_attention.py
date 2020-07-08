"""Self attention network block.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttention, self).__init__()
        self.feature_size = feature_size
        self.att_wt = nn.Parameter(torch.randn(1, 1, feature_size))

    def forward(self, feature_block, mask=False):
        """Self attends a feature block.

        Args:
            feature_block: Input of size Batch_shape x N_steps x Embed_size
            mask: Boolean mask to ignore the feature_block (B x N)

        Returns:
            att_features: Self attended features from the feature block (B X E)
        """
        # Compute attention scores.
        batch_size = feature_block.shape[0]
        new_size = (batch_size, 1, self.feature_size)
        att_logits = torch.bmm(
            feature_block, self.att_wt.expand(new_size).transpose(1, 2)
        )
        if mask is not None:
            att_logits.masked_fill_(mask.unsqueeze(-1), float("-inf"))
        att_wts = nn.functional.softmax(att_logits, dim=1)
        att_features = (att_wts * feature_block).sum(1)
        return att_features
