"""Implements seq2seq encoder that is history-agnostic.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn

from tools import rnn_support as rnn
from tools import torch_support as support
import models
import models.encoders as encoders


@encoders.register_encoder("history_agnostic")
class HistoryAgnosticEncoder(nn.Module):
    def __init__(self, params):
        super(HistoryAgnosticEncoder, self).__init__()
        self.params = params

        self.word_embed_net = nn.Embedding(
            params["vocab_size"], params["word_embed_size"]
        )
        encoder_input_size = params["word_embed_size"]
        if params["text_encoder"] == "transformer":
            layer = nn.TransformerEncoderLayer(
                params["word_embed_size"],
                params["num_heads_transformer"],
                params["hidden_size_transformer"],
            )
            self.encoder_unit = nn.TransformerEncoder(
                layer, params["num_layers_transformer"]
            )
            self.pos_encoder = models.PositionalEncoding(params["word_embed_size"])
        elif params["text_encoder"] == "lstm":
            self.encoder_unit = nn.LSTM(
                encoder_input_size,
                params["hidden_size"],
                params["num_layers"],
                batch_first=True,
            )
        else:
            raise NotImplementedError("Text encoder must be transformer or LSTM!")

    def forward(self, batch):
        """Forward pass through the encoder.

        Args:
            batch: Dict of batch variables.

        Returns:
            encoder_outputs: Dict of outputs from the forward pass.
        """
        encoder_out = {}
        # Flatten for history_agnostic encoder.
        batch_size, num_rounds, max_length = batch["user_utt"].shape
        encoder_in = support.flatten(batch["user_utt"], batch_size, num_rounds)
        encoder_len = support.flatten(batch["user_utt_len"], batch_size, num_rounds)
        word_embeds_enc = self.word_embed_net(encoder_in)
        # Text encoder: LSTM or Transformer.
        if self.params["text_encoder"] == "lstm":
            all_enc_states, enc_states = rnn.dynamic_rnn(
                self.encoder_unit, word_embeds_enc, encoder_len, return_states=True
            )
            encoder_out["hidden_states_all"] = all_enc_states
            encoder_out["hidden_state"] = enc_states

        elif self.params["text_encoder"] == "transformer":
            enc_embeds = self.pos_encoder(word_embeds_enc).transpose(0, 1)
            enc_pad_mask = batch["user_utt"] == batch["pad_token"]
            enc_pad_mask = support.flatten(enc_pad_mask, batch_size, num_rounds)
            enc_states = self.encoder_unit(
                enc_embeds, src_key_padding_mask=enc_pad_mask
            )
            encoder_out["hidden_states_all"] = enc_states.transpose(0, 1)
        return encoder_out
