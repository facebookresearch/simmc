#!/usr/bin/env python3
"""Assistant Model for Furniture Genie.

Author(s): Satwik Kottur
"""

import torch
import torch.nn as nn

from tools import weight_init, torch_support
import models
import models.encoders as encoders


class Assistant(nn.Module):
    """SIMMC Assistant Agent.
    """

    def __init__(self, params):
        super(Assistant, self).__init__()
        self.params = params

        self.encoder = encoders.ENCODER_REGISTRY[params["encoder"]](params)
        self.decoder = models.GenerativeDecoder(params)

        if params["encoder"] == "pretrained_transformer":
            self.decoder.word_embed_net = (
                self.encoder.models.decoder.bert.embeddings.word_embeddings
            )
            self.decoder.decoder_unit = self.encoder.models.decoder

        # Learn to predict and execute actions.
        self.action_executor = models.ActionExecutor(params)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        # Initialize weights.
        weight_init.weight_init(self)
        if params["use_gpu"]:
            self = self.cuda()
        # Sharing word embeddings across encoder and decoder.
        if self.params["share_embeddings"]:
            if hasattr(self.encoder, "word_embed_net") and hasattr(
                self.decoder, "word_embed_net"
            ):
                self.decoder.word_embed_net = self.encoder.word_embed_net

    def forward(self, batch, mode=None):
        """Forward propagation.

        Args:
          batch: Dict of batch input variables.
          mode: None for training or teaching forcing evaluation;
                BEAMSEARCH / SAMPLE / MAX to generate text
        """
        outputs = self.encoder(batch)
        action_output = self.action_executor(batch, outputs)
        outputs.update(action_output)
        decoder_output = self.decoder(batch, outputs)
        if mode:
            generation_output = self.decoder.forward_beamsearch_multiple(
                batch, outputs, mode
            )
            outputs.update(generation_output)

        # If evaluating by retrieval, construct fake batch for each candidate.
        # Inputs from batch used in decoder:
        #   assist_in, assist_out, assist_in_len, assist_mask
        if self.params["retrieval_evaluation"] and not self.training:
            option_scores = []
            batch_size, num_rounds, num_candidates, _ = batch["candidate_in"].shape
            replace_keys = ("assist_in", "assist_out", "assist_in_len", "assist_mask")
            for ii in range(num_candidates):
                for key in replace_keys:
                    new_key = key.replace("assist", "candidate")
                    batch[key] = batch[new_key][:, :, ii]
                decoder_output = self.decoder(batch, outputs)
                log_probs = torch_support.unflatten(
                    decoder_output["loss_token"], batch_size, num_rounds
                )
                option_scores.append(-1 * log_probs.sum(-1))
            option_scores = torch.stack(option_scores, 2)
            outputs["candidate_scores"] = [
                {
                    "dialog_id": batch["dialog_id"][ii].item(),
                    "candidate_scores": [
                        list(option_scores[ii, jj].cpu().numpy())
                        for jj in range(batch["dialog_len"][ii])
                    ]
                }
                for ii in range(batch_size)
            ]

        # Local aliases.
        loss_token = decoder_output["loss_token"]
        pad_mask = decoder_output["pad_mask"]
        if self.training:
            loss_token = loss_token.sum() / (~pad_mask).sum().item()
            loss_action = action_output["action_loss"]
            loss_action_attr = action_output["action_attr_loss"]
            loss_total = loss_action + loss_token + loss_action_attr
            return {
                "token": loss_token,
                "action": loss_action,
                "action_attr": loss_action_attr,
                "total": loss_total,
            }
        else:
            outputs.update(
                {"loss_sum": loss_token.sum(), "num_tokens": (~pad_mask).sum()}
            )
            return outputs
