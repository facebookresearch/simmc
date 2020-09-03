"""Implements decoder for hoste assistant neural conversational model.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import torch
import torch.nn as nn

import models
from tools import rnn_support as rnn
from tools import torch_support as support


class GenerativeDecoder(nn.Module):
    def __init__(self, params):
        super(GenerativeDecoder, self).__init__()
        self.params = params

        # Dialog context encoders.
        self.DIALOG_CONTEXT_ENCODERS = ("hierarchical_recurrent", "memory_network")

        # Word embedding.
        self.word_embed_net = nn.Embedding(
            params["vocab_size"], params["word_embed_size"]
        )
        # Text encoder.
        if params["text_encoder"] == "transformer":
            if params["encoder"] != "pretrained_transformer":
                decoder_layer = nn.TransformerDecoderLayer(
                    params["word_embed_size"],
                    params["num_heads_transformer"],
                    params["hidden_size_transformer"],
                )
                self.decoder_unit = nn.TransformerDecoder(
                    decoder_layer, params["num_layers_transformer"]
                )
                self.pos_encoder = models.PositionalEncoding(
                    params["word_embed_size"]
                )
            else:
                self.decoder_unit = None
            self.no_peek_mask = None
        elif params["text_encoder"] == "lstm":
            input_size = params["word_embed_size"]
            if params["encoder"] in self.DIALOG_CONTEXT_ENCODERS:
                input_size += params["hidden_size"]
            if params["use_bahdanau_attention"]:
                input_size += params["hidden_size"]
            if params["encoder"] == "tf_idf":
                # If encoder is tf_idf, simple decoder.
                input_size = params["word_embed_size"]
            self.decoder_unit = nn.LSTM(
                input_size,
                params["hidden_size"],
                params["num_layers"],
                batch_first=True,
            )
        else:
            raise NotImplementedError("Text encoder must be not transformer or LSTM!")

        input_size = params["hidden_size"]
        if params["use_bahdanau_attention"] and params["text_encoder"] == "lstm":
            output_size = params["hidden_size"]
            # if self.params['use_action_output']:
            #     input_size += hidden_size
            self.attention_net = nn.Linear(input_size, output_size)
            input_size += params["hidden_size"]

        # If action outputs are to be used.
        if params["use_action_output"]:
            self.action_fusion_net = nn.Linear(
                3 * params["hidden_size"], params["hidden_size"]
            )
        # Reset the input_size if tf_idf.
        if params["encoder"] == "tf_idf":
            input_size = params["hidden_size"]
        self.inv_word_net = nn.Linear(input_size, params["vocab_size"])
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _generate_no_peek_mask(self, size):
        """Generates square masks for transformers to avoid peeking.
        """
        # host = torch.cuda if self.params['use_gpu'] else torch
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        if self.params["use_gpu"]:
            mask = mask.cuda()
        mask = mask.float().masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, batch, encoder_output):
        """Forward pass through the decoder.

        Args:
            batch: Dict of batch variables.
            encoder_output: Dict of outputs from the encoder.

        Returns:
            decoder_outputs: Dict of outputs from the forward pass.
        """
        # Flatten for history_agnostic encoder.
        batch_size, num_rounds, max_length = batch["assist_in"].shape
        decoder_in = support.flatten(batch["assist_in"], batch_size, num_rounds)
        decoder_out = support.flatten(batch["assist_out"], batch_size, num_rounds)
        decoder_len = support.flatten(batch["assist_in_len"], batch_size, num_rounds)
        word_embeds_dec = self.word_embed_net(decoder_in)

        if self.params["encoder"] in self.DIALOG_CONTEXT_ENCODERS:
            dialog_context = support.flatten(
                encoder_output["dialog_context"], batch_size, num_rounds
            ).unsqueeze(1)
            dialog_context = dialog_context.expand(-1, max_length, -1)
            decoder_steps_in = torch.cat([dialog_context, word_embeds_dec], -1)
        else:
            decoder_steps_in = word_embeds_dec

        # Encoder states conditioned on action outputs, if need be.
        if self.params["use_action_output"]:
            action_out = encoder_output["action_output_all"].unsqueeze(1)
            time_steps = encoder_output["hidden_states_all"].shape[1]
            fusion_out = torch.cat(
                [
                    encoder_output["hidden_states_all"],
                    action_out.expand(-1, time_steps, -1),
                ],
                dim=-1,
            )
            encoder_output["hidden_states_all"] = self.action_fusion_net(fusion_out)

        if self.params["text_encoder"] == "transformer":
            # Check the status of no_peek_mask.
            if self.no_peek_mask is None or self.no_peek_mask.size(0) != max_length:
                self.no_peek_mask = self._generate_no_peek_mask(max_length)

            hidden_state = encoder_output["hidden_states_all"]
            enc_pad_mask = batch["user_utt"] == batch["pad_token"]
            enc_pad_mask = support.flatten(enc_pad_mask, batch_size, num_rounds)
            dec_pad_mask = batch["assist_in"] == batch["pad_token"]
            dec_pad_mask = support.flatten(dec_pad_mask, batch_size, num_rounds)
            if self.params["encoder"] != "pretrained_transformer":
                dec_embeds = self.pos_encoder(decoder_steps_in).transpose(0, 1)
                outputs = self.decoder_unit(
                    dec_embeds,
                    hidden_state.transpose(0, 1),
                    memory_key_padding_mask=enc_pad_mask,
                    tgt_mask=self.no_peek_mask,
                    tgt_key_padding_mask=dec_pad_mask,
                )
                outputs = outputs.transpose(0, 1)
            else:
                outputs = self.decoder_unit(
                    inputs_embeds=decoder_steps_in,
                    attention_mask=~dec_pad_mask,
                    encoder_hidden_states=hidden_state,
                    encoder_attention_mask=~enc_pad_mask,
                )
                outputs = outputs[0]
        else:
            hidden_state = encoder_output["hidden_state"]
            if self.params["encoder"] == "tf_idf":
                hidden_state = None

            # If Bahdahnue attention is to be used.
            if (
                self.params["use_bahdanau_attention"]
                and self.params["encoder"] != "tf_idf"
            ):
                encoder_states = encoder_output["hidden_states_all"]
                max_decoder_len = min(
                    decoder_in.shape[1], self.params["max_decoder_len"]
                )
                encoder_states_proj = self.attention_net(encoder_states)
                enc_mask = (batch["user_utt"] == batch["pad_token"]).unsqueeze(-1)
                enc_mask = support.flatten(enc_mask, batch_size, num_rounds)
                outputs = []
                for step in range(max_decoder_len):
                    previous_state = hidden_state[0][-1].unsqueeze(1)
                    att_logits = previous_state * encoder_states_proj
                    att_logits = att_logits.sum(dim=-1, keepdim=True)
                    # Use encoder mask to replace <pad> with -Inf.
                    att_logits.masked_fill_(enc_mask, float("-Inf"))
                    att_wts = nn.functional.softmax(att_logits, dim=1)
                    context = (encoder_states * att_wts).sum(1, keepdim=True)
                    # Run through LSTM.
                    concat_in = [context, decoder_steps_in[:, step : step + 1, :]]
                    step_in = torch.cat(concat_in, dim=-1)
                    decoder_output, hidden_state = self.decoder_unit(
                        step_in, hidden_state
                    )
                    concat_out = torch.cat([decoder_output, context], dim=-1)
                    outputs.append(concat_out)
                outputs = torch.cat(outputs, dim=1)
            else:
                outputs = rnn.dynamic_rnn(
                    self.decoder_unit,
                    decoder_steps_in,
                    decoder_len,
                    init_state=hidden_state,
                )
        if self.params["encoder"] == "pretrained_transformer":
            output_logits = outputs
        else:
            # Logits over vocabulary.
            output_logits = self.inv_word_net(outputs)
        # Mask out the criterion while summing.
        pad_mask = support.flatten(batch["assist_mask"], batch_size, num_rounds)
        loss_token = self.criterion(output_logits.transpose(1, 2), decoder_out)
        loss_token.masked_fill_(pad_mask, 0.0)
        return {"loss_token": loss_token, "pad_mask": pad_mask}

    def forward_beamsearch_multiple(self, batch, encoder_output, mode_params):
        """Performs beamsearch for multilength batch.

        Args:
            batch: Dictionary of inputs, with batch size of 1
            beam_size: Number of beams

        Returns:
            top_beams: Dictionary of top beams
        """
        # Construct new batch and new encoder_output of batch_size 1.
        beam_outputs = batch["user_utt"].clone()
        beam_outputs.fill_(batch["pad_token"])
        batch_size, num_rounds, _ = batch["user_utt"].shape

        # Encoder states conditioned on action outputs, if need be.
        if self.params["use_action_output"]:
            action_out = encoder_output["action_output_all"].unsqueeze(1)
            time_steps = encoder_output["hidden_states_all"].shape[1]
            fusion_out = torch.cat(
                [
                    encoder_output["hidden_states_all"],
                    action_out.expand(-1, time_steps, -1),
                ],
                dim=-1,
            )
            encoder_output["hidden_states_all"] = self.action_fusion_net(fusion_out)

        for inst_id in range(batch_size):
            for round_id in range(num_rounds):
                new_output = {}
                new_batch = {
                    "user_utt": batch["user_utt"][inst_id, round_id].view(1, 1, -1),
                    "pad_token": batch["pad_token"],
                }
                DIALOG_CONTEXT = "dialog_context"
                if DIALOG_CONTEXT in encoder_output:
                    dialog_context = encoder_output[DIALOG_CONTEXT][inst_id][round_id]
                    new_output[DIALOG_CONTEXT] = dialog_context.view(1, 1, -1)
                # Hidden States all.
                index = num_rounds * inst_id + round_id
                if "hidden_states_all" in encoder_output:
                    hidden_state = encoder_output["hidden_states_all"][index]
                    new_output["hidden_states_all"] = hidden_state.unsqueeze(0)
                # Hidden state.
                if "hidden_state" in encoder_output:
                    hidden_state = encoder_output["hidden_state"]
                    if self.params["encoder"] == "tf_idf":
                        new_output["hidden_state"] = None
                    else:
                        new_output["hidden_state"] = tuple(
                            ii[:, index, :].unsqueeze(1).contiguous()
                            for ii in encoder_output["hidden_state"]
                        )
                beamsearch_outputs = self.forward_beamsearch_single(
                    new_batch, new_output, mode_params
                )
                top_beam = beamsearch_outputs["top_beams"][0]
                seq_len = min(len(top_beam), beam_outputs.shape[-1])
                beam_outputs[inst_id, round_id, :seq_len] = top_beam[:seq_len, 0]
        return {"beam_output": beam_outputs}

    def forward_beamsearch_single(self, batch, encoder_output, mode_params):
        """Evaluates the model using beam search with batch size 1.

        NOTE: Current implementation only supports beam search for batch size 1
              and for RNN text_encoder (will be extended for transformers)

        Args:
            batch: Dictionary of inputs, with batch size of 1
            beam_size: Number of beams

        Returns:
            top_beams: Dictionary of top beams
        """
        # Initializations and aliases.
        # Tensors are either on CPU or GPU.
        LENGTH_NORM = True
        self.host = torch.cuda if self.params["use_gpu"] else torch
        end_token = self.params["end_token"]
        start_token = self.params["start_token"]
        beam_size = mode_params["beam_size"]
        max_decoder_len = self.params["max_decoder_len"]

        if self.params["text_encoder"] == "transformer":
            hidden_state = encoder_output["hidden_states_all"].transpose(0, 1)
            max_enc_len, batch_size, enc_embed_size = hidden_state.shape
            hidden_state_expand = hidden_state.expand(
                max_enc_len, beam_size, enc_embed_size
            )
            enc_pad_mask = batch["user_utt"] == batch["pad_token"]
            enc_pad_mask = support.flatten(enc_pad_mask, 1, 1)
            enc_pad_mask_expand = enc_pad_mask.expand(beam_size, max_enc_len)
            if (
                self.no_peek_mask is None
                or self.no_peek_mask.size(0) != max_decoder_len
            ):
                self.no_peek_mask = self._generate_no_peek_mask(max_decoder_len)
        elif self.params["text_encoder"] == "lstm":
            hidden_state = encoder_output["hidden_state"]

        if (
            self.params["use_bahdanau_attention"]
            and self.params["encoder"] != "tf_idf"
            and self.params["text_encoder"] == "lstm"
        ):
            encoder_states = encoder_output["hidden_states_all"]
            encoder_states_proj = self.attention_net(encoder_states)
            enc_mask = (batch["user_utt"] == batch["pad_token"]).unsqueeze(-1)
            enc_mask = support.flatten(enc_mask, 1, 1)

        # Per instance initializations.
        # Copy the hidden state beam_size number of times.
        if hidden_state is not None:
            hidden_state = [ii.repeat(1, beam_size, 1) for ii in hidden_state]
        beams = {-1: self.host.LongTensor(1, beam_size).fill_(start_token)}
        beam_scores = self.host.FloatTensor(beam_size, 1).fill_(0.)
        finished_beams = self.host.ByteTensor(beam_size, 1).fill_(False)
        zero_tensor = self.host.LongTensor(beam_size, 1).fill_(end_token)
        reverse_inds = {}
        # Generate beams until max_len time steps.
        for step in range(max_decoder_len - 1):
            if self.params["text_encoder"] == "transformer":
                beams, tokens_list = self._backtrack_beams(beams, reverse_inds)
                beam_tokens = torch.cat(tokens_list, dim=0).transpose(0, 1)
                beam_tokens_embed = self.word_embed_net(beam_tokens)

                if self.params["encoder"] != "pretrained_transformer":
                    dec_embeds = self.pos_encoder(beam_tokens_embed).transpose(0, 1)
                    output = self.decoder_unit(
                        dec_embeds,
                        hidden_state_expand,
                        tgt_mask=self.no_peek_mask[: step + 1, : step + 1],
                        memory_key_padding_mask=enc_pad_mask_expand,
                    )
                    logits = self.inv_word_net(output[-1])
                else:
                    outputs = self.decoder_unit(
                        inputs_embeds=beam_tokens_embed,
                        encoder_hidden_states=hidden_state_expand.transpose(0, 1),
                        encoder_attention_mask=~enc_pad_mask_expand,
                    )
                    logits = outputs[0][:, -1, :]

            elif self.params["text_encoder"] == "lstm":
                beam_tokens = beams[step - 1].t()
                beam_tokens_embed = self.word_embed_net(beam_tokens)
                # Append dialog context if exists.
                if self.params["encoder"] in self.DIALOG_CONTEXT_ENCODERS:
                    dialog_context = encoder_output["dialog_context"]
                    beam_tokens_embed = torch.cat(
                        [dialog_context.repeat(beam_size, 1, 1), beam_tokens_embed],
                        dim=-1,
                    )
                # Use bahdanau attention over encoder hidden states.
                if (
                    self.params["use_bahdanau_attention"]
                    and self.params["encoder"] != "tf_idf"
                ):
                    previous_state = hidden_state[0][-1].unsqueeze(1)
                    att_logits = previous_state * encoder_states_proj
                    att_logits = att_logits.sum(dim=-1, keepdim=True)
                    # Use encoder mask to replace <pad> with -Inf.
                    att_logits.masked_fill_(enc_mask, float("-Inf"))
                    att_wts = nn.functional.softmax(att_logits, dim=1)
                    context = (encoder_states * att_wts).sum(1, keepdim=True)
                    # Run through LSTM.
                    step_in = torch.cat([context, beam_tokens_embed], dim=-1)
                    decoder_output, new_state = self.decoder_unit(
                        step_in, hidden_state
                    )
                    output = torch.cat([decoder_output, context], dim=-1)
                else:
                    output, new_state = self.decoder_unit(
                        beam_tokens_embed, hidden_state
                    )
                logits = self.inv_word_net(output).squeeze(1)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            # Compute the new beam scores.
            alive = finished_beams.eq(0).float()
            if LENGTH_NORM:
                # Add (current log probs / (step + 1))
                cur_weight = alive / (step + 1)
                # Add (previous log probs * (t/t+1) ) <- Mean update
                prev_weight = alive * step / (step + 1)
            else:
                # No length normalization.
                cur_weight = alive
                prev_weight = alive

            # Compute the new beam extensions.
            if step == 0:
                # For the first step, make all but first beam
                # probabilities -inf.
                log_probs[1:, :] = float("-inf")
            new_scores = log_probs * cur_weight + beam_scores * prev_weight
            finished_beam_scores = beam_scores * finished_beams.float()
            new_scores.scatter_add_(1, zero_tensor, finished_beam_scores)
            # Finished beams scores are set to -inf for all words but one.
            new_scores.masked_fill_(new_scores.eq(0), float("-inf"))

            num_candidates = new_scores.shape[-1]
            new_scores_flat = new_scores.view(1, -1)
            beam_scores, top_inds_flat = torch.topk(new_scores_flat, beam_size)
            beam_scores = beam_scores.t()
            top_beam_inds = (top_inds_flat / num_candidates).squeeze(0)
            top_tokens = top_inds_flat % num_candidates

            # Prepare for next step.
            beams[step] = top_tokens
            reverse_inds[step] = top_beam_inds
            finished_beams = finished_beams[top_beam_inds]
            if self.params["text_encoder"] == "lstm":
                hidden_state = tuple(
                    ii.index_select(1, top_beam_inds) for ii in new_state
                )

            # Update if any of the latest beams are finished, ie, have <END>.
            # new_finished_beams = beams[step].eq(end_token)
            new_finished_beams = beams[step].eq(end_token).type(self.host.ByteTensor)
            finished_beams = finished_beams | new_finished_beams.t()
            if torch.sum(finished_beams).item() == beam_size:
                break

        # Backtrack the beam through indices.
        beams, tokens_list = self._backtrack_beams(beams, reverse_inds)
        # Add an <END> token at the end.
        tokens_list.append(self.host.LongTensor(1, beam_size).fill_(end_token))

        sorted_beam_tokens = torch.cat(tokens_list, 0).t()
        sorted_beam_lengths = sorted_beam_tokens.ne(end_token).long().sum(dim=1)
        # Trim all the top beams.
        top_beams = []
        for index in range(beam_size):
            beam_length = sorted_beam_lengths[index].view(-1)
            beam = sorted_beam_tokens[index].view(-1, 1)[1:beam_length]
            top_beams.append(beam)
        return {"top_beams": top_beams}

    def _backtrack_beams(self, beam_tokens, reverse_inds):
        """Subroutine backtracks beams based on the reverse lookup indices.

        Args:
            beam_tokens: Dict of tokens sampled for the beam at each time step
            reverse_inds: Reverse index lookup for beams at each step

        Return:
            backtrack_beam: Beam reconstructed through backtracking
        """
        backtrack_beams = copy.deepcopy(beam_tokens)
        max_time_step = max(beam_tokens)
        beam_size = beam_tokens[max_time_step].shape[1]
        start_token = self.params["start_token"]
        for step in reversed(range(1, max_time_step + 1)):
            cur_reverse_inds = reverse_inds[step]
            if step - 1 in reverse_inds:
                prev_reverse_inds = reverse_inds[step - 1]
                reverse_inds[step - 1] = prev_reverse_inds[cur_reverse_inds]
            # Replace after index selecting.
            prev_tokens = beam_tokens[step - 1]
            backtrack_beams[step - 1] = prev_tokens.index_select(1, cur_reverse_inds)
        # Get everything together.
        backtrack_beams[-1] = self.host.LongTensor(1, beam_size).fill_(start_token)
        tokens_list = [
            ii for _, ii in sorted(backtrack_beams.items(), key=lambda x: x[0])
        ]
        return backtrack_beams, tokens_list
