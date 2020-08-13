"""Executes the actions and predicts action attributes for SIMMC.

Author(s): Satwik Kottur
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import json
import torch
import torch.nn as nn

import loaders
import models
from tools import torch_support as support


class ActionExecutor(nn.Module):
    def __init__(self, params):
        """Initialize classifiers.
        """
        super(ActionExecutor, self).__init__()
        self.params = params

        input_size = self.params["hidden_size"]
        if self.params["text_encoder"] == "transformer":
            input_size = self.params["word_embed_size"]
        self.action_net = self._get_classify_network(input_size, params["num_actions"])
        if params["use_action_attention"]:
            self.attention_net = models.SelfAttention(input_size)
        # If multimodal input state is to be used.
        if self.params["use_multimodal_state"]:
            input_size += self.params["hidden_size"]
        self.action_net = self._get_classify_network(input_size, params["num_actions"])
        # Read action metadata.
        with open(params["metainfo_path"], "r") as file_id:
            action_metainfo = json.load(file_id)["actions"]
            action_dict = {ii["name"]: ii["id"] for ii in action_metainfo}
            self.action_metainfo = {ii["name"]: ii for ii in action_metainfo}
            self.action_map = loaders.Vocabulary(immutable=True, verbose=False)
            sorted_actions = sorted(action_dict.keys(), key=lambda x: action_dict[x])
            self.action_map.set_vocabulary_state(sorted_actions)
        # Read action attribute metadata.
        with open(params["attr_vocab_path"], "r") as file_id:
            self.attribute_vocab = json.load(file_id)
        # Create classifiers for action attributes.
        self.classifiers = {}
        for key, val in self.attribute_vocab.items():
            self.classifiers[key] = self._get_classify_network(
                input_size, len(val)
            )
        self.classifiers = nn.ModuleDict(self.classifiers)

        # Model multimodal state.
        if params["use_multimodal_state"]:
            if params["domain"] == "furniture":
                self.multimodal_embed = models.CarouselEmbedder(params)
            elif params["domain"] == "fashion":
                self.multimodal_embed = models.UserMemoryEmbedder(params)
            else:
                raise ValueError("Domain neither of furniture/fashion")

        # NOTE: Action output is modeled as multimodal state.
        if params["use_action_output"]:
            if params["domain"] == "furniture":
                self.action_output_embed = models.CarouselEmbedder(params)
            elif params["domain"] == "fashion":
                self.action_output_embed = models.UserMemoryEmbedder(params)
            else:
                raise ValueError("Domain neither of furniture/fashion")
        self.criterion_mean = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.criterion_multi = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, batch, prev_outputs):
        """Forward pass a given batch.

        Args:
            batch: Batch to forward pass
            prev_outputs: Output from previous modules.

        Returns:
            outputs: Dict of expected outputs
        """
        outputs = {}
        if self.params["use_action_attention"] and self.params["encoder"] != "tf_idf":
            encoder_state = prev_outputs["hidden_states_all"]
            batch_size, num_rounds, max_len = batch["user_mask"].shape
            encoder_mask = batch["user_utt"].eq(batch["pad_token"])
            encoder_mask = support.flatten(encoder_mask, batch_size, num_rounds)
            encoder_state = self.attention_net(encoder_state, encoder_mask)
        else:
            encoder_state = prev_outputs["hidden_state"][0][-1]

        encoder_state_old = encoder_state
        # Multimodal state.
        if self.params["use_multimodal_state"]:
            if self.params["domain"] == "furniture":
                encoder_state = self.multimodal_embed(
                    batch["carousel_state"],
                    encoder_state,
                    batch["dialog_mask"].shape[:2]
                )
            elif self.params["domain"] == "fashion":
                multimodal_state = {}
                for ii in ["memory_images", "focus_images"]:
                    multimodal_state[ii] = batch[ii]
                encoder_state = self.multimodal_embed(
                    multimodal_state, encoder_state, batch["dialog_mask"].shape[:2]
                )

        # Predict and execute actions.
        action_logits = self.action_net(encoder_state)
        dialog_mask = batch["dialog_mask"]
        batch_size, num_rounds = dialog_mask.shape
        loss_action = self.criterion(action_logits, batch["action"].view(-1))
        loss_action.masked_fill_((~dialog_mask).view(-1), 0.0)
        loss_action_sum = loss_action.sum() / dialog_mask.sum().item()
        outputs["action_loss"] = loss_action_sum
        if not self.training:
            # Check for action accuracy.
            action_logits = support.unflatten(action_logits, batch_size, num_rounds)
            actions = action_logits.argmax(dim=-1)
            action_logits = nn.functional.log_softmax(action_logits, dim=-1)
            action_list = self.action_map.get_vocabulary_state()
            # Convert predictions to dictionary.
            action_preds_dict = [
                {
                    "dialog_id": batch["dialog_id"][ii].item(),
                    "predictions": [
                        {
                            "action": self.action_map.word(actions[ii, jj].item()),
                            "action_log_prob": {
                                action_token: action_logits[ii, jj, kk].item()
                                for kk, action_token in enumerate(action_list)
                            },
                            "attributes": {}
                        }
                        for jj in range(batch["dialog_len"][ii])
                    ]
                }
                for ii in range(batch_size)
            ]
            outputs["action_preds"] = action_preds_dict
        else:
            actions = batch["action"]

        # Run classifiers based on the action, record supervision if training.
        if self.training:
            assert (
                "action_super" in batch
            ), "Need supervision to learn action attributes"
        attr_logits = collections.defaultdict(list)
        attr_loss = collections.defaultdict(list)
        encoder_state_unflat = support.unflatten(
            encoder_state, batch_size, num_rounds
        )

        host = torch.cuda if self.params["use_gpu"] else torch
        for inst_id in range(batch_size):
            for round_id in range(num_rounds):
                # Turn out of dialog length.
                if not dialog_mask[inst_id, round_id]:
                    continue

                cur_action_ind = actions[inst_id, round_id].item()
                cur_action = self.action_map.word(cur_action_ind)
                cur_state = encoder_state_unflat[inst_id, round_id]
                supervision = batch["action_super"][inst_id][round_id]
                # If there is no supervision, ignore and move on to next round.
                if supervision is None:
                    continue

                # Run classifiers on attributes.
                # Attributes overlaps completely with GT when training.
                if self.training:
                    classifier_list = self.action_metainfo[cur_action]["attributes"]
                    if self.params["domain"] == "furniture":
                        for key in classifier_list:
                            cur_gt = (
                                supervision.get(key, None)
                                if supervision is not None
                                else None
                            )
                            new_entry = (cur_state, cur_gt, inst_id, round_id)
                            attr_logits[key].append(new_entry)
                    elif self.params["domain"] == "fashion":
                        for key in classifier_list:
                            cur_gt = supervision.get(key, None)
                            gt_indices = host.FloatTensor(
                                len(self.attribute_vocab[key])
                            ).fill_(0.)
                            gt_indices[cur_gt] = 1
                            new_entry = (cur_state, gt_indices, inst_id, round_id)
                            attr_logits[key].append(new_entry)
                    else:
                        raise ValueError("Domain neither of furniture/fashion!")
                else:
                    classifier_list = self.action_metainfo[cur_action]["attributes"]
                    action_pred_datum = action_preds_dict[
                        inst_id
                    ]["predictions"][round_id]
                    if self.params["domain"] == "furniture":
                        # Predict attributes based on the predicted action.
                        for key in classifier_list:
                            classifier = self.classifiers[key]
                            model_pred = classifier(cur_state).argmax(dim=-1)
                            attr_pred = self.attribute_vocab[key][model_pred.item()]
                            action_pred_datum["attributes"][key] = attr_pred
                    elif self.params["domain"] == "fashion":
                        # Predict attributes based on predicted action.
                        for key in classifier_list:
                            classifier = self.classifiers[key]
                            model_pred = classifier(cur_state) > 0
                            attr_pred = [
                                self.attribute_vocab[key][index]
                                for index, ii in enumerate(model_pred)
                                if ii
                            ]
                            action_pred_datum["attributes"][key] = attr_pred
                    else:
                        raise ValueError("Domain neither of furniture/fashion!")

        # Compute losses if training, else predict.
        if self.training:
            for key, values in attr_logits.items():
                classifier = self.classifiers[key]
                prelogits = [ii[0] for ii in values if ii[1] is not None]
                if not prelogits:
                    continue
                logits = classifier(torch.stack(prelogits, dim=0))
                if self.params["domain"] == "furniture":
                    gt_labels = [ii[1] for ii in values if ii[1] is not None]
                    gt_labels = host.LongTensor(gt_labels)
                    attr_loss[key] = self.criterion_mean(logits, gt_labels)
                elif self.params["domain"] == "fashion":
                    gt_labels = torch.stack(
                        [ii[1] for ii in values if ii[1] is not None], dim=0
                    )
                    attr_loss[key] = self.criterion_multi(logits, gt_labels)
                else:
                    raise ValueError("Domain neither of furniture/fashion!")

            total_attr_loss = host.FloatTensor([0.0])
            if len(attr_loss.values()):
                total_attr_loss = sum(attr_loss.values()) / len(attr_loss.values())
            outputs["action_attr_loss"] = total_attr_loss

        # Obtain action outputs as memory cells to attend over.
        if self.params["use_action_output"]:
            if self.params["domain"] == "furniture":
                encoder_state_out = self.action_output_embed(
                    batch["action_output"],
                    encoder_state_old,
                    batch["dialog_mask"].shape[:2],
                )
            elif self.params["domain"] == "fashion":
                multimodal_state = {}
                for ii in ["memory_images", "focus_images"]:
                    multimodal_state[ii] = batch[ii]
                # For action output, advance focus_images by one time step.
                # Output at step t is input at step t+1.
                feature_size = batch["focus_images"].shape[-1]
                zero_tensor = host.FloatTensor(batch_size, 1, feature_size).fill_(0.)
                multimodal_state["focus_images"] = torch.cat(
                    [batch["focus_images"][:, 1:, :], zero_tensor], dim=1
                )
                encoder_state_out = self.multimodal_embed(
                    multimodal_state, encoder_state_old, batch["dialog_mask"].shape[:2]
                )
            else:
                raise ValueError("Domain neither furniture/fashion!")
            outputs["action_output_all"] = encoder_state_out

        outputs.update(
            {"action_logits": action_logits, "action_attr_loss_dict": attr_loss}
        )
        return outputs

    def _get_classify_network(self, input_size, num_classes):
        """Construct network for predicting actiosn and attributes.
        """
        return nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, num_classes),
        )
