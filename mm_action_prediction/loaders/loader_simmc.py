"""Dataloader for SIMMC Dataset.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import numpy as np
from nltk.tokenize import word_tokenize

import loaders
from tools import support, torch_support
from tools.action_evaluation import evaluate_action_prediction
from tools.response_evaluation import evaluate_response_generation
from tools.retrieval_evaluation import evaluate_response_retrieval


class DataloaderSIMMC(loaders.LoaderParent):
    """Loads data for SIMMC datasets.
    """

    def __init__(self, params):
        self.params = params
        # Load the dataset.
        raw_data = np.load(params["data_read_path"], allow_pickle=True)
        self.raw_data = raw_data[()]
        if self.params["encoder"] != "pretrained_transformer":
            self.words = loaders.Vocabulary()
            self.words.set_vocabulary_state(self.raw_data["vocabulary"]["word"])
            # Aliases.
            self.start_token = self.words.index("<start>")
            self.end_token = self.words.index("<end>")
            self.pad_token = self.words.index("<pad>")
            self.unk_token = self.words.index("<unk>")
        else:
            from transformers import BertTokenizer
            self.words = BertTokenizer.from_pretrained(self.raw_data["vocabulary"])
            # Aliases.
            self.start_token = self.words.added_tokens_encoder["[start]"]
            self.end_token = self.words.added_tokens_encoder["[end]"]
            self.pad_token = self.words.pad_token_id
            self.unk_token = self.words.unk_token_id
            self.words.word = self.words.convert_ids_to_tokens
            self.words.index = self.words.convert_tokens_to_ids

        # Read the metainfo for the dataset.
        with open(params["metainfo_path"], "r") as file_id:
            self.metainfo = json.load(file_id)
        self.action_map = {ii["name"]: ii["id"] for ii in self.metainfo["actions"]}

        # Read the attribute vocabulary for the dataset.
        with open(params["attr_vocab_path"], "r") as file_id:
            attribute_map = json.load(file_id)
        print("Loading attribute vocabularies..")
        self.attribute_map = {}
        for attr, attr_vocab in attribute_map.items():
            self.attribute_map[attr] = loaders.Vocabulary(
                immutable=True, verbose=False
            )
            self.attribute_map[attr].set_vocabulary_state(attr_vocab)
        # Encode attribute supervision.
        for d_id, super_datum in enumerate(self.raw_data["action_supervision"]):
            for r_id, round_datum in enumerate(super_datum):
                if round_datum is None:
                    continue
                if self.params["domain"] == "furniture":
                    new_supervision = {
                        key: self.attribute_map[key].index(val)
                        for key, val in round_datum.items()
                        if key in self.attribute_map
                    }
                elif self.params["domain"] == "fashion":
                    ATTRIBUTE_FIXES = {
                        "embellishment": "embellishments", "hemlength": "hemLength"
                    }
                    new_supervision = {}
                    for key, val in round_datum.items():
                        # No dictionary to map attributes to indices.
                        # (Non-classification/categorical fields)
                        if key not in self.attribute_map:
                            continue
                        # Encode each attribute -- multi-class classification.
                        fixed_keys = [ATTRIBUTE_FIXES.get(ii, ii) for ii in val]
                        new_supervision[key] = [
                            self.attribute_map[key].index(ii)
                            if ii in self.attribute_map[key]
                            else self.attribute_map[key].index("other")
                            for ii in fixed_keys
                        ]
                else:
                    raise ValueError("Domain must be either furniture/fashion!")
                self.raw_data["action_supervision"][d_id][r_id] = new_supervision

        if self.params["domain"] == "furniture":
            if self.params["use_multimodal_state"]:
                # Read embeddings for furniture assets to model carousel state.
                self._prepare_carousel_states()
            if self.params["use_action_output"]:
                # Output for the actions.
                self._prepare_carousel_states(key="action_output_state")
        elif self.params["domain"] == "fashion":
            # Prepare embeddings for fashion items.
            self._prepare_asset_embeddings()
        else:
            raise ValueError("Domain must be either furniture/fashion!")


        # Additional data constructs (post-processing).
        if params["encoder"] == "memory_network":
            self._construct_fact()
        elif params["encoder"] == "tf_idf":
            self.compute_idf_features()
        super(DataloaderSIMMC, self).__init__()

    def load_one_batch(self, sample_ids):
        """Loads a batch, given the sample ids.

        Args:
            sample_ids: List of instance ids to load data for.

        Returns:
            batch: Dictionary with relevant fields for training/evaluation.
        """
        batch = {
            "pad_token": self.pad_token,
            "start_token": self.start_token,
            "sample_ids": sample_ids,
        }
        batch["dialog_len"] = self.raw_data["dialog_len"][sample_ids]
        batch["dialog_id"] = self.raw_data["dialog_id"][sample_ids]
        max_dialog_len = max(batch["dialog_len"])

        user_utt_id = self.raw_data["user_utt_id"][sample_ids]
        batch["user_utt"], batch["user_utt_len"] = self._sample_utterance_pool(
            user_utt_id,
            self.raw_data["user_sent"],
            self.raw_data["user_sent_len"],
            self.params["max_encoder_len"],
        )

        for key in ("assist_in", "assist_out"):
            batch[key], batch[key + "_len"] = self._sample_utterance_pool(
                self.raw_data["assist_utt_id"][sample_ids],
                self.raw_data[key],
                self.raw_data["assist_sent_len"],
                self.params["max_decoder_len"],
            )
        actions = self.raw_data["action"][sample_ids]
        batch["action"] = np.vectorize(lambda x: self.action_map[x])(actions)
        # Construct user, assistant, and dialog masks.
        batch["dialog_mask"] = user_utt_id != -1
        batch["user_mask"] = (batch["user_utt"] == batch["pad_token"]) | (
            batch["user_utt"] == batch["start_token"]
        )
        batch["assist_mask"] = (batch["assist_out"] == batch["pad_token"]) | (
            batch["assist_out"] == batch["start_token"]
        )

        # Get retrieval candidates if needed.
        if self.params["get_retrieval_candidates"]:
            retrieval_inds = self.raw_data["retrieval_candidates"][sample_ids]
            batch_size, num_rounds, _ = retrieval_inds.shape
            flat_inds = torch_support.flatten(
                retrieval_inds, batch_size, num_rounds
            )
            for key in ("assist_in", "assist_out"):
                new_key = key.replace("assist", "candidate")
                cands, cands_len = self._sample_utterance_pool(
                    flat_inds,
                    self.raw_data[key],
                    self.raw_data["assist_sent_len"],
                    self.params["max_decoder_len"],
                )
                batch[new_key] = torch_support.unflatten(
                    cands, batch_size, num_rounds
                )
                batch[new_key + "_len"] = torch_support.unflatten(
                    cands_len, batch_size, num_rounds
                )
            batch["candidate_mask"] = (
                (batch["candidate_out"] == batch["pad_token"])
                | (batch["candidate_out"] == batch["start_token"])
            )

        # Action supervision.
        batch["action_super"] = [
            self.raw_data["action_supervision"][ii] for ii in sample_ids
        ]

        # Fetch facts if required.
        if self.params["encoder"] == "memory_network":
            batch["fact"] = self.raw_data["fact"][sample_ids]
            batch["fact_len"] = self.raw_data["fact_len"][sample_ids]

        # Trim to the maximum dialog length.
        for key in (
            "assist_in",
            "assist_out",
            "candidate_in",
            "candidate_out",
            "user_utt",
            "fact",
            "user_mask",
            "assist_mask",
            "candidate_mask"
        ):
            if key in batch:
                batch[key] = batch[key][:, :max_dialog_len]
        for key in (
            "action",
            "assist_in_len",
            "assist_out_len",
            "candidate_in_len",
            "candidate_out_len",
            "user_utt_len",
            "dialog_mask",
            "fact_len",
        ):
            if key in batch:
                batch[key] = batch[key][:, :max_dialog_len]
        # TF-IDF features.
        if self.params["encoder"] == "tf_idf":
            batch["user_tf_idf"] = self.compute_tf_features(
                batch["user_utt"], batch["user_utt_len"]
            )

        # Domain-specific processing.
        if self.params["domain"] == "furniture":
            # Carousel states.
            if self.params["use_multimodal_state"]:
                batch["carousel_state"] = [
                    self.raw_data["carousel_state"][ii] for ii in sample_ids
                ]
            # Action output.
            if self.params["use_action_output"]:
                batch["action_output"] = [
                    self.raw_data["action_output_state"][ii] for ii in sample_ids
                ]
        elif self.params["domain"] == "fashion":
            # Asset embeddings -- memory, database, focus images.
            for dtype in ["memory", "database", "focus"]:
                indices = self.raw_data["{}_inds".format(dtype)][sample_ids]
                image_embeds = self.embed_data["embedding"][indices]
                batch["{}_images".format(dtype)] = image_embeds
        else:
            raise ValueError("Domain must be either furniture/fashion!")
        return self._ship_torch_batch(batch)

    def _sample_utterance_pool(
        self, utterance_ids, pool, pool_len, max_utterance_len=None
    ):
        """Sample ids from a pool along with utterance lengths.

        Args:
            utterance_ids: Utterance ids to index
            pool: Utterance pool to index from
            pool_len: Utterance pool length
            max_utterance_len: Maximum utterance length to use
        """
        if max_utterance_len:
            max_len = min(pool.shape[1], max_utterance_len)
        else:
            max_len = min(pool.shape[1], self.params["max_encoder_len"])
        new_size = utterance_ids.shape + (max_len,)
        utterances = np.zeros(new_size, dtype="int32")
        utterances.fill(self.pad_token)
        # NOTE: Empty sentence have one valid word <start>.
        utterances[:, :, 0] = self.start_token
        utterance_lens = np.ones(utterance_ids.shape, dtype="int32")

        for dialog_id in range(utterance_ids.shape[0]):
            for round_id in range(utterance_ids.shape[1]):
                pool_index = utterance_ids[dialog_id, round_id]
                if pool_index == -1:
                    break
                utterances[dialog_id, round_id] = pool[pool_index][:max_len]
                utterance_lens[dialog_id, round_id] = min(pool_len[pool_index], max_len)
        return utterances, utterance_lens

    def _prepare_carousel_states(self, key="carousel_state"):
        """Prepares carousel states for action generation conditioning.
        """
        embed_data = np.load(self.params["asset_embed_path"], allow_pickle=True)[()]
        self.asset_embeds = self._ship_helper(embed_data["embedding"])
        self.asset_id_map = {
            asset_id: index for index, asset_id in enumerate(embed_data["asset_id"])
        }
        self.asset_feature_size = embed_data["asset_feature_size"]
        # Simplify and prepare carousel states.
        carousel_states = self.raw_data[key]
        for inst_id, inst_datum in enumerate(carousel_states):
            for round_id, round_datum in enumerate(inst_datum):
                if round_datum is None:
                    carousel_states[inst_id][round_id] = None
                    continue
                focus_id = round_datum["focus"]
                if focus_id is not None and focus_id != "":
                    carousel_states[inst_id][round_id] = {
                        "focus": self.asset_embeds[self.asset_id_map[int(focus_id)]]
                    }
                elif len(round_datum["carousel"]) != 0:
                    asset_inds = [
                        self.asset_id_map[int(ii)] for ii in round_datum["carousel"]
                    ]
                    carousel_states[inst_id][round_id] = {
                        "carousel": self.asset_embeds[asset_inds]
                    }
                else:
                    carousel_states[inst_id][round_id] = None

    def _prepare_asset_embeddings(self):
        """Get memory and database image embeddings for given sample ids.
        """
        # Read embeddings for fashion items.
        self.embed_data = np.load(self.params["asset_embed_path"], allow_pickle=True)[
            ()
        ]
        self.embed_data["asset_map"] = {
            key: index for index, key in enumerate(self.embed_data["asset_id"])
        }

        def map_asset_indices(asset_indices):
            """Given asset indices, map them to embedding indices.
            """
            return [self.embed_data["asset_map"][ii] for ii in asset_indices]

        # Append with a zero vector to enable indexing empty values with -1.
        self.asset_feature_size = self.embed_data["asset_feature_size"]
        zero_vector = np.zeros((1, self.asset_feature_size), dtype=np.int32)
        self.embed_data["embedding"] = np.concatenate(
            [self.embed_data["embedding"], zero_vector], axis=0
        )
        # Additional checks for Focus Images.
        for inst_id in range(self.num_instances):
            num_turns = self.raw_data["dialog_len"][inst_id]
            assert (
                len(self.raw_data["focus_images"][inst_id]) == num_turns
            ), "Number of turns do not match in focus images"
        # Go over data and get the indices for memory, database, focus images.
        for name in ["memory", "database", "focus"]:
            images = self.raw_data["{}_images".format(name)]
            # Get max size for the array and initialize indices.
            max_size = max(len(ii) for ii in images)
            indices = np.full((self.num_instances, max_size), -1, np.int32)
            for inst_id in range(self.num_instances):
                num_images = len(images[inst_id])
                indices[inst_id, :num_images] = map_asset_indices(images[inst_id])
            self.raw_data["{}_inds".format(name)] = indices

    def _construct_fact(self):
        """Method to construct facts.

        Facts are previous utterance + response concatenated as one. These
        serve as memory units that the model can refer back to.

        For example, 'What is the color of the couch? A: Red.' is a fact.
        """
        print("Constructing facts...", end=""),
        num_dialogs, num_rounds = self.raw_data["user_utt_id"].shape
        # Concatenate both user + assistance utterances.
        max_len = 2 * self.params["max_encoder_len"]
        fact = np.zeros((num_dialogs, num_rounds, max_len), dtype=np.int32)
        fact_len = np.zeros((num_dialogs, num_rounds), dtype=np.int32)
        # Fact for the first round is just <pad> token.
        fact_len[:, 0] = 1
        fact.fill(self.pad_token)
        for diag_id in range(num_dialogs):
            for r_id in range(num_rounds - 1):
                # User utterance.
                utt_id = self.raw_data["user_utt_id"][diag_id, r_id]
                user_utt = self.raw_data["user_sent"][utt_id]
                user_utt_len = self.raw_data["user_sent_len"][utt_id]
                # Assistant utterance.
                utt_id = self.raw_data["assist_utt_id"][diag_id, r_id]
                assist_utt = self.raw_data["assist_sent"][utt_id]
                assist_utt_len = self.raw_data["assist_sent_len"][utt_id]
                # Handle overflow.
                bound = min(user_utt_len, max_len)
                fact[diag_id, r_id + 1, :bound] = user_utt[:bound]
                if bound < max_len:
                    bound = min(user_utt_len + assist_utt_len, max_len)
                    fact[diag_id, r_id + 1, user_utt_len:bound] = assist_utt[
                        : bound - user_utt_len
                    ]
                fact_len[diag_id, r_id + 1] = bound
        self.raw_data["fact"] = fact
        self.raw_data["fact_len"] = fact_len
        print("done")

    def _construct_history(self):
        """Method to construct history.

        History is concatenation of previous utterances + responses
        They serve as memory units that the model can use to encode the current
        utterance.

        For example,
        'What is the color of the couch? Red. 'What about the table? Blue' is
        a fact.
        """
        print("Constructing history...", end=""),
        num_dialogs, num_rounds = self.raw_data["user_utt_id"].shape
        # Concatenate all user + assistant utterances.
        max_len = self.params["max_history_len"]
        history = np.full(
            (num_dialogs, num_rounds, max_len), self.pad_token, dtype=np.int32
        )
        history[:, :, 0] = self.start_token
        history_len = np.ones((num_dialogs, num_rounds), dtype=np.int32)
        for diag_id in range(num_dialogs):
            run_history = np.array([], dtype=np.int32)
            run_history_len = 0
            for r_id in range(num_rounds - 1):
                # User utterance.
                utt_id = self.raw_data["user_utt_id"][diag_id, r_id]
                user_utt = self.raw_data["user_sent"][utt_id]
                user_utt_len = self.raw_data["user_sent_len"][utt_id]
                # Assistant utterance.
                utt_id = self.raw_data["assist_utt_id"][diag_id, r_id]
                assist_utt = self.raw_data["assist_sent"][utt_id]
                assist_utt_len = self.raw_data["assist_sent_len"][utt_id]
                # Append user utterance.
                run_history = np.concatenate(
                    [run_history, user_utt[:user_utt_len]], axis=-1
                )
                run_history_len += user_utt_len

                copy_len = min(max_len, run_history_len)
                history[diag_id, r_id, :copy_len] = run_history[-copy_len:]
                history_len[diag_id, r_id] = copy_len
                # Append assistant utterance.
                run_history = np.concatenate(
                    [run_history, assist_utt[:assist_utt_len]], axis=-1
                )
                run_history_len += assist_utt_len
        self.raw_data["history"] = history
        self.raw_data["history_len"] = history_len
        print("done")

    def interactive_batch(self, input_str, round_id):
        """Generate a batch out of interactive chat.

        Args:
          input_str: Input string from the interactive chat.

        Returns:
          batch: Dictionary to feedforward to the model.
        """
        tokens = word_tokenize(input_str.lower())
        enc_in = np.array(
            [self.words.index(ii, unk_default=True) for ii in tokens], dtype="int32"
        )
        enc_in = enc_in.reshape(1, 1, -1)
        enc_len = np.array([len(tokens)], dtype="int32").reshape(1, 1)
        # Dialog mask, action.
        batch = {"pad_token": self.pad_token, "start_token": self.start_token}
        batch.update(
            {
                "user_utt": enc_in,
                "user_utt_len": enc_len,
                "dialog_mask": np.ones((1, 1), dtype=np.bool),
                "round_id": np.array([round_id], dtype="int32"),
            }
        )
        return self._ship_torch_batch(batch)

    def stringify_beam_outputs(self, beam_outputs, batch):
        """Stringifies beamsearch outputs.

        Args:
            beam_outputs: Outputs of beamsearch generation
            batch: Current batch
        """
        def stringify(indices):
            tokens = []
            # Convert to numpy array.
            for ii in indices.cpu().numpy():
                if ii == self.end_token:
                    break
                if ii == self.pad_token:
                    continue
                tokens.append(self.words.word(int(ii)))
            return " ".join(tokens)
        stringified_beam_output = [
            {
                "dialog_id": batch["dialog_id"][ii].item(),
                "predictions": [
                    {"response": stringify(beam_outputs[ii][jj])}
                    for jj in range(batch["dialog_len"][ii])
                ]
            }
            for ii in range(beam_outputs.shape[0])
        ]
        return stringified_beam_output

    def evaluate_response_generation(self, model_responses):
        """Evaluates response generation comparing against ground truth.

        Args:
            model_responses: Model responses
        """
        assert "data" in self.raw_data["paths"], "Cannot find data file!"
        if not hasattr(self, "gt_responses"):
            with open(self.raw_data["paths"]["data"][0], "r") as file_id:
                self.gt_responses = json.load(file_id)
        return evaluate_response_generation(self.gt_responses, model_responses)

    def evaluate_response_retrieval(self, candidate_scores):
        """Evaluates response retrieval comparing against ground truth.

        Args:
            candidate_scores: Candidate scores
        """
        assert "retrieval" in self.raw_data["paths"], "Cannot find data file!"
        if not hasattr(self, "retrieval_candidates"):
            with open(self.raw_data["paths"]["retrieval"], "r") as file_id:
                self.retrieval_candidates = json.load(file_id)
        return evaluate_response_retrieval(
            self.retrieval_candidates, candidate_scores
        )

    def evaluate_action_prediction(self, model_actions):
        """Evaluate action prediction comparing against ground truth.

        Args:
            model_actions: Model predictions
        """
        assert "action" in self.raw_data["paths"], "Cannot find GT action file!"
        if not hasattr(self, "gt_actions"):
            with open(self.raw_data["paths"]["action"], "r") as file_id:
                self.gt_actions = json.load(file_id)
        return evaluate_action_prediction(self.gt_actions, model_actions)

    def additional_analysis(self):
        """Performs additional analysis after reading the data.
        """
        distribution = {}
        # Percent of attributes are in the supervision.
        for d_id, super_datum in enumerate(self.raw_data["action_supervision"]):
            for r_id, round_datum in enumerate(super_datum):
                (d_id, r_id)
                if round_datum is None:
                    continue
                for key, val in round_datum.items():
                    if key not in distribution:
                        distribution[key] = {}
                    distribution[key][val] = distribution[key].get(val, 0) + 1
        for key, val in distribution.items():
            support.print_distribution(val, key)

    @property
    def num_instances(self):
        """Get number of data instances in the current object.
        """
        return self.raw_data["user_utt_id"].shape[0]

    @property
    def vocab_size(self):
        """Return the vocabulary size for the dataloader.
        """
        return self.words.vocab_size

    @property
    def num_actions(self):
        """Return the number of possible actions.
        """
        return len(self.action_map)
