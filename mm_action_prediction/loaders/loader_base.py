"""Parent class for data loaders.

Author: Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import threading
import queue

import numpy as np
import torch


class LoaderParent:
    def __init__(self):
        """Class constructor.
        """
        # Assert the presence of mandatory attributes to setup prefetch daemon.
        mandatory_attrs = ["single_pass", "shuffle", "use_gpu"]
        assert hasattr(self, "params"), "Params is mandatory attribute!"
        for attr in mandatory_attrs:
            assert attr in self.params, "{0} is mandatory!".format(attr)
        self.params["prefetch_num"] = self.params.get("prefetch_num", 1)
        self._setup_prefetching()

    def load_one_batch(self, sample_ids):
        """Load one batch given the sample indices.

        Args:
          sample_ids: Ids of the instances -- either train/val.

        Return:
          batch: Dictionary of features for the instance with sample_ids.
        """
        raise NotImplementedError

    def _setup_prefetching(self):
        """Prefetches batches to save time.
        """
        # Setup and start prefetching daemon.
        self._prefetch_queue = queue.Queue(maxsize=self.params["prefetch_num"])
        self._prefetch_thread = threading.Thread(target=self._run_prefetch)
        self._prefetch_thread.daemon = True
        self._prefetch_thread.start()

    def get_batch(self):
        """Batch generator depending on train/eval mode.
        """
        while True:
            # Get a batch from the prefetching queue
            if self._prefetch_queue.empty():
                pass
                # print('DataLoader: Waiting for data loading (IO is slow)...')
            batch = self._prefetch_queue.get(block=True)
            if batch is None:
                assert self.params["single_pass"], "Mode set to one pass!"
                return
            yield batch

    def _run_prefetch(self):
        batch_size = self.params["batch_size"]
        fetch_order = np.arange(self.num_instances)
        n_sample = 0
        while True:
            # Shuffle the sample order for every epoch.
            if n_sample == 0 and self.params["shuffle"]:
                fetch_order = np.random.permutation(self.num_instances)
            # Load batch from file
            # note that len(sample_ids) <= batch_size, not necessarily equal.
            sample_ids = fetch_order[n_sample : n_sample + batch_size]
            batch = self.load_one_batch(sample_ids)
            self._prefetch_queue.put(batch, block=True)
            n_sample += len(sample_ids)
            if n_sample >= self.num_instances:
                # Put in a None batch to indicate a whole pass is over.
                if self.params["single_pass"]:
                    self._prefetch_queue.put(None, block=True)
                n_sample = 0

    def _ship_torch_batch(self, batch):
        """Ship a batch in PyTorch.

      Useful for cross-package dataloader.

      Args:
        batch: Dictionary of the batch.

      Returns:
        Batch members changed in place to torch Tensors (with GPU, if needed)
      """
        for key, value in batch.items():
            # Check if numpy array or list of numpy arrays.
            if isinstance(value, np.ndarray):
                batch[key] = self._ship_helper(value)
            elif isinstance(value, list) and isinstance(value[0], np.ndarray):
                for index, element in enumerate(value):
                    batch[key][index] = self._ship_helper(element)
        return batch

    def _ship_helper(self, numpy_array):
        """Helper to ship numpy arrays to torch.
      """
        # int32 get mapped to int64 and float to double
        if numpy_array.dtype == np.int32 or numpy_array.dtype == np.int64:
            new_type = torch.int64
        elif numpy_array.dtype == np.bool:
            new_type = torch.bool
        else:
            new_type = torch.float
        torch_tensor = torch.tensor(numpy_array, dtype=new_type)
        if self.params["use_gpu"]:
            torch_tensor = torch_tensor.cuda()
        return torch_tensor

    def compute_idf_features(self):
        """Computes idf scores based on train set.
        """
        # Should not be invoked if mandatory fields are absent.
        mandatory_fields = [
            "user_sent",
            "user_sent_len",
            "user_utt_id",
            "assist_sent",
            "assist_sent_len",
            "assist_utt_id",
        ]
        for field in mandatory_fields:
            assert field in self.raw_data, "{} missing!".format(field)
        # Get document frequency of words for both user / assistant utterances.
        IDF = np.ones(self.vocab_size)
        num_inst, max_len = self.raw_data["user_utt_id"].shape
        for _, dialog_utt in enumerate(self.raw_data["user_utt_id"]):
            for _, utt_id in enumerate(dialog_utt):
                if utt_id == -1:
                    break
                utt_len = self.raw_data["user_sent_len"][utt_id]
                utterance = self.raw_data["user_sent"][utt_id, :utt_len]
                IDF[np.unique(utterance)] += 1
        for _, dialog_utt in enumerate(self.raw_data["assist_utt_id"]):
            for _, utt_id in enumerate(dialog_utt):
                if utt_id == -1:
                    break
                utt_len = self.raw_data["assist_sent_len"][utt_id]
                utterance = self.raw_data["assist_sent"][utt_id, :utt_len]
                IDF[np.unique(utterance)] += 1
        num_utterances = (self.raw_data["user_utt_id"] != -1).sum() + (
            self.raw_data["user_utt_id"] != -1
        ).sum()
        self.IDF = np.log(num_utterances / IDF)

    def compute_tf_features(self, utterances, utterance_lens):
        """Compute TF features for either train/val/test set.

        Args:
            Utterances: arguments to compute TF features
            utterance_lens: Length of the utterances

        Returns:
            tf_idf_features: tf_idf features
        """
        assert hasattr(self, "IDF"), "IDF has not been computed/loaded!"
        batch_size, num_rounds, max_len = utterances.shape
        num_utterances = batch_size * num_rounds
        utterances = utterances.reshape(-1, max_len)
        utterance_lens = utterance_lens.reshape(-1)
        tf_features = np.zeros((num_utterances, self.vocab_size))
        for utt_id, utterance in enumerate(utterances):
            tokens = utterance[: utterance_lens[utt_id]]
            for tt in tokens:
                tf_features[utt_id, tt] += 1.0 / utterance_lens[utt_id]
        return tf_features.reshape(batch_size, num_rounds, -1)

    def get_data_related_arguments(self):
        """Get data related arguments like vocab_size, etc.

        Complete list: Vocab size, pad_token, start_token, end_token,
            num_actions, asset_feature_size (if exists).

        Returns:
            related_args: Dictionary containing the above arguments
        """
        related_args = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "num_actions": self.num_actions,
        }
        if self.params["encoder"] == "pretrained_transformer":
            related_args["vocab_path"] = self.raw_data["vocabulary"]
            related_args["vocab_size"] += len(self.words.added_tokens_encoder)
        if hasattr(self, "asset_feature_size"):
            related_args["asset_feature_size"] = self.asset_feature_size
        return related_args

    @staticmethod
    def numpy(batch_torch):
        """Convert a batch into numpy arrays.

        Args:
          batch_torch: A batch with torch tensors

        Returns:
          batch_numpy: batch_torch with all tensors moved to numpy
        """
        batch_numpy = {}
        for key, value in batch_torch.items():
            # Check if numpy array or list of numpy arrays.
            if isinstance(value, torch.Tensor):
                batch_numpy[key] = value.cpu().numpy()
            elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
                batch_numpy[key] = [None] * len(batch_torch[key])
                for index, element in enumerate(value):
                    batch_numpy[key][index] = element.cpu().numpy()
            else:
                batch_numpy[key] = value
        return batch_numpy

    @property
    def num_instances(self):
        """Number of instances in the dataloader.
        """
        raise NotImplementedError
