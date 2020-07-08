"""Ingests DSTC format data and creates multimodal input files for baselines.

Authors(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from absl import flags
from absl import app
import collections
import copy
import json
import os
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm as progressbar

from tools import support


FLAGS = flags.FLAGS
flags.DEFINE_spaceseplist(
    "json_path", "data/furniture_data.json", "JSON containing the dataset"
)
flags.DEFINE_string(
    "action_json_path",
    "data/dialog_action_supervision.json",
    "JSON containing API action supervision",
)
flags.DEFINE_string(
    "vocab_file", "data/furniture_vocabulary.json", "Vocabulary for furniture"
)
flags.DEFINE_string(
    "retrieval_candidate_file", None, "Path to retrieval candidates"
)
flags.DEFINE_string("save_path", "data/", "Folder to save multimodal input files")
flags.DEFINE_enum(
    "domain", "furniture", ["furniture", "fashion"], "Domain"
)
flags.DEFINE_boolean(
    "pretrained_tokenizer", False, "Use pretrained tokenizer instead of nltk"
)


def build_multimodal_inputs(input_json_file):
    """Convert splits into format injested by the dataloader.

    Args:
        input_json_file: Path to the JSON file to injest

    Returns:
        mm_inputs: Dictionary of multimodal inputs to train/evaluate
    """

    # Read the raw data.
    print("Reading: {}".format(input_json_file))
    with open(input_json_file, "r") as file_id:
        data = json.load(file_id)

    # Read action supervision.
    print("Reading action supervision: {}".format(FLAGS.action_json_path))
    with open(FLAGS.action_json_path, "r") as file_id:
        extracted_actions = json.load(file_id)
    # Convert into a dictionary.
    extracted_actions = {ii["dialog_id"]: ii for ii in extracted_actions}

    # Obtain maximum dialog length.
    dialog_lens = np.array(
        [len(ii["dialogue"]) for ii in data["dialogue_data"]], dtype="int32"
    )
    max_dialog_len = np.max(dialog_lens)
    num_dialogs = len(data["dialogue_data"])
    # Setup datastructures for recoding utterances, actions, action supervision,
    # carousel states, and outputs.
    encoded_dialogs = {
        "user": np.full((num_dialogs, max_dialog_len), -1, dtype="int32"),
        "assistant": np.full((num_dialogs, max_dialog_len), -1, dtype="int32")
    }
    empty_action_list = [
        [None for _ in range(max_dialog_len)] for _ in range(num_dialogs)
    ]
    action_info = {
        "action": np.full((num_dialogs, max_dialog_len), "None", dtype="object_"),
        "action_supervision": copy.deepcopy(empty_action_list),
        "carousel_state": copy.deepcopy(empty_action_list),
        "action_output_state": copy.deepcopy(empty_action_list)
    }
    dialog_ids = np.zeros(num_dialogs, dtype="int32")
    action_counts = collections.defaultdict(lambda : 0)

    # Compile dictionaries for user and assitant utterances separately.
    utterance_dict = {"user": {}, "assistant": {}}
    action_keys = ("action",)
    if FLAGS.domain == "furniture":
        action_keys += ("carousel_state", "action_output_state")
    elif FLAGS.domain == "fashion":
        task_mapping = {ii["task_id"]: ii for ii in data["task_mapping"]}
        dialog_image_ids = {
            "memory_images": [], "focus_images": [], "database_images": []
        }

    # If retrieval candidates file is available, encode the candidates.
    if FLAGS.retrieval_candidate_file:
        print("Reading retrieval candidates: {}".format(
            FLAGS.retrieval_candidate_file)
        )
        with open(FLAGS.retrieval_candidate_file, "r") as file_id:
            candidates_data = json.load(file_id)
        candidate_pool = candidates_data["system_transcript_pool"]
        candidate_ids = candidates_data["retrieval_candidates"]
        candidate_ids = {ii["dialogue_idx"]: ii for ii in candidate_ids}

        def get_candidate_ids(dialog_id, round_id):
            """Given the dialog_id and round_id, get the candidates.

            Args:
                candidate_ids: Dictionary of candidate ids
                dialog_id: Dialog id
                round_id: Round id

            Returns:
                candidates: List of candidates, indexed by the pool
            """
            candidates = candidate_ids[dialog_id]["retrieval_candidates"]
            candidates = candidates[round_id]["retrieval_candidates"]
            return candidates

        # Read the first dialog to get number of candidates.
        random_dialog_id = list(candidate_ids.keys())[0]
        num_candidates = len(get_candidate_ids(random_dialog_id, 0))
        encoded_candidates = np.full(
            (num_dialogs, max_dialog_len, num_candidates), -1, dtype=np.int32
        )

    for datum_id, datum in enumerate(data["dialogue_data"]):
        dialog_id = datum["dialogue_idx"]
        dialog_ids[datum_id] = dialog_id
        # Get action supervision.
        dialog_action_data = extracted_actions[dialog_id]["actions"]

        # Record images for fashion.
        if FLAGS.domain == "fashion":
            # Assign random task if not found (1-2 dialogs).
            if "dialogue_task_id" not in datum:
                print("Dialog task id not found, using 1874 (random)!")
            task_info = task_mapping[datum.get("dialogue_task_id", 1874)]
            for key in ("memory_images", "database_images"):
                dialog_image_ids[key].append(task_info[key])
            dialog_image_ids["focus_images"].append(
                extracted_actions[dialog_id]["focus_images"]
            )

        for round_id, round_datum in enumerate(datum["dialogue"]):
            for key, speaker in (
                ("transcript", "user"), ("system_transcript", "assistant")
            ):
                utterance_clean = round_datum[key].lower().strip(" ")
                speaker_pool = utterance_dict[speaker]
                if utterance_clean not in speaker_pool:
                    speaker_pool[utterance_clean] = len(speaker_pool)
                encoded_dialogs[speaker][datum_id, round_id] = (
                    speaker_pool[utterance_clean]
                )

            # Record action related keys.
            action_datum = dialog_action_data[round_id]
            cur_action_supervision = action_datum["action_supervision"]
            if FLAGS.domain == "furniture":
                if cur_action_supervision is not None:
                    # Retain only the args of supervision.
                    cur_action_supervision = cur_action_supervision["args"]

            action_info["action_supervision"][datum_id][round_id] = (
                cur_action_supervision
            )
            for key in action_keys:
                action_info[key][datum_id][round_id] = action_datum[key]
            action_counts[action_datum["action"]] += 1
    support.print_distribution(action_counts, "Action distribution:")

    # Record retrieval candidates, if path is provided.
    if FLAGS.retrieval_candidate_file:
        for datum_id, datum in enumerate(data["dialogue_data"]):
            dialog_id = datum["dialogue_idx"]
            for round_id, _ in enumerate(datum["dialogue"]):
                round_candidates = get_candidate_ids(dialog_id, round_id)
                encoded_round_candidates = []
                for cand_ind in round_candidates:
                    cand_str = candidate_pool[cand_ind].lower().strip(" ")
                    pool_ind = utterance_dict["assistant"][cand_str]
                    encoded_round_candidates.append(pool_ind)
                encoded_candidates[datum_id, round_id] = encoded_round_candidates

    # Sort utterance list for consistency.
    utterance_list = {
        key: sorted(value.keys(), key=lambda x: value[x])
        for key, value in utterance_dict.items()
    }

    # Convert the pools into matrices.
    mm_inputs = {}
    mm_inputs.update(action_info)

    # If token-wise encoding is to be used.
    print("Vocabulary: {}".format(FLAGS.vocab_file))
    if not FLAGS.pretrained_tokenizer:
        with open(FLAGS.vocab_file, "r") as file_id:
            vocabulary = json.load(file_id)
        mm_inputs["vocabulary"] = vocabulary
        word2ind = {word: index for index, word in enumerate(vocabulary["word"])}

        mm_inputs["user_sent"], mm_inputs["user_sent_len"] = convert_pool_matrices(
            utterance_list["user"], word2ind
        )
        mm_inputs["assist_sent"], mm_inputs["assist_sent_len"] = convert_pool_matrices(
            utterance_list["assistant"], word2ind
        )
        # Token aliases.
        pad_token = word2ind["<pad>"]
        start_token = word2ind["<start>"]
        end_token = word2ind["<end>"]
    else:
        # Use pretrained BERT tokenizer.
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(FLAGS.vocab_file)
        mm_inputs["vocabulary"] = FLAGS.vocab_file
        mm_inputs["user_sent"], mm_inputs["user_sent_len"] = (
            convert_pool_matrices_pretrained_tokenizer(
                utterance_list["user"], tokenizer
            )
        )
        mm_inputs["assist_sent"], mm_inputs["assist_sent_len"] = (
            convert_pool_matrices_pretrained_tokenizer(
                utterance_list["assistant"], tokenizer
            )
        )
        # Token aliases.
        pad_token = tokenizer.pad_token_id
        start_token = tokenizer.added_tokens_encoder["[start]"]
        end_token = tokenizer.added_tokens_encoder["[end]"]

    # Get the input and output version for RNN for assistant_sent.
    extra_slice = np.full((len(mm_inputs["assist_sent"]), 1), start_token, np.int32)
    mm_inputs["assist_in"] = np.concatenate(
        [extra_slice, mm_inputs["assist_sent"]], axis=1
    )
    extra_slice.fill(pad_token)
    mm_inputs["assist_out"] = np.concatenate(
        [mm_inputs["assist_sent"], extra_slice], axis=1
    )
    for ii in range(len(mm_inputs["assist_out"])):
        mm_inputs["assist_out"][ii, mm_inputs["assist_sent_len"][ii]] = end_token
    mm_inputs["assist_sent_len"] += 1

    # Save the memory and dataset image_ids for each instance.
    if FLAGS.domain == "fashion":
        mm_inputs.update(dialog_image_ids)

    # Save the retrieval candidates.
    if FLAGS.retrieval_candidate_file:
        mm_inputs["retrieval_candidates"] = encoded_candidates

    # Save the dialogs by user/assistant utterances.
    mm_inputs["user_utt_id"] = encoded_dialogs["user"]
    mm_inputs["assist_utt_id"] = encoded_dialogs["assistant"]
    mm_inputs["dialog_len"] = dialog_lens
    mm_inputs["dialog_id"] = dialog_ids
    mm_inputs["paths"] = {
        "data": FLAGS.json_path,
        "action": FLAGS.action_json_path,
        "retrieval": FLAGS.retrieval_candidate_file,
        "vocabulary": FLAGS.vocab_file
    }
    return mm_inputs


def convert_pool_matrices(pool_input, word2ind):
    """Converts a dictionary of pooled captions/questions into matrices.

    Args:
        pool_input: Dictionary of pooled captions/questions
        word2ind: Dictionary of word -> vocabulary index conversion.

    Returns:
        item_tokens: Items in the pool tokenized and converted into a matrix.
        item_lens: Length of items in the matrix.
    """
    unk_token = word2ind["<unk>"]

    def tokenizer(x):
        return [word2ind.get(ii, unk_token) for ii in word_tokenize(x.lower())]

    if isinstance(pool_input, dict):
        pool_list = sorted(pool_input, key=lambda x: pool_input[x])
    else:
        pool_list = pool_input

    tokenized_items = [tokenizer(item) for item in progressbar(pool_list)]
    max_item_len = max(len(ii) for ii in tokenized_items)
    item_tokens = np.zeros((len(tokenized_items), max_item_len)).astype("int32")
    item_tokens.fill(word2ind["<pad>"])
    item_lens = np.zeros(len(tokenized_items)).astype("int32")
    for item_id, tokens in progressbar(enumerate(tokenized_items)):
        item_lens[item_id] = len(tokens)
        item_tokens[item_id, : item_lens[item_id]] = np.array(tokens)
    return item_tokens, item_lens


def convert_pool_matrices_pretrained_tokenizer(pool_input, pretrained_tokenizer):
    """Converts a dictionary of pooled captions/questions into matrices.

    Args:
        pool_input: Dictionary of pooled captions/questions
        pretrained_tokenizer: Huggingface tokenizer for pretrained models.

    Returns:
        item_tokens: Items in the pool tokenized and converted into a matrix.
        item_lens: Length of items in the matrix.
    """
    def tokenizer(x):
        return pretrained_tokenizer.encode(x, add_special_tokens=True)

    if isinstance(pool_input, dict):
        pool_list = sorted(pool_input, key=lambda x: pool_input[x])
    else:
        pool_list = pool_input

    tokenized_items = [tokenizer(item) for item in progressbar(pool_list)]
    max_item_len = max(len(ii) for ii in tokenized_items)
    item_tokens = np.zeros((len(tokenized_items), max_item_len)).astype("int32")
    item_tokens.fill(pretrained_tokenizer.pad_token_id)
    item_lens = np.zeros(len(tokenized_items)).astype("int32")
    for item_id, tokens in progressbar(enumerate(tokenized_items)):
        item_lens[item_id] = len(tokens)
        item_tokens[item_id, : item_lens[item_id]] = np.array(tokens)
    return item_tokens, item_lens


def get_save_path(save_folder, input_json_file):
    """Get the save path given save folder and input_json_file.

    Args:
    save_folder: Folder to save the processed data matrices
    input_json_file: JSON input file name

    Returns:
    save_file_name: File name to save the matrices under
    """
    save_name = input_json_file.rsplit("/")[-1].replace(".json", "_mm_inputs.npy")
    if FLAGS.pretrained_tokenizer:
        save_name = save_name.replace(".npy", "_pretrained.npy")
    save_name = os.path.join(save_folder, save_name)
    return save_name


def main(_):
    for input_json_file in FLAGS.json_path:
        save_file_name = get_save_path(FLAGS.save_path, input_json_file)
        mm_inputs_split = build_multimodal_inputs(input_json_file)
        print("Saving multimodal inputs: {}".format(save_file_name))
        np.save(save_file_name, np.array(mm_inputs_split))


if __name__ == "__main__":
    app.run(main)
