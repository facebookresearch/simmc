"""Extracts vocabulary for SIMMC dataset.

Author: Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import argparse
from nltk.tokenize import word_tokenize


def main(args):
    # Read the data, parse the datapoints.
    print("Reading: {}".format(args["train_json_path"]))
    with open(args["train_json_path"], "r") as file_id:
        train_data = json.load(file_id)
    dialog_data = train_data["dialogue_data"]

    counts = {}
    for datum in dialog_data:
        dialog_utterances = [
            ii[key] for ii in datum["dialogue"]
            for key in ("transcript", "system_transcript")
        ]
        dialog_tokens = [
            word_tokenize(ii.lower()) for ii in dialog_utterances
        ]
        for turn in dialog_tokens:
            for word in turn:
                counts[word] = counts.get(word, 0) + 1

    # Add <pad>, <unk>, <start>, <end>.
    counts["<pad>"] = args["threshold_count"] + 1
    counts["<unk>"] = args["threshold_count"] + 1
    counts["<start>"] = args["threshold_count"] + 1
    counts["<end>"] = args["threshold_count"] + 1

    word_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    words = [ii[0] for ii in word_counts if ii[1] >= args["threshold_count"]]
    vocabulary = {"word": words}
    # Save answers and vocabularies.
    print("Identified {} words..".format(len(words)))
    print("Saving dictionary: {}".format(args["vocab_save_path"]))
    with open(args["vocab_save_path"], "w") as file_id:
        json.dump(vocabulary, file_id)


if __name__ == "__main__":
    # Read the commandline arguments.
    parser = argparse.ArgumentParser(description="Extract vocabulary")
    parser.add_argument(
        "--train_json_path",
        default="data/furniture_data.json",
        help="Path to read the vocabulary (train) JSON",
    )
    parser.add_argument(
        "--vocab_save_path",
        default="data/furniture_vocabulary.json",
        help="Path to read the vocabulary (train) JSON",
    )
    parser.add_argument(
        "--threshold_count",
        default=0,
        type=int,
        help="Words are included if beyond this threshold",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
