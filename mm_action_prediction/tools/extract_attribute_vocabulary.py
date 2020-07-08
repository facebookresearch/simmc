"""Extracts attribute vocabulary for SIMMC Furniture.

Author: Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import json
import argparse
import numpy as np


# Furniture.
EXCLUDE_KEYS_FURNITURE = [
    "minPrice",
    "maxPrice",
    "furniture_id",
    "material",
    "decorStyle",
    "intendedRoom",
    "raw_matches",
]
# Furniture
EXCLUDE_KEYS_FASHION = ["focus", "memory"]
INCLUDE_ATTRIBUTES_FASHION = [
    "availableSizes", "price", "brand", "customerRating", "info", "color"
]


# Key aliases.
DOMAIN = "domain"
FURNITURE = "furniture"
FASHION = "fashion"


def extract_action_attributes(args):
    """Read training multimodal input, extract attribute vocabulary (furniture)
    """
    # Read the data, parse the datapoints.
    data = np.load(args["train_npy_path"], allow_pickle=True)[()]
    actions = data["action"]
    num_instances, num_rounds = actions.shape

    # Get action attributes.
    attr_vocab = {}
    for ii in range(num_instances):
        for jj in range(num_rounds):
            cur_action = actions[ii, jj]
            if cur_action == "None":
                continue
            if cur_action not in attr_vocab:
                if args[DOMAIN] == FURNITURE:
                    attr_vocab[cur_action] = collections.defaultdict(dict)
                elif args[DOMAIN] == FASHION:
                    attr_vocab[cur_action] = collections.defaultdict(
                        lambda: collections.defaultdict(lambda: 0)
                    )

            cur_super = data["action_supervision"][ii][jj]
            if cur_super is None:
                continue
            for key, val in cur_super.items():
                if args[DOMAIN] == FURNITURE:
                    if key in EXCLUDE_KEYS_FURNITURE:
                        continue
                    if isinstance(val, list):
                        val = tuple(val)
                    new_count = attr_vocab[cur_action][key].get(val, 0) + 1
                    attr_vocab[cur_action][key][val] = new_count

                elif args[DOMAIN] == FASHION:
                    if key in EXCLUDE_KEYS_FASHION:
                        continue
                    if isinstance(val, list):
                        val = tuple(val)
                        for vv in val:
                            # If vv not in INCLUDE_ATTRIBUTES_FASHION,
                            # assign it to "other."
                            if vv not in INCLUDE_ATTRIBUTES_FASHION:
                                vv = "other"
                            attr_vocab[cur_action][key][vv] += 1
                    else:
                        # If val not in INCLUDE_ATTRIBUTES_FASHION,
                        # assign it to other.
                        if val not in INCLUDE_ATTRIBUTES_FASHION:
                            val = "other"
                        attr_vocab[cur_action][key][val] += 1

    attr_vocab = {
        key: sorted(val)
        for attr_values in attr_vocab.values()
        for key, val in attr_values.items()
    }
    print(attr_vocab)
    print("Saving attribute dictionary: {}".format(args["vocab_save_path"]))
    with open(args["vocab_save_path"], "w") as file_id:
        json.dump(attr_vocab, file_id)


def print_fashion_attributes(attribute_vocabulary):
    """Prints fashion attributes (for visualization).

    Args:
        attribute_vocabulary: Extracted attribute vocabulary count dict.
    """
    for key, val in attribute_vocabulary.items():
        print(key)
        print(val.keys())
        for attr, attr_val_dict in val.items():
            print('Name: {}'.format(attr))
            for ii in sorted(
                attr_val_dict.items(), key=lambda x: x[1], reverse=True
            ):
                print("\t{}: {}".format(*ii))
        print("-" * 50)


if __name__ == "__main__":
    # Read the commandline arguments.
    parser = argparse.ArgumentParser(description="Extract vocabulary")
    parser.add_argument(
        "--train_json_path",
        default=None,
        help="Path to read the vocabulary (train) JSON",
    )
    parser.add_argument(
        "--train_npy_path",
        default=None,
        help="Path to read the vocabulary (train) Numpy file",
    )
    parser.add_argument(
        "--vocab_save_path",
        default="data/vocabulary_genie.json",
        help="Path to read the vocabulary (train) JSON",
    )
    parser.add_argument(
        "--domain",
        default="furniture",
        choices=["furniture", "fashion"],
        help="Domain to extract attribute vocabulary"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    # Extract action API attributes using training file.
    extract_action_attributes(parsed_args)
