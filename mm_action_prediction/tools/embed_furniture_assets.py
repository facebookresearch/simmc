"""Create furniture assest embeddings by concatenating attribute Glove embeddings.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import ast

import numpy as np
from tools import data_support
import spacy


# Attributes to encode.
EMBED_ATTRIBUTES = [
    "class_name", "color", "decor_style", "intended_room", "material"
]


def main(args):
    assets = data_support.read_furniture_metadata(args["input_csv_file"])
    cleaned_assets = []
    # Quick fix dictionary.
    correction = {
        "['Traditional', 'Modern'']": "['Traditional', 'Modern']",
        "[Brown']": "['Brown']",
    }
    for _, asset in assets.items():
        clean_asset = {}
        for key in EMBED_ATTRIBUTES:
            val = asset[key]
            val = correction.get(val, val).lower()
            val = ast.literal_eval(val) if "[" in val else val
            clean_asset[key] = val if isinstance(val, list) else [val]
        clean_asset["id"] = int(asset["obj"].split("/")[-1].strip(".zip"))
        cleaned_assets.append(clean_asset)

    # Vocabulary for each field.
    vocabulary = {key: {} for key in EMBED_ATTRIBUTES}
    for asset in cleaned_assets:
        for attr in EMBED_ATTRIBUTES:
            attr_val = asset[attr]
            for val in attr_val:
                vocabulary[attr][val] = vocabulary[attr].get(val, 0) + 1

    # Embedding for each item.
    nlp = spacy.load(args["spacy_model"])
    embeddings = []
    id_list = []
    for asset in cleaned_assets:
        embed_vector = []
        for attr in EMBED_ATTRIBUTES:
            attr_val = asset[attr]
            feature_vector = np.stack([nlp(val).vector for val in attr_val])
            embed_vector.append(feature_vector.mean(0))
        embeddings.append(np.concatenate(embed_vector))
        id_list.append(asset["id"])
    embeddings = np.stack(embeddings)
    print("Saving embeddings: {}".format(args["embed_path"]))
    feature_size = embeddings.shape[1]
    np.save(
        args["embed_path"],
        {
            "asset_id": id_list,
            "embedding": embeddings,
            "asset_feature_size": feature_size,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed furniture assets")
    parser.add_argument(
        "--input_csv_file",
        default="data/furniture_metadata.csv",
        help="Furniture metadata file for assets",
    )
    parser.add_argument(
        "--embed_path",
        default="data/furniture_metadata_embed.npy",
        help="Embeddings for furniture assets",
    )
    parser.add_argument(
        "--spacy_model",
        default="en_vectors_web_lg",
        help="Spacy model to use for language model",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
