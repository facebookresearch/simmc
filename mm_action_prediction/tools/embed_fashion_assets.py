"""Create fashion assest embeddings by concatenating attribute Glove embeddings.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import ast
import json

import numpy as np
import spacy


# Attributes to encode.
EMBED_ATTRIBUTES = ["type", "color", "embellishments", "pattern"]


def main(args):
    with open(args["input_asset_file"], "r") as file_id:
        assets = json.load(file_id)

    # Select and embed only the top attributes.
    cleaned_assets = []
    for image_id, asset in assets.items():
        clean_asset = {}
        asset_info = asset["metadata"]
        for key in EMBED_ATTRIBUTES:
            if key in asset_info:
                val = asset_info[key]
                # val = correction.get(val, val).lower()
                val = ast.literal_eval(val) if "[" in val else val
                clean_asset[key] = val if isinstance(val, list) else [val]
        clean_asset["id"] = int(image_id)
        cleaned_assets.append(clean_asset)
    # Vocabulary for each field.
    vocabulary = {key: {} for key in EMBED_ATTRIBUTES}
    for asset in cleaned_assets:
        for attr in EMBED_ATTRIBUTES:
            attr_val = asset.get(attr, [])
            for val in attr_val:
                vocabulary[attr][val] = vocabulary[attr].get(val, 0) + 1

    # Embedding for each item.
    nlp = spacy.load(args["spacy_model"])
    sample_feature = nlp("apple").vector
    feature_size = sample_feature.size
    zero_features = np.zeros(feature_size)
    embeddings = []
    id_list = []
    for asset in cleaned_assets:
        embed_vector = []
        for attr in EMBED_ATTRIBUTES:
            if attr in asset and len(asset[attr]) > 0:
                attr_val = asset[attr]
                feature_vector = np.stack(
                    [nlp(val).vector for val in attr_val]
                ).mean(0)
            else:
                feature_vector = zero_features
            embed_vector.append(feature_vector)
        embeddings.append(np.concatenate(embed_vector))
        id_list.append(asset["id"])
    embeddings = np.stack(embeddings)
    print("Saving embeddings: {}".format(args["embed_path"]))
    np.save(
        args["embed_path"],
        {
            "asset_id": id_list,
            "embedding": embeddings,
            "asset_feature_size": embeddings.shape[1],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed Fashion assets")
    parser.add_argument(
        "--input_asset_file",
        default="data/fashion_assets.json",
        help="Fashion metadata file for assets",
    )
    parser.add_argument(
        "--embed_path",
        default="data/fashion_metadata_embed.npy",
        help="Embeddings for Fashion assets",
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
