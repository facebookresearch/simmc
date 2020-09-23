"""Script evaluates action prediction along with attributes.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import collections
import json

import numpy as np



IGNORE_ATTRIBUTES = [
    "minPrice",
    "maxPrice",
    "furniture_id",
    "material",
    "decorStyle",
    "intendedRoom",
    "raw_matches",
    "focus"  # fashion
]


def evaluate_action_prediction(gt_actions, model_actions, single_round_eval=False):
    """Evaluates action prediction using the raw data and model predictions.

    Args:
        gt_actions: Ground truth actions + action attributes
        model_actions: Actions + attributes predicted by the model
        single_round_eval: Evaluate only for the last turn.
    """
    gt_actions_pool = {ii["dialog_id"]: ii for ii in gt_actions}
    matches = {"action": [], "attributes": [], "perplexity": []}
    confusion_dict = collections.defaultdict(list)
    for model_datum in model_actions:
        dialog_id = model_datum["dialog_id"]
        num_gt_rounds = len(gt_actions_pool[dialog_id]["actions"])
        for round_datum in model_datum["predictions"]:
            round_id = round_datum["turn_id"]
            # Skip if single_round_eval and this is not the last round.
            if single_round_eval and round_id != num_gt_rounds - 1:
                continue

            gt_datum = gt_actions_pool[dialog_id]["actions"][round_id]
            action_match = gt_datum["action"] == round_datum["action"]
            # Record matches and confusion.
            matches["action"].append(action_match)
            matches["perplexity"].append(
                round_datum["action_log_prob"][gt_datum["action"]]
            )
            confusion_dict[gt_datum["action"]].append(round_datum["action"])

            # Get supervision for action attributes.
            supervision = gt_datum["action_supervision"]
            if supervision is not None and "args" in supervision:
                supervision = supervision["args"]
            if supervision is None:
                continue
            # Case 1: Action mismatch -- record False for all attributes.
            if not action_match:
                for key in supervision.keys():
                    if key in IGNORE_ATTRIBUTES:
                        continue
                    matches["attributes"].append(False)
            # Case 2: Action matches -- use model predictions for attributes.
            else:
                for key in supervision.keys():
                    if key in IGNORE_ATTRIBUTES:
                        continue
                    gt_key_vals = supervision[key]
                    model_key_vals = round_datum["attributes"][key]
                    if not len(gt_key_vals):
                        continue
                    # For fashion, this is a list -- multi label prediction.
                    if isinstance(gt_key_vals, list):
                        assert isinstance(model_key_vals, list), (
                            "Model should also predict a list for attributes"
                        )
                        recall = np.mean(
                            [(ii in model_key_vals) for ii in gt_key_vals]
                        )
                        if len(model_key_vals):
                            precision = np.mean(
                                [(ii in gt_key_vals) for ii in model_key_vals]
                            )
                        else:
                            precision = 0.
                        f1_score = (2 * recall * precision) / (recall + precision + 1e-5)
                        matches["attributes"].append(f1_score)
                    else:
                        # For furniture, this is a string -- single label prediction.
                        matches["attributes"].append(gt_key_vals == model_key_vals)

    print("#Instances evaluated API: {}".format(len(matches["action"])))

    # Compute the confusion matrix.
    all_actions = sorted(
        set(confusion_dict.keys()).union(
            {jj for ii in confusion_dict.values() for jj in ii}
        )
    )
    matrix = np.zeros((len(all_actions), len(all_actions)))
    for index, action in enumerate(all_actions):
        labels, counts = np.unique(confusion_dict[action], return_counts=True)
        for label, count in zip(labels, counts):
            matrix[all_actions.index(label), index] += count

    return {
        "action_accuracy": np.mean(matches["action"]),
        "action_perplexity": np.exp(-1 * np.mean(matches["perplexity"])),
        "attribute_accuracy": np.mean(matches["attributes"]),
        "confusion_matrix": matrix
    }


def main(args):
    print("Reading: {}".format(args["action_json_path"]))
    with open(args["action_json_path"], "r") as file_id:
        gt_actions = json.load(file_id)
    print("Reading: {}".format(args["model_output_path"]))
    with open(args["model_output_path"], "r") as file_id:
        model_actions = json.load(file_id)
    action_metrics = evaluate_action_prediction(
        gt_actions, model_actions, args["single_round_evaluation"]
    )
    print(action_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Call Action Evaluation")
    parser.add_argument(
        "--action_json_path",
        default="data/furniture_api_calls.json",
        help="Ground truth API calls"
    )
    parser.add_argument(
        "--model_output_path",
        default=None,
        help="API calls generated by the model"
    )
    parser.add_argument(
        "--single_round_evaluation",
        dest="single_round_evaluation",
        action="store_true",
        default=False,
        help="Single round evaluation for hidden split",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
