"""Extract action API supervision for the SIMMC Fashion dataset.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from absl import flags
from absl import app
import ast
import json
import os


FLAGS = flags.FLAGS
flags.DEFINE_spaceseplist(
    "json_path", "data/furniture_pilot_oct24.json", "JSON containing the dataset"
)
flags.DEFINE_string(
    "save_root", "data/", "Folder path to save extraced api annotations"
)
flags.DEFINE_string(
    "metadata_path", "data/fashion_metadata.json", "Path to fashion metadata"
)


def extract_actions(input_json_file):
    """Extract action API for SIMMC fashion.

    Args:
        input_json_file: JSON data file to extraction actions
    """
    print("Reading: {}".format(input_json_file))
    with open(input_json_file, "r") as file_id:
        raw_data = json.load(file_id)

    task_mapping = {ii["task_id"]: ii for ii in raw_data["task_mapping"]}
    dialogs = []
    for dialog_datum in raw_data["dialogue_data"]:
        dialog_id = dialog_datum["dialogue_idx"]
        # If task id is missing for the dialog, assign a random task.
        # Could lead to problems but it is for < 0.1% of the data
        if "dialogue_task_id" not in dialog_datum:
            # Assign a random task for missing ids.
            print("Dialogue task Id missing: {}".format(dialog_id))
            mm_state = task_mapping[1874]
        else:
            mm_state = task_mapping[dialog_datum["dialogue_task_id"]]
        focus_image = mm_state["focus_image"]
        focus_images = []
        roundwise_actions = []

        for round_datum in dialog_datum["dialogue"]:
            focus_images.append(focus_image)
            # Default None action.
            insert_item = {
                "turn_idx": round_datum["turn_idx"],
                "action": "None",
                "action_supervision": None
            }
            keystrokes = round_datum.get("raw_assistant_keystrokes", [])
            # Get information attributes given the asset id.
            attributes = extract_info_attributes(round_datum)
            if keystrokes:
                focus_image = int(keystrokes[0]["image_id"])
                # Change of focus image -> Search in dataset or memory.
                if focus_image in mm_state["memory_images"]:
                    insert_item["action"] = "SearchMemory"
                    insert_item["action_supervision"] = {
                        "focus": focus_image,
                        "attributes": attributes,
                    }
                elif focus_image in mm_state["database_images"]:
                    insert_item["action"] = "SearchDatabase"
                    insert_item["action_supervision"] = {
                        "focus": focus_image,
                        "attributes": attributes,
                    }
                else:
                    print("Undefined action; using None instead")
                roundwise_actions.append(insert_item)
            else:
                # Check for SpecifyInfo action.
                # Get information attributes given the asset id.
                attributes = extract_info_attributes(round_datum)
                if len(attributes):
                    insert_item["action"] = "SpecifyInfo"
                    insert_item["action_supervision"] = {
                        "attributes": attributes
                    }
                else:
                    # AddToCart action.
                    for intent_info in ast.literal_eval(
                        round_datum["transcript_annotated"]
                    ):
                        if "DA:REQUEST:ADD_TO_CART" in intent_info["intent"]:
                            insert_item["action"] = "AddToCart"
                            insert_item["action_supervision"] = None
                roundwise_actions.append(insert_item)

        dialogs.append(
            {
                "dialog_id": dialog_id,
                "actions": roundwise_actions,
                "focus_images": focus_images,
            }
        )

    # Save extracted API calls.
    save_path = input_json_file.split("/")[-1].replace(".json", "_api_calls.json")
    save_path = os.path.join(FLAGS.save_root, save_path)
    print("Saving: {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(dialogs, f)


def extract_info_attributes(round_datum):
    """Extract information attributes for current round using NLU annotations.

    Args:
        round_datum: Current round information

    Returns:
        get_attribute_matches: Information attributes
    """
    user_annotation = ast.literal_eval(round_datum["transcript_annotated"])
    # assistant_annotation = ast.literal_eval(
    #     round_datum["system_transcript_annotated"]
    # )
    # annotation = user_annotation + assistant_annotation
    annotation = user_annotation
    all_intents = [ii["intent"] for ii in annotation]
    get_attribute_matches = []
    for index, intent in enumerate(all_intents):
        if any(
            ii in intent
            for ii in ("DA:ASK:GET", "DA:ASK:CHECK", "DA:INFORM:GET")
        ):
            # If there is no attribute added, default to info.
            if "." not in intent:
                get_attribute_matches.append("info")
                continue

            attribute = intent.split(".")[-1]
            if attribute == "info":
                new_matches = [
                    ii["id"].split(".")[-1]
                    for ii in annotation[index]["slots"]
                    if "INFO" in ii["id"]
                ]
                if len(new_matches):
                    get_attribute_matches.extend(new_matches)
                else:
                    get_attribute_matches.append("info")
            elif attribute != "":
                get_attribute_matches.append(attribute)
    return sorted(set(get_attribute_matches))


def main(_):
    for input_json_file in FLAGS.json_path:
        extract_actions(input_json_file)


if __name__ == "__main__":
    app.run(main)
