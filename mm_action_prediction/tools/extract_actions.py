"""Ingests DSTC format data and extracts API actions for SIMMC furniture.

Author(s): Ankita De, Paul Crook, Satwik Kottur
"""

#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import copy
import json
import math
import os
import re

import data_support

from absl import app, flags
from nltk.tokenize import word_tokenize


FLAGS = flags.FLAGS
flags.DEFINE_spaceseplist(
    "json_path", "data/furniture_raw_data.json", "JSON containing the dataset"
)
flags.DEFINE_string(
    "save_root", "data/", "Folder path to save extracted api annotations"
)
flags.DEFINE_string(
    "metadata_path",
    "data/furniture_metadata.csv",
    "Path to the furniture metadata CSV",
)
flags.DEFINE_enum(
    "subtask",
    "dominant-action",
    ["dominant-action", "multi-action"],
    "Selects output format; dominant-action (single action) and multi-action",
)


# sub-tasks
DOMINANT_ACTION = "dominant-action"
MULTI_ACTION = "multi-action"

### log field names ###
DOMAIN = "furniture"
ACTION_NAME = "actionName"
RAW_MATCHES = "raw_matches"
MATCHES = "matches"
ACTION_METADATA = "actionMetadata"
FURNITURE_TYPE = "furnitureType"
MIN_PRICE = "minPrice"
MAX_PRICE = "maxPrice"
NEXT_STATE = "nextState"
PREVIOUS_STATE = "previousState"
TEXT_PREFABS_CAROUSEL = "textPrefabsInCarousel"
PREFAB_IN_FOCUS = "prefabInFocus"
PREFABS_IN_CAROUSEL = "prefabsInCarousel"
SHARED_FOCUS = "sharedPrefabInFocus"
SHARED_CAROUSEL = "sharedPrefabsInCarousel"

# Keystroke names
SEARCH_FURNITURE = "SearchFurniture"
BRING_OBJECT_TO_FOCUS = "BringObjectToFocus"
REMOVE_OBJECT_FROM_FOCUS = "RemoveObjectInFocus"
ROTATE = "Rotate"  # RotateRight, RotateUp, RotateDown, RotateBack, etc.
FURNITURE_CLICK = "FurnitureClick"
PREVIOUS = "Previous"
NEXT = "Next"
SHARE = "Share"

# API Action names
SEARCH_FURNITURE_ACTION = "SearchFurniture"
NAVIGATE_CAROUSEL_ACTION = "NavigateCarousel"
FOCUS_ON_FURNITURE_ACTION = "FocusOnFurniture"
ROTATE_ACTION = "Rotate"
ADD_TO_CART_ACTION = "AddToCart"
GET_INFO_ACTION = "SpecifyInfo"
NONE_ACTION = "None"

# Keep only these matching attributes for GET_INFO
FILTER_MATCHES = ["price", "dimensions", "info", "material", "color"]

# Action preference order for dominant action sub-task
PREFERENCE_ORDER = [
    ADD_TO_CART_ACTION,
    GET_INFO_ACTION,
    ROTATE_ACTION,
    SEARCH_FURNITURE_ACTION,
    FOCUS_ON_FURNITURE_ACTION,
    NAVIGATE_CAROUSEL_ACTION,
    NONE_ACTION,
]

# api / arg field names
API = "api"
ARGS = "args"
DIRECTION = "direction"
FURNITURE_ID = "furniture_id"
NAVIGATE_DIRECTION = "navigate_direction"
POSITION = "position"


def get_args_for_furniture_click(stroke):
    entry = json.loads(stroke)
    text_next_state = entry[NEXT_STATE][TEXT_PREFABS_CAROUSEL]
    text_prev_state = entry[PREVIOUS_STATE][TEXT_PREFABS_CAROUSEL]
    arg = [text for text in text_next_state if text not in text_prev_state]
    return arg


def get_keystrokes_with_args(raw_keystroke_list, price_dict):
    """Gets the keystrokes + args from the raw keystrokes in the logs after some processing

    Args:
        raw_keystroke_list: list extracted from the logs
        price_dict : dict from furniture type to the default min & max prices

    Returns:
        list of keystrokes where each keystroke is an api name + corresponding args
    """
    keystrokes_with_args = []
    for stroke in raw_keystroke_list:
        keystroke = json.loads(stroke)[ACTION_NAME]

        if keystroke == SEARCH_FURNITURE:
            furniture_type_arg = json.loads(stroke)[ACTION_METADATA][FURNITURE_TYPE]

            if furniture_type_arg != "":
                action_metadata = json.loads(stroke)[ACTION_METADATA]

                # check if the prices are close to the default populated prices.
                # if yes, replace by -1
                min_price_arg = action_metadata[MIN_PRICE]
                max_price_arg = action_metadata[MAX_PRICE]
                if math.isclose(
                    min_price_arg, price_dict[furniture_type_arg][0], abs_tol=0.9
                ):
                    action_metadata[MIN_PRICE] = -1
                if math.isclose(
                    max_price_arg, price_dict[furniture_type_arg][1], abs_tol=0.9
                ):
                    action_metadata[MAX_PRICE] = -1
                keystrokes_with_args.append(
                    {
                        API: keystroke,
                        NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                        PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                        ARGS: action_metadata,
                    }
                )

        elif keystroke == BRING_OBJECT_TO_FOCUS:
            keystrokes_with_args.append(
                {
                    API: keystroke,
                    NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                    PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                    ARGS: json.loads(stroke)[NEXT_STATE][PREFAB_IN_FOCUS],
                }
            )
        elif keystroke.startswith(ROTATE):
            if (
                json.loads(stroke)[NEXT_STATE][PREFAB_IN_FOCUS] is not None
                and json.loads(stroke)[NEXT_STATE][PREFAB_IN_FOCUS] != ""
            ):
                keystrokes_with_args.append(
                    {
                        API: keystroke,
                        NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                        PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                        ARGS: json.loads(stroke)[NEXT_STATE][PREFAB_IN_FOCUS],
                    }
                )
            else:
                keystrokes_with_args.append(
                    {
                        API: keystroke,
                        NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                        PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                        ARGS: json.loads(stroke)[NEXT_STATE]["sharedPrefabInFocus"],
                    }
                )
        elif keystroke == FURNITURE_CLICK:
            keystrokes_with_args.append(
                {
                    API: keystroke,
                    NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                    PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                    ARGS: get_args_for_furniture_click(stroke),
                }
            )
        elif keystroke == SHARE:
            keystrokes_with_args.append(
                {
                    API: keystroke,
                    NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                    PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                    ARGS: None,
                }
            )
            # fix some PREVIOUS_STATE log strangeness
            keystrokes_with_args[-1][PREVIOUS_STATE][PREFAB_IN_FOCUS] = \
                keystrokes_with_args[-1][NEXT_STATE][SHARED_FOCUS]
            keystrokes_with_args[-1][PREVIOUS_STATE][PREFABS_IN_CAROUSEL] = \
                keystrokes_with_args[-1][NEXT_STATE][SHARED_CAROUSEL]
        else:
            keystrokes_with_args.append(
                {
                    API: keystroke,
                    NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                    PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                    ARGS: None,
                }
            )
    return keystrokes_with_args


def get_turn_keystrokes(keystroke_sequence_with_args):
    """ Gets the keystrokes that were shared this turn

    Args:
        keystroke_sequence_with_args : list of unshared keystrokes from this and previous turns

    Returns:
        list of keystrokes shared this turn, remaining unshared keystrokes
    """
    share_index = -1
    this_turn_keystrokes = []
    for index, keystroke in reversed(list(enumerate(keystroke_sequence_with_args))):
        keystroke_name = keystroke[API]
        # check for the last share in this turn.
        # Shares are not included in the filtered keystrokes
        if keystroke_name == SHARE and index > 0:
            share_index = index
            i = index - 1

            # get the keystrokes just before the last share
            # back to the most recent search furniture
            while i >= 0:
                if keystroke_sequence_with_args[i][API] == SHARE:
                    i = i - 1
                    continue
                this_turn_keystrokes.insert(0, keystroke_sequence_with_args[i])

                # fix a logging bug with NEXT_STATE of REMOVE_OBJECT_FROM_FOCUS
                if this_turn_keystrokes[0][API] == REMOVE_OBJECT_FROM_FOCUS:
                    # update NEXT_STATE with the PREVIOUS_STATE of the
                    # following action (there should always exist an i+1 action
                    # as the sequence ends with a SHARE)
                    this_turn_keystrokes[0][NEXT_STATE] = \
                        keystroke_sequence_with_args[i + 1][PREVIOUS_STATE]

                # anything that happened prior to an Search which wasn't shared
                # can be discarded
                if this_turn_keystrokes[0][API] == SEARCH_FURNITURE:
                    break
                i = i - 1

            break

    # if there are keystrokes after the last share,
    # retain them to be used in the next turn
    if share_index < len(keystroke_sequence_with_args) - 1:
        keystroke_sequence_with_args = keystroke_sequence_with_args[(share_index + 1) :]
    else:
        keystroke_sequence_with_args = []

    return this_turn_keystrokes, keystroke_sequence_with_args


def is_prefab_in_focus(state):
    """ Check if a prefab is proposed to be in-focus, or there is a prefab
    currently in focus which will remain in focus.

    Args:
        state: keystroke/action api state

    Returns:
        True if a prefab is proposed to be in focus or will remain in focus.

    Applies the following logic:

    1.  "prefabInFocus" = []
        "prefabsInCarousel" = []
        "sharedPrefabInFocus" = [something]
            ==> share will do nothing, i.e. 'something' stays in focus

    2.  "prefabInFocus" = [something-else]
        "prefabsInCarousel" = []
        "sharedPrefabInFocus" = [something]
            ==> share will share something-else as being in focus

    3. "prefabInFocus" = []
        "prefabsInCarousel" = [other stuff]
        "sharedPrefabInFocus" = [something]
            ==> share will share the carousel containing something else

    4.  "prefabInFocus" = [something-else]
        "prefabsInCarousel" = [other stuff]
        "sharedPrefabInFocus" = [something]
            ==> share will share something-else as being in focus
    """
    return state[PREFAB_IN_FOCUS] or \
            (not state[PREFABS_IN_CAROUSEL] and state[SHARED_FOCUS])


def get_prefab_in_focus(state):
    """ Get the prefab that is in focus, if there is one.
    Returns None if nothing currently in focus. Applies logic
    from is_prefab_in_focus(.) to determine if something is
    in focus.

    Args:
        state: keystroke/action api state

    Returns:
        prefab in focus or None
    """
    if is_prefab_in_focus(state):
        if state[PREFAB_IN_FOCUS]:
            return state[PREFAB_IN_FOCUS]
        elif state[SHARED_FOCUS]:
            return state[SHARED_FOCUS]
    return None


def get_carousel_prefabs(state):
    """ Get the set of prefabs in the carousel.

    Args:
        state: keystroke/action api state

    Returns:
        list of prefabs in the carousel

    The presumption is that the sequence of actions in one turn always
    terminates with a share in the fictional UI. Thus to determine what the
    "state" of the carousel could look like it's important to first look at
    what pending updates to the carousel will become visible if this state
    were shared, i.e. the list of prefabs in the proposed carousel. If there
    are no pending changes, e.g. in the real interface the human wizard already
    shared updates to the carousel mid-turn, then return the set of existing
    shared carousel prefabs from this state.

    Note that this function doesn't check if the visual associated with the
    state would only consist of an in-focus item.
    """
    return state[PREFABS_IN_CAROUSEL] if state[PREFABS_IN_CAROUSEL] \
            else state[SHARED_CAROUSEL]


def matching_carousels(state1, state2):
    """ Compares prefabs in the carousel between two states.

    Args:
        state1: keystroke/action api state (either previous or next)
        state2: keystroke/action api state (either previous or next)

    Returns:
        do the set of prefabs in the carousel match in both states

    Compares prefabs in the carousel between two state.

    Function doesn't check if the visual associated with either state would
    only consist of an in-focus item. The comparsion result is thus only valid
    when an in-focus item is not expected in either state.

    In extracting the carousel for each state there is an presumption that
    in the fictional (target) UI the sequence of actions in one turn always
    terminates with a share. Thus to determine what the "state" of the
    carousel could look like it's important to first look at what pending
    updates to the carousel will become visible if this state were shared,
    i.e. the list of prefabs in the proposed carousel. If there are no pending
    changes, e.g. in the real interface the human wizard already shared
    updates to the carousel mid-turn, then use the set of already shared
    carousel prefabs from this state.
    """
    carousel1 = get_carousel_prefabs(state1)
    carousel2 = get_carousel_prefabs(state2)
    return carousel1 == carousel2


def get_relevant_actions(
        turn_keystrokes,
        search_results,
        last_search_args,
        furniture_db
):
    """ Gets the minimal set of actions needed to reach the scene visible to
    the user at the end of the turn.

    Args:
        turn_keystrokes : list keystrokes (api + args) from this turns
        search_results: current set of search results
        last_search_args: last set of SearchFurniture arguments
        furniture_db: object wrapping the furniture database

    Returns:
        minimal set of scene setting actions,
        furniture clicks potentially related to viewing text,
        search results,
        search arguments

    The control surface is imagined as a fictional machine-friendly interface
    (not limited by human UI design) where all the items in the search results
    are available to the Assistant, and which allows the following actions:
     - SearchFurniture (args: furniture attributes)
     - FocusOnFurniture (arg: furniture_id)
     - NavigateCarousel (arg: furniture_id)
     - Rotate (args: furniture_id, direction)
     - [GetInfo (arg: furniture_id)]
     - [AddToCart (arg: furniture_id)]
    Note the latter two (in square brackets) don't change the visual scene.

    Given this setup, the setting for the final scene is established by the
    last FocusOnFurniture, NavigateCarousel, or SearchFurniture action. After
    which only Rotate actions can occur if the the scene was established by a
    FocusOnFurniture action. A SearchFurniture will always be the first
    action if it is present.

    In this setup, navigation sequences of NEXT and/or PREVIOUS keystrokes, as
    well as the REMOVE_OBJECT_FROM_FOCUS keystroke, are transformed into a
    single NavigateCarousel action. This action taking as an argument of one
    of the furniture item ids which appear at that point in the carousel.

    GetInfo actions can occur at any point in the turn. As could AddToCart,
    though the latter probably makes most sense a the last action in any turn.
    """
    relevant_keystrokes = []
    relevant_actions = []
    viewed_text_keystrokes = []

    # no keystrokes this turn -- nothing to do
    if len(turn_keystrokes) == 0:
        return relevant_actions, viewed_text_keystrokes, search_results,\
            last_search_args

    # split:
    #  - furniture clicks; as potential 'viewed_text' indicators,
    #  - others keystrokes; as potentially leading to relevant actions
    for keystroke in turn_keystrokes:
        if keystroke[API] == FURNITURE_CLICK:
            viewed_text_keystrokes.append(
                {
                    API: FURNITURE_CLICK,
                    PREVIOUS_STATE: keystroke[PREVIOUS_STATE],
                    NEXT_STATE: keystroke[NEXT_STATE],
                    ARGS: keystroke[ARGS]
                }
            )
        else:
            relevant_keystrokes.append(keystroke)

    # gather information about the scene at the start of the turn
    keystroke = turn_keystrokes[0]
    starts_with_search = keystroke[API] == SEARCH_FURNITURE
    if starts_with_search:
        opening_scene = keystroke[NEXT_STATE]
        # start with SearchFurniture action
        # TODO: use last_search_args and carousel position to determine if
        #       this search is a null-op, or is equivalent to NavigateCarousel
        #       back to the start of existing carousel.
        relevant_actions = [
            {
                API: SEARCH_FURNITURE_ACTION,
                PREVIOUS_STATE: keystroke[PREVIOUS_STATE],
                NEXT_STATE: keystroke[NEXT_STATE],
                ARGS: keystroke[ARGS],
            }
        ]
        # update search_results and last_search_args
        if keystroke[ARGS] != last_search_args:
            search_results = furniture_db.search_furniture(keystroke[ARGS])
            last_search_args = keystroke[ARGS]
    else:
        opening_scene = keystroke[PREVIOUS_STATE]

    # if only one action and it is SearchFurniture then nothing more to do
    if starts_with_search and len(relevant_keystrokes) == 1:
        return relevant_actions, viewed_text_keystrokes, search_results,\
            last_search_args

    index = 0
    # find last scene setting action
    for _index, keystroke in reversed(list(enumerate(relevant_keystrokes))):
        keystroke_name = keystroke[API]
        # SEARCH_FURNITURE keystroke only occurs at the start so no need to
        # include in this check
        if keystroke_name == NEXT or keystroke_name == PREVIOUS or \
                keystroke_name == REMOVE_OBJECT_FROM_FOCUS or \
                keystroke_name == BRING_OBJECT_TO_FOCUS:
            index = _index  # (keeping lint happy)
            break

    # for keystrokes which are transformed to the NavigateCarousel action,
    # check did the scene really change, i.e. do the carousel prefabs differ
    # with those from the opening
    if keystroke_name == NEXT or keystroke_name == PREVIOUS or \
            keystroke_name == REMOVE_OBJECT_FROM_FOCUS:
        new_scene = keystroke[NEXT_STATE]
        if is_prefab_in_focus(opening_scene) or \
                not matching_carousels(opening_scene, new_scene):
            prefabs = get_carousel_prefabs(new_scene)
            relevant_actions.append(
                {
                    API: NAVIGATE_CAROUSEL_ACTION,
                    PREVIOUS_STATE: opening_scene,
                    NEXT_STATE: new_scene,
                    ARGS: {
                        FURNITURE_ID: prefabs[1] if len(prefabs) > 1
                        else prefabs[0],
                        NAVIGATE_DIRECTION: keystroke_name if keystroke_name == NEXT
                        or keystroke_name == PREVIOUS else 'Here'
                    },
                }
            )
        # return as there should be no further scene changing actions
        # after a NavigateCarousel (no further valid actions)
        return relevant_actions, viewed_text_keystrokes, search_results,\
                last_search_args

    # process BRING_OBJECT_TO_FOCUS keystroke
    if keystroke_name == BRING_OBJECT_TO_FOCUS:
        relevant_actions.append(
            {
                API: FOCUS_ON_FURNITURE_ACTION,
                NEXT_STATE: keystroke[NEXT_STATE],
                PREVIOUS_STATE: opening_scene,
                ARGS: {FURNITURE_ID: keystroke[ARGS]},
            }
        )
        index += 1

    # process remaining relevant keystrokes -- there should be only rotate
    # actions left to deal with
    while index < len(relevant_keystrokes):
        keystroke = relevant_keystrokes[index]

        if keystroke[API].startswith(ROTATE):
            direction = keystroke[API].split(ROTATE)[1].lower()
            furniture_id = keystroke[ARGS]
            next_state = relevant_keystrokes[index][NEXT_STATE]
            previous_state = relevant_keystrokes[index][PREVIOUS_STATE]
            rotate_index = index + 1

            # compress all consecutive rotates into
            # one rotate with the final rotate direction
            while rotate_index < len(relevant_keystrokes) and \
                    relevant_keystrokes[rotate_index][API].startswith(ROTATE):
                direction = relevant_keystrokes[rotate_index][API].split(ROTATE)[1].lower()
                furniture_id = relevant_keystrokes[rotate_index][ARGS]
                next_state = relevant_keystrokes[rotate_index][NEXT_STATE]
                rotate_index += 1

            # ROTATE('front') is redudant when directly following a
            # FOCUS_ON_FURNITURE_ACTION (checking only within turns,
            # not across turns)
            if direction == 'front' and len(relevant_actions) > 0 and \
                    relevant_actions[-1][API] == FOCUS_ON_FURNITURE_ACTION:
                pass
            else:
                relevant_actions.append(
                    {
                        API: ROTATE_ACTION,
                        PREVIOUS_STATE: previous_state,
                        NEXT_STATE: next_state,
                        ARGS: {DIRECTION: direction, FURNITURE_ID: furniture_id},
                    }
                )
            index = rotate_index

        else:
            index = index + 1

    return relevant_actions, viewed_text_keystrokes, search_results,\
            last_search_args


def get_viewed_text_actions(viewed_text_keystrokes):
    """ Examine FURNITURE_CLICK keystrokes and determine which are potential
    actions by the assistant where they viewed the text descriptions of items.

    Args:
        viewes_text_keystrokes: list of FURNITURE_CLICK keystrokes

    Returns:
        list of 'potential' viewed text actions
    """
    # Not yet implemented -- not needed for Dominant action sub-task
    return []


def examine_get_info_action(
    turn, raw_action_list, action_sequence_with_args, relevant_actions, furniture_metadata
):
    """Examines the presence of getInfo action for a given turn.

    Args:
        turn: Turn information
        raw_action_list: List of raw actions from the data logs
        action_sequence_with_args: Cleaned up actions with arguments
        relevant_actions: Extracted list of relevant actions
        furniture_metadata: Metadata for furniture

    Returns:
        new_action: New GetInfo action if needed to be added, else None.
    """
    # For assistant utterances, get item description overlap.
    # Go through all the FurnitureClicks.
    all_furniture_clicks = set()
    for key in action_sequence_with_args:
        if key[API] == FURNITURE_CLICK:
            [all_furniture_clicks.add(ii) for ii in key[ARGS]]
    utterance_words = word_tokenize(turn["message"])
    matches = collections.defaultdict(list)
    for item_id in all_furniture_clicks:
        item_info = furniture_metadata[int(item_id)]
        for attr in ["sale_price", "x_dim", "y_dim", "z_dim"]:
            if item_info[attr] in utterance_words:
                matches[item_id].append(attr)
        # info_words = word_tokenize(item_info['product_description'])
        # num_matches += sum([ii in utterance_words for ii in info_words])
        # print(num_matches, len(utterance_words), utterance_words)

    new_action = None
    if len(matches) > 0:
        furniture_id = sorted(
            matches.keys(), key=lambda x: len(matches[x]), reverse=True
        )[0]
        new_action = {
            API: GET_INFO_ACTION,
            ARGS: {FURNITURE_ID: furniture_id, MATCHES: matches[furniture_id]},
        }
    elif len(all_furniture_clicks) > 0 and len(relevant_actions) == 0:
        # If union of textPrefabInFocus and textPrefabsInCarousel
        # is non-empty, add a new action -- GetInfo.
        parsed_actions = [json.loads(ii)["nextState"] for ii in raw_action_list]
        open_boxes = [
            ([ii["textPrefabInFocus"]] + ii["textPrefabsInCarousel"])
            for ii in parsed_actions
        ]
        open_boxes = {jj for ii in open_boxes for jj in ii if jj != ""}
        if len(open_boxes):
            new_action = {API: GET_INFO_ACTION, ARGS: None}
            if len(open_boxes) == 1:
                new_action[ARGS] = {FURNITURE_ID: list(open_boxes)[0], MATCHES: []}
    return new_action


def extract_actions(input_json_file, save_root, furniture_db, subtask):
    """Extract assistant API calls from keystrokes and NLU/NLG annotations.

    Args:
        input_json_file: JSON data file
        save_root: Folder to save the extracted API calls
        furniture_db: object wrapping the furniture database
        subtask: Single dominant or multiple actions
    """
    # Read the raw data.
    print("Reading: {}".format(input_json_file))
    with open(input_json_file, "r") as file_id:
        data = json.load(file_id)

    dialogs = []
    price_dict = furniture_db.get_min_max_price_per_class()
    for datum in data["dialogue_data"]:
        dialog_id = datum["dialogue_idx"]
        dialog_datum = datum["dialogue"]
        dialog_coref_map = datum["dialogue_coref_map"]
        reversed_dialog_coref_map = {v: k for k, v in dialog_coref_map.items()}
        keystroke_sequence_with_args = []
        chat_utterances = []
        search_results = []
        last_search_args = {}

        for round_datum in dialog_datum:
            insert_item = {"turn_idx": round_datum["turn_idx"]}
            raw_keystroke_list = round_datum["raw_assistant_keystrokes"]
            keystrokes_with_args = get_keystrokes_with_args(
                raw_keystroke_list, price_dict
            )
            keystroke_sequence_with_args.extend(keystrokes_with_args)
            this_turn_keystrokes, keystroke_sequence_with_args = get_turn_keystrokes(
                keystroke_sequence_with_args
            )
            # get min. set of actions required to update the scene and
            # secondly furniture clicks which could signal viewed-text
            relevant_actions, viewed_text_keystrokes,\
                    search_results, last_search_args = \
                get_relevant_actions(
                    this_turn_keystrokes,
                    search_results,
                    last_search_args,
                    furniture_db
                )
            viewed_text_actions = get_viewed_text_actions(
                viewed_text_keystrokes
            )

            # get additional actions based on NLU/NLG annotation
            getinfo_actions = gen_getinfo_from_annotation(
                round_datum,
                reversed_dialog_coref_map
            )
            addtocart_actions = gen_addtocart_from_annotation(
                round_datum,
                reversed_dialog_coref_map
            )

            # collate the different sets of actions into insert_item
            collate_and_insert_actions(
                subtask,
                insert_item,
                relevant_actions,
                getinfo_actions,
                addtocart_actions,
                viewed_text_actions,
                round_datum
            )
            insert_item["raw_action_with_args"] = copy.deepcopy(keystrokes_with_args)
            insert_item["current_search_results"] = copy.deepcopy(search_results)
            chat_utterances.append(insert_item)

        roundwise_actions = get_roundwise_dialog_actions(
            subtask,
            chat_utterances
        )
        dialogs.append(
            {
                "dialog_id": dialog_id,
                "turns": chat_utterances,
                "actions": roundwise_actions,
            }
        )

    save_path = input_json_file.split("/")[-1].replace(".json", "_api_calls.json")
    save_path = os.path.join(save_root, save_path)
    print("Saving: {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(dialogs, f)


def collate_and_insert_actions(
    subtask,
    insert_item,
    relevant_actions,
    getinfo_actions,
    addtocart_actions,
    viewed_text_actions,
    round_datum
):
    """ Collate the different sets of actions according to the proposed
    'subtask' and insert into 'insert_item' (directly modified).

    Args:
        subtask: sub-task, one of DOMINANT_ACTION or MULTI_ACTION
        insert_item: dictionary actions are added to
        relevant_actions: scene setting actions
        getinfo_actions: get info actions
        addtocart_actions: add-to-cart actions
        viewed_text_actions: 'potentially' viewed text box actions
        round_datum: user-assistant turn-pair, including annotations
    """
    if subtask == DOMINANT_ACTION:
        candidate_actions = []
        candidate_actions.extend(relevant_actions)
        candidate_actions.extend(getinfo_actions)
        candidate_actions.extend(addtocart_actions)
        # viewed_text_actions are ignored in the dominant action sub-task

        # get the list of candidate actions types (names)
        candidate_action_types = [
            action[API] for action in candidate_actions
        ]

        # if candidate actions includes a SEARCH_FURNITURE_ACTION and
        # assistant intents include DA:INFORM:GET (without attribute)
        # then override priority ordering by dropping all actions except
        # SEARCH_FURNITURE_ACTION and ADD_TO_CART_ACTION
        if SEARCH_FURNITURE_ACTION in candidate_action_types:
            assistant_intents = data_support.get_intents(
                data_support.ASSISTANT,
                round_datum
            )
            search_furniture = any(
                ("DA:INFORM:GET" in intent) and ("." not in intent)
                for intent in assistant_intents
            )
            if search_furniture:
                candidate_actions = [
                    action for action in relevant_actions
                    if action[API] == SEARCH_FURNITURE_ACTION
                    or action[API] == ADD_TO_CART_ACTION
                ]

        # find highest preference action type
        candidate_action_types = [
            action[API] for action in candidate_actions
        ]
        target_action_type = None
        for action_type in PREFERENCE_ORDER:
            if action_type in candidate_action_types:
                target_action_type = action_type
                break

        # filter to retain only the highest preference action types
        candidate_actions = [
            action for action in candidate_actions
            if action[API] == target_action_type
        ]

        # If there are multiple GEN_INFO_ACTIONs then check if the last one
        # is part of a set of requests for multiple attributes. Only the last
        # action needs to be checked as only it is retained below.
        if target_action_type == GET_INFO_ACTION \
                and len(candidate_actions) > 1:
            last_action_attributes = set()
            last_action_furniture_id = \
                    candidate_actions[-1][ARGS][FURNITURE_ID]
            for action in candidate_actions:
                if action[ARGS][FURNITURE_ID] == last_action_furniture_id:
                    last_action_attributes.add(action[ARGS][MATCHES])
            # replace multi-attribute requests with a single "info" request
            if len(last_action_attributes) > 1:
                candidate_actions[-1][ARGS][MATCHES] = "info"

        # if multiple candidates remain take the last action
        if candidate_actions:
            dominant_action = candidate_actions[-1]
        else:
            dominant_action = {
                API: NONE_ACTION,
                # TODO: fix up  NEXT_STATE and PREVIOUS_STATE
                PREVIOUS_STATE: None,
                NEXT_STATE: None,
                ARGS: None,
            }
        insert_item["relevant_apis_with_args"] = copy.deepcopy([dominant_action])

    elif subtask == MULTI_ACTION:
        raise NotImplementedError
    else:
        raise Exception(f"Unexpected sub-task '{subtask}'")


def gen_getinfo_from_annotation(round_datum, reversed_dialog_coref_map):
    """ Generate GET_INFO_ACTIONs inferred based on NLU/NGL annotation.

    Args:
        round_datum: user-assistant turn-pair, including annotation
        reversed_dialog_coref_map: maps per dialog object ids to furniture ids

    Returns:
        list of inferred GET_INFO_ACTIONs for this turn.
    """
    da_ask_regex = re.compile("DA:ASK:GET:([^.]+)[.](.+)")
    da_inform_regex = re.compile("DA:INFORM:GET:([^.]+)[.](.+)")

    user_action_refs = data_support.get_object_references(
        round_datum['turn_label'],
        reversed_dialog_coref_map
    )
    system_action_refs = data_support.get_object_references(
        round_datum['system_turn_label'],
        reversed_dialog_coref_map
    )
    # For GetInfo, look for DA:ASK:GET intent with attributes and extract
    # the attributes, and check system responded
    get_info_attributes = []
    for intent, obj_refs in user_action_refs.items():
        da_ask_match = da_ask_regex.match(intent)
        if da_ask_match:
            obj = da_ask_match[1]
            attribute = da_ask_match[2]
            furniture_id = [r[1] for r in obj_refs]
            # if object reference(s) missing try to find references in the
            # assistant action
            if not furniture_id:
                last_resort_id = None
                for sys_intent, sys_obj_refs in system_action_refs.items():
                    if sys_obj_refs:
                        da_inform_match = da_inform_regex.match(sys_intent)
                        # if any object id found in any DA:INFORM:GET, keep the
                        # first one found as a last resort fall-back
                        if not last_resort_id:
                            last_resort_id = [r[1] for r in sys_obj_refs]
                        if da_inform_match:
                            # try strict match first; match object and attribute
                            if (
                                da_inform_match[1] == obj
                                and da_inform_match[2] == attribute
                            ):
                                furniture_id = [r[1] for r in sys_obj_refs]
                                break  # take the first strict match found.
                            # else object matches and system attribute is a
                            # super set of the attribute asked, i.e. 'info'
                            elif (
                                da_inform_match[1] == obj
                                and da_inform_match[2] == 'info'
                            ):
                                furniture_id = [r[1] for r in sys_obj_refs]
                                # keep looping in case a strict match exists.
                            # if nothing better so far, accept object mis-match
                            # (allows for annotation error)
                            elif (
                                not furniture_id
                                and (
                                    da_inform_match[2] == attribute
                                    or da_inform_match[2] == 'info'
                                )
                            ):
                                furniture_id = [r[1] for r in sys_obj_refs]
                                # keep looping in case something better
                # if nothing better found
                if not furniture_id and last_resort_id:
                    furniture_id = last_resort_id
            get_info_attributes.append((attribute, furniture_id))

    # Check the system responded; at least one DA:INFORM:GET intent with an
    # attribute
    system_responded = any(
        da_inform_regex.match(intent) for intent in system_action_refs.keys()
    )

    get_info_actions = []
    if get_info_attributes and system_responded:
        for attribute, furniture_ids in get_info_attributes:
            if attribute not in FILTER_MATCHES:
                attribute = "info"
            new_action = {
                API: GET_INFO_ACTION,
                ARGS: {
                    MATCHES: attribute,
                    FURNITURE_ID: furniture_ids
                },
            }
            get_info_actions.append(new_action)

    return get_info_actions


def gen_addtocart_from_annotation(round_datum, reversed_dialog_coref_map):
    """ Generate ADD_TO_CART_ACTIONs inferred based on NLU/NGL annotation.

    Args:
        round_datum: user-assistant turn-pair, including annotations
        reversed_dialog_coref_map: maps per dialog object ids to furniture ids

    Returns:
        list of inferred ADD_TO_CART_ACTIONs for this turn.
    """
    intents = data_support.get_intents(data_support.ASSISTANT, round_datum)
    for intent in intents:
        if (
            "DA:CONFIRM:ADD_TO_CART" in intent
            or "DA:INFORM:ADD_TO_CART" in intent
        ):
            system_action_refs = data_support.get_object_references(
                round_datum['system_turn_label'],
                reversed_dialog_coref_map
            )
            user_action_refs = data_support.get_object_references(
                round_datum['turn_label'],
                reversed_dialog_coref_map
            )
            furniture_id = []
            # check for object reference(s) that accompanies the assistant action
            for act, refs in system_action_refs.items():
                if 'ADD_TO_CART' in act and refs:
                    furniture_id = [r[1] for r in refs]
                    break
            # if object reference missing for assistant action, check if
            # preceding user action mentions ADD_TO_CART and has object refs
            if not furniture_id:
                for act, refs in user_action_refs.items():
                    if 'ADD_TO_CART' in act and refs:
                        furniture_id = [r[1] for r in refs]
            new_action = {
                API: ADD_TO_CART_ACTION,
                PREVIOUS_STATE: None,
                NEXT_STATE: None,
                ARGS: {
                    FURNITURE_ID: furniture_id
                }
            }
            # only expect one add-to-cart per turn
            return [new_action]
    return []


def get_roundwise_dialog_actions(subtask, dialog_actions):
    """From NLU / NLG + Assistant logs, get finalized actions + carousel states.

    Args:
        subtask: sub-task, one of DOMINANT_ACTION or MULTI_ACTION
        dialog_actions: Dialog actions with metainformation
    """
    # Initialize the turn carousel state as empty.
    carousel_pos = ["left", "center", "right"]
    ignore_actions = ["BringObjectToFocus", "FurnitureClick"]
    turn_carousel_state = get_carousel_state(None)
    roundwise_actions = []
    for _turn_id, turn_datum in enumerate(dialog_actions):
        # Report action and supervision for assistant.
        action_datum = {}
        # Defaults.
        cur_action = NONE_ACTION
        cur_supervision = None
        action_output_state = None
        action_datum["carousel_state"] = copy.deepcopy(turn_carousel_state)
        relevant_actions = turn_datum["relevant_apis_with_args"]
        # search_results = turn_datum["current_search_results"]
        if len(relevant_actions) == 1:
            cur_action = relevant_actions[0]["api"]
            if cur_action in ignore_actions:
                cur_action = NONE_ACTION
                action_datum["carousel_state"] = copy.deepcopy(turn_carousel_state)
            else:
                cur_supervision = relevant_actions[0]
                # Update carousel state and store.
                insert_state, action_output_state = update_carousel_state(
                    relevant_actions[0], turn_carousel_state
                )
                action_datum["carousel_state"] = insert_state
        elif len(relevant_actions) > 1:
            combo_actions = [ii["api"] for ii in relevant_actions]
            # For combos, use the following priority:
            combo_args = {ii["api"]: ii for ii in relevant_actions}
            preference_order = [
                ADD_TO_CART_ACTION,
                GET_INFO_ACTION,
                ROTATE_ACTION,
                SEARCH_FURNITURE_ACTION,
                FOCUS_ON_FURNITURE_ACTION,
                NAVIGATE_CAROUSEL_ACTION,
                NONE_ACTION,
            ]
            cur_supervision = None
            for order in preference_order:
                if order in combo_actions:
                    cur_action = order
                    break
            if order == NONE_ACTION:
                action_datum["carousel_state"] = copy.deepcopy(turn_carousel_state)
            else:
                cur_supervision = combo_args[order]
                # Update carousel state and store.
                insert_state, action_output_state = update_carousel_state(
                    combo_args[order], turn_carousel_state
                )
                action_datum["carousel_state"] = insert_state

        if subtask == DOMINANT_ACTION:
            # for DOMINANT_ACTION subtask remove absolute reference from ARGS
            if cur_action == NAVIGATE_CAROUSEL_ACTION:
                cur_supervision[ARGS] = {
                    NAVIGATE_DIRECTION: cur_supervision[ARGS][NAVIGATE_DIRECTION]
                }
            # if cur_action == FOCUS_ON_FURNITURE_ACTION:
            #    cur_supervision[ARGS] = {
            #        POSITION: cur_supervision[ARGS][POSITION]
            #    }  <-- clean up, and align with below FOCUS_ON_FURNITURE_ACTION
            if cur_action == ROTATE_ACTION:
                cur_supervision[ARGS] = {
                    DIRECTION: cur_supervision[ARGS][DIRECTION]
                }
            if (
                cur_action == GET_INFO_ACTION
                or cur_action == ADD_TO_CART_ACTION
            ):
                # for GET_INFO_ACTION and ADD_TO_CART_ACTION subtask create a
                # single reference relative to the carousel
                furniture_ids = cur_supervision[ARGS][FURNITURE_ID]
                # {'focus': '', 'carousel': ['901712', '1215555']}
                if turn_carousel_state["focus"]:
                    if turn_carousel_state["focus"] in furniture_ids:
                        cur_supervision[ARGS][FURNITURE_ID] = "focus"
                    else:
                        cur_supervision[ARGS][FURNITURE_ID] = ""
                else:
                    in_carousel = False
                    for id in furniture_ids:
                        if id in turn_carousel_state["carousel"]:
                            index = turn_carousel_state["carousel"].index(id)
                            cur_supervision[ARGS][FURNITURE_ID] = \
                                carousel_pos[index]
                            in_carousel = True
                            break  # take first only
                    if not in_carousel:
                        cur_supervision[ARGS][FURNITURE_ID] = ""

        if cur_action == FOCUS_ON_FURNITURE_ACTION:
            furniture_id = cur_supervision["args"]["furniture_id"]
            # Check if furniture is in shared carousel.
            shared_carousel = cur_supervision["nextState"]["sharedPrefabsInCarousel"]
            if furniture_id in shared_carousel:
                position = shared_carousel.index(furniture_id)
            # Else, check assistant carousel (did not share)!
            else:
                assistant_carousel = cur_supervision["nextState"]["prefabsInCarousel"]
                position = assistant_carousel.index(furniture_id)
            cur_supervision["args"] = {
                # "furniture_id": furniture_id,
                "position": carousel_pos[position],
            }
        action_datum["action"] = cur_action
        action_datum["action_supervision"] = cur_supervision
        action_datum["action_output_state"] = action_output_state
        roundwise_actions.append(action_datum)

        # Go through all the raw_actions to get the next turn_state.
        raw_actions = turn_datum["raw_action_with_args"]
        if len(raw_actions):
            turn_carousel_state = {
                "focus": raw_actions[-1][NEXT_STATE][SHARED_FOCUS],
                "carousel": raw_actions[-1][NEXT_STATE][SHARED_CAROUSEL],
            }
    return roundwise_actions


def update_carousel_state(filtered_action, current_state):
    """Update the carousel state from filtered_action.

    Args:
        filtered_action: Relevant action for the current round
        current_state: Current state of the carousel

    Returns:
        insert_carousal_state: Carousel state to insert for the action
        new_current_state: State of carousel after executing the action
    """
    if filtered_action[API] != GET_INFO_ACTION:
        current_state = get_carousel_state(
            filtered_action["previousState"], filtered_action
        )
    insert_carousel_state = copy.deepcopy(current_state)
    if filtered_action[API] != GET_INFO_ACTION:
        current_state = get_carousel_state(
            filtered_action["nextState"], filtered_action
        )
    return insert_carousel_state, copy.deepcopy(current_state)


def get_carousel_state(state=None, action_args=None):
    """Obtain carousel state from dictionary of States.

    Args:
        state: The original state to convert into new format.
        action_args: Action arguments.

    Returns:
        new_state: Return state in a new format.
    """
    new_state = {"focus": None, "carousel": []}
    if state is None:
        return new_state

    # Check if the state is empty. If yes, return None.
    empty_lists = [
        "prefabsInCarousel",
        "sharedPrefabsInCarousel",
        "textPrefabsInCarousel",
    ]
    empty_strs = ["prefabInFocus", "sharedPrefabInFocus", "textPrefabInFocus"]
    empty_state = all(state[ii] == "" for ii in empty_strs)
    if empty_state:
        for key in empty_lists:
            if len(state[key]):
                empty_state = False
                break
    if empty_state:
        return new_state

    # Get items in the carousel.
    action = action_args["api"]
    if action == "SearchFurniture":
        new_state["carousel"] = state["sharedPrefabsInCarousel"]
    elif action in ["Rotate", "FocusOnFurniture"]:
        focus_id = action_args["args"]["furniture_id"]
        search_order = ["prefabsInCarousel", "sharedPrefabsInCarousel"]
        for order in search_order:
            if focus_id in state[order]:
                new_state["carousel"] = state[order]
    elif action in ["Previous", "Next"]:
        new_state["carousel"] = state["prefabsInCarousel"]
    # Focus object.
    search_order = ["prefabInFocus", "sharedPrefabInFocus"]
    for key in search_order:
        if state[key] != "":
            new_state["focus"] = state[key]
            break
    return new_state


def main(_):
    furniture_db = data_support.FurnitureDatabase(FLAGS.metadata_path)
    for input_json_file in FLAGS.json_path:
        extract_actions(
            input_json_file,
            FLAGS.save_root,
            furniture_db,
            FLAGS.subtask
        )
    furniture_db.shutdown()


if __name__ == "__main__":
    app.run(main)
