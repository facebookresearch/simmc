#!/usr/bin/env python3


def evaluate_from_flat_list(d_true, d_pred):
    """
        <list>d_true and <list>d_pred are in the following format:
        (Each element represents a single turn, with (multiple) frames)
        [
            [
                {
                    'act': <str>,
                    'slots': [
                        [
                            SLOT_NAME, SLOT_VALUE
                        ], ...
                    ]
                },
                [End of a frame]
                ...
            ],
            [End of a turn]
            ...
        ]
    """
    c = {
        'n_frames': 0.0,
        'n_true_acts': 0.0,
        'n_pred_acts': 0.0,
        'n_correct_acts': 0.0,
        'n_true_slots': 0.0,
        'n_pred_slots': 0.0,
        'n_correct_slots': 0.0,
        'n_correct_beliefs': 0.0,
    }

    # Count # corrects & # wrongs
    for turn_idx in range(len(d_true)):
        true_turn = d_true[turn_idx]
        pred_turn = d_pred[turn_idx]

        c = add_dicts(
            c,
            evaluate_turn(true_turn, pred_turn))

    # Calculate metrics
    joint_accuracy = c['n_correct_beliefs'] / c['n_frames']
    act_rec = c['n_correct_acts'] / c['n_true_acts']
    act_prec = c['n_correct_acts'] / c['n_pred_acts']
    act_f1 = \
        2 * act_prec * act_rec / (act_prec + act_rec) \
            if (act_prec + act_rec) != 0 else 0

    slot_rec = c['n_correct_slots'] / c['n_true_slots']
    slot_prec = c['n_correct_slots'] / c['n_pred_slots']
    slot_f1 = \
        2 * slot_prec * slot_rec / (slot_prec + slot_rec) \
            if (slot_prec + slot_rec) != 0 else 0

    return {
        'joint_accuracy': joint_accuracy,
        'act_rec': act_rec,
        'act_prec': act_prec,
        'act_f1': act_f1,
        'slot_rec': slot_rec,
        'slot_prec': slot_prec,
        'slot_f1': slot_f1,
    }


def evaluate_turn(true_turn, pred_turn):

    count_dict = {
        'n_frames': 0,
        'n_true_acts': 0,
        'n_pred_acts': 0,
        'n_correct_acts': 0,
        'n_true_slots': 0,
        'n_pred_slots': 0,
        'n_correct_slots': 0,
        'n_correct_beliefs': 0,
    }

    # Must preserve order in which frames appear.
    for frame_idx in range(len(true_turn)):
        # For each frame
        true_frame = true_turn[frame_idx]
        if frame_idx >= len(pred_turn):
            pred_frame = {}
        else:
            pred_frame = pred_turn[frame_idx]

        count_dict = add_dicts(
            count_dict,
            evaluate_frame(true_frame, pred_frame, strict=False))

    return count_dict


def evaluate_frame(true_frame, pred_frame, strict=True):
    """
        If strict=True,
            For each dialog_act (frame), set(slot values) must match.
            If dialog_act is incorrect, its set(slot values) is considered wrong.
    """
    count_dict = {
        'n_frames': 1,
        'n_true_acts': 0,
        'n_pred_acts': 0,
        'n_correct_acts': 0,
        'n_true_slots': 0,
        'n_pred_slots': 0,
        'n_correct_slots': 0,
        'n_correct_beliefs': 0,
    }

    # Compare Dialog Actss
    true_act = true_frame['act'] if 'act' in true_frame else None
    pred_act = pred_frame['act'] if 'act' in pred_frame else None
    b_correct_act = true_act == pred_act
    count_dict['n_correct_acts'] += b_correct_act
    count_dict['n_true_acts'] += 'act' in true_frame
    count_dict['n_pred_acts'] += 'act' in pred_frame

    # Compare Slots
    true_frame_slot_values = {f'{k}={v}' for k, v in true_frame.get('slots', [])}
    pred_frame_slot_values = {f'{k}={v}' for k, v in pred_frame.get('slots', [])}

    count_dict['n_true_slots'] += len(true_frame_slot_values)
    count_dict['n_pred_slots'] += len(pred_frame_slot_values)

    if strict and not b_correct_act:
        pass
    else:
        count_dict['n_correct_slots'] += \
            len(true_frame_slot_values.intersection(pred_frame_slot_values))

    count_dict['n_correct_beliefs'] += \
        (b_correct_act and true_frame_slot_values == pred_frame_slot_values)

    return count_dict


def add_dicts(d1, d2):
    return {k: d1[k] + d2[k] for k in d1}
