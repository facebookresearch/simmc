**Description**

`transcript_annotated` provides the detailed structural intents, slots and values for each USER turn, including the text spans. `system_transcript_annotated` provides the similar information for ASSISTANT turns.

`turn_label` expands `transcript_annotated` with the coreference labels annotated as well. `objects` field in `turn_label` includes a list of objects referred to in each turn - each marked with a local index throughout the dialog (`obj_idx`) and `obj_type`. `system_turn_label` provides the similar information for ASSISTANT turns.

`belief_state` provides the intents, slots, and values, where their slots and values are cumulative throughout the dialog whenever applicable. Each slot name is prepended with its domain name, e.g. `{domain}-{slot_name}`. Specifically, we include an object slot called `{domain}-O` whose values are `OBJECT_{local_idx}`. For instance, a `belief_state` with `act: DA:REQUEST:ADD_TO_CART:CLOTHING` with a slot `[[‘fashion-O’, ‘OBJECT_2’], [‘fashion-O’, ‘OBJECT_3’]]` would annotate a user belief state with the intention of adding objects 2 and 3 to the cart. 

**Allowed Inputs**

| Key | Description | Subtask #1 </br>(API Prediction) | Subtask #2 <br>(Response Generation) | Subtask #3 <br> (MM-DST) | 
|:---|:---:|:---:|:---:|:---:|
| `belief_state` |  | ✗ | ✗ | ✗ <br> (prediction target) |
|  `domain` | Fashion / Furniture |
| `state_graph_0` | Before the round | | |  ✗ | 
|`state_graph_1`| After the user turn | ✗ | ✗ | ✗ |
|`state_graph_2`| After the system turn| ✗ | ✗ | ✗ |
|`system_transcript`| | ✗ | ✗<br>(prediction target) | ✗ |
|`system_transcript_annotated`| | ✗ | ✗ | ✗ |
|`system_turn_label`| | ✗ | ✗ | ✗ |
|`transcript`| | |  | |
| `transcript_annotated` | | ✗ | ✗ | ✗ |
|`turn_idx`| | | | |
|`turn_label`| | ✗ | ✗ | ✗ |
|`visual_objects`| | | | |
|`raw_assistant_keystrokes`| | ✗ | ✗ | ✗ |
|**API Call File**| | | | |
|`action`| | ✗  <br> (prediction target) |  | ✗ |
|`action_supervision`| | ✗ |  | ✗ |
|`focus_images (Fashion)`| | | | |
|`carousel_state (Furniture)`| | | | |
|`action_output_state(Furniture)`| | ✗ |  | ✗ |
