**Allowed Inputs**

* The guideline below shows the input fields that are allowed (default) and disallowed (marked as 'X') at **inference time**, for each subtask.
* Participants are free to use any of the fields below during **training** though as additional supervision signals, and *e.g.* at the inference time use the reconstructed / predicted values instead.


| Key |  Subtask #1 </br>(API Prediction) | Subtask #2 <br>(Response Generation) | Subtask #3 <br> (MM-DST) | 
|:---|:---:|:---:|:---:|
|**JSON File (Turn Level Input Fields)**| | | |
| `belief_state` | ✗ | ✗ | ✗ <br> (prediction target) |
|  `domain` | 
| `state_graph_0` |  | |  ✗ | 
|`state_graph_1`|  ✗ | ✗ | ✗ |
|`state_graph_2`|  ✗ | ✗ | ✗ |
|`system_transcript`<br>(current turn) | ✗ | ✗<br>(prediction target) | ✗ |
|`system_transcript`<br>(previous turns)|  |  |  |
|`system_transcript_annotated`| ✗ | ✗ | ✗ |
|`system_turn_label`| ✗ | ✗ | ✗ |
|`transcript`| | |  |
| `transcript_annotated` | ✗ | ✗ | ✗ |
|`turn_idx`| | | |
|`turn_label`| ✗ | ✗ | ✗ |
|`visual_objects`| | | |
|`raw_assistant_keystrokes`| ✗ | ✗ | ✗ |
|**JSON File (Dialog Level Input Fields)**| | | |
|`dialogue_coref_map`| ✗ | ✗ | ✗ |
| `dialogue_idx` | 
| `domains` | 
|**API Call File**| | | |
|`action`<br>(current turn)| ✗  <br> (prediction target) |  | ✗ |
|`action`<br>(previous turns)|  |  |  |
|`action_supervision`<br>(current turn)| ✗ |  | ✗ |
|`action_supervision`<br>(previous turns)|  |  |  |
|`focus_images (Fashion)`| | | |
|`carousel_state (Furniture)`| | | |
|`action_output_state(Furniture)`| ✗ |  | ✗ |
|**Metadata Files**| | | |
|`fashion_metadata.json`| | | |
|`furniture_metadata.csv`| | | |

**Notes**

`transcript_annotated` provides the detailed structural intents, slots and values for each USER turn, including the text spans. `system_transcript_annotated` provides the similar information for ASSISTANT turns.

`turn_label` expands `transcript_annotated` with the coreference labels annotated as well. `objects` field in `turn_label` includes a list of objects referred to in each turn - each marked with a local index throughout the dialog (`obj_idx`) and `obj_type`. `system_turn_label` provides the similar information for ASSISTANT turns.

`belief_state` provides the intents, slots, and values, where their slots and values are cumulative throughout the dialog whenever applicable. Each slot name is prepended with its domain name, e.g. `{domain}-{slot_name}`. Specifically, we include an object slot called `{domain}-O` whose values are `OBJECT_{local_idx}`. For instance, a `belief_state` with `act: DA:REQUEST:ADD_TO_CART:CLOTHING` with a slot `[[‘fashion-O’, ‘OBJECT_2’], [‘fashion-O’, ‘OBJECT_3’]]` would annotate a user belief state with the intention of adding objects 2 and 3 to the cart. 

The entire catalog information is stored in either `fashion_metadata.json` or `furniture_metadata.csv`. The API calls provide the state of the carousel (`furniture`) or focus item (`fashion`) after the ground truth API / actions have been called. By using these two, one should be able to retrieve the entire information about the catalog items that are potentially described in the system response.

For more details, please refer to the full description in the [data README document](https://github.com/facebookresearch/simmc/tree/master/data).
