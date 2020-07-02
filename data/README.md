# SIMMC Datasets

## Summary

Our challenge focuses on two SIMMC datasets, both in the shopping domain:
(a) furniture (grounded in a shared virtual environment) and, 
(b) fashion (grounded in an evolving set of images).   

Both datasets were collected through the SIMMC Platform, an extension to ParlAI for multimodal conversational data collection and system evaluation that allows human annotators to each play the role of either the assistant or the user.

The following papers describe in detail the dataset, the collection platform, and the NLU/NLG/Coref annotations we provide:

Seungwhan Moon*, Satwik Kottur*, Paul A. Crook^, Ankita De^, Shivani Poddar^, Theodore Levin, David Whitney, Daniel Difranco, Ahmad Beirami, Eunjoon Cho, Rajen Subba, Alborz Geramifard. ["Situated and Interactive Multimodal Conversations"](https://arxiv.org/pdf/2006.01460.pdf) (2020).

Paul A. Crook*, Shivani Poddar*, Ankita De, Semir Shafi, David Whitney, Alborz Geramifard, Rajen Subba. ["SIMMC: Situated Interactive Multi-Modal Conversational Data Collection And Evaluation Platform"](https://arxiv.org/pdf/1911.02690.pdf) (2020).

If you want to publish experimental results with our datasets or use the baseline models, please cite the following articles:
```
@article{moon2020situated,
  title={Situated and Interactive Multimodal Conversations},
  author={Moon, Seungwhan and Kottur, Satwik and Crook, Paul A and De, Ankita and Poddar, Shivani and Levin, Theodore and Whitney, David and Difranco, Daniel and Beirami, Ahmad and Cho, Eunjoon and Subba, Rajen and Geramifard, Alborz},
  journal={arXiv preprint arXiv:2006.01460},
  year={2020}
}

@article{crook2019simmc,
  title={SIMMC: Situated Interactive Multi-Modal Conversational Data Collection And Evaluation Platform},
  author={Crook, Paul A and Poddar, Shivani and De, Ankita and Shafi, Semir and Whitney, David and Geramifard, Alborz and Subba, Rajen},
  journal={arXiv preprint arXiv:1911.02690},
  year={2019}
```

### Dataset Splits

We randomly split each of our SIMMC-Furniture and SIMMC-Fashion datasets into four components:

| **Split** | **Furniture** | **Fashion** |
| :--: | :--: | :--: |
| Train (60%)   | 3839 | 3929 | 
| Dev (10%)     | 640 | 655 |
| Test-Dev (15%) | 960 | 982 |
| Test-Std (15%) | 960 | 983 |

**NOTE**
* **Dev** is for hyperparameter selection and other modeling choices.  
* **Test-Dev** is the publicly available test set to measure model performance and report results outside the challenge.  
* **Test-Std** is used as the main test set for evaluation for Challenge Phase 2 (to be released on Sept 28).

## Download the Datasets
We are hosting our datasets in this Github Repository (with [Git LFS](https://git-lfs.github.com/)).
First, install Git LFS
```
$ git lfs install
```

Clone our repository to download both the dataset and the code:
```
$ git clone https://github.com/facebookresearch/simmc.git
```

## Overview of the Dataset Repository 

The data are made available for each `domain` (`simmc_furniture` | `simmc_fashion`) in the following files:
```
[Main Data]
- full dialogs: ./{domain}/{train|dev|devtest|test}_dials.json
- list of dialog IDs per split: ./{domain}/{train|dev|devtest|test}_dialog_ids

[Metadata]
- Fashion metadta: ./simmc_fashion/fashion_metadata.json
- Furniture metadata: ./simmc_furniture/furniture_metadata.csv
- images: ./simmc-furniture/figures/{object_id}.png
```
**NOTE**: The test set will be made available after DSTC9.

## Data Format

For each `{train|dev|devtest}` split, the JSON data (`./{domain}/{train|dev|devtest}_dials.json`
) is formatted as follows:


```
{
  "split": support.extract_split_from_filename(json_path),
  "version": 1.0,
  "year": 2020,
  "domain": FLAGS.domain,
  "dialogue_data": [
  {
    “dialogue”: [
      {
        “belief_state”: [
          {
            “act”: <str>,
            “slots”: [
              [ <str> slot_name, <str> slot_value  ], // end of a slot name-value pair
              ...
            ]
          }, // end of an act-slot pair
          ...
        ],
        “domain”: <str>,
        “raw_assistant_keystrokes”: <dict>,
        “state_graph_{idx}”: <dict>,
        “syste_belief_state”: <dict>,
        “system_transcript”: <str>,
        “system_transcript_annotated”: <str>,
        “transcript”: <str>,
        “transcript_annotated”: <str>,
        “turn_idx”: <int>,
        “turn_label”: [ <dict> ],
        “visual_objects”: <dict>
      }, // end of a turn (always sorted by turn_idx)
      ...
    ],
    “dialogue_coref_map”: {
      // map from object_id to local_id re-indexed for each dialog
      <str>: <int>
    },
    “dialogue_idx”: <int>,
    “domains”: [ <str> ]
  }
]
}
```
The data can be processed with respective data readers / preprocessing scripts for each sub-task (please refer to the respective README documents). Each sub-task will describe which fields can be used as input.

**NOTES**

`visual_objects` refer to the list of objects and their visual attributes that are shown to the user at each given turn (via a VR environment or an image). 
```
{
    <str> obj_name: {
        <str> attribute_name: <list> or <str> attribute_values
    }
}
```

`state_graph_{idx}` refers to the graph representation of the cumulative dialog and the multimodal contexts known to the user, each at a different phase during the dialog (e.g. via a multimodal action of showing items, an assistant providing information, a user providing preferences, etc.). 
- state_graph_0: initial state before the start of the user utterance
- state_graph_1: state modified after the user utterance 
- state_graph_2: final state modified after the assistant utterance & assistant action.

Participants may use this information for inspection, or as additional training signals for some of the sub-tasks (but not at inference time). `belief_state`, `system_beilef_state`, and `visual_objects` provide the same information. 

Each state graph is represented as follows:
```
{
    <str> obj_name: {
        <str> attribute_name: <list> or <str> attribute_values
    }
}
```

`raw_assistant_keystrokes` are the raw UI interactions made by the human Assistant (wizard) using the Unity interface during data collection. We distil target actions for the action prediction task (sub-task #1) from these raw keystrokes and NLU/NLG annotation

We also release the metadata for each object referred in the dialog data:
```
<fashion_metadata.json>
{
    <int> object_id: {
        “metadata”: {dict},
        “url”: <str> source image
    }, // end of an object
}

<furniture_metadata.csv>
columns:
- product_name
- product_description
- product_thumbnail_image_url
- material
- color
- obj ({object_id}.zip)
...
```
Attributes for each object either pulled from the original sources or annotated manually. Note that some of the catalog-specific attributes (e.g. availableSizes, brand, etc.) were randomly and synthetically generated. 

Each item in a catalog metadata has a unique `<int> object_id`. `dialog_coref_map` defines the mapping from the `local_idx` (local to each dialog), to its canonical `object_id` reference, for each dialog. This `local_idx` is used in `belief_state` as an object slot. For example, given a `dialog_coref_map = {0: 123, 1: 234, 2: 345}` -- the belief state: `{‘act’: ‘DA:REQUEST:ADD_TO_CART’, ‘slots’: [‘O’: ‘OBJECT_2’]}` would indicate this particular dialog act performed upon `OBJECT_2` (`2 == local_idx`), which has a canonical reference to an object with `object_id: 345`. We are including this information in case you want to refer to the additional information provided in the `metadata.{json|csv}` file. 
