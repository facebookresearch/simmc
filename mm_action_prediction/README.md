# DSTC Track 4: SIMMC | Sub-Task #1: Multimodal Assistant API Prediction

This directory contains the code and the scripts for running the baseline models for Sub-Task #1: Multimodal Assistant API Prediction.

This subtask involves predicting the assistant actions through API calls along with the necessary arguments using dialog history, multimodal context, and the current user utterance as inputs.
For example, enquiring about an attribute value (e.g., price) for a shared furniture item is realized through a call to the *SpecifyInfo* API with the price argument.
A comprehensive set of APIs for our SIMMC dataset is given in the [paper][simmc_arxiv]. 

Please check the [task input](./TASK_INPUTS.md) file for a full description of inputs
for each subtask.

## Evaluation
Currently, we evaluate action prediction as a round-wise, multiclass classification problem over the set of APIs, and measure the *accuracy* of the most **dominant action**. 
In addition, we also use *action perplexity* (defined as the exponential of the mean loglikelihood of the dominant action) to allow situations where several actions are equally valid in a given context.
We also measure the correctness of the predicted action (API) arguments using attribute *accuracy* (for Furniture) and *f1 score* (for Fashion).
Specifically, the following API classes and attributes are evaluated.

**SIMMC-Furniture**
|  API   |  API Attributes |
|:------:| :--------: |
| `SearchFurniture`  | `furnitureType`, `color` |
| `FocusOnFurniture` | `position` | 
| `SpecifyInfo` | `attributes`|
| `Rotate` | `direction`|
| `NavigateCarousel` | `navigateDirection` |
| `AddToCart` | - |
| `None` | - |

Each of the above attributes is a categorical variable, modeled as multiclass classification problem, and evaluated using attribute accuracy.
**Note:** `minPrice` and `maxPrice` attributes corresponding to the `SpecifyInfo` action for Furniture are excluded in the current evaluation.

**SIMMC-Fashion**
| API  |   API Attributes |
|:--------: | :------: |
| `SearchDatabase` | `attributes` |
| `SearchMemory` | `attributes` |
| `SpecifyInfo`| `attributes` |
| `AddToCart` | - |
| `None` | - |

Each of the attributes takes multiple values from a fixed set, modeled as multilabel classification problem, and evaluated using attribute F1 score.

The code to evaluate Sub-Task #1 is given in `tools/action_evaluation.py`.
The model outputs are expected in the following format:

```
[
	{
		"dialog_id": ...,  
		"predictions": [
			{
				"action": <predicted_action>,
				"action_log_prob": {
					<action_token>: <action_log_prob>,
					...
				},
				"attributes": {
					<attribute_label>: <attribute_val>,
					...
				}
			}
		]
	}
]
```
where `attribute_label` corresponds to the API attribute(s) predicted for each API (refer to the table above) and
`attribute_val` contains the list of values taken by the key `attribute_label`.

**NOTE**: We plan to extend the Multimodal Assistant API Prediction from the most dominant assistant action to allow the prediction of a series of multiple actions per turn. Please follow the [**Latest News**](https://github.com/facebookresearch/simmc/#latest-news) section in the main README of the repository for updates.

For more details on the task definition and the baseline models we provide, please refer to our SIMMC paper:

```
@article{moon2020situated,
  title={Situated and Interactive Multimodal Conversations},
  author={Moon, Seungwhan and Kottur, Satwik and Crook, Paul A and De, Ankita and Poddar, Shivani and Levin, Theodore and Whitney, David and Difranco, Daniel and Beirami, Ahmad and Cho, Eunjoon and Subba, Rajen and Geramifard, Alborz},
  journal={arXiv preprint arXiv:2006.01460},
  year={2020}
}
```
**NOTE**: The [paper][simmc_arxiv] reports the results from an earlier version of the dataset and with different train-dev-test splits, hence the baseline performances on the challenge resources will be slightly different. 

## Installation (Same across all sub-tasks)

* Git clone the repository:
```
$ git lfs install
$ git clone https://github.com/facebookresearch/simmc.git
```
**NOTE**: We recommend installation in a virtual environment ([user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)). Create a new virtual environment and activate it prior to installing the packages. 

* Install the required Python packages:
  * [Python 3.6+](https://www.python.org/downloads/)
  * [PyTorch 1.5+](https://pytorch.org/get-started/locally/#start-locally)
  * [Transformers](https://huggingface.co/transformers/installation.html)

## Baselines
The baselines for API Prediction (Sub-Task #1) and Assistant Response Generation & Retrieval (Sub-Task #2) 
are jointly learnt.
For these baselines, the following are the additional dependencies:

```
pip install absl-py
pip install numpy
pip install json
pip install nltk
pip install spacy
```
Code also uses spaCy's `en_vectors_web_lg` dataset for GloVE embeddings. To install:

```
python -m spacy download en_vectors_web_lg
```
Code also uses NLTK's `punkt`. To install:

```
python
>>> import nltk
>>> nltk.download('punkt')
```

### Overview

Contains the following baselines models:

1. History-agnostic Encoder (HAE)
2. Hierarchical Recurrent Encoder (HRE)
3. Memory Network Encoder (MN)
4. Transformer-based History-agnostic Encoder (T-HAE)
5. TF-IDF-based Encoder (TF-IDF)
6. LSTM-based Language Model (LSTM)

Please see our [paper][simmc_arxiv] for more details about the models.


### Code Structure

* `options.py`: Command line arguments to control behavior
* `train_simmc_agent.py`: Trains SIMMC baselines
* `eval_simmc_agent.py` or `eval_genie.py`: Evaluates trained checkpoints
* `loaders/`: Dataloaders for SIMMC
* `models/`: Model files
	* `assistant.py`: SIMMC Assistant Wrapper Class
	* `encoders/`: Different types of encoders
		* `history_agnostic.py`
		* `hierarchical_recurrent.py`
		* `memory_network.py`
		* `tf_idf_encoder.py`
	* `decoder.py`: Response decoder, language model with LSTM or Transformers
	* `action_executor.py`: Learns action and action attributes
	* `carousel_embedder.py`: Learns multimodal embedding for furniture
	* `user_memory_embedder.py`: Learns multimodal embedding for fashion
	* `positional_encoding.py`: Positional encoding unit for transformers
	* `self_attention.py`: Self attention model unit
	* `{fashion|furniture}_model_metainfo.json`: Model action/attribute configurations for SIMMC
* `tools/`: Supporting scripts for preprocessing and other utilites
* `scripts/`: Bash scripts to run preprocessing, training, evaluation

### Pretraining

Run `scripts/preprocess_simmc.sh` with appropriate `$DOMAIN` (either "furniture" or "fashion") to
run through the following steps:

1. Extract supervision for **dominant** Assistant Action API
2. Extract word vocabulary from the *train* split
3. Read and embed the shopping assets into a feature vector
4. Convert all the above information into a multimodal numpy input array for dataloader consumption
5. Extract action attribute vocabulary from train split

Please see `scripts/preprocess_simmc.sh` to better understand the inputs/outputs for each
of the above steps.


### Training and Evaluation

To train a model or evaluate a saved checkpoints, please check examples in 
`scripts/train_simmc_model.sh`. 

You can also train all the above baselines at once using `scripts/train_all_simmc_models.sh`.
For description and usage of necessary options/flags, please refer to `options.py` or one of the above two
scripts.

### Results

The baselines trained through the code obtain the following results for Sub-Task #1.

**SIMMC-Furniture**

| Model  |     Action Accuracy      | Action Perplexity | Attribute Accuracy |
|----------| :-------------: | :------: | :------: |
| TF-IDF | 77.1 | 2.59 | 57.5 |
| HAE    | 79.7 | 1.70 | 53.6 |        
| HRE    | 80.0 | 1.66 | 54.7 |
| MN     | 79.2 | 1.71 | 53.3 |        
| T-HAE  | 78.4 | 1.83 | 53.6 |

**SIMMC-Fashion**

| Model  |     Action Accuracy      | Action Perplexity | Attribute Accuracy |
|----------| :-------------: | :------: | :------: |
| TD-IDF | 78.1 | 3.51 | 57.9 |
| HAE    | 81.0 | 1.75 | 60.2 |
| HRE    | 81.9 | 1.76 | 62.1 |
| MN     | 81.6 | 1.74 | 61.6 |
| T-HAE  | 81.4 | 1.78 | 62.1 |


## Rules for Sub-task #1 Submissions
* Disallowed Input: `belief_state`, `system_transcript`, `system_transcript_annotated`, `state_graph_1`, `state_graph_2`, and anything from future turns.
* If you would like to use any other external resources, please consult with the track organizers (simmc@fb.com). Generally, we allow the use of publicly available pre-trained language models, such as BERT, GPT-2, etc.

[simmc_arxiv]:https://arxiv.org/abs/2006.01460



[1]: https://drive.google.com/file/d/0Bx4CHsnRHDmJLWVObHBtcnBYVFA1dUVCY2ZwRUFvMWx0clVj/view
[2]: https://www.dropbox.com/sh/bivp8lvsy3ff9x5/AACBLGJ2gR1qmz4x_4f6PuNIa?dl=0
[3]: https://our.internmc.facebook.com/intern/wiki/PyTorch/Using_PyTorch/PyTorch+DevGPU+Conda/
[4]: https://pytorch.org/
