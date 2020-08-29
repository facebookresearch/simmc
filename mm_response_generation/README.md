# DSTC Track 4: SIMMC | Sub-Task #2: Multimodal Assistant Response Generation

This directory contains the code and the scripts for running the baseline models for Sub-Task #2: Multimodal Assistant Response Generation.

This subtask measures the generation (or retrieval) of the assistant response given the dialog history, multimodal context, ground truth assistant API call and the current utterance.

Please check the [task input](./TASK_INPUTS.md) file for a full description of inputs
for each subtask.

## Evaluation
For generation, we use BLEU-4 score and for retrieval, we use recall@1, recall@5, recall@10, mean reciprocal rank (MRR), and mean rank.

The code to evaluate Sub-Task #2 is given in `mm_action_prediction/tools/response_evaluation.py` and 
`mm_action_prediction/tools/retrieval_evaluation.py`.
The model outputs are expected in the following format:

**Response Generation Evaluation**

```
[
	{
		"dialog_id": batch["dialog_id"][ii].item(),
		"predictions": [
			{
				"response": ...
			}
			...
		]
	}
	...
]
```

**Retrieval Evaluation**

```
[
	{
		"dialog_id": <dialog_id>,
		"candidate_scores": [
			<list of 100 scores for 100 candidates for round 1>
			<list of 100 scores for 100 candidates for round 2>
			...
		]
	}
	...
]
```


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

* Install the required Python packages:
  * [Python 3.6+](https://www.python.org/downloads/)
  * [PyTorch 1.5+](https://pytorch.org/get-started/locally/#start-locally)
  * [Transformers](https://huggingface.co/transformers/installation.html)

**NOTE**: We recommend installation in a virtual environment ([user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)). Create a new virtual environment and activate it prior to installing the packages. 

## Run Baselines

Baselines for Sub-Task #2 jointly train for both Sub-Task #2 and Sub-Task #1.
Please see Sub-Task #1 for instructions to run the baselines.

### Results
The baselines trained through the code obtain the following results for Sub-Task #2.

**SIMMC-Furniture**

| Model  |     BLEU-4     | R@1 | R@5 | R@10 | Mean Rank | MRR |
|----------| :-------------: | :------: | :------: | :------: | :------: |:------: |        
| LSTM  | 0.022 | 4.1 | 11.1 | 17.3 | 46.4 | 0.094 |
| HAE   | 0.075 | 12.9 | 28.9 | 38.4 | 31.0 | 0.218 |
| HRE   | 0.075 | 13.8 | 30.5 | 40.2 | 30.0 | 0.229 |
| MN    | 0.084 | 15.3 | 31.8 | 42.2 | 29.1 | 0.244 |
| T-HAE | 0.044 | 8.5  | 20.3 | 28.9 | 37.9 | 0.156 |
 
 
 **SIMMC-Fashion**

| Model  |     BLEU-4     | R@1 | R@5 | R@10 | Mean Rank | MRR |
|----------| :-------------: | :------: | :------: | :------: | :------: |:------: |
| LSTM  | 0.022 | 5.3  | 11.4 | 16.5 | 46.9 | 0.102 |
| HAE   | 0.059 | 10.5 | 25.3 | 34.1 | 33.5 | 0.190 |
| HRE   | 0.079 | 16.3 | 33.1 | 41.7 | 27.4 | 0.253 |        
| MN    | 0.065 | 16.1 | 31.0 | 39.4 | 29.3 | 0.245 |
| T-HAE | 0.051 | 10.3 | 23.2 | 31.1 | 37.1 | 0.178 |

MRR = Mean Reciprocal Rank  
**Higher is better:** BLEU-4, R@1, R@5, R@10, MRR  
**Lower is better:** Mean Rank


## Rules for Sub-task #2 Submissions
* Disallowed Input: `belief_state`, `system_transcript`, `system_transcript_annotated`, `state_graph_1`, `state_graph_2`, and anything from future turns.
* If you would like to use any other external resources, please consult with the track organizers (simmc@fb.com). Generally, we allow the use of publicly available pre-trained language models, such as BERT, GPT-2, etc.

[simmc_arxiv]:https://arxiv.org/abs/2006.01460
