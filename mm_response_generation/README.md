# DSTC Track 4: SIMMC | Sub-Task #2: Multimodal Assistant Response Generation

This directory contains the code and the scripts for running the baseline models for Sub-Task #2: Multimodal Assistant Response Generation.

This subtask measures the generation (or retrieval) of the assistant response given the dialog history, multimodal context, ground truth assistant API call and the current utterance.

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
| LSTM  | 0.094 | 4.0 | 11.0 | 17.1 | 46.5 | 0.093 |
| HAE   | 0.180 | 11.5 | 27.0 | 37.2 | 32.0 | 0.202 |
| HRE   | 0.202 | 11.8 | 28.3 | 38.3 | 31.1 | 0.208 |
| MN    | 0.207 | 13.1 | 29.4 | 39.2 | 30.3 | 0.221 |
| T-HAE | 0.129 | 8.7  | 21.0 | 30.0 | 35.1 | 0.160 |
 
 
 **SIMMC-Furniture**

| Model  |     BLEU-4     | R@1 | R@5 | R@10 | Mean Rank | MRR |
|----------| :-------------: | :------: | :------: | :------: | :------: |:------: |
| LSTM  | 0.074 | 5.3  | 11.3 | 16.6 | 46.9 | 0.102 |
| HAE   | 0.221 | 12.0 | 29.5 | 40.0 | 29.6 | 0.215 |
| HRE   | 0.218 | 11.7 | 29.6 | 39.2 | 29.9 | 0.212 |        
| MN    | 0.208 | 13.1 | 28.2 | 36.6 | 31.8 | 0.216 |
| T-HAE | 0.157 | 10.4 | 22.0 | 29.9 | 37.3 | 0.174 |

MRR = Mean Reciprocal Rank  
**Higher is better:** BLEU-4, R@1, R@5, R@10, MRR  
**Lower is better:** Mean Rank


## Rules for Sub-task #2 Submissions
* Disallowed Input: `belief_state`, `system_transcript`, `system_transcript_annotated`, `state_graph_1`, `state_graph_2`, and anything from future turns.
* If you would like to use any other external resources, please consult with the track organizers (simmc@fb.com). Generally, we allow the use of publicly available pre-trained language models, such as BERT, GPT-2, etc.

[simmc_arxiv]:https://arxiv.org/abs/2006.01460
