# DSTC Track 4: SIMMC | Sub-Task #2: Multimodal Assistant Response Generation

This directory contains the code and the scripts for running the baseline models for Sub-Task #2: Multimodal Assistant Response Generation.

This subtask measures the generation (or retrieval) of the assistant response given the dialog history, multimodal context, ground truth assistant API call and the current utterance.

## Evaluation
For generation, we use BLEU-4 score and for retrieval, we use recall@1, recall@5, recall@10, mean reciprocal rank (MRR), and mean rank.

**NOTE**: We plan to extend the evaluation of response generation conditioned on the model generated API calls. Please follow the [**Latest News**](https://github.com/facebookresearch/simmc/#latest-news) section in the main README of the repository for updates.

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
$ git clone https://github.com/facebookresearch/simmc.git
```

* Install the required Python packages:
  * [Python 3.6+](https://www.python.org/downloads/)
  * [PyTorch 1.5+](https://pytorch.org/get-started/locally/#start-locally)
  * [Transformers](https://huggingface.co/transformers/installation.html)

**NOTE**: We recommend installation in a virtual environment ([user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)). Create a new virtual environment and activate it prior to installing the packages. 

## Run Baselines
(Will be updated soon, please follow the **Latest News** section in the main README of the repository for updates.)

## Rules for Sub-task #2 Submissions
* Disallowed Input: `belief_state`, `system_transcript`, `system_transcript_annotated`, `state_graph_1`, `state_graph_2`, and anything from future turns.
* If you would like to use any other external resources, please consult with the track organizers (simmc@fb.com). Generally, we allow the use of pre-trained language models such as BERT, GPT-2, etc.

[simmc_arxiv]:https://arxiv.org/abs/2006.01460