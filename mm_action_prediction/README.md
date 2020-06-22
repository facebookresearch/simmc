# DSTC Track 4: SIMMC | Sub-Task #1: Multimodal Assistant API Prediction

This directory contains the code and the scripts for running the baseline models for Sub-Task #1: Multimodal Assistant API Prediction.

This subtask involves predicting the assistant actions through API calls along with the necessary arguments using dialog history, multimodal context, and the current user utterance as inputs.
For example, enquiring about an attribute value (e.g., price) for a shared furniture item is realized through a call to the *SpecifyInfo* API with the price argument.
A comprehensive set of APIs for our SIMMC dataset is given in the [paper][simmc_arxiv]. 

## Evaluation
Currently, we evaluate action prediction as a round-wise, multiclass classification problem over the set of APIs, and measure the *accuracy* of the most **dominant action**. 
In addition, we also use *action perplexity* (defined as the exponential of the mean loglikelihood of the dominant action) to allow situations where several actions are equally valid in a given context. We also measure the correctness of the predicted action (API) arguments using attribute *accuracy*.

**NOTE**: We plan to extend the Multimodal Assistant API Prediction from the most dominant assistant action to allow the prediction of a series of multiple actions per turn. Please follow the [**Latest News**](https://github.com/facebookresearch/simmc/#latest-news) section in the main README of the repository for updates.

For more details on the task definition and the baseline models we provide, please refer to our SIMMC paper:
```
@article{moon2020situated,
  title={Situated and Interactive Multimodal Conversations},
  author={Moon, Seungwhan and Kottur, Satwik and Crook, Paul A and De, Ankita and Poddar, Shivani and Levin, Theodore and Whitney, David and Difranco, Daniel and Beirami, Ahmad and Cho, Eunjoon and others},
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
**NOTE**: We recommend installation in a virtual environment ([user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)). Create a new virtual environment and activate it prior to installing the packages. 

* Install the required Python packages:
  * [Python 3.6+](https://www.python.org/downloads/)
  * [PyTorch 1.5+)](https://pytorch.org/get-started/locally/#start-locally)
  * [Transformers](https://huggingface.co/transformers/installation.html)

## Run Baselines
Will be updated soon, please follow the [**Latest News**](https://github.com/facebookresearch/simmc/#latest-news) section in the main README of the repository for updates.

## Rules for Sub-task #1 Submissions
* Disallowed Input: `belief_state`, `system_transcript`, `system_transcript_annotated`, `state_graph_1`, `state_graph_2`, and anything from future turns.
* If you would like to use any other external resources, please consult with the track organizers (simmc@fb.com). Generally, we allow the use of pre-trained language models such as BERT, GPT-2, etc.
