# DSTC Track 4: SIMMC | Sub-Task #3: Multimodal Dialog State Tracking (MM-DST)

This directory contains the code and the scripts for running the baseline models for Sub-Task #3: Multimodal DST.

The Multimodal Dialog State Tracking (MM-DST) task involves systematically tracking the attributes of dialog act labels cumulative across multiple turns.
Multimodal belief states at each turn should encode sufficient information for handling user utterances in the downstream dialog components (e.g. Dialog Policy).

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

### Baseline: GPT-2 Based DST

1. **Preprocess** the datasets to reformat the data for GPT-2 input.

```
$ cd mm_dst
$ ./run_preprocess_gpt2.sh
```

The shell script above repeats the following for all {train|dev|devtest} splits and both {furniture|fashion} domains.

```
$ python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json={path_dir}/data/simmc-fashion/fashion_train_dials.json \
    --output_path_predict={path_dir}/mm_dst/gpt2_dst/data/fashion/fashion_train_dials_predict.txt \
    --output_path_target={path_dir}/mm_dst/gpt2_dst/data/fashion/fashion_train_dials_target.txt \
    --output_path_special_tokens={path_dir}/mm_dst/gpt2_dst/data/fashion/special_tokens.json
    --len_context=2 \
    --use_multimodal_contexts=1 \
```

2. **Train** the baseline model

```
$ ./run_train_gpt2.sh
```

The shell script above repeats the following for both {furniture|fashion} domains.

```
$ python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir={path_dir}/save/fashion \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens={path_dir}/mm_dst/gpt2_dst/data/fashion/special_tokens.json \
    --do_train \
    --train_data_file={path_dir}/mm_dst/gpt2_dst/data/fashion/fashion_train_dials_target.txt \
    --do_eval \
    --eval_data_file={path_dir}/mm_dst/gpt2_dst/data/fashion/fashion_dev_dials_target.txt \
    --num_train_epochs=1 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    #--no_cuda

```

3. **Generate** prediction for `devtest` data

```
$ ./run_generate_gpt2.sh
```

The shell script above repeats the following for both {furniture|fashion} domains.
```
$ python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path={path_dir}/mm_dst/gpt2_dst/save/furniture/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file={path_dir}/mm_dst/gpt2_dst/data/furniture/furniture_devtest_dials_predict.txt \
    --path_output={path_dir}/mm_dst/gpt2_dst/results/furniture/furniture_devtest_dials_predicted.txt
```

Here is an example output:
```
System : Yes, here's another one you might like. User : Oh yeah I think my niece would really like that. Does it come in any other colors?  System : I'm sorry I don't have that information. User : Ah well. I like this color. I'd like to go ahead and buy it. Can you add it to my cart please?  => Belief State :
 DA:INFORM:PREFER:JACKET  [ fashion-O_2  = obj ] DA:REQUEST:ADD_TO_CART:JACKET  [ fashion-O_2  = obj ] <EOB>  Of course, you now have this
```

The generation results are saved in the `/mm_dst/results` folder. Change the `path_output` to a desired path accordingly.


4. **Evaluate** predictions for `devtest` data

```
$ ./run_evaluate_gpt2.sh
```

The shell script above repeats the following for both {furniture|fashion} domains.
```
python -m gpt2_dst.scripts.evaluate \
    --input_path_target={path_dir}/mm_dst/gpt2_dst/data/furniture/furniture_devtest_dials_target.txt \
    --input_path_predicted={path_dir}/mm_dst/gpt2_dst/results/furniture/furniture_devtest_dials_predicted.txt \
    --output_path_report={path_dir}/mm_dst/gpt2_dst/results/furniture/furniture_devtest_dials_report.json

```

Evaluation reports are saved in the `/mm_dst/results` folder as JSON files.

*Important*: For any of the models you build, please make sure that you use the function `simmc.mm_dst.utils.evaluate_dst.evaluate_from_flat_list` to obtain the evaluation reports.

Please also note that the GPT2 fine-tuning is highly sensitive to the batch size (which `n_gpu` of your machine may affect), hence it may need some hyperparameter tuning to obtain the best results (and avoid over/under fitting). Please feel free to change the hyperparameter of the default settings (provided) to compare results.

Below is the summary of the [published models](https://github.com/facebookresearch/simmc/releases/download/1.0/mm_dst_gpt2_baselines.tar.gz) we provide:

| Baseline | Dialog Act F1 | Slot F1 |
|--------|-------|-------|
| GPT2 - Furniture (text-only) | 69.9 | 52.5 |
| GPT2 - Furniture (multimodal) | 69.5 | 63.9 |
| GPT2 - Fashion (text-only) | 61.2 | 52.1 |
| GPT2 - Fashion (multimodal) | 61.1 | 60.6 |

## Rules for Sub-task #3 Submissions
* Disallowed input per each turn: `belief_state`, `system_transcript`, `system_transcript_annotated`, `state_graph_1`, `state_graph_2`, and anything from future turns.
* If you would like to use any other external resources, please consult with the track organizers (simmc@fb.com). Generally, we allow the use of publicly available pre-trained language models, such as BERT, GPT-2, etc.

[dstc9]:https://sites.google.com/dstc.community/dstc9/home
[simmc_arxiv]:https://arxiv.org/abs/2006.01460
