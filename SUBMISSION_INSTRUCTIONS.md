# Final Evaluation

Below we describe how the participants can submit their results, and how the winner(s) will be announced.

## Evaluation Dataset

Final evaluation for the SIMMC DSTC9 track will be on the `test-std` split, different from the `devtest` split. Each test instance in `test-std` contains only `K` number of rounds (not necessarily the entire dialog), where we release the user utterances from `1` to `K` rounds, and system utterances from `1` to `K-1` utterances. Please refer to [this table](./TASK_INPUTS.md) that lists the set of allowed inputs for each subtask.

For subtask 1, evaluation is on the assistant action (API call) for `K`th round.
For subtask 2, evaluation is on the assistant utterance generation for `K`th round.
For subtask 3, evaluation is on dialog state prediction based on user utterances from `1` through `K`.

For subtasks 1 and 2 there are 1.2K predictions (1 per dialogue). For subtask 3 there are mean(`K`) * number of dialogues predictions.

We provide:

* **`devtest`, in the `test-std` format**: to give participants an early heads-up on how the `test-std` dataset will look like, we re-formatted the already-released `devtest` set in the format of the soon-to-be-released `test-std` file. Please ensure that your script and model can run on [fashion_devtest_dials_teststd_format_public.json](./data/simmc_fashion/fashion_devtest_dials_teststd_format_public.json) and [furniture_devtest_dials_teststd_format_public.json](./data/simmc_furniture/furniture_devtest_dials_teststd_format_public.json).

* **`test-std`**: <Download link will be available on Sept 28>.


## Evaluation Criteria

| **Subtask** | **Evaluation** | **Metric Priority List** |
| :-- | :-- | :-- |
| Subtask 1 (Multimodal Assistant API Prediction) | On assistant action (API call) for `K`th round | Action Accuracy, Attribute Accuracy,  Action Perplexity |
| Subtask 2 (Multimodal Assistant Response Generation) | On assistant utterance generation for `K`th round | * Generative category: BLEU-4 <br> * Retrieval category: MRR, R@1, R@5, R@10, Mean Rank |
| Subtask 3 (Multimodal Dialog State Tracking) | On dialog state based on user utterances from 1 through `K` | Slot F1, Intent F1 |

**Separate winners** will be announced for each subtask based on the respective performance, with the exception of subtask 2 (response generation) that will have two winners based on two categories -- generative metrics and retrieval metrics.

Rules to select the winner for each subtask (and categories) are given below:

* For each subtask, we enforce a **priority over the respective metrics** (shown above) to highlight the model behavior desired by this challenge

* The entry with the most favorable (higher or lower) performance on the metric will be labelled as a winner candidate. Further, all other entries within one standard error of this candidate’s performance will also be considered as candidates. If there are more than one candidate according to the metric, we will move to the next metric in the priority list and repeat this process until we have a single winner candidate, which would be declared as the "**subtask winner**".

* In case of multiple candidates even after running through the list of metrics in the priority order, all of them will be declared as "**joint subtask winners**".

**NOTE**: Only entries that are able to open-sourced their code will be considered for the final evaluation. In all other cases, we can only give “honorable mentions” depending on the devtest performance and cannot declare them as winners of any subtask.


## Submission Format

Participants must submit the model prediction results in JSON format that can be scored with the automatic scripts provided for that sub-task. Specifically, please name your JSON output as follows (format for subtask1 and 2 is given in the respective READMEs):

```
<Subtask 1>
dstc9-simmc-teststd-{domain}-subtask-1.json

<Subtask 2>
dstc9-simmc-teststd-{domain}-subtask-2-generation.json
dstc9-simmc-teststd-{domain}-subtask-2-retrieval.json

<Subtask 3>
dstc9-simmc-teststd-{domain}-subtask-3.txt (line-separated output)
or
dstc9-simmc-teststd-{domain}-subtask-3.json (JSON format)
```

The SIMMC organizers will then evaluate them internally using the following scripts:

```
<Subtask 1>
python tools/action_evaluation.py \
    --action_json_path={PATH_TO_API_CALLS} \
    --model_output_path={PATH_TO_MODEL_PREDICTIONS} \
    --single_round_evaluation

<Subtask 2 Generation>
python tools/response_evaluation.py \
    --data_json_path={PATH_TO_GOLD_RESPONSES} \
    --model_response_path={PATH_TO_MODEL_RESPONSES} \
    --single_round_evaluation

<Subtask 2 Retrieval>
python tools/retrieval_evaluation.py \
    --retrieval_json_path={PATH_TO_GROUNDTRUTH_RETRIEVAL} \
    --model_score_path={PATH_TO_MODEL_CANDIDATE_SCORES} \
    --single_round_evaluation

<Subtask 3>
(line-by-line evaluation)
python -m gpt2_dst.scripts.evaluate \
  --input_path_target={PATH_TO_GROUNDTRUTH_TARGET} \
  --input_path_predicted={PATH_TO_MODEL_PREDICTIONS} \
  --output_path_report={PATH_TO_REPORT}

(Or, dialog level evaluation)
python -m utils.evaluate_dst \
    --input_path_target={PATH_TO_GROUNDTRUTH_TARGET} \
    --input_path_predicted={PATH_TO_MODEL_PREDICTIONS} \
    --output_path_report={PATH_TO_REPORT}
```

## Submission Instructions and Timeline

<table>
  <tbody>
    <tr>
      <td rowspan=4><ins>Before</ins> Sept 28th 2020</td>
      <td rowspan=4>Each Team</td>
      <td>Each participating team should create a repository, e.g. in github.com, that can be made public under a permissive open source license (MIT License preferred). Repository doesn’t need to be publicly viewable at that time.</td>
    </tr>
    <tr>
      <td>Before Sept 28th <a href='https://git-scm.com/book/en/v2/Git-Basics-Tagging'>tag a repository commit</a> that contains both runable code and model parameter files that are the team’s entries for all sub-tasks attempted.</td>
    </tr>
    <tr>
      <td>Tag commit with `dstc9-simmc-entry`.</td>
    </tr>
    <tr>
      <td>Models (model parameter files) and code should have associated date-time stamps which are before Sept 27 23:59:59 anywhere on Earth.</td>
    </tr>
    <tr>
      <td>Sept 28th 2020</td>
      <td>SIMMC Organizers</td>
      <td>Test-Std data released (during US Pacific coast working hours).</td>
    </tr>
    <tr>
      <td rowspan=2><ins>Before</ins> Oct 5th 2020</td>
      <td rowspan=2>Each Team</td>
      <td>Generate test data predictions using the code & model versions tagged previously with `dstc9-simmc-entry`.</td>
    </tr>
    <tr>
      <td>For each sub-task attempted, create a PR and check-in to the team’s repository where:
        <ul>
          <li>The PR/check-in contains an output directory with the model output in JSON format that can be scored with the automatic scripts provided for that sub-task.</li>
          <li>The PR comments contain a short technical summary of model and a copy-paste of the results of running the automatic test script for that sub-task.</li>
          <li>Tag the commit with `dstc9-simmc-test-subtask-{N}`; where `{N}` is the sub-task number.</li>
        </ul>
      </td>
    </tr>
    <tr>    
      <td rowspan=2>By Oct 5th 2020</td>
      <td rowspan=2>Each Team</td>
      <td>Make the team repository public under a permissive Open Source license (MIT license is prefered).</td>
    </tr>
    <tr>
      <td>Email the SIMMC Organizers a link to the repository at simmc@fb.com</td>
    </tr>
    <tr>
      <td>Oct 5th - Oct 12th 2020</td>
      <td>SIMMC Organizers</td>
      <td>SIMMC organizers to validate sub-task results.</td>
    </tr>
    <tr>
      <td>Oct 12th 2020</td>
      <td>SIMMC Organizers</td>
      <td>Publish anonymized team rankings on the SIMMC track github and email each team their anonymized team identity.</td>
    </tr>
    <tr>
      <td>Post Oct 12th 2020</td>
      <td>SIMMC Organizers</td>
      <td>Our plan is to write up a challenge summary paper. In this we may conduct error analysis of the results and may look to extend, e.g. possibly with human scoring, the submitted results.</td>
    </tr>
  </tbody>
</table>
