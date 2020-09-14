"""Script evaluates response generation using GT responses.

Author(s): Satwik Kottur
"""


from absl import app, flags
import json

import nltk
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "data_json_path", "data/furniture_train.json", "Data with gold responses"
)
flags.DEFINE_string(
    "model_response_path", None, "Responses generated by the model"
)


def normalize_sentence(sentence):
    """Normalize the sentences and tokenize.
    """
    return nltk.tokenize.word_tokenize(sentence.lower())


def evaluate_response_generation(gt_responses, model_responses):
    """Evaluates response generation using the raw data and model predictions.
    """
    gt_responses_pool = {
        ii["dialogue_idx"]: ii for ii in gt_responses["dialogue_data"]
    }
    bleu_scores = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    for model_datum in model_responses:
        dialog_id = model_datum["dialog_id"]
        for round_id, round_datum in enumerate(model_datum["predictions"]):
            response = round_datum["response"]
            gt_datum = gt_responses_pool[dialog_id]["dialogue"][round_id]
            gt_response = gt_datum["system_transcript"]

            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                [normalize_sentence(gt_response)],
                normalize_sentence(response),
                smoothing_function=chencherry.method1
            )
            bleu_scores.append(bleu_score)
    return np.mean(bleu_scores)


def main(_):
    print("Reading: {}".format(FLAGS.data_json_path))
    with open(FLAGS.data_json_path, "r") as file_id:
        gt_responses = json.load(file_id)
    print("Reading: {}".format(FLAGS.model_response_path))
    with open(FLAGS.model_response_path, "r") as file_id:
        model_responses = json.load(file_id)
    bleu_score = evaluate_response_generation(gt_responses, model_responses)
    print(bleu_score)


if __name__ == "__main__":
    app.run(main)
