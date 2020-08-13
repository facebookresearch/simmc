"""Evaluate SIMMC agent for Furniture and Fashion datasets.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import math
import torch
from tqdm import tqdm as progressbar

import loaders
import models
from tools import support


def main(args):
    """Evaluate model and save the results.
    """
    # Read the checkpoint and train args.
    print("Loading checkpoint: {}".format(args["checkpoint"]))
    checkpoint = torch.load(args["checkpoint"], map_location=torch.device("cpu"))
    saved_args = checkpoint["args"]
    saved_args.update(args)
    support.pretty_print_dict(saved_args)

    # Dataloader for evaluation.
    dataloader_args = {
        "single_pass": True,
        "shuffle": False,
        "data_read_path": args["eval_data_path"],
        "get_retrieval_candidates": True
    }
    dataloader_args.update(saved_args)
    val_loader = loaders.DataloaderSIMMC(dataloader_args)
    saved_args.update(val_loader.get_data_related_arguments())

    # Model.
    wizard = models.Assistant(saved_args)
    # Load the checkpoint.
    wizard.load_state_dict(checkpoint["model_state"])

    # Evaluate the SIMMC model.
    eval_dict, eval_outputs = evaluate_agent(wizard, val_loader, saved_args)
    save_path = saved_args["checkpoint"].replace(".tar", "_eval.json")
    print("Saving results: {}".format(save_path))
    with open(save_path, "w") as file_id:
        json.dump(eval_dict, file_id)


def evaluate_agent(wizard, val_loader, args):
    """Evaluate a SIMMC agent given a dataloader.

    Args:
        wizard: SIMMC model
        dataloader: Dataloader to use to run the model on
        args: Arguments for evaluation
    """
    total_iters = int(val_loader.num_instances / args["batch_size"])
    # Turn autograd off for evaluation -- light-weight and faster.
    with torch.no_grad():
        wizard.eval()
        matches = []
        for batch in progressbar(val_loader.get_batch(), total=int(total_iters)):
            if args["bleu_evaluation"]:
                mode = {"next_token": "ARGMAX", "beam_size": 5}
            else:
                mode = None
            batch_outputs = wizard(batch, mode)
            # Stringify model responses.
            if args["bleu_evaluation"]:
                batch_outputs["model_response"] = (
                    val_loader.stringify_beam_outputs(
                        batch_outputs["beam_output"], batch
                    )
                )
                # Remove beam output to avoid memory issues.
                del batch_outputs["beam_output"]
            matches.append(batch_outputs)
    wizard.train()

    # Compute perplexity.
    total_loss_sum = sum(ii["loss_sum"].item() for ii in matches)
    num_tokens = sum(ii["num_tokens"].item() for ii in matches)
    avg_loss_eval = total_loss_sum / num_tokens

    # Compute BLEU score.
    if args["bleu_evaluation"]:
        model_responses = [jj for ii in matches for jj in ii["model_response"]]
        bleu_score = val_loader.evaluate_response_generation(model_responses)
    else:
        model_responses = None
        bleu_score = -1.

    # Evaluate retrieval score.
    if args["retrieval_evaluation"]:
        candidate_scores = [jj for ii in matches for jj in ii["candidate_scores"]]
        retrieval_metrics = val_loader.evaluate_response_retrieval(candidate_scores)
        print(retrieval_metrics)
    else:
        retrieval_metrics = {}

    # Evaluate action prediction.
    action_predictions = [jj for ii in matches for jj in ii["action_preds"]]
    action_metrics = val_loader.evaluate_action_prediction(action_predictions)
    print(action_metrics["confusion_matrix"])
    print_str = (
        "\nEvaluation\n\tLoss: {:.2f}\n\t"
        "Perplexity: {:.2f}\n\tBLEU: {:.3f}\n\t"
        "Action: {:.2f}\n\t"
        "Action Perplexity: {:.2f}\n\t"
        "Action Attribute Accuracy: {:.2f}"
    )
    print(
        print_str.format(
            avg_loss_eval,
            math.exp(avg_loss_eval),
            bleu_score,
            100 * action_metrics["action_accuracy"],
            action_metrics["action_perplexity"],
            100 * action_metrics["attribute_accuracy"]
        )
    )
    # Save the results to a file.
    eval_dict = {
        "loss": avg_loss_eval,
        "perplexity": math.exp(avg_loss_eval),
        "bleu": bleu_score,
        "action_accuracy": action_metrics["action_accuracy"],
        "action_perplexity": action_metrics["action_perplexity"],
        "action_attribute": action_metrics["attribute_accuracy"]
    }
    eval_dict.update(retrieval_metrics)
    eval_outputs = {
        "model_actions": action_predictions,
        "model_responses": model_responses
    }
    return eval_dict, eval_outputs


if __name__ == "__main__":
    # Read command line options.
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to load")
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument(
        "--eval_data_path", required=True, help="Evaluation data split"
    )
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument(
        "--skip_bleu_evaluation",
        dest="bleu_evaluation",
        action="store_false",
        default=True,
        help="Use beamsearch to compute BLEU score when evaluation"
    )
    parser.add_argument(
        "--skip_retrieval_evaluation",
        dest="retrieval_evaluation",
        action="store_false",
        default=True,
        help="Evaluation response generation through retrieval"
    )
    parser.add_argument(
        "--domain",
        default=None,
        choices=["furniture", "fashion"],
        help="Domain to train the model on",
    )
    try:
        args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    # Setup CUDA environment.
    args["use_gpu"] = support.setup_cuda_environment(args["gpu_id"])

    main(args)
