#!/usr/bin/env python3
"""
    Scripts for evaluating the GPT-2 DST model predictions.

    First, we parse the line-by-line stringified format into
    the structured DST output.

    We then run the main DST Evaluation script to get results.
"""
import argparse
import json
from gpt2_dst.utils.convert import parse_flattened_results_from_file
from utils.evaluate_dst import evaluate_from_flat_list


if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_target',
                        help='path for target, line-separated format (.txt)')
    parser.add_argument('--input_path_predicted',
                        help='path for model prediction output, line-separated format (.txt)')
    parser.add_argument('--output_path_report',
                        help='path for saving evaluation summary (.json)')

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report

    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_flattened_results_from_file(input_path_target)
    list_predicted = parse_flattened_results_from_file(input_path_predicted)

    # Evaluate
    report = evaluate_from_flat_list(list_target, list_predicted)

    # Save report
    with open(output_path_report, 'w') as f_out:
        json.dump(report, f_out)
