"""Collection of support tools.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import os
import nltk
import numpy as np


class ExponentialSmoothing:
    """Exponentially smooth and track losses.
    """

    def __init__(self):
        self.value = None
        self.blur = 0.95
        self.op = lambda x, y: self.blur * x + (1 - self.blur) * y

    def report(self, new_val):
        """Add a new score.

        Args:
            new_val: New value to record.
        """
        if self.value is None:
            self.value = new_val
        else:
            self.value = {
                key: self.op(value, new_val[key]) for key, value in self.value.items()
            }
        return self.value


def setup_cuda_environment(gpu_id):
    """Setup the GPU/CPU configuration for PyTorch.
    """
    if gpu_id < 0:
        print("Running on CPU...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return False
    else:
        print("Running on GPU {0}...".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return True


def pretty_print_dict(parsed):
    """Pretty print a parsed dictionary.
    """
    max_len = max(len(ii) for ii in parsed.keys())
    format_str = "\t{{:<{width}}}: {{}}".format(width=max_len)
    print("Arguments:")
    # Sort in alphabetical order and print.
    for key in sorted(parsed.keys()):
        print(format_str.format(key, parsed[key]))
    print("")


def print_distribution(counts, label=None):
    """Prints distribution for a given histogram of counts.

    Args:
        counts: Dictionary of count histograms
    """
    total_items = sum(counts.values())
    max_length = max(len(str(ii)) for ii in counts.keys())
    if label is not None:
        print(label)
    format_str = "\t{{:<{width}}} [{{:.0f}}%]: {{}}".format(width=max_length)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for key, val in sorted_counts:
        print(format_str.format(key, 100 * float(val) / total_items, val))


def sort_eval_metrics(eval_metrics):
    """Sort a dictionary of evaluation metrics.

    Args:
        eval_metrics: Dict of evaluation metrics.

    Returns:
        sorted_evals: Sorted evaluated metrics, best first.
    """
    # Sort based on 'perplexity' (lower is better).
    # sorted_evals = sorted(eval_metrics.items(), key=lambda x: x[1]['perplexity'])
    # return sorted_evals

    # Sort based on average %increase across all metrics (higher is better).
    def mean_relative_increase(arg1, arg2):
        _, metric1 = arg1
        _, metric2 = arg2
        rel_gain = []
        # higher_better is +1 if true and -1 if false.
        for higher_better, key in [
            (-1, "perplexity"),
            (1, "action_accuracy"),
            (1, "action_attribute"),
        ]:
            rel_gain.append(
                higher_better
                * (metric1[key] - metric2[key])
                / (metric1[key] + metric2[key] + 1e-5)
            )
        return np.mean(rel_gain)

    sorted_evals = sorted(
        eval_metrics.items(),
        key=functools.cmp_to_key(mean_relative_increase),
        reverse=True,
    )
    return sorted_evals


def extract_split_from_filename(file_name):
    """Extract the split from the filename.

    Args:
        file_name: JSON path to the split
    Return:
        split: Name of the split (train | dev | devtest | test)
    """
    for split in ("train", "devtest", "dev", "test"):
        if split in file_name.split('/')[-1]:
            return split
