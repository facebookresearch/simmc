"""Script to read command line flags using ArgParser.

Author(s): Satwik Kottur
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import torch
from tools import support


def read_command_line():
    """Read and parse commandline arguments to run the program.

    Returns:
    parsed_args: Dictionary of parsed arguments.
    """
    title = "Train assistant model for furniture genie"
    parser = argparse.ArgumentParser(description=title)

    # Data input settings.
    parser.add_argument(
        "--train_data_path", required=True, help="Path to compiled training data"
    )
    parser.add_argument(
        "--eval_data_path", default=None, help="Path to compiled evaluation data"
    )
    parser.add_argument(
        "--snapshot_path", default="checkpoints/", help="Path to save checkpoints"
    )
    parser.add_argument(
        "--metainfo_path",
        default="data/furniture_metainfo.json",
        help="Path to file containing metainfo",
    )
    parser.add_argument(
        "--attr_vocab_path",
        default="data/attr_vocab_file.json",
        help="Path to attribute vocabulary file",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["furniture", "fashion"],
        help="Domain to train the model on",
    )
    # Asset embedding.
    parser.add_argument(
        "--asset_embed_path",
        default="data/furniture_asset_path.npy",
        help="Path to asset embeddings",
    )
    # Specify encoder/decoder flags.
    # Model hyperparameters.
    parser.add_argument(
        "--encoder",
        required=True,
        choices=[
            "history_agnostic",
            "history_aware",
            "pretrained_transformer",
            "hierarchical_recurrent",
            "memory_network",
            "tf_idf",
        ],
        help="Encoder type to use for text",
    )
    parser.add_argument(
        "--text_encoder",
        required=True,
        choices=["lstm", "transformer"],
        help="Encoder type to use for text",
    )
    parser.add_argument(
        "--word_embed_size", default=128, type=int, help="size of embedding for text"
    )
    parser.add_argument(
        "--hidden_size",
        default=128,
        type=int,
        help=(
            "Size of hidden state in LSTM/transformer."
            "Must be same as word_embed_size for transformer"
        ),
    )
    # Parameters for transformer text encoder.
    parser.add_argument(
        "--num_heads_transformer",
        default=-1,
        type=int,
        help="Number of heads in the transformer",
    )
    parser.add_argument(
        "--num_layers_transformer",
        default=-1,
        type=int,
        help="Number of layers in the transformer",
    )
    parser.add_argument(
        "--hidden_size_transformer",
        default=2048,
        type=int,
        help="Hidden Size within transformer",
    )
    parser.add_argument(
        "--num_layers", default=1, type=int, help="Number of layers in LSTM"
    )
    parser.add_argument(
        "--use_action_attention",
        dest="use_action_attention",
        action="store_true",
        default=False,
        help="Use attention over all encoder statesfor action",
    )
    parser.add_argument(
        "--use_action_output",
        dest="use_action_output",
        action="store_true",
        default=False,
        help="Model output of actions as decoder memory elements",
    )
    parser.add_argument(
        "--use_multimodal_state",
        dest="use_multimodal_state",
        action="store_true",
        default=False,
        help="Use multimodal state for action prediction (fashion)",
    )
    parser.add_argument(
        "--use_bahdanau_attention",
        dest="use_bahdanau_attention",
        action="store_true",
        default=False,
        help="Use bahdanau attention for decoder LSTM",
    )
    parser.add_argument(
        "--skip_retrieval_evaluation",
        dest="retrieval_evaluation",
        action="store_false",
        default=True,
        help="Evaluation response generation through retrieval"
    )
    parser.add_argument(
        "--skip_bleu_evaluation",
        dest="bleu_evaluation",
        action="store_false",
        default=True,
        help="Use beamsearch to evaluate BLEU score"
    )
    parser.add_argument(
        "--max_encoder_len",
        default=24,
        type=int,
        help="Maximum encoding length for sentences",
    )
    parser.add_argument(
        "--max_history_len",
        default=100,
        type=int,
        help="Maximum encoding length for history encoding",
    )
    parser.add_argument(
        "--max_decoder_len",
        default=26,
        type=int,
        help="Maximum decoding length for sentences",
    )
    parser.add_argument(
        "--max_rounds",
        default=30,
        type=int,
        help="Maximum number of rounds for the dialog",
    )
    parser.add_argument(
        "--share_embeddings",
        dest="share_embeddings",
        action="store_true",
        default=True,
        help="Encoder/decoder share emebddings",
    )

    # Optimization hyperparameters.
    parser.add_argument(
        "--batch_size",
        default=30,
        type=int,
        help="Training batch size (adjust based on GPU memory)",
    )
    parser.add_argument(
        "--learning_rate", default=1e-3, type=float, help="Learning rate for training"
    )
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout")
    parser.add_argument(
        "--num_epochs",
        default=20,
        type=int,
        help="Maximum number of epochs to run training",
    )
    parser.add_argument(
        "--eval_every_epoch",
        default=1,
        type=int,
        help="Number of epochs to evaluate every",
    )
    parser.add_argument(
        "--save_every_epoch",
        default=-1,
        type=int,
        help="Epochs to save the model every, -1 does not save",
    )
    parser.add_argument(
        "--save_prudently",
        dest="save_prudently",
        action="store_true",
        default=False,
        help="Save checkpoints prudently (only best models)",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=-1, help="GPU id to use, -1 for CPU"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    # For transformers, hidden size must be same as word_embed_size.
    if parsed_args["text_encoder"] == "transformer":
        assert (
            parsed_args["word_embed_size"] == parsed_args["hidden_size"]
        ), "hidden_size should be same as word_embed_size for transformer"
        if not parsed_args["use_bahdanau_attention"]:
            print("Bahdanau attention must be off!")
            parsed_args["use_bahdanau_attention"] = False

    # If action output is to be used for LSTM, bahdahnau attention must be on.
    if parsed_args["use_action_output"] and parsed_args["text_encoder"] == "lstm":
        assert parsed_args["use_bahdanau_attention"], (
            "Bahdanau attention " "must be on for action output to be used!"
        )
    # For tf_idf, ignore the action_output flag.
    if parsed_args["encoder"] == "tf_idf":
        parsed_args["use_action_output"] = False
    # Prudent save is not possible without evaluation.
    if parsed_args["save_prudently"]:
        assert parsed_args[
            "eval_data_path"
        ], "Prudent save needs a non-empty eval_data_path"

    # Set the cuda environment variable for the gpu to use and get context.
    parsed_args["use_gpu"] = support.setup_cuda_environment(parsed_args["gpu_id"])
    # Force cuda initialization
    # (otherwise results in weird race conditions in PyTorch 1.4).
    if parsed_args["use_gpu"]:
        _ = torch.Tensor([1.0]).cuda()
    support.pretty_print_dict(parsed_args)
    return parsed_args
