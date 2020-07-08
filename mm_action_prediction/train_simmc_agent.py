"""Train baselines for SIMMC dataset (furniture and fashion).

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import math
import time
import os
import torch

import loaders
import models
import options
import eval_simmc_agent as evaluation
from tools import support


# Arguments.
args = options.read_command_line()

# Dataloader.
dataloader_args = {
    "single_pass": False,
    "shuffle": True,
    "data_read_path": args["train_data_path"],
    "get_retrieval_candidates": False
}
dataloader_args.update(args)
train_loader = loaders.DataloaderSIMMC(dataloader_args)
args.update(train_loader.get_data_related_arguments())
# Initiate the loader for val (DEV) data split.
if args["eval_data_path"]:
    dataloader_args = {
        "single_pass": True,
        "shuffle": False,
        "data_read_path": args["eval_data_path"],
        "get_retrieval_candidates": args["retrieval_evaluation"]
    }
    dataloader_args.update(args)
    val_loader = loaders.DataloaderSIMMC(dataloader_args)
else:
    val_loader = None

# Model.
wizard = models.Assistant(args)
wizard.train()
if args["encoder"] == "tf_idf":
    wizard.encoder.IDF.data = train_loader._ship_helper(train_loader.IDF)

# Optimizer.
optimizer = torch.optim.Adam(wizard.parameters(), args["learning_rate"])

# Training iterations.
smoother = support.ExponentialSmoothing()
num_iters_per_epoch = train_loader.num_instances / args["batch_size"]
print("Number of iterations per epoch: {:.2f}".format(num_iters_per_epoch))
eval_dict = {}
best_epoch = -1

# first_batch = None
for iter_ind, batch in enumerate(train_loader.get_batch()):
    epoch = iter_ind / num_iters_per_epoch
    batch_loss = wizard(batch)
    batch_loss_items = {key: val.item() for key, val in batch_loss.items()}
    losses = smoother.report(batch_loss_items)

    # Optimization steps.
    optimizer.zero_grad()
    batch_loss["total"].backward()
    torch.nn.utils.clip_grad_value_(wizard.parameters(), 1.0)
    optimizer.step()

    if iter_ind % 50 == 0:
        cur_time = time.strftime("%a %d%b%y %X", time.gmtime())
        print_str = (
            "[{}][Ep: {:.2f}][It: {:d}][A: {:.2f}][Aa: {:.2f}]" "[L: {:.2f}][T: {:.2f}]"
        )
        print_args = (
            cur_time,
            epoch,
            iter_ind,
            losses["action"],
            losses["action_attr"],
            losses["token"],
            losses["total"],
        )
        print(print_str.format(*print_args))

    # Perform evaluation, every X number of epochs.
    if (
        val_loader
        and int(epoch) % args["eval_every_epoch"] == 0
        and (iter_ind == math.ceil(int(epoch) * num_iters_per_epoch))
    ):
        eval_dict[int(epoch)], eval_outputs = evaluation.evaluate_agent(
            wizard, val_loader, args
        )
        # Print the best epoch so far.
        best_epoch, best_epoch_dict = support.sort_eval_metrics(eval_dict)[0]
        print("\nBest Val Performance: Ep {}".format(best_epoch))
        for item in best_epoch_dict.items():
            print("\t{}: {:.2f}".format(*item))

    # Save the model every epoch.
    if (
        args["save_every_epoch"] > 0
        and int(epoch) % args["save_every_epoch"] == 0
        and (iter_ind == math.ceil(int(epoch) * num_iters_per_epoch))
    ):
        # Create the folder if it does not exist.
        os.makedirs(args["snapshot_path"], exist_ok=True)
        # If prudent, save only if best model.
        checkpoint_dict = {
            "model_state": wizard.state_dict(),
            "args": args,
            "epoch": best_epoch,
        }
        if args["save_prudently"]:
            if best_epoch == int(epoch):
                save_path = os.path.join(args["snapshot_path"], "epoch_best.tar")
                print("Saving the model: {}".format(save_path))
                torch.save(checkpoint_dict, save_path)
        else:
            save_path = os.path.join(
                args["snapshot_path"], "epoch_{}.tar".format(int(epoch))
            )
            print("Saving the model: {}".format(save_path))
            torch.save(checkpoint_dict, save_path)
        # Save the file with evaluation metrics.
        eval_file = os.path.join(args["snapshot_path"], "eval_metrics.json")
        with open(eval_file, "w") as file_id:
            json.dump(eval_dict, file_id)
    # Exit if number of epochs exceed.
    if epoch > args["num_epochs"]:
        break
