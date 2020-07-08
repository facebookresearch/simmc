"""Utilities to run dynamic RNN.

Adapted from: VisDial-RL PyTorch Codebase
Author: Satwik Kottur
"""

import enum

import torch
import torch.nn as nn


# Enums to denote the output type.
class OutputForm(enum.Enum):
    ALL = 1
    ALL_CONCISE = 2
    LAST = 3
    PACKED = 4
    NONE = 5


def get_sorted_order(lengths):
    """Sorts based on the lengths.

  Args:
    lengths: Lengths to perform sorting on.

  Returns:
    sorted_len: Lengths sorted according to descending order.
    fwd_order: Forward order of the sorting.
    bwd_order: Backward order of the sorting.
  """
    sorted_len, fwd_order = torch.sort(lengths, dim=0, descending=True)
    _, bwd_order = torch.sort(fwd_order)
    return sorted_len, fwd_order, bwd_order


def rearrange(rearrange_order, dim=0, *inputs):
    """Rearrages input tensors based on an order and along a given dimension.

  Args:
    rearrange_order: Order to use while rearranging.
    dim: Dimension along which to rearrange.
    *inputs: List of input tensors.
  """
    rearranged_inputs = []
    for input_tensor in inputs:
        assert (
            input_tensor.shape[dim] == rearrange_order.shape[0]
        ), "Rearrange " "along dim {0} is incompatible!".format(dim)
        rearranged_inputs.append(input_tensor.index_select(dim, rearrange_order))
    return tuple(rearranged_inputs)


def dynamic_rnn(
    rnn_model,
    seq_input,
    seq_len,
    init_state=None,
    return_states=False,
    return_output=OutputForm.ALL,
):
    """
    Inputs:
        rnnModel     : Any torch.nn RNN model
        seqInput     : (batchSize, maxSequenceLength, embedSize)
                        Input sequence tensor (padded) for RNN model
        seqLens      : batchSize length torch.LongTensor or numpy array
        initialState : Initial (hidden, cell) states of RNN
        return_output: LAST time step or ALL time steps (ALL has ALL_CONCISE).

    Output:
        A single tensor of shape (batchSize, rnnHiddenSize) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence. If returnStates is True, also return a tuple of hidden
        and cell states at every layer of size (num_layers, batchSize,
        rnnHiddenSize)
    """

    # Perform the sorting operation to ensure sequences are in decreasing order
    # of lengths, as required by PyTorch packed sequence.
    sorted_len, fwd_order, bwd_order = get_sorted_order(seq_len)
    sorted_seq_input = seq_input.index_select(0, fwd_order)
    packed_seq_in = nn.utils.rnn.pack_padded_sequence(
        sorted_seq_input, lengths=sorted_len, batch_first=True
    )

    # If initial state is given, re-arrange according to the initial sorting.
    if init_state is not None:
        sorted_init_state = [ii.index_select(1, fwd_order) for ii in init_state]
        # Check for number of layers match.
        assert (
            sorted_init_state[0].size(0) == rnn_model.num_layers
        ), "Number of hidden layers do not match in dynamic rnn!"
    else:
        sorted_init_state = None
    output, (h_n, c_n) = rnn_model(packed_seq_in, sorted_init_state)

    # Undo the sorting operation.
    if return_output == OutputForm.ALL:
        max_seq_len = seq_input.shape[1]
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=max_seq_len
        )
        rnn_output = output.index_select(0, bwd_order)
    elif return_output == OutputForm.ALL_CONCISE:
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        rnn_output = output.index_select(0, bwd_order)
    elif return_output == OutputForm.LAST:
        rnn_output = h_n[-1].index_select(0, bwd_order)
    elif return_output == OutputForm.NONE:
        rnn_output = None
    elif return_output == OutputForm.PACKED:
        raise NotImplementedError
    else:
        raise TypeError("Only LAST and ALL are supported in dynamic_rnn!")

    if return_states:
        h_n = h_n.index_select(1, bwd_order)
        c_n = c_n.index_select(1, bwd_order)
        return rnn_output, (h_n, c_n)
    else:
        return rnn_output
