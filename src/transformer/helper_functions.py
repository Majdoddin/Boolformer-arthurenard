import torch
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
from src.ConfigClasses import ConfigFormula

def find_specific_id(tensor: torch.Tensor, target_id: int) -> int | None:
    """
    Finds the position of a specific ID in a tensor. Returns None if the ID is not found.

    Args:
        tensor (torch.Tensor): The tensor to search.
        target_id (int): The ID to find in the tensor.

    Returns:
        int | None: The index of the target ID in the tensor, or None if not found.
    """
    for idx, value in enumerate(tensor):
        if value.item() == target_id:
            return idx
    return None

def get_number_of_candidates(config: ConfigFormula, tensor: torch.Tensor, pad_id: int) -> int:
    """
    Computes the number of candidates in a tensor, taking into account padding and noise configuration.

    Args:
        config (ConfigFormula): Configuration object containing relevant settings.
        tensor (torch.Tensor): The tensor containing candidates.
        pad_id (int): The ID representing padding in the tensor.

    Returns:
        int: The number of candidates.
    """
    result_size_adjustment = 1 if config.NOISY else 0

    # Find the first occurrence of the pad ID
    first_pad_index = find_specific_id(tensor, pad_id)

    if first_pad_index is None:
        return len(tensor) - result_size_adjustment

    return first_pad_index

def dynamic_encoder_collate_fn(batch, input_pad_id: int):
    """
    Custom collate function for dynamic padding of encoder inputs only.

    Args:
        batch: List of (input, target, less_freq_rslt) tuples
        input_pad_id: The ID of the PAD token for input vocabulary

    Returns:
        Tuple of (padded_inputs, targets, less_freq_results)
    """
    inputs, targets, less_freq_results = zip(*batch)

    # Dynamically pad encoder inputs using pad_sequence
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=input_pad_id)

    # Keep targets and less_freq_results unchanged
    targets_batch = torch.stack(targets, dim=0)

    return inputs_padded, targets_batch, less_freq_results