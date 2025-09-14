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

def dynamic_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for dynamic batch padding.

    Pads target sequences to the longest sequence in the batch instead of
    using fixed EXPR_SIZE_MAX padding.

    Args:
        batch: List of (input, target) tensor tuples
        pad_id: The ID of the PAD token for padding

    Returns:
        Tuple of (inputs, targets) with targets dynamically padded
    """
    inputs, targets = zip(*batch)

    # Stack inputs (already have consistent size from tokenize_regression_mode)
    inputs_batch = torch.stack(inputs)

    # Dynamically pad targets to longest sequence in batch
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_id)

    return inputs_batch, targets_padded 