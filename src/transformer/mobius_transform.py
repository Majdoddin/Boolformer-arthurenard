import torch
from typing import Tuple

def mobius_transform(f: torch.Tensor) -> torch.Tensor:
    """
    Fast Möbius transform on boolean lattice using SOS DP.

    Computes μ(S) = sum over all subsets T of S: (-1)^|S\T| * f(T)

    Args:
        f: Function values for all 2^n subsets [2^n]

    Returns:
        μ: Möbius transform values [2^n]
    """
    n = int(torch.log2(torch.tensor(len(f))).item())
    mu = f.clone()

    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                mu[mask] -= mu[mask ^ (1 << i)]

    return mu

def select_dominant_coefficients(coeffs: torch.Tensor, threshold: float = 0.1,
                                min_count: int = 8, max_count: int = 64) -> torch.Tensor:
    """
    Select dominant Möbius coefficients and create scaled input tensor.

    Args:
        coeffs: All Möbius coefficients [2^n]
        threshold: Minimum normalized score threshold
        min_count: Minimum number of coefficients to include (including empty set)
        max_count: Maximum number of coefficients to include (including empty set)

    Returns:
        coeff_input: Tensor [K, n+1] with subset encodings and scaled coefficients
    """
    n_vars = int(torch.log2(torch.tensor(len(coeffs))).item())

    # Compute normalized scores for all non-empty subsets
    candidates = []
    for subset in range(1, len(coeffs)):
        subset_size = bin(subset).count('1')  # Count number of 1s = |S|
        normalized_score = abs(coeffs[subset].item()) / (2 ** (subset_size / 2))
        candidates.append((normalized_score, subset))

    # Sort candidates by score (descending)
    candidates.sort(reverse=True)

    # Filter: keep those above threshold, but at least min_count-1 (reserve 1 for empty set)
    min_needed = min_count - 1  # -1 for empty set
    selected_indices = [0]  # Always include empty set

    for i, (score, subset) in enumerate(candidates[:max_count-1]):
        if score > threshold or i < min_needed:
            selected_indices.append(subset)

    # Create final input tensor with subset encodings and scaled coefficients
    coeff_input = []

    for subset in selected_indices:
        # Convert subset index to binary encoding
        encoding = []
        size = 0
        for i in range(n_vars):
            if subset & (1 << i):
                encoding.append(1.0)  # Variable is in subset
                size += 1
            else:
                encoding.append(-1.0)  # Variable is not in subset

        # Scale coefficient by 2^{|S|-1}
        scaled_coeff = coeffs[subset].item()
        if size > 0:  # Don't scale empty set
            scaled_coeff /= (2 ** (size - 1))

        # Combine encoding with scaled coefficient
        row = encoding + [scaled_coeff]
        coeff_input.append(row)

    return torch.tensor(coeff_input, dtype=torch.float32)