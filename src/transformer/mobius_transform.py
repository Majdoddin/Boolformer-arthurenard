import torch
from typing import Tuple

def mobius_transform(f: torch.Tensor) -> torch.Tensor:
    """
    Fast Möbius transform on boolean lattice using SOS DP.

    Computes μ(S) = sum over all subsets T of S: (-1)^|S\\T| * f(T)

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

def select_dominant_coefficients(mu: torch.Tensor, evaluations: torch.Tensor, min_count: int = 8, max_count: int = 64, threshold: float = 0.1) -> torch.Tensor:
    """Select dominant Möbius coefficients and create input tensor.

    Args:
        mu: Möbius coefficients tensor of shape [2^n]
        evaluations: Truth table tensor of shape [2^n, n+1] where last column is function values
        min_count: Minimum number of coefficients to select (including empty set)
        max_count: Maximum number of coefficients to select
        threshold: Threshold for normalized magnitude filtering

    Returns:
        coeff_input: Tensor [K, n+1] with subset encodings and scaled coefficients
    """
    n = evaluations.shape[1] - 1  # Number of variables (excluding function value column)

    # Calculate normalized magnitudes |a_S| / 2^{|S|/2}
    subset_sizes = torch.tensor([bin(i).count('1') for i in range(len(mu))], dtype=torch.float32)
    normalized_magnitudes = torch.abs(mu) / (2.0 ** (subset_sizes / 2))

    # Sort by normalized magnitude (excluding empty set which is always included)
    sorted_indices = torch.argsort(normalized_magnitudes[1:], descending=True) + 1

    # Take top max_count candidates
    candidates = sorted_indices[:max_count-1]  # -1 for empty set

    # Filter by threshold and ensure minimum count
    above_threshold = normalized_magnitudes[candidates] >= threshold
    selected_candidates = candidates[above_threshold]

    # Ensure we have at least min_count-1 candidates (excluding empty set)
    if len(selected_candidates) < min_count - 1:
        selected_candidates = candidates[:min_count-1]

    # Always include empty set (index 0) at the beginning
    selected_indices = torch.cat([torch.tensor([0]), selected_candidates])

    # Scale coefficients by a_S / 2^{|S|-1}
    selected_sizes = subset_sizes[selected_indices]
    scaling_factors = 1.0 / (2.0 ** torch.clamp(selected_sizes - 1, min=0))  # Clamp to avoid division by 2^{-1}
    scaled_coefficients = mu[selected_indices] * scaling_factors

    # Extract encodings directly from truth table rows (convert 0.0→-1.0, 1.0→+1.0)
    encodings = 2 * evaluations[selected_indices, :-1] - 1

    # Combine encodings with scaled coefficients
    coeff_input = torch.cat([encodings, scaled_coefficients.unsqueeze(1)], dim=1)

    # TODO: Implement dynamic padding in DataLoader collate function
    # For now, pad to max_count to ensure consistent size across batches
    current_count = coeff_input.shape[0]
    if current_count < max_count:
        padding_rows = max_count - current_count
        padding = torch.zeros(padding_rows, coeff_input.shape[1], dtype=coeff_input.dtype)
        coeff_input = torch.cat([coeff_input, padding], dim=0)

    return coeff_input