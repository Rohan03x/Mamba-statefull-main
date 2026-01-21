"""
Correlation matrix utilities
"""
import numpy as np


def build_correlation_matrix(n_params: int,
                             correlation_values: np.ndarray) -> np.ndarray:
    """
    Build correlation matrix from flat array of correlation values

    Args:
        n_params: Number of parameters
        correlation_values: Flat array of correlation values

    Returns:
        Correlation matrix
    """
    corr_matrix = np.eye(n_params)
    idx = 0

    for i in range(n_params):
        for j in range(i+1, n_params):
            corr_matrix[i, j] = correlation_values[idx]
            corr_matrix[j, i] = correlation_values[idx]
            idx += 1

    return corr_matrix


def ensure_valid_correlation_matrix(
        corr_matrix: np.ndarray,
        epsilon: float = 1e-6) -> np.ndarray:
    """
    Ensure correlation matrix is valid (positive semidefinite)

    Args:
        corr_matrix: Correlation matrix
        epsilon: Small value to add to diagonal if needed

    Returns:
        Valid correlation matrix
    """
    try:
        # Add small diagonal to ensure positive definiteness if needed
        min_eig = np.min(np.linalg.eigvals(corr_matrix))

        if min_eig < 0:
            # Add small value to diagonal to make positive definite
            adjustment = abs(min_eig) + epsilon
            corr_matrix += np.eye(len(corr_matrix)) * adjustment

            # Normalize to ensure diagonal is 1
            d = np.sqrt(np.diag(np.diag(corr_matrix)))
            corr_matrix = np.linalg.inv(d) @ corr_matrix @ np.linalg.inv(d)
    except (np.linalg.LinAlgError, ValueError):
        # If eigenvalue decomposition fails, add small perturbation
        corr_matrix += np.eye(len(corr_matrix)) * epsilon

    return corr_matrix
