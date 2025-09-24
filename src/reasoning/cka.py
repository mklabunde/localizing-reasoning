import torch

# #############################################
# CKA Implementation converted to pytorch
# (From https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb)
# Apache License 2.0
# #############################################


def gram_linear(x: torch.Tensor) -> torch.Tensor:
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
      x: A num_examples x num_features matrix of features.

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    return torch.matmul(x, x.T)


def gram_rbf(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
      x: A num_examples x num_features matrix of features.
      threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = torch.matmul(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0)
    sq_median_distance = torch.median(sq_distances).item()
    return torch.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram(gram: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Center a symmetric Gram matrix.

    This is equivalent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    # Convert to float64 and clone to match numpy's precision for these calculations
    # and to ensure the original input tensor is not modified.
    gram_centered = gram.to(torch.float64).clone()

    if not torch.allclose(gram_centered, gram_centered.T, atol=1e-4):
        torch.save(gram_centered, "asym_gram.pt")
        raise ValueError("Input must be a symmetric matrix.")

    n = gram_centered.shape[0]

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).

        # The unbiased estimator for HSIC requires n > 2.
        if n <= 2:
            raise ValueError(
                f"Unbiased HSIC estimator requires n > 2 examples. Got n = {n}."
                " For n <= 2, the biased estimator should typically be used."
            )

        gram_centered.fill_diagonal_(0)  # Set diagonal elements to 0 in-place

        means = torch.sum(gram_centered, dim=0) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))

        gram_centered -= means.unsqueeze(1)  # Subtract column vector
        gram_centered -= means.unsqueeze(0)  # Subtract row vector
        gram_centered.fill_diagonal_(0)  # Set diagonal to 0 again after centering
    else:
        means = torch.mean(gram_centered, 0)
        means -= torch.mean(means) / 2

        gram_centered -= means.unsqueeze(1)
        gram_centered -= means.unsqueeze(0)

    return gram_centered


def cka(gram_x: torch.Tensor, gram_y: torch.Tensor, debiased: bool = True) -> torch.Tensor:
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased. (range -1,1)

    Returns:
      The value of CKA between X and Y.
    """
    # center_gram handles dtype conversion and cloning internally.
    gram_x_centered = center_gram(gram_x, unbiased=debiased)
    gram_y_centered = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    # torch.flatten() is equivalent to numpy.ravel()
    # torch.dot performs dot product for 1D tensors.
    scaled_hsic = torch.dot(gram_x_centered.flatten(), gram_y_centered.flatten())

    # torch.linalg.norm computes the Frobenius norm by default for matrices.
    normalization_x = torch.linalg.norm(gram_x_centered)
    normalization_y = torch.linalg.norm(gram_y_centered)

    denominator = normalization_x * normalization_y

    # Handle cases where norms might be zero to avoid division by zero.
    # If both centered matrices are zero (e.g., all input features were constant),
    # CKA is typically considered 0 or undefined.
    if denominator == 0:
        return torch.tensor(0.0, device=gram_x.device, dtype=gram_x_centered.dtype)

    return scaled_hsic / denominator
