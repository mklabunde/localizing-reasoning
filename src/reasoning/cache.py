import logging
import os
from pathlib import Path

import safetensors
import torch
from safetensors.torch import load_file

log = logging.getLogger(__name__)

# #############################################
# Data Loading Helpers
# #############################################


def get_subdirectories(path: str) -> list[str]:
    """Returns a sorted list of subdirectories in a given path."""
    if not os.path.exists(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def parse_layer_name(layer_dir):
    """Extracts integer from layer directory name like 'layer_6'."""
    return int(layer_dir.replace("layer_", ""))


def get_token_types(file_path):
    """Inspects a safetensors file to get available token types (keys)."""
    tensors = load_file(file_path)
    # Filter out keys that are not hidden states (like 'toks')
    return sorted([k for k in tensors.keys() if "tok" not in k])


def load_representations(model_path, layer_dir, token_type, common_samples, device):
    """
    Loads all representations for a given model, layer, and token type
    across all common samples.
    """
    all_vectors = []
    for sample_id in common_samples:
        file_path = os.path.join(model_path, sample_id, layer_dir, "hidden_states.safetensors")
        if not os.path.exists(file_path):
            log.warning(f"File not found, skipping: {file_path}")
            return None  # Indicates a sample was missing
        try:
            tensors = load_file(file_path, device=device)
            # Ensure the key exists before trying to access it
            if token_type in tensors:
                all_vectors.append(tensors[token_type])
            else:
                log.warning(f"Token type '{token_type}' not found in {file_path}")
                return None
        except Exception as e:
            log.error(f"Error loading {file_path}: {e}")
            return None

    if not all_vectors:
        return None

    # Stack into a single matrix of shape (num_samples, hidden_dim)
    return torch.stack(all_vectors).to(torch.float64)


def all_vecs_identical(x: torch.Tensor) -> bool:
    if x.size(0) <= 1:
        return True

    # Compare all rows to the first row
    first_row = x[0]
    return torch.all(torch.all(x == first_row, dim=1)).item()  # type:ignore


def load_tokens(cache_sample_path: str) -> torch.Tensor:
    """`cache_sample_path` must point to a directory per sample, e.g., modelName/sample_12"""
    cache_dir = Path(cache_sample_path)
    layer_dir = [p for p in cache_dir.iterdir() if p.name.startswith("layer")][0]
    with safetensors.safe_open(layer_dir / "hidden_states.safetensors", "pt") as ts:
        tokens = ts.get_tensor("toks")
    return tokens
