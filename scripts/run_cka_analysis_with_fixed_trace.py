import logging
import os
from itertools import combinations
from typing import Any

import hydra
import omegaconf
import pandas as pd
import safetensors
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from reasoning.cache import (
    get_subdirectories,
    parse_layer_name,
)
from reasoning.cka import cka, gram_linear, gram_rbf
from reasoning.generation import get_hidden_states, get_model_and_tokenizer

log = logging.getLogger(__name__)


def get_model_cfg(cfg: DictConfig, model_name: str) -> DictConfig:
    if source_model_keys := [
        model_key for model_key in cfg.source_model_selection if cfg.get(model_key).name == model_name
    ]:
        model_key = source_model_keys[0]
    elif other_model_keys := [
        model_key for model_key in cfg.comparison_model_selection if cfg.get(model_key).name == model_name
    ]:
        model_key = other_model_keys[0]
    else:
        raise KeyError(f"{model_name=} subconfig not found in cfg")
    return cfg.get(model_key)


def compare_representations(
    token_source_model_name: str,
    model1_name: str,
    model2_name: str,
    tokens: torch.Tensor,
    models: dict[str, Any],
    sample_id: str,
    device: str,
) -> list[dict[str, Any]]:
    results = []

    # 3b. Run models to get hidden_states
    h1 = get_hidden_states(models[model1_name], tokens)
    h2 = get_hidden_states(models[model2_name], tokens)

    if len(h1) != len(h2):
        log.warning(
            f"Different number of layers detected ({len(h1), len(h2)}). Matching naively 1-to-1. "
            "Late layers of the deeper model will be ignored."
        )

    # 4. Compute CKA for all layers
    for layer_idx in range(1, min(len(h1), len(h2))):  # we start with 1 to skip the embedding layer
        # these tensors have size (batch, tokens, dim). batch is 1 for us
        X = h1[layer_idx].to(device).squeeze(0).to(torch.float32)
        Y = h2[layer_idx].to(device).squeeze(0).to(torch.float32)

        # Linear CKA
        gram_x_linear = gram_linear(X)
        gram_y_linear = gram_linear(Y)
        linear_cka_score = cka(gram_x_linear, gram_y_linear)

        # RBF CKA. The CKA paper uses different threshold, including 0.8
        rbf_threshold = 0.8
        gram_x_rbf = gram_rbf(X, rbf_threshold)
        gram_y_rbf = gram_rbf(Y, rbf_threshold)
        rbf_cka_score = cka(gram_x_rbf, gram_y_rbf)

        # 5. Store results
        results.append(
            {
                "model_1": model1_name,
                "model_2": model2_name,
                "layer": layer_idx,
                "sample": sample_id,
                "token_origin": token_source_model_name,
                "cka_linear": linear_cka_score.item(),
                "cka_rbf": rbf_cka_score.item(),
            }
        )
    return results


# #############################################
# Main Analysis Function
# #############################################


@hydra.main(version_base=None, config_path="../conf", config_name="cka_same_input")
def main(cfg: DictConfig):
    print(omegaconf.OmegaConf.to_yaml(cfg))

    if cfg.device:
        device = cfg.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    cache_dir: str = cfg.cache_dir
    output_file: str = cfg.output_file

    results = []

    # 1. Discover models, layers, and token types from the cache structure
    source_model_names = get_subdirectories(cache_dir)
    if not source_model_names:
        log.error(f"No model directories found in '{cache_dir}'. Please check the path.")
        return

    log.info(f"Found source models: {source_model_names}. Filtering based on cfg.source_model_selection...")

    source_model_names = [
        name
        for name in source_model_names
        for model_cfg_key in cfg.source_model_selection
        if name == cfg.get(model_cfg_key, {}).get("name")
    ]
    log.info(f"Remaining source models: {source_model_names}.")

    comparison_model_names = (
        [cfg.get(model_cfg_key, {}).get("name") for model_cfg_key in cfg.comparison_model_selection]
        if cfg.comparison_model_selection
        else source_model_names
    )
    log.info(
        f"Comparison models: {comparison_model_names}. Assuming they all have the same tokenizer and the "
        "tokenizer is consistent with source models."
    )

    # Loading all models if specified
    if cfg.keep_models_in_mem:
        if len(cfg.model_devices) == 1:
            cfg.model_devices = [device] * len(comparison_model_names)
        else:
            if len(cfg.model_devices) != len(comparison_model_names) or None in cfg.model_devices:
                raise ValueError(
                    f"Trying to keep models in memory, but number of specified devices does not match. "
                    f"Needs to be 1 device or {len(comparison_model_names)} devices."
                )
        model_cfgs = {name: get_model_cfg(cfg, name) for name in comparison_model_names}
        models = {
            name: get_model_and_tokenizer(
                model_cfg.model_id, device=device, torch_dtype=hydra.utils.instantiate(model_cfg.dtype)
            )[0]
            for (name, model_cfg), device in zip(model_cfgs.items(), cfg.model_devices)
        }
    else:
        raise NotImplementedError("Need to load all models into memory.")

    # Use combinations_with_replacement to get all pairs (A,B), (A,C), (B,C), etc.
    model_pairs = list(combinations(comparison_model_names, 2))

    # Find all different generation tokens from an arbitrary layer cache. They are all the same.
    for model_name in source_model_names:
        model_path = os.path.join(cache_dir, model_name)
        samples = get_subdirectories(model_path)
        for sample_id in tqdm(samples):
            layer_dirs = sorted(get_subdirectories(os.path.join(model_path, sample_id)), key=parse_layer_name)

            # 3a. Load answer in tokens
            file_path = os.path.join(model_path, sample_id, layer_dirs[0], "hidden_states.safetensors")
            with safetensors.safe_open(file_path, "pt") as ts:
                tokens = ts.get_tensor("toks")

            # Comparing the model representations
            pbar_model_pairs = tqdm(model_pairs, desc="Comparing Model Pairs")
            for model1_name, model2_name in pbar_model_pairs:
                pbar_model_pairs.set_postfix_str(f"{model1_name} vs {model2_name}")

                # 3.1 Regenerate hidden_states for both models and compare them for the answers generated by model1
                results.extend(
                    compare_representations(
                        token_source_model_name=model_name,
                        model1_name=model1_name,
                        model2_name=model2_name,
                        tokens=tokens,
                        models=models,
                        sample_id=sample_id,
                        device=device,
                    )
                )

    # 4. Save final results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        log.info(f"Analysis complete. Results saved to '{output_file}'.")
        log.info(f"DataFrame head:\n{df.head()}")
    else:
        log.warning("No results were generated. Please check your cache directory and logs.")


if __name__ == "__main__":
    main()
