import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch
from loguru import logger
from scipy.stats import pearsonr
from tqdm import tqdm

from reasoning.cache import load_tokens
from reasoning.cka import cka, gram_linear
from reasoning.generation import get_hidden_states, get_model_and_tokenizer


def pca_svd(data):
    """
    Computes principal components and explained variance using the SVD method.

    Args:
        data (torch.Tensor): A tensor of shape (N, D) where N is the number of samples
                            and D is the number of features.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The principal components (right singular vectors), shape (D, D).
            - torch.Tensor: The explained variance for each component.
    """
    # 1. Center the data
    mean = torch.mean(data, dim=0)
    centered_data = data - mean

    # 2. Perform SVD
    # torch.linalg.svd returns U, S, and Vh (V transpose)
    # The singular values are returned in descending order. [3]
    _, S, Vh = torch.linalg.svd(centered_data)

    # The principal components are the right singular vectors (V), which is Vh.T
    principal_components = Vh.T

    # 3. Calculate explained variance
    # The eigenvalues of the covariance matrix are the square of the singular values
    # divided by (n_samples - 1).
    eigenvalues = (S**2) / (data.shape[0] - 1)
    total_variance = torch.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance

    return principal_components, explained_variance


@torch.no_grad()
def pc_projection(data: torch.Tensor, principal_components: torch.Tensor, k: int):
    if k < 1:
        raise ValueError(f"Need to project at least one dimension, but {k=}.")
    return data @ principal_components[:, :k]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run CKA analysis between reasoning models")

    # Device settings
    parser.add_argument("--cka-device", default="cuda:7", help="Device for CKA computation")
    parser.add_argument("--device1", default="cuda:5", help="Device for first model")
    parser.add_argument("--device2", default="cuda:5", help="Device for second model")

    # Experiment parameters
    parser.add_argument("--k", type=int, default=10, help="Number of top PC components to remove")
    parser.add_argument(
        "--output-dir", default="/ceph-ssd/reasoning/outputs/cka_in_depth", help="Output directory for results"
    )
    parser.add_argument("--sample-start", type=int, default=0, help="Start sample ID")
    parser.add_argument("--sample-end", type=int, default=50, help="End sample ID")

    # Model override options
    parser.add_argument("--token-origin", help="Token origin model name", default="Qwen2.5-Math-7B")
    parser.add_argument("--descendant", help="HuggingFace model path")
    parser.add_argument("--base-model", help="HuggingFace base model path", default="Qwen/Qwen2.5-Math-7B")

    return parser.parse_args()


def massive_activation_plots(
    activation: torch.Tensor,
    tokenizer,
    tokens: torch.Tensor,
    layer_idx: int | None = None,
    k: int = 3,
    color: Literal["royalblue", "orange"] = "royalblue",
):
    X = activation.to("cpu").squeeze(0).to(torch.float32)

    # original paper talked about activations in specific dimensions as massive activation
    # but we should more or less find the right tokens when we look by norm as the massive activations blow up the norm
    topk = torch.linalg.norm(X, dim=1).topk(k)

    fig = plt.figure(figsize=plt.figaspect(1 / k))
    for i, idx in enumerate(topk.indices):
        ax = fig.add_subplot(1, k, i + 1, projection="3d")
        surrounding_tokens = torch.tensor(
            [idx.item() + i for i in range(-4, 4) if 0 < idx.item() + i < tokens.size(0)]
        )

        acts = X[surrounding_tokens].cpu()
        acts.size()

        dims = torch.arange(acts.size(1))

        surrounding_tokens = surrounding_tokens.unsqueeze(0)
        strs = [f"{tokenizer.decode(tokens[i]).replace('\n', '\\n')} ({i})" for i in surrounding_tokens[0]]
        dims = dims.unsqueeze(-1)
        ax.plot_wireframe(surrounding_tokens, dims, acts.abs().T, rstride=0, color=color, linewidth=2.5)
        ax.set_xticks(surrounding_tokens[0])
        ax.set_xticklabels(strs, rotation=45, ha="right")
        ax.set_title(f"{layer_idx=}, tok_pos={idx.item()}")
    return fig


def main():
    """Main experiment function."""
    args = parse_args()

    # Convert args to constants used in the experiment
    cka_device = args.cka_device
    device1 = args.device1
    device2 = args.device2
    k = args.k
    output_dir = Path(args.output_dir)
    sample_ids = range(args.sample_start, args.sample_end)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DataFrames to store results
    pc_stats_data = []
    projections_data = []
    cka_results_data = []
    correlations_data = []
    top_tokens_data = []

    cka_records = []

    model_name_descendant = args.descendant
    model_name_base = args.base_model
    model_descendant, tokenizer_descendant = get_model_and_tokenizer(model_id=model_name_descendant, device=device1)
    model_base, tokenizer_base = get_model_and_tokenizer(model_id=model_name_base, device=device2)
    model_base.requires_grad_(False)
    model_descendant.requires_grad_(False)

    for sample_id in tqdm(sample_ids):
        cache_dir = f"cache_representations_2/{args.token_origin}/sample_{sample_id}"
        tokens = load_tokens(cache_dir)
        logger.info(f"{sample_id=}: {tokens.size()=}")

        h_base = get_hidden_states(model_base, tokens)
        h_descendant = get_hidden_states(model_descendant, tokens)

        for layer_idx in tqdm(
            range(1, min(len(h_base), len(h_descendant)))
        ):  # we start with 1 to skip the embedding layer
            # these tensors have size (batch, tokens, dim). batch is 1 for us
            X = h_base[layer_idx].to(cka_device).squeeze(0).to(torch.float32)
            Y = h_descendant[layer_idx].to(cka_device).squeeze(0).to(torch.float32)

            gram_x_linear = gram_linear(X)
            gram_y_linear = gram_linear(Y)
            linear_cka_score = cka(gram_x_linear, gram_y_linear)

            # --- Taking stats of explained variance by principal components
            pc_x, expvar_x = pca_svd(X)
            pc_y, expvar_y = pca_svd(Y)
            pc_stats_data.append(
                {
                    "base_model": model_name_base,
                    "descendant_model": model_name_descendant,
                    "sample_id": sample_id,
                    "layer": layer_idx,
                    "expvar_base_full": expvar_x.cpu().numpy(),
                    "expvar_descendant_full": expvar_y.cpu().numpy(),
                }
            )
            one_dim_proj_x = pc_projection(X, pc_x, 1)
            one_dim_proj_y = pc_projection(Y, pc_y, 1)
            projections_data.append(
                {
                    "base_model": model_name_base,
                    "descendant_model": model_name_descendant,
                    "sample_id": sample_id,
                    "layer": layer_idx,
                    "proj_base_tensor": one_dim_proj_x.cpu().numpy(),
                    "proj_descendant_tensor": one_dim_proj_y.cpu().numpy(),
                }
            )

            # --- Testing effect of high-pc1-projecting samples on CKA by removing them.
            topk_x = one_dim_proj_x.squeeze(-1).topk(k)
            topk_y = one_dim_proj_y.squeeze(-1).topk(k)
            top_indices = torch.tensor(list(set(topk_x.indices.tolist()) | set(topk_y.indices.tolist())))

            X_removed = X[~top_indices]
            Y_removed = Y[~top_indices]
            gram_x_linear = gram_linear(X_removed)
            gram_y_linear = gram_linear(Y_removed)
            linear_cka_score_wo_highpc_samples = cka(gram_x_linear, gram_y_linear)

            cka_records.append({"layer": layer_idx, "cka_type": "linear", "score": linear_cka_score.item()})
            cka_records.append(
                {
                    "layer": layer_idx,
                    "cka_type": f"linear_wo_top{k}_pcs",
                    "score": linear_cka_score_wo_highpc_samples.item(),
                }
            )
            cka_results_data.append(
                {
                    "base_model": model_name_base,
                    "descendant_model": model_name_descendant,
                    "sample_id": sample_id,
                    "layer": layer_idx,
                    "linear_cka_score": linear_cka_score.item(),
                    "linear_cka_wo_highpc": linear_cka_score_wo_highpc_samples.item(),
                }
            )

            # --- Testing correlation between pc1 projection magnitude and activation magnitude
            normsX = torch.linalg.norm(X, dim=1).detach().cpu()
            normsY = torch.linalg.norm(Y, dim=1).detach().cpu()
            corr_x = pearsonr(one_dim_proj_x.cpu().squeeze(-1), normsX)
            corr_y = pearsonr(one_dim_proj_y.cpu().squeeze(-1), normsY)
            # -- Checking overlap between top k pc1 projection tokens and activating tokens
            topk_proj_x_indices = set(topk_x.indices.tolist())
            topk_norms_x_indices = set(normsX.topk(k).indices.tolist())
            overlap_x = len(topk_proj_x_indices.intersection(topk_norms_x_indices)) / len(topk_proj_x_indices)

            topk_proj_y_indices = set(topk_y.indices.tolist())
            topk_norms_y_indices = set(normsY.topk(k).indices.tolist())
            overlap_y = len(topk_proj_y_indices.intersection(topk_norms_y_indices)) / len(topk_proj_y_indices)
            correlations_data.append(
                {
                    "base_model": model_name_base,
                    "descendant_model": model_name_descendant,
                    "sample_id": sample_id,
                    "layer": layer_idx,
                    "corr_base_statistic": corr_x.statistic,
                    "corr_base_pvalue": corr_x.pvalue,
                    "overlap_base": overlap_x,
                    "corr_descendant_statistic": corr_y.statistic,
                    "corr_descendant_pvalue": corr_y.pvalue,
                    "overlap_descendant": overlap_y,
                }
            )

            # --- Inspecting tokens with highest pc1 projection
            top_tokens_data.append(
                {
                    "base_model": model_name_base,
                    "descendant_model": model_name_descendant,
                    "sample_id": sample_id,
                    "layer": layer_idx,
                    "top_tokens": [tokenizer_base.decode(t) for t in tokens[top_indices]],
                }
            )

        # Convert to DataFrames for current sample
        cka_df_sample = pd.DataFrame([d for d in cka_results_data if d["sample_id"] == sample_id])
        corr_df_sample = pd.DataFrame([d for d in correlations_data if d["sample_id"] == sample_id])

        # Show tables
        # if not cka_df_sample.empty:
        #     print(
        #         tabulate(
        #             cka_df_sample[["layer", "linear_cka_score", "linear_cka_wo_highpc"]].values,
        #             headers=["layer", "old cka", "new cka"],
        #         )
        #     )
        # if not corr_df_sample.empty:
        #     print(
        #         tabulate(
        #             corr_df_sample[["layer", "corr_base_statistic", "corr_descendant_statistic"]].values,
        #             headers=["layer", "base corr", "descendant corr"],
        #         )
        #     )

        # Plot corr(pc1_proj_magnitude, token_act_magnitude) for base model and descendant as well as CKA between them
        # this basically checks whether massive activations also have the biggest influence on the linear CKA score
        plt.figure(figsize=(4 * 1.6, 4))
        if not corr_df_sample.empty:
            plt.plot(corr_df_sample["layer"], corr_df_sample["corr_base_statistic"], label="base")
            plt.plot(corr_df_sample["layer"], corr_df_sample["corr_descendant_statistic"], label="descendant")
        if not cka_df_sample.empty:
            plt.plot(cka_df_sample["layer"], cka_df_sample["linear_cka_score"], label="cka")

        plt.xlabel("layer")
        plt.ylabel("pearsonr")
        plt.title(f"Pearsonr PC1 projection with representation norm (sample={sample_id})")
        plt.legend()
        plt.savefig(
            output_dir
            / f"corr_{model_name_base.replace('/', '__')}_{model_name_descendant.replace('/', '__')}_{sample_id}.png"
        )
        plt.close()

        # Plot 1-explained variance by top1 PC for base and descendant as well as CKA and CKA w/o max pc1 projecting tokens
        pc_df_sample = pd.DataFrame([d for d in pc_stats_data if d["sample_id"] == sample_id])

        plt.figure(figsize=(4 * 1.6, 4))
        if not pc_df_sample.empty:
            expvar_base_pc1 = [arr[0] for arr in pc_df_sample["expvar_base_full"]]
            expvar_desc_pc1 = [arr[0] for arr in pc_df_sample["expvar_descendant_full"]]
            plt.plot(pc_df_sample["layer"], [1 - x for x in expvar_base_pc1], label="base_model")
            plt.plot(pc_df_sample["layer"], [1 - x for x in expvar_desc_pc1], label="descendant")
        if not cka_df_sample.empty:
            plt.plot(cka_df_sample["layer"], cka_df_sample["linear_cka_score"], label="cka")
            plt.plot(cka_df_sample["layer"], cka_df_sample["linear_cka_wo_highpc"], label="cka w/o max pc samples")
        plt.title(f"sample {sample_id}")
        secax = plt.gca().secondary_yaxis(
            "right",
        )
        secax.set_ylabel("CKA")
        plt.legend()
        plt.ylabel("1 - Explained variance by PC1 in representations")
        plt.savefig(
            output_dir
            / f"expvar_{model_name_base.replace('/', '__')}_{model_name_descendant.replace('/', '__')}_{sample_id}.png"
        )
        plt.close()

        # savedir = output_dir / "massacts" / str(sample_id)
        # savedir.mkdir(exist_ok=True, parents=True)
        # for layer_idx in range(len(h_base)):
        #     # descendant
        #     fig = massive_activation_plots(h_descendant[layer_idx], tokenizer_base, tokens, layer_idx, k=3)
        #     fig.savefig(savedir / f"{model_name_descendant.replace('/', '__')}_l{layer_idx}_{sample_id}.png")
        #     plt.close()

        #     # base model
        #     fig = massive_activation_plots(h_base[layer_idx], tokenizer_base, tokens, layer_idx, k=3, color="orange")
        #     fig.savefig(savedir / f"{model_name_base.replace('/', '__')}_l{layer_idx}_{sample_id}.png")
        #     plt.close()

    # Save all results to DataFrames
    pc_stats_df = pd.DataFrame(pc_stats_data)
    projections_df = pd.DataFrame(projections_data)
    cka_results_df = pd.DataFrame(cka_results_data)
    correlations_df = pd.DataFrame(correlations_data)
    top_tokens_df = pd.DataFrame(top_tokens_data)

    # Create filename suffix with model names
    base_name = model_name_base.replace("/", "__")
    desc_name = model_name_descendant.replace("/", "__")
    filename_suffix = f"{base_name}_vs_{desc_name}"

    # Save to CSV files
    pc_stats_df.to_csv(output_dir / f"pc_stats_{filename_suffix}.csv", index=False)
    projections_df.to_csv(output_dir / f"projections_{filename_suffix}.csv", index=False)
    cka_results_df.to_csv(output_dir / f"cka_results_{filename_suffix}.csv", index=False)
    correlations_df.to_csv(output_dir / f"correlations_{filename_suffix}.csv", index=False)
    top_tokens_df.to_csv(output_dir / f"top_tokens_{filename_suffix}.csv", index=False)

    logger.info(f"Saved DataFrames to {output_dir}")
    logger.info(f"PC stats shape: {pc_stats_df.shape}")
    logger.info(f"Projections shape: {projections_df.shape}")
    logger.info(f"CKA results shape: {cka_results_df.shape}")
    logger.info(f"Correlations shape: {correlations_df.shape}")
    logger.info(f"Top tokens shape: {top_tokens_df.shape}")


if __name__ == "__main__":
    main()
