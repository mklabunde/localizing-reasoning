import logging
import os
import re

import datasets
import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from safetensors.torch import save_file
from tqdm import tqdm

from reasoning.generation import get_model_and_tokenizer

# Set up logging
log = logging.getLogger(__name__)


def get_target_layers(model, layer_config: list[int]) -> list[int]:
    """Resolves layer indices, handling negative indices."""
    num_layers = model.config.num_hidden_layers
    target_layers = []
    for layer_idx in layer_config:
        if layer_idx < 0:
            target_layers.append(num_layers + layer_idx)
        else:
            target_layers.append(layer_idx)
    log.info(f"Total model layers: {num_layers}. Extracting from: {target_layers}")
    return target_layers


def add_think_token_to_chat_template(tokenizer):
    # By default the tokenizer does not force thinking for some models. We modify the template.
    current_template = tokenizer.get_chat_template()

    # Pattern 1: Handle <｜Assistant｜> format
    pattern1 = r"(\{\{'<｜Assistant｜>'\}\})({% endif %})"
    replacement1 = r"{{'<｜Assistant｜><think>'}}\2"
    modified_template = re.sub(pattern1, replacement1, current_template)

    # Pattern 2: Handle <|im_start|>assistant format
    # Look for the pattern: {{- '<|im_start|>assistant\n' }}
    pattern2 = r"(\{\{-?\s*'<\|im_start\|>assistant\\n'\s*\}\})"
    replacement2 = r"\1\n{{- '<think>' }}"

    # Apply pattern2 if pattern1 didn't make changes
    if modified_template == current_template:
        modified_template = re.sub(pattern2, replacement2, current_template)
        if modified_template != current_template:
            log.debug("Applied <|im_start|>assistant pattern modification.")
    else:
        log.debug("Applied <｜Assistant｜> pattern modification.")

    tokenizer.chat_template = modified_template

    return tokenizer


def format_prompt(sample: dict, tokenizer, model_name: str) -> torch.Tensor:
    """Formats the MATH problem into a chat template for the model."""
    problem = sample["problem"]

    # Different models were trained with different instructions for different tasks
    # acereason: Please reason step by step, and put your final answer within \boxed{}.
    # openreasoner: ??
    # boba2: qwen3-think , see https://inclusionai.github.io/AReaL/tutorial/eval.html#command-line-parameters
    #  "qwen3-think": (
    #     "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}./think<|im_end|>\n"  # noqa:E501
    #     "<|im_start|>assistant\n<think>",
    #     "{output}",
    #     "\n\n",
    # ),  this is supposed to be an (input_template, output_template, splitter) tuple
    # lightr1: Please reason step by step, and put your final answer within \boxed{}.
    if model_name in [
        "AceReason-Nemotron-7B",
        "Light-R1-7B-DS",
        "Skywork-OR1-7B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "AceReason-Nemotron-1.1-7B",
        "OpenR1-Distill",
    ]:
        # No information on Skywork-OR1, but assuming it uses the format of the base deepseek model
        instruction = "Please reason step by step, and put your final answer within \\boxed{{}}."
    elif model_name in ["AReaL-boba-2-8B-Open"]:
        instruction = "Please reason step by step, and put your final answer within \\boxed{{}}./think"
    elif model_name in [
        "Qwen2.5-Math-7B",
        "Qwen2.5-Math-7B-RoPE-300k",
        "Qwen2.5-Math-7B-Oat-Zero",
    ]:
        instruction = ""  # They get a default system prompt with CoT instructions.
    else:
        raise ValueError(f"Attempting to create instruction for {model_name=}, but model is not recognized.")

    messages = [
        {
            "role": "user",
            "content": f"{problem}\n{instruction}",
        }
    ]

    # apply_chat_template will correctly format the input for the specific model
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    return input_ids


@hydra.main(version_base=None, config_path="../conf", config_name="cache_representations")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the representation generation and caching experiment.
    """
    log.info("Starting representation caching experiment...")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # --- 1. Setup ---
    if cfg.device:
        device = cfg.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = get_model_and_tokenizer(
        cfg.model.model_id, device=device, torch_dtype=hydra.utils.instantiate(cfg.model.dtype)
    )
    if cfg.model.name in ["Light-R1-7B-DS", "Skywork-OR1-7B", "AReaL-boba-2-8B-Open", "AceReason-Nemotron-1.1-7B"]:
        # by default thinking is not forced, we make every answer start with <think>
        tokenizer = add_think_token_to_chat_template(tokenizer)
    target_layers = get_target_layers(model, cfg.layers_to_extract)

    # --- 2. Load Dataset ---
    dataset = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    assert isinstance(dataset, datasets.Dataset)

    # --- 3. Main Loop: Generate and Cache ---
    for i, sample in enumerate(
        tqdm(dataset.select(range(cfg.dataset.max_samples)), desc=f"Processing {cfg.model.name}")
    ):
        sample_id = f"sample_{i}"

        # Check if cache already exists for this sample
        final_file_check_path = os.path.join(
            cfg.cache_dir, cfg.model.name, sample_id, f"layer_{target_layers[-1]}", "hidden_states.safetensors"
        )
        if os.path.exists(final_file_check_path):
            log.info(f"Skipping {sample_id} for {cfg.model.name}, cache found.")
            continue

        with torch.no_grad():
            # --- 3a. Prepare input ---
            input_ids = format_prompt(sample, tokenizer, cfg.model.name).to(device)  # type:ignore
            prompt_len = input_ids.shape[1]

            # --- 3b. Generate reasoning steps ---
            generated_outputs = model.generate(
                input_ids,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=False,  # Use greedy decoding for reproducibility
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # Deepseek recommends temp=0.6 and sampling to avoid bad output

            # --- 3c. Identify token positions ---
            generated_token_ids: torch.Tensor = generated_outputs.sequences[0]
            if not cfg.model.end_think_token_id and not cfg.model.start_think_token_id:
                num_reasoning_tokens = 0
            elif isinstance(cfg.model.end_think_token_id, int) and isinstance(cfg.model.start_think_token_id, int):
                idx_stop_thinking = (generated_token_ids == cfg.model.end_think_token_id).int().argmax()
                idx_start_thinking = (generated_token_ids == cfg.model.start_think_token_id).int().argmax()
                num_reasoning_tokens = idx_stop_thinking - idx_start_thinking
            elif isinstance(cfg.model.end_think_token_id, ListConfig) and isinstance(
                cfg.model.start_think_token_id, ListConfig
            ):
                start_found = False
                stop_found = False
                num_reasoning_tokens = 0
                for i in range(len(generated_token_ids)):
                    if not start_found:
                        potential_start = generated_token_ids[i : i + len(cfg.model.start_think_token_id)].tolist()
                        if potential_start == cfg.model.start_think_token_id:
                            idx_start_thinking = i + len(cfg.model.start_think_token_id) - 1
                            start_found = True

                    if not stop_found:
                        potential_stop = generated_token_ids[i : i + len(cfg.model.end_think_token_id)].tolist()
                        if potential_stop == cfg.model.end_think_token_id:
                            idx_stop_thinking = i + len(cfg.model.end_think_token_id) - 1
                            stop_found = True

                    if start_found and stop_found:
                        num_reasoning_tokens = idx_stop_thinking - idx_start_thinking  # type:ignore
                        break
            else:
                raise ValueError(
                    f"Unknown type of thinking tokens: {type(cfg.model.start_think_token_id)=} and "
                    f"{type(cfg.model.end_think_token_id)=}"
                )

            # Non-reasoning models or models not forced to emit thinking tokens could simply not generate these tokens
            if num_reasoning_tokens == 0:
                log.warning("Did not find start/stop thinking tokens. Using whole output as reasoning steps.")
                idx_stop_thinking = len(generated_token_ids) - 1
                idx_start_thinking = prompt_len
                num_reasoning_tokens = idx_stop_thinking - idx_start_thinking

            # print(f"{len(generated_outputs.hidden_states)=}")
            # print(f"{generated_outputs.hidden_states[0][0].size()=}")
            # print(f"{generated_outputs.hidden_states[1][0].size()=}")
            # print(f"{generated_outputs.hidden_states[2][0].size()=}")
            # print(f"{generated_outputs.hidden_states[3][0].size()=}")
            # print(f"{prompt_len=}")
            # print(f"{num_reasoning_tokens=}")
            # print(f"{len(generated_token_ids)=}")
            # print(f"{idx_start_thinking=}, {idx_stop_thinking=}")

            token_indices = {}
            for name, percentage in cfg.token_positions.items():
                if "reasoning" in name:
                    idx = min(
                        max(0, idx_start_thinking + int(num_reasoning_tokens * percentage)),  # type:ignore
                        len(generated_token_ids) - 1,
                    )
                    token_indices[name] = idx

            # --- 3d. Extract and organize hidden states ---
            # This dictionary will hold {layer_idx: {token_name: tensor}}
            layer_representations = {layer: {} for layer in target_layers}

            # Extract from the PROMPT
            if "assistant_start" in cfg.token_positions:
                for layer_idx in target_layers:
                    # Get the state of the last token of the prompt [batch, toks, hidden] -> [hidden]
                    vec = generated_outputs.hidden_states[0][layer_idx][0, max(0, idx_start_thinking - 1), :].cpu()  # type:ignore
                    layer_representations[layer_idx]["assistant_start"] = vec

            # Extract from the REASONING
            for name, target_idx in token_indices.items():
                for layer_idx in target_layers:
                    # generated_outputs.hidden_states is a tuple of (num_gen_tokens)
                    # each element is a tuple of (num_layers) (including embedding output)
                    # each tensor is [batch_size, sequence_length_at_step, hidden_dim].
                    # sequence_length_at_step is len(input prompt) at hidden_states[0], else 1
                    if target_idx < prompt_len:
                        hidden_state_at_step = generated_outputs.hidden_states[0][
                            layer_idx + 1
                        ]  # + 1 to ignore embed
                    else:
                        # TODO: fix index out of range error with openr1
                        hidden_state_at_step = generated_outputs.hidden_states[target_idx - prompt_len][layer_idx + 1]
                    # We want the state of the token that was just generated, which is the last one
                    vec = hidden_state_at_step[0, -1, :].cpu()
                    layer_representations[layer_idx][name] = vec

            # --- 3e. Save to cache using safetensors ---
            for layer_idx, tensors_dict in layer_representations.items():
                if not tensors_dict:
                    continue  # Skip if no tensors were extracted for this layer
                tensors_dict["toks"] = generated_token_ids

                # e.g., cache_representations/nvidia_AceReason.../sample_0/layer_31/
                output_dir = os.path.join(cfg.cache_dir, cfg.model.name, sample_id, f"layer_{layer_idx}")
                os.makedirs(output_dir, exist_ok=True)

                output_path = os.path.join(output_dir, "hidden_states.safetensors")
                save_file(
                    tensors_dict,
                    output_path,
                    metadata={
                        "input_toks": str(prompt_len),
                        "reasoning_toks": str(
                            num_reasoning_tokens.item()
                            if isinstance(num_reasoning_tokens, torch.Tensor)
                            else int(num_reasoning_tokens)
                        ),
                        "total_toks": str(len(generated_token_ids)),
                    },
                )

        log.info(f"Saved representations for {cfg.model.name}, {sample_id}")

    log.info("Representation caching finished successfully!")


if __name__ == "__main__":
    main()
