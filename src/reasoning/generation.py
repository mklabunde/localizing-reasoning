import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def get_model_and_tokenizer(model_id: str, device: str = "auto", torch_dtype: torch.dtype = torch.bfloat16):
    """Loads the model and tokenizer with appropriate settings."""
    log.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_hidden_states(model, tokens, max_length: int = 8096) -> tuple[torch.Tensor, ...]:
    inp_ids = tokens.to(model.device)
    if tokens.ndim == 1:
        inp_ids = inp_ids.unsqueeze(0)

    # We limit the tokens to keep inference on a single gpu and fast. Otherwise OOM.
    # TODO: this could be improved by regenerating token-by-token if it is too long.
    if inp_ids.size(1) > max_length:
        print(inp_ids.size())
        log.warning(f"Excessive answer length. Only using first {max_length}.")
        inp_ids = inp_ids[:, :max_length]

    att_mask = torch.ones_like(inp_ids)
    output = model.forward(
        input_ids=inp_ids,
        attention_mask=att_mask,
        output_hidden_states=True,
        do_sample=False,
        max_new_tokens=1,
    )
    return output.hidden_states
