import torch


def get_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    elif name == "bfloat16":
        return torch.bfloat16
    else:
        raise NotImplementedError(f"Tried to get dtype for '{name}', but only implemented for fp16 and bf16.")
