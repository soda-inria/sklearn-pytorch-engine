import os
from functools import lru_cache

import array_api_compat
import numpy as np
import torch

_TORCH_ARRAY_API_NAMESPACE = (lambda: array_api_compat.get_namespace(torch.empty(0)))()


def get_torch_array_api_namespace():
    return _TORCH_ARRAY_API_NAMESPACE


@lru_cache
def to_pytorch_dtype(dtype):
    return torch.from_numpy(np.empty(0, dtype=dtype)).dtype


# NB: value is sensitive to `torch.device` context or previous
# `torch.set_default_device` calls
def get_torch_default_device():
    return torch.empty(0).device.type


def get_sklearn_pytorch_engine_default_device():
    device = os.getenv("SKLEARN_PYTORCH_ENGINE_DEFAULT_DEVICE", None)
    if device is None:
        device = get_torch_default_device()
    return device


@lru_cache
def has_fp64_support(device):
    try:
        torch.zeros(1, dtype=torch.float64, device=device)
        return True
    except RuntimeError as runtime_error:
        if "data type is unsupported" in str(runtime_error):
            return False