import torch
from functools import lru_cache
import array_api_compat

import numpy as np
import pytest


_TORCH_ARRAY_API_NAMESPACE = (
    lambda: array_api_compat.get_namespace(torch.zeros(1, 1))
)()
_TORCH_DEFAULT_DEVICE = torch.Tensor(0).device
_DEVICE_TYPE = _TORCH_DEFAULT_DEVICE.type
_SUPPORTED_DTYPE = [np.float32]


@lru_cache
def has_fp64_support(device):
    try:
        torch.zeros(1, 1, dtype=torch.float64, device=device)
        return True
    except RuntimeError as runtime_error:
        if "data type is unsupported" in str(runtime_error):
            return False


def get_torch_default_device():
    return _TORCH_DEFAULT_DEVICE


def get_torch_array_api_namespace():
    return _TORCH_ARRAY_API_NAMESPACE


@lru_cache
def to_pytorch_dtype(dtype):
    return torch.from_numpy(np.array(0, dtype=dtype)).dtype


if has_fp64_support(_TORCH_DEFAULT_DEVICE):
    _SUPPORTED_DTYPE.append(np.float64)


float_dtype_params = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            dtype not in _SUPPORTED_DTYPE,
            reason=(
                f"The default device {_DEVICE_TYPE} does not have support for"
                f" {np.dtype(dtype).name} operations."
            ),
        ),
    )
    for dtype in [np.float32, np.float64]
]
