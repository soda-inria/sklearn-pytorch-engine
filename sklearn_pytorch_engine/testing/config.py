import os
from functools import lru_cache

import array_api_compat
import numpy as np
import pytest
import torch

_TORCH_ARRAY_API_NAMESPACE = (
    lambda: array_api_compat.get_namespace(torch.zeros(1, 1))
)()


@lru_cache
def has_fp64_support(device):
    try:
        torch.zeros(1, 1, dtype=torch.float64, device=device)
        return True
    except RuntimeError as runtime_error:
        if "data type is unsupported" in str(runtime_error):
            return False


def get_torch_array_api_namespace():
    return _TORCH_ARRAY_API_NAMESPACE


def get_torch_default_device():
    return torch.zeros(1).device.type


@lru_cache
def to_pytorch_dtype(dtype):
    return torch.from_numpy(np.array(0, dtype=dtype)).dtype


def get_sklearn_pytorch_engine_default_device():
    device = os.getenv("SKLEARN_PYTORCH_ENGINE_DEFAULT_DEVICE", None)
    if device is None:
        device = get_torch_default_device()
    return device


def get_sklearn_pytorch_engine_test_inputs_device():
    device = os.getenv("SKLEARN_PYTORCH_ENGINE_TEST_INPUTS_DEVICE", None)
    if device is None:
        device = get_torch_default_device()
    return device


def _supported_dtypes(device):
    if has_fp64_support(device):
        return [np.float32, np.float64]
    else:
        return [np.float32]


float_dtype_params = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            (
                dtype
                not in _supported_dtypes(
                    DEVICE := get_sklearn_pytorch_engine_default_device()
                )
            )
            or (
                dtype
                not in _supported_dtypes(
                    DEVICE := get_sklearn_pytorch_engine_test_inputs_device()
                )
            ),
            reason=(
                f"The device {DEVICE} does not have support for"
                f" {np.dtype(dtype).name} operations."
            ),
        ),
    )
    for dtype in [np.float32, np.float64]
]
