import os

import numpy as np
import pytest
import torch

from sklearn_pytorch_engine._utils import (
    get_sklearn_pytorch_engine_default_device,
    get_torch_default_device,
    has_fp64_support,
    to_pytorch_dtype,
)


def get_sklearn_pytorch_engine_test_inputs_device():
    device = os.getenv("SKLEARN_PYTORCH_ENGINE_TEST_INPUTS_DEVICE", None)
    if device is None:
        device = get_torch_default_device()
    return device


def _torch_array_constr_on_device(X, dtype):
    return torch.asarray(
        X,
        dtype=to_pytorch_dtype(dtype),
        device=get_sklearn_pytorch_engine_test_inputs_device(),
    )


DEVICE = None

float_dtype_params = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            (dtype is np.float64)
            and not (
                has_fp64_support(DEVICE := get_sklearn_pytorch_engine_default_device())
                and has_fp64_support(
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
