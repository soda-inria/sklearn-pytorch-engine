try:
    # ensure xpu backend is loaded if available
    import intel_extension_for_pytorch as ipex  # noqa
    import torch

    torch.zeros(1, 1, device="xpu")
except ModuleNotFoundError:
    pass
