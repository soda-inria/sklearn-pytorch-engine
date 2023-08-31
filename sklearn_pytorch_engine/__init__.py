try:
    # ensure xpu backend is loaded
    import intel_extension_for_pytorch as ipex  # noqa
    import torch

    truc = torch.zeros(1, 1, device="xpu")
except ModuleNotFoundError:
    pass
