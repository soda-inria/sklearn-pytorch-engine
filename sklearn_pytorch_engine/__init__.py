try:
    # ensure xpu backend is loaded
    import intel_extension_for_pytorch as ipex  # noqa
except ModuleNotFoundError:
    pass
