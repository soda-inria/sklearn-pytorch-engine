# sklearn-pytorch-engine

Experimental plugin for scikit-learn that implements a backend for (some) scikit-learn
estimators, written in `pytorch`, so that it benefits from `pytorch` ability to
dispatch data and compute to many devices, providing the appropriate pytorch extensions
are installed.

This package requires working with the following experimental branch of scikit-learn:

- `feature/engine-api` branch on https://github.com/scikit-learn/scikit-learn

## List of Included Engines

- `sklearn.cluster.KMeans` for the standard LLoyd's algorithm on dense data arrays,
  including `kmeans++` support.

## Getting started:

### Pre-requisites

#### Step 1: Install PyTorch

Getting started requires a working python environment for using `pytorch`. Depending on
the device you target, install PyTorch extensions accordingly, including (but not
limited to):

- using one of the [native distributions](https://pytorch.org/get-started/locally/) for
cuda (for nvidia gpus), rocm (amd gpus) or mps (apple gpus) support

- using [Intel distributions](https://intel.github.io/intel-extension-for-pytorch/xpu/2.0.110+xpu/tutorials/installations/linux.html#install-via-prebuilt-wheel-files)
for xpu (for intel gpus), has experimental (unofficial) support for igpus if compiling
from source
[with appropriate flags](https://intel.github.io/intel-extension-for-pytorch/xpu/2.0.110+xpu/tutorials/installations/linux.html#configure-the-aot-optional)

#### Step 2: Install scikit-learn from source

Using the plugin requires the experimental development branch `feature/engine-api` of
scikit-learn that implements the compatible plugin system. The `sklearn_pytorch_engine`
plugin is compatible with the commit 2ccfc8c4bdf66db005d7681757b4145842944fb9 available
in the fork [fcharras/scikit-learn](https://github.com/fcharras/scikit-learn/) .

Please refer to the relevant [scikit-learn documentation page](https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge)
for a comprehensive guide regarding installing from source. For instance, using `pip`
and `apt` (assuming `apt`-based environment):

```
apt-get update --quiet
# Install prerequisites
apt-get install -y build-essential python3-dev git
pip install cython numpy scipy joblib threadpoolctl
# Build and install
pip install git+https://github.com/fcharras/scikit-learn.git@2ccfc8c4bdf66db005d7681757b4145842944fb9#egg=scikit-learn
```

#### Step 3: Install this plugin

When loaded into your PyTorch + scikit-learn environment, run:

```
git clone https://github.com/soda-inria/sklearn-pytorch-engine
cd sklearn-pytorch-engine
pip install -e .
```

## Using the plugin

See the `sklearn_pytorch_engine/kmeans/tests` folder for example usage.

🚧 TODO: write some examples here instead.

### Running the tests

To run the tests run the following from the root of the `sklearn_pytorch_engine`
repository:

```bash
pytest sklearn_pytorch_engine
```

To run the `scikit-learn` tests with the `sklearn_pytorch_engine` engine you can run the
following:

```bash
SKLEARN_PYTORCH_ENGINE_TESTING_MODE=1 pytest --sklearn-engine-provider sklearn_pytorch_engine --pyargs sklearn.cluster.tests.test_k_means
```

(change the `--pyargs` option accordingly to select other test suites).

The `--sklearn-engine-provider sklearn_pytorch_engine` option offered by the sklearn
pytest plugin will automatically activate the `sklearn_pytorch_engine` engine for all
tests.

Tests covering unsupported features (that trigger
`sklearn.exceptions.FeatureNotCoveredByPluginError`) will be automatically marked as
_xfailed_.

### Additional environment variables for device selection behavior

By default, the engine will use the _compute follow data_ principle, meaning that it
will run the compute on the device that manages the data. For instance `kmeans.fit(X)`
will run compute on corresponding xpu device if `X` is a `torch.tensor` array such that
`X.device.type` is `"xpu"`, and will run on cpu if `X.device.type` is `"cpu"`, etc.

It's  possible to alter this behavior and have the engine force offload the compute to
a specific device, using the environment variable
`SKLEARN_PYTORCH_ENGINE_DEFAULT_DEVICE`. For instance, on a compatible computer,
`SKLEARN_PYTORCH_ENGINE_DEFAULT_DEVICE=mps` will force the compute to the
`mps`-compatible device, even if it requires copying the input data under the hood to
do so.

Both internal and scikit-learn test suites can run with any value of
`SKLEARN_PYTORCH_ENGINE_DEFAULT_DEVICE` as long as the compatible pytorch extension
is available and that the host hardware is compatible, for instance:

```bash
export SKLEARN_PYTORCH_ENGINE_DEFAULT_DEVICE=xpu
pytest sklearn_pytorch_engine
SKLEARN_PYTORCH_ENGINE_TESTING_MODE=1 pytest --sklearn-engine-provider sklearn_pytorch_engine --pyargs sklearn.cluster.tests.test_k_means
```

will run all compute on the relevant `xpu` device.

At the moment, both tests suite will create test data that is hosted on the CPU by
default. For internal tests, this behavior can be changed with the environment variable
`SKLEARN_PYTORCH_ENGINE_TEST_INPUTS_DEVICE`, for instance the command

```bash
SKLEARN_PYTORCH_ENGINE_TEST_INPUTS_DEVICE=cuda SKLEARN_PYTORCH_ENGINE_DEFAULT_DEVICE=cpu pytest sklearn_pytorch_engine
```

will run the tests while enforcing that the test data is generated on the cuda device
but the compute is done on cpu (since `SKLEARN_PYTORCH_ENGINE_DEFAULT_DEVICE` is set
to `cpu`).

All combinations of those two environment variables makes for a reasonably exhaustive
test matrix regarding internal data conversions.

### Notes about the preferred floating point precision (float32)

In many machine learning applications, operations using single-precision (float32)
floating point data require twice as less memory that double-precision (float64), are
regarded as faster, accurate enough and more suitable for GPU compute. Besides, most
GPUs used in machine learning projects are significantly faster with float32 than with
double-precision (float64) floating point data.

To leverage the full potential of GPU execution, it's strongly advised to use a float32
data type.

By default, unless specified otherwise numpy array are created with type float64, so be
especially careful to the type whenever the loader does not explicitly document the
type nor expose a type option.

Transforming NumPy arrays from float64 to float32 is also possible using
[`numpy.ndarray.astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html),
although it is less recommended to prevent avoidable data copies. `numpy.ndarray.astype`
can be used as follows:

```python
X = my_data_loader()
X_float32 = X.astype(float32)
my_gpu_compute(X_float32)
```
