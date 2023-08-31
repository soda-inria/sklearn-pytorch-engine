from functools import lru_cache

import numpy as np
import pytest
import torch
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.cluster._kmeans import _is_same_clustering
from sklearn.cluster.tests.test_k_means import X as X_sklearn_test
from sklearn.cluster.tests.test_k_means import n_clusters as n_clusters_sklearn_test
from sklearn.datasets import make_blobs as _make_blobs
from sklearn.utils._testing import assert_allclose

from sklearn_pytorch_engine._utils import to_pytorch_dtype
from sklearn_pytorch_engine.kmeans.engine import KMeansEngine
from sklearn_pytorch_engine.testing.config import (
    _torch_array_constr_on_device,
    float_dtype_params,
)


def asnumpy(X):
    return X.cpu().numpy()


@lru_cache
def make_blobs(random_state):
    return _make_blobs(random_state=random_state)


@pytest.mark.parametrize(
    "array_constr,test_attributes_auto_convert",
    [
        (np.asarray, False),
        (_torch_array_constr_on_device, False),
        (_torch_array_constr_on_device, True),
    ],
    ids=["numpy", "torch", "torch+convert"],
)
@pytest.mark.parametrize("dtype", float_dtype_params)
def test_kmeans_same_results(dtype, array_constr, test_attributes_auto_convert):
    random_seed = 42
    X, _ = make_blobs(random_state=random_seed)
    X_array = array_constr(X, dtype=dtype)
    X = X.astype(dtype)

    kmeans_truth = KMeans(algorithm="lloyd", max_iter=2, n_init=2, init="random")
    kmeans_engine = clone(kmeans_truth)

    # NB: force use of initialization with np.random.RandomState even with
    # sklearn_pytorch_engine engine by explicitly passing a RandomState object.
    # RandomState objects must be independant.
    kmeans_truth.set_params(random_state=np.random.RandomState(random_seed))
    kmeans_engine.set_params(random_state=np.random.RandomState(random_seed))

    if test_attributes_auto_convert:
        vanilla_engine = "sklearn_pytorch_engine"
        attribute_conversion = "sklearn_types"
        X_vanilla = X_array
    else:
        vanilla_engine = attribute_conversion = None
        X_vanilla = X

    with config_context(
        engine_provider=vanilla_engine, engine_attributes=attribute_conversion
    ):
        kmeans_truth.fit(X_vanilla)

    with config_context(engine_provider="sklearn_pytorch_engine"):
        kmeans_engine.fit(X_array)

    # ensure same results
    assert_array_equal(kmeans_truth.labels_, asnumpy(kmeans_engine.labels_))
    assert_allclose(
        kmeans_truth.cluster_centers_, asnumpy(kmeans_engine.cluster_centers_)
    )
    assert_allclose(dtype(kmeans_truth.inertia_), dtype(kmeans_engine.inertia_))

    # test fit_predictz
    y_labels = kmeans_truth.fit_predict(X)
    with config_context(engine_provider="sklearn_pytorch_engine"):
        y_labels_engine = kmeans_engine.fit_predict(X_array)
    assert_array_equal(y_labels, asnumpy(y_labels_engine))
    assert_array_equal(kmeans_truth.labels_, asnumpy(kmeans_engine.labels_))
    assert_allclose(
        kmeans_truth.cluster_centers_, asnumpy(kmeans_engine.cluster_centers_)
    )
    assert_allclose(dtype(kmeans_truth.inertia_), dtype(kmeans_engine.inertia_))

    # test fit_transform
    y_transform = kmeans_truth.fit_transform(X)
    with config_context(engine_provider="sklearn_pytorch_engine"):
        y_transform_engine = kmeans_engine.fit_transform(X_array)
    assert_allclose(
        y_transform,
        asnumpy(y_transform_engine),
        rtol=1.2e-4 if dtype == np.float32 else None,
    )
    assert_array_equal(kmeans_truth.labels_, asnumpy(kmeans_engine.labels_))
    assert_allclose(
        kmeans_truth.cluster_centers_, asnumpy(kmeans_engine.cluster_centers_)
    )
    assert_allclose(dtype(kmeans_truth.inertia_), dtype(kmeans_engine.inertia_))

    # test predict method (returns labels)
    y_labels = kmeans_truth.predict(X)
    with config_context(engine_provider="sklearn_pytorch_engine"):
        y_labels_engine = kmeans_engine.predict(X_array)
    assert_array_equal(y_labels, asnumpy(y_labels_engine))

    # test score method (returns negative inertia for each sample)
    y_scores = kmeans_truth.score(X)
    with config_context(engine_provider="sklearn_pytorch_engine"):
        y_scores_engine = kmeans_engine.score(X_array)
    assert_allclose(dtype(y_scores), dtype(y_scores_engine))

    # test transform method (returns euclidean distances)
    y_transform = kmeans_truth.transform(X)
    with config_context(engine_provider="sklearn_pytorch_engine"):
        y_transform_engine = kmeans_engine.transform(X_array)
    assert_allclose(
        y_transform,
        asnumpy(y_transform_engine),
        rtol=1.2e-4 if dtype == np.float32 else None,
    )


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_kmeans_predict_centers(dtype):
    kmeans = KMeans(n_clusters=10)
    kmeans._n_threads = 1
    cluster_centers = kmeans.cluster_centers_ = torch.asarray(
        torch.reshape(torch.arange(0, 100, dtype=to_pytorch_dtype(dtype)), (10, 10))
    )
    with config_context(engine_provider="sklearn_pytorch_engine"):
        cluster_centers_score = asnumpy(kmeans.predict(cluster_centers))
    assert_array_equal(cluster_centers_score, np.arange(10))


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_kmeans_relocated_clusters(dtype):
    """Copied and adapted from sklearn's test_kmeans_relocated_clusters"""

    # check that empty clusters are relocated as expected
    X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)

    # second center too far from others points will be empty at first iter
    init_centers = np.array([[0.5, 0.5], [3, 3]], dtype=dtype)

    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
    with config_context(engine_provider="sklearn_pytorch_engine"):
        kmeans.fit(X)

    expected_n_iter = 3
    expected_inertia = 0.25
    assert_allclose(kmeans.inertia_, expected_inertia)
    assert kmeans.n_iter_ == expected_n_iter

    labels = asnumpy(kmeans.labels_)
    cluster_centers = asnumpy(kmeans.cluster_centers_)
    # There are two acceptable ways of relocating clusters in this example, the output
    # depends on how the argpartition strategy break ties. It might not be deterministic
    # (might depend on thread concurrency) so we accept both outputs.
    try:
        expected_labels = [0, 0, 1, 1]
        expected_centers = [[0.25, 0], [0.75, 1]]
        assert_array_equal(labels, expected_labels)
        assert_allclose(cluster_centers, expected_centers)
    except AssertionError:
        expected_labels = [1, 1, 0, 0]
        expected_centers = [[0.75, 1.0], [0.25, 0.0]]
        assert_array_equal(labels, expected_labels)
        assert_allclose(cluster_centers, expected_centers)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_euclidean_distance(dtype):
    """Test adapted from sklearn's test_euclidean_distance"""

    random_seed = 42
    rng = default_rng(random_seed)
    a = rng.random(size=(1, 100), dtype=dtype)
    b = rng.standard_normal((1, 100), dtype=dtype)

    expected = np.sqrt(((a - b) ** 2).sum())

    estimator = KMeans(n_clusters=len(b))
    estimator.cluster_centers_ = b
    engine = KMeansEngine(estimator)

    result = engine.get_euclidean_distances(a)

    rtol = 1e-4 if dtype == np.float32 else 1e-7
    assert_allclose(asnumpy(result), expected, rtol=rtol)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_inertia(dtype):
    """Test adapted from sklearn's test_inertia"""

    random_seed = 42
    rng = default_rng(random_seed)
    X = rng.random((100, 10), dtype=dtype)
    sample_weight = rng.standard_normal(100, dtype=dtype)
    centers = rng.standard_normal((5, 10), dtype=dtype)

    estimator = KMeans(n_clusters=len(centers))
    estimator.cluster_centers_ = torch.asarray(centers)
    engine = KMeansEngine(estimator)
    X_prepared, sample_weight_prepared = engine.prepare_prediction(X, sample_weight)
    labels = engine.get_labels(X_prepared, sample_weight_prepared)

    closest_centers = centers[asnumpy(labels), :]
    distances = ((X - closest_centers) ** 2).sum(axis=1)
    expected = float(np.sum(distances * sample_weight))

    inertia = engine.get_score(X_prepared, sample_weight_prepared)

    rtol = 1e-4 if dtype == np.float32 else 1e-6
    assert_allclose(float(inertia), expected, rtol=rtol)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_relocate_empty_clusters(dtype):
    """Copied and adapted from sklearn's test_relocate_empty_clusters"""

    # Synthetic dataset with 3 obvious clusters of different sizes
    X = np.array(
        [-10.0, -9.5, -9.0, -8.5, -8.0, -1.0, 1.0, 9.0, 9.5, 10.0],
        dtype=dtype,
    ).reshape(-1, 1)
    # centers all initialized to the first point of X
    # With this initialization, all points will be assigned to the first center
    init_centers = np.array([-10.0, -10.0, -10.0]).reshape(-1, 1)

    kmeans_truth = KMeans(
        n_clusters=3, n_init=1, max_iter=1, init=init_centers, algorithm="lloyd"
    )
    kmeans_engine = clone(kmeans_truth)

    kmeans_truth.fit(X)
    with config_context(engine_provider="sklearn_pytorch_engine"):
        kmeans_engine.fit(X)

    expected_n_iter = 1
    expected_labels = [0, 0, 0, 0, 0, 0, 0, 2, 2, 1]
    truth_labels = kmeans_truth.labels_
    engine_labels = asnumpy(kmeans_engine.labels_).astype(np.int32)
    assert kmeans_truth.n_iter_ == expected_n_iter
    assert kmeans_engine.n_iter_ == expected_n_iter
    assert _is_same_clustering(
        truth_labels,
        engine_labels,
        n_clusters=3,
    )
    assert _is_same_clustering(
        truth_labels, np.asarray(expected_labels, dtype=np.int32), n_clusters=3
    )
    assert_allclose(
        kmeans_truth.cluster_centers_[truth_labels],
        asnumpy(kmeans_engine.cluster_centers_)[engine_labels],
    )
    assert_allclose(dtype(kmeans_truth.inertia_), dtype(kmeans_engine.inertia_))


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_kmeans_plusplus_same_quality(dtype):
    X = X_sklearn_test.astype(dtype)
    n_clusters = n_clusters_sklearn_test

    # HACK: to compare the quality of the initialization, it's convenient to use the
    # `score` method of an estimator whose fitted attribute cluster_centers_ has been
    # set to the result of the initialization, without running the remaining steps of
    # the  KMeans algorithm. For this purpose, since KMeans does not support passing
    # `max_iter=0` we forcefully set fitted attribute values without actually running
    # `fit`.
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
    )
    kmeans._n_threads = 1

    def _get_score_with_centers(centers):
        kmeans.cluster_centers_ = np.ascontiguousarray(centers)
        return kmeans.score(X)

    scores_vanilla_kmeans_plusplus = []
    scores_engine_kmeans_plusplus = []
    scores_random_init = []

    for random_state in range(10):
        random_centers = np.random.default_rng(random_state).choice(
            X, size=n_clusters, replace=False
        )
        scores_random_init.append(_get_score_with_centers(random_centers))

        vanilla_kmeans_plusplus_centers, _ = kmeans_plusplus(
            X, n_clusters, random_state=random_state
        )
        scores_vanilla_kmeans_plusplus.append(
            _get_score_with_centers(vanilla_kmeans_plusplus_centers)
        )

        kmeans.set_params(random_state=random_state)
        engine = KMeansEngine(kmeans)
        X_prepared, _, sample_weight = engine.prepare_fit(X)
        engine_kmeans_plusplus_centers = engine.init_centroids(
            X_prepared, sample_weight
        )
        engine.unshift_centers(X_prepared, engine_kmeans_plusplus_centers)
        scores_engine_kmeans_plusplus.append(
            _get_score_with_centers(asnumpy(engine_kmeans_plusplus_centers))
        )

    # Those results confirm that both sklearn KMeans++ and ours have similar quality,
    # and are both very significantly better than random init.

    assert_allclose(
        [
            np.mean(scores_random_init),
            np.mean(scores_vanilla_kmeans_plusplus),
        ],
        [
            -1827.22702,
            -892.39115,
        ],
    )

    # Since RNG is different depending on the device that, only
    # check for an upper bound on the value rather than hardcoding the value
    assert np.mean(scores_engine_kmeans_plusplus) < 1000


@pytest.mark.parametrize("dtype", float_dtype_params)
@pytest.mark.parametrize(
    "array_constr",
    [np.asarray, _torch_array_constr_on_device],
    ids=["numpy", "torch"],
)
def test_kmeans_plusplus_output(array_constr, dtype):
    """Test adapted from sklearn's test_kmeans_plusplus_output"""
    random_state = 42

    # Check for the correct number of seeds and all positive values
    X = array_constr(X_sklearn_test, dtype=dtype)

    sample_weight = default_rng(random_state).random(X.shape[0], dtype=dtype)

    estimator = KMeans(
        init="k-means++", n_clusters=n_clusters_sklearn_test, random_state=random_state
    )
    engine = KMeansEngine(estimator)
    X_prepared, _, sample_weight = engine.prepare_fit(X, sample_weight=sample_weight)

    centers, indices = engine._kmeans_plusplus(X_prepared, sample_weight)
    engine.unshift_centers(X_prepared, centers)
    centers = asnumpy(centers)
    indices = asnumpy(indices)

    # Check there are the correct number of indices and that all indices are
    # positive and within the number of samples
    assert indices.shape[0] == n_clusters_sklearn_test
    assert (indices >= 0).all()
    assert (indices <= X.shape[0]).all()

    # Check for the correct number of seeds and that they are bound by the data
    assert centers.shape[0] == n_clusters_sklearn_test
    assert (centers.max(axis=0) <= X_sklearn_test.max(axis=0)).all()
    assert (centers.min(axis=0) >= X_sklearn_test.min(axis=0)).all()
    # NB: dtype can change depending on the device, so we accept all valid dtypes.
    assert centers.dtype.type in {np.float32, np.float64}

    # Check that indices correspond to reported centers
    assert_allclose(X_sklearn_test[indices].astype(dtype), centers)


def test_kmeans_plusplus_dataorder():
    """Test adapted from sklearn's test_kmeans_plusplus_dataorder"""
    # Check that memory layout does not effect result
    random_state = 42

    estimator = KMeans(
        init="k-means++", n_clusters=n_clusters_sklearn_test, random_state=random_state
    )
    engine = KMeansEngine(estimator)
    X_sklearn_test_prepared, _, sample_weight = engine.prepare_fit(X_sklearn_test)
    centers_c = engine.init_centroids(X_sklearn_test_prepared, sample_weight)
    centers_c = asnumpy(centers_c)

    X_fortran = np.asfortranarray(X_sklearn_test)
    # The engine is re-created to reset random state
    engine = KMeansEngine(estimator)
    X_fortran_prepared, _, sample_weight = engine.prepare_fit(X_fortran)
    centers_fortran = engine.init_centroids(X_fortran_prepared, sample_weight)
    centers_fortran = asnumpy(centers_fortran)

    assert_allclose(centers_c, centers_fortran)


def test_xpu_supports_torch_generator():
    """This test will fail once the xpu backend supports on-device RNG with
    torch.Generator RNG. From then on, please adapt the KMeans engine so that
    it uses on-device torch.Generator rather than np.randon.RandomState, then
    remove this unit test"""
    with pytest.raises(Exception):
        torch.Generator(device="xpu")
