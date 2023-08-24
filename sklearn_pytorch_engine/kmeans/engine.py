import torch

import math

import contextlib
import numbers
import os
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.utils.validation as sklearn_validation
from sklearn.cluster._kmeans import KMeansCythonEngine
from sklearn.exceptions import NotSupportedByEngineError
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _is_arraylike_not_scalar
from sklearn.utils.extmath import stable_cumsum

from sklearn_pytorch_engine.testing import override_attr_context


from sklearn_pytorch_engine.testing.config import (
    has_fp64_support,
    get_torch_default_device,
    get_torch_array_api_namespace,
    to_pytorch_dtype,
)


class KMeansEngine(KMeansCythonEngine):
    """Implementation of Lloyd's k-means in pure pytorch.

    It uses `torch.cdist` and `torch.argmin` to find the closest centroid,
    and then `torch.expand` and `torch.scatter_add_` to update the centroids.

    Those two steps are ran one batch of data at a time and the size of the
    batches is set so that no more than `max_compute_buffer_bytes` extra
    memory have to be allocated during compute for intermediary objects.

    This class instantiates into a callable that mimics the interface of sklearn's
    private function `_kmeans_single_lloyd` .
    """

    # NB: by convention, all public methods override the corresponding method from
    # `sklearn.cluster._kmeans.KMeansCythonEngine`. All methods that do not override a
    # method in `sklearn.cluster._kmeans.KMeansCythonEngine` are considered private to
    # this implementation.

    # This class attribute can alter globally the attributes `device` and `order` of
    # future instances. It is only used for testing purposes, using
    # `sklearn_numba_dpex.testing.config.override_attr_context` context, for instance
    # in the benchmark script.
    # For normal usage, the compute will follow the *compute follows data* principle.
    _CONFIG: Dict[str, Any] = dict()

    engine_name = "kmeans"

    @staticmethod
    def convert_to_sklearn_types(name, value):
        if name in ["cluster_centers_", "labels_"]:
            return value.cpu().numpy()
        return value

    def __init__(self, estimator):
        self.device = self._CONFIG.get("device", None)
        self.max_compute_buffer_bytes = self._CONFIG.get(
            "max_compute_buffer_bytes", 1073741824
        )

        self.estimator = estimator

        _is_in_testing_mode = os.getenv("SKLEARN_PYTORCH_ENGINE_TESTING_MODE", "0")
        if _is_in_testing_mode not in {"0", "1"}:
            raise ValueError(
                "If the environment variable SKLEARN_PYTORCH_ENGINE_TESTING_MODE is"
                f' set, it is expected to take values in {"0", "1"}, but got'
                f" {_is_in_testing_mode} instead."
            )
        self._is_in_testing_mode = _is_in_testing_mode == "1"

    def accepts(self, X, y, sample_weight):
        if (algorithm := self.estimator.algorithm) not in ("lloyd", "auto", "full"):
            if self._is_in_testing_mode:
                raise NotSupportedByEngineError(
                    "The sklearn_pytorch_engine engine for KMeans only support the"
                    f" Lloyd algorithm, {algorithm} is not supported."
                )
            else:
                return False

        if sp.issparse(X):
            return self._is_in_testing_mode

        return True

    def prepare_fit(self, X, y=None, sample_weight=None):
        estimator = self.estimator

        X = self._validate_data(X)
        estimator._check_params_vs_input(X)

        sample_weight = self._check_sample_weight(sample_weight, X)

        init = self.estimator.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = self._check_init(init, X)

        X_mean = X.mean(dim=0)
        X -= X_mean

        if init_is_array_like:
            init -= X_mean

        self.init = init

        if (tol := estimator.tol) != 0:
            tol = torch.mean(torch.var(X, dim=0)) * tol
        self.tol = tol

        if self._is_in_testing_mode and X_mean is not None:
            X_mean = X_mean.cpu().numpy()

        self.X_mean = X_mean

        use_numpy_random = isinstance(estimator.random_state, np.random.RandomState)
        self.random_state = check_random_state(estimator.random_state)
        if not self._is_in_testing_mode and not use_numpy_random:
            self.random_state = torch.Generator(device=X.device).manual_seed(
                self.random_state.randint(2**30, 2**31 - 1, size=1).item()
            )

        return X, y, sample_weight

    def unshift_centers(self, X, best_centers):
        if self._is_in_testing_mode:
            X = X.cpu().numpy()

        return super().unshift_centers(X, best_centers)

    def init_centroids(self, X, sample_weight):
        init = self.init
        n_clusters = self.estimator.n_clusters

        if isinstance(init, torch.Tensor):
            centers = init

        elif isinstance(init, str) and init == "k-means++":
            centers, _ = self._kmeans_plusplus(X, sample_weight)

        elif callable(init):
            centers = init(X, self.estimator.n_clusters, random_state=self.random_state)
            centers = self._check_init(centers, X)

        else:
            n_samples = X.shape[0]
            if isinstance(self.random_state, np.random.RandomState):
                # use numpy rng to ensure reproducibility consistent with
                # scikit learn tests
                # TODO: what happens if we just keep torch rng ?
                if self.sample_weight_is_uniform:
                    sample_weight_ = np.full(
                        (n_samples,), sample_weight, dtype=np.float32
                    )
                else:
                    sample_weight_ = sample_weight.cpu().numpy()

                centers_idx = self.random_state.choice(
                    n_samples,
                    size=n_clusters,
                    replace=False,
                    p=sample_weight_ / sample_weight_.sum(),
                )
            else:
                if self.sample_weight_is_uniform:
                    p = torch.tensor(1, dtype=X.dtype, device=X.device).expand(
                        n_samples
                    )
                else:
                    p = sample_weight

                centers_idx = torch.multinomial(p, n_clusters, replacement=False)

            centers = X[centers_idx, :]

        return centers

    def _kmeans_plusplus(self, X, sample_weight):
        device = X.device
        n_samples, n_features = X.shape
        n_clusters = self.estimator.n_clusters
        compute_dtype = X.dtype

        # Same retrial heuristic as scikit-learn (at least until <1.2)
        n_local_trials = 2 + int(np.log(n_clusters))

        centers = torch.empty(
            n_clusters, n_features, dtype=compute_dtype, device=device
        )
        indices = torch.empty(n_clusters, dtype=torch.int32, device=device)

        first_center_idx = torch.randint(
            n_samples,
            (1,),
            generator=self.random_state,
            out=indices[0],
            device=device,
            dtype=indices.dtype,
        )

        first_center = centers[0] = X[first_center_idx]

        closest_dist_sq = torch.cdist(first_center, X)

        current_pot = closest_dist_sq * sample_weight

        if isinstance(self.random_state, np.random.RandomState):

            def _sample_candidate_ids(current_pot):
                current_pot_ = current_pot.cpu().numpy()
                rand_vals = (
                    self.random_state.uniform(size=n_local_trials) * current_pot_
                )
                candidate_ids = np.searchsorted(stable_cumsum(current_pot_), rand_vals)
                # XXX: numerical imprecision can result in a candidate_id out of range
                np.clip(
                    candidate_ids, None, len(closest_dist_sq) - 1, out=candidate_ids
                )

                return torch.asarray(candidate_ids, device=device)

        else:

            def _sample_candidate_ids(current_pot):
                return torch.multinomial(
                    current_pot,
                    n_local_trials,
                    replacement=False,
                    generator=self.random_state,
                )

        # Pick the remaining n_clusters-1 points
        for c in range(1, n_clusters):
            # First, let's sample indices of candidates using a empirical cumulative
            # density function built using the potential of the samples and squared
            # distances to each sample's closest centroids.
            candidate_ids = _sample_candidate_ids(current_pot)

            # Now, for each (sample, candidate)-pair, compute the minimum between
            # their distance and the previous minimum.

            # XXX: at the cost of one additional pass on data AND batching the
            # pairwise_distance + minimum + argmin sequence, like it is done in lloyd,
            # we could avoid storing entirely distance_to_candidates_t in memory, and
            # limit memory allocations up to a user-facing parameter value.
            # Which is better ?
            distance_to_candidates = torch.cdist(X[candidate_ids], X)
            torch.minimum(
                closest_dist_sq, distance_to_candidates, out=distance_to_candidates
            )
            candidates_pot = torch.matmul(
                distance_to_candidates, sample_weight.reshape(-1, 1)
            )

            best_candidate = torch.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            centers[c] = X[best_candidate]
            indices[c] = best_candidate

        return centers, indices

    def kmeans_single(self, X, sample_weight, centers_init):
        centroids = centers_init
        max_iter = self.estimator.max_iter
        verbose = self.estimator.verbose
        tol = self.tol
        max_compute_buffer_bytes = self.max_compute_buffer_bytes

        device = X.device
        compute_dtype = X.dtype

        n_samples, n_features = X.shape
        n_clusters = centroids.shape[0]

        compute_dtype_itemsize = X[-1, -1].cpu().numpy().dtype.itemsize

        # The computation of nearest centroids will be batched and the size of each
        # batch is set so that the size of the buffer of pairwise distances computed
        # for this batch do not exceed `maximum_comnpute_buffer_size`
        (
            assignment_batch_size,
            assignment_n_batches,
            assignment_n_full_batches,
            assignment_last_batch_size,
        ) = _get_batch_properties(
            expected_bytes_per_sample=n_clusters * compute_dtype_itemsize,
            max_compute_buffer_bytes=max_compute_buffer_bytes,
            dataset_n_samples=n_samples,
        )

        if not self.sample_weight_is_uniform:
            # Batching the update of the centroids is also necessary to support
            # non-uniform sample weights.
            (
                update_batch_size,
                update_n_batches,
                update_n_full_batches,
                update_last_batch_size,
            ) = _get_batch_properties(
                expected_bytes_per_sample=n_features * compute_dtype_itemsize,
                max_compute_buffer_bytes=max_compute_buffer_bytes,
                dataset_n_samples=n_samples,
            )

        # Pre-allocate buffers that will be reused accross iterations (rather than re-
        # allocated)
        new_centroids = torch.zeros_like(centroids)  # TODO: test memory layouts ?

        weight_in_clusters = torch.zeros(n_clusters, dtype=compute_dtype, device=device)
        new_weight_in_clusters = torch.zeros_like(weight_in_clusters)

        # Those buffers that will store centroid assignments for each sample are
        # over-allocated with `n_clusters` extra values ranging for 0 to `n_clusters`,
        # that are used to detect empty clusters later on, using torch.unique
        assignments_idx_extended = torch.empty(
            (n_samples + n_clusters, 1), dtype=torch.int64, device=device
        )
        assignments_idx_extended[n_samples:] = torch.arange(
            n_clusters, dtype=assignments_idx_extended.dtype, device=device
        ).unsqueeze(1)
        assignments_idx = assignments_idx_extended[:n_samples]

        new_assignments_idx_extended = torch.empty_like(assignments_idx_extended)
        new_assignments_idx_extended[n_samples:] = assignments_idx_extended[n_samples:]
        new_assignments_idx = new_assignments_idx_extended[:n_samples]

        dist_to_nearest_centroid = torch.empty(
            (n_samples, 1), dtype=compute_dtype, device=device
        )
        dist_to_nearest_centroid_sqz = dist_to_nearest_centroid.squeeze(1)

        n_iteration = 0
        strict_convergence = False
        centroid_shifts_sum = torch.inf

        while (n_iteration < max_iter) and (centroid_shifts_sum > tol):
            # NB: this implementation of _min_over_pairwise_distance is suboptimal
            # because for each batch it materializes in memory the pairwise distance
            # matrix, before searching the closest centroid. The IO from writing and
            # reading from global memory is likely to be the bottleneck. It can be
            # considerably faster if the pairwise distance and the min lookup are fused
            # together in a way that global memory is not used anymore. That requires a
            # custom low level implementation (e.g using triton directly).
            # `torch.compiler` doesn't seem to support automatically fusing
            # `torch.cdist` and `torch.min`. An optimized implementation is available
            # at the `sklearn_numba_dpex` project
            _min_over_pairwise_distance(
                X,
                centroids,
                assignment_n_batches,
                assignment_n_full_batches,
                assignment_batch_size,
                assignment_last_batch_size,
                # OUT
                dist_to_nearest_centroid,
                new_assignments_idx,
            )

            # ???: should we pass `sorted=False` ?
            unique_clusters, counts = torch.unique(
                new_assignments_idx_extended, return_counts=True
            )
            empty_clusters_list = unique_clusters[counts == 1]

            new_centroids[:] = 0
            new_weight_in_clusters[:] = 0

            # relocate empty clusters if such clusters are detected
            if (n_empty_clusters := len(empty_clusters_list)) > 0:
                # ???: should we pass `sorted=False` ?
                samples_far_from_center = torch.topk(
                    dist_to_nearest_centroid_sqz, n_empty_clusters
                ).indices
                new_centroids[empty_clusters_list] = X[samples_far_from_center]
                new_assignments_idx[
                    samples_far_from_center
                ] = empty_clusters_list.unsqueeze(1)
                dist_to_nearest_centroid[samples_far_from_center] = 0

            if verbose:
                inertia = (
                    (
                        sample_weight
                        * dist_to_nearest_centroid_sqz
                        * dist_to_nearest_centroid_sqz
                    )
                    .sum()
                    .item()
                )
                print(f"Iteration {n_iteration}, inertia {inertia:5.3e}")

            # update centers
            # NB: (same comment than for `_min_over_pairwise_distance`)

            if self.sample_weight_is_uniform:
                new_centroids.scatter_add_(
                    dim=0, index=new_assignments_idx.expand(-1, n_features), src=X
                )
                new_centroids *= sample_weight
                new_weight_in_clusters.scatter_add_(
                    dim=0,
                    index=new_assignments_idx.squeeze(),
                    src=sample_weight.expand(n_samples),
                )

            else:
                # Multipliying with weights and then using `scatter_add_` could be
                # fused together, yet again with a x2 - x3 speedup.
                batch_start_idx = batch_end_idx = 0
                for batch_idx in range(update_n_batches):
                    if batch_idx == update_n_full_batches:
                        batch_end_idx += update_last_batch_size
                    else:
                        batch_end_idx += update_batch_size
                    batch_slice = slice(batch_start_idx, batch_end_idx)
                    X_weighted = X[batch_slice] * sample_weight[batch_slice].unsqueeze(
                        1
                    )
                    new_centroids.scatter_add_(
                        dim=0,
                        # NB: expand does not allocate memory, it's like a
                        # "repeated view"
                        index=new_assignments_idx[batch_slice].expand(-1, n_features),
                        src=X_weighted,
                    )
                    del X_weighted
                    # HACK: force synchronization to avoid memory overflow
                    # Similar to torch.cuda.synchronize(X.device) but with device
                    # interoperability for a negligible cost.
                    new_centroids[-1, -1].item()
                    batch_start_idx += update_batch_size

                new_weight_in_clusters.scatter_add_(
                    dim=0, index=new_assignments_idx.squeeze(), src=sample_weight
                )

            new_centroids /= new_weight_in_clusters.unsqueeze(1)

            centroids, new_centroids = new_centroids, centroids
            assignments_idx, new_assignments_idx = new_assignments_idx, assignments_idx
            assignments_idx_extended, new_assignments_idx_extended = (
                new_assignments_idx_extended,
                assignments_idx_extended,
            )

            n_iteration += 1

            if (n_iteration > 1) and (
                strict_convergence := bool(
                    (assignments_idx == new_assignments_idx).all()
                )
            ):
                break

            new_centroids -= centroids
            new_centroids *= new_centroids
            centroid_shifts_sum = new_centroids.sum().item()

        if verbose:
            converged_at = n_iteration - 1
            # NB: possible if tol = 0
            if strict_convergence or (centroid_shifts_sum == 0):
                print(f"Converged at iteration {converged_at}: strict convergence.")

            elif centroid_shifts_sum <= tol:
                print(
                    f"Converged at iteration {converged_at}: center shift "
                    f"{centroid_shifts_sum} within tolerance {tol}."
                )

        if not strict_convergence:
            _min_over_pairwise_distance(
                X,
                centroids,
                assignment_n_batches,
                assignment_n_full_batches,
                assignment_batch_size,
                assignment_last_batch_size,
                # OUT
                dist_to_nearest_centroid,
                assignments_idx,
            )

        inertia = (
            (
                sample_weight
                * dist_to_nearest_centroid_sqz
                * dist_to_nearest_centroid_sqz
            )
            .sum()
            .item()
        )

        return assignments_idx.squeeze(), inertia, centroids, n_iteration

    def is_same_clustering(self, labels, best_labels, n_clusters):
        if self._is_in_testing_mode:
            return super().is_same_clustering(labels, best_labels, n_clusters)

        # TODO: this implementation relies on sort that is nlogn complexity. Is there
        # a solution for linear complexity just using pytorch functions ?
        sorting_index = torch.argsort(labels)

        groupby_equal_labels = torch.nonzero(torch.diff(labels[sorting_index]))
        groupby_equal_best_labels = torch.nonzero(
            torch.diff(best_labels[sorting_index])
        )

        if len(groupby_equal_labels) != len(groupby_equal_best_labels):
            return False

        return (groupby_equal_labels == groupby_equal_best_labels).all()

    def count_distinct_clusters(self, best_labels):
        if self._is_in_testing_mode:
            return super().count_distinct_clusters(best_labels)

        return len(torch.unique(best_labels))

    def prepare_prediction(self, X, sample_weight):
        X = self._validate_data(X, reset=False)
        sample_weight = self._check_sample_weight(sample_weight, X)
        return X, sample_weight

    def get_labels(self, X, sample_weight):
        # TODO: sample_weight actually not used for get_labels. Fix in sklearn ?
        # Relevant issue: https://github.com/scikit-learn/scikit-learn/issues/25066
        labels, _ = self._get_labels_inertia(X, sample_weight, with_inertia=False)

        if self._is_in_testing_mode:
            labels = labels.cpu().numpy().astype(np.int32)

        return labels

    def get_score(self, X, sample_weight):
        _, inertia = self._get_labels_inertia(X, sample_weight, with_inertia=True)
        return inertia

    def _get_labels_inertia(self, X, sample_weight, with_inertia=True):
        cluster_centers = self._check_init(
            self.estimator.cluster_centers_, X, copy=False
        )

        device = X.device
        compute_dtype = X.dtype

        n_samples, n_features = X.shape
        n_clusters = cluster_centers.shape[0]

        max_compute_buffer_bytes = self.max_compute_buffer_bytes

        compute_dtype_itemsize = X[-1, -1].cpu().numpy().dtype.itemsize

        (
            batch_size,
            n_batches,
            n_full_batches,
            last_batch_size,
        ) = _get_batch_properties(
            expected_bytes_per_sample=n_clusters * compute_dtype_itemsize,
            max_compute_buffer_bytes=max_compute_buffer_bytes,
            dataset_n_samples=n_samples,
        )

        dist_to_nearest_centroid = torch.empty(
            (n_samples, 1), dtype=compute_dtype, device=device
        )
        assignments_idx = torch.empty((n_samples, 1), dtype=torch.int64, device=device)

        _min_over_pairwise_distance(
            X,
            cluster_centers,
            n_batches,
            n_full_batches,
            batch_size,
            last_batch_size,
            # OUT
            dist_to_nearest_centroid,
            assignments_idx,
        )

        if with_inertia:
            dist_to_nearest_centroid = dist_to_nearest_centroid.squeeze()
            inertia = (
                (sample_weight * dist_to_nearest_centroid * dist_to_nearest_centroid)
                .sum()
                .item()
            )

        else:
            inertia = None

        return assignments_idx.squeeze(), inertia

    def prepare_transform(self, X):
        # TODO: fix fit_transform in sklearn: need to call prepare_transform
        # inbetween fit and transform ? or remove prepare_transform ?
        return X

    def get_euclidean_distances(self, X):
        X = self._validate_data(X, reset=False)
        cluster_centers = self._check_init(
            self.estimator.cluster_centers_, X, copy=False
        )

        euclidean_distances = torch.cdist(X, cluster_centers)

        if self._is_in_testing_mode:
            euclidean_distances = (
                euclidean_distances.cpu().numpy().astype(self.estimator._output_dtype)
            )

        return euclidean_distances

    def _validate_data(self, X, reset=True):
        accepted_dtypes = [np.float32]

        if self.device is not None:
            device = torch.device(self.device)

        elif isinstance(X, torch.Tensor):
            device = X.device

        else:
            device = get_torch_default_device()

        # NB: one could argue that `float32` is a better default, but sklearn defaults
        # to `np.float64` and we apply the same for consistency.
        if has_fp64_support(device):
            accepted_dtypes = [torch.float64, torch.float32]
        else:
            accepted_dtypes = [torch.float32]

        if self._is_in_testing_mode and reset:
            if (X_dtype := np.dtype(X.dtype)) not in accepted_dtypes:
                self.estimator._output_dtype = np.float64
            else:
                self.estimator._output_dtype = X_dtype

        with _validate_with_array_api(device):
            try:
                # TODO: investigate use of order for passing contiguous
                # format option, and impact on performance ?
                X = self.estimator._validate_data(
                    X,
                    accept_sparse=False,
                    dtype=accepted_dtypes,
                    copy=self.estimator.copy_x,
                    reset=reset,
                    force_all_finite=True,
                    estimator=self.estimator,
                )
                return X
            except TypeError as type_error:
                if "A sparse matrix was passed, but dense data is required" in str(
                    type_error
                ):
                    raise NotSupportedByEngineError from type_error

    def _check_sample_weight(self, sample_weight, X):
        """Adapted from sklearn.utils.validation._check_sample_weight to be compatible
        with Array API dispatch"""
        n_samples = X.shape[0]
        dtype = X.dtype
        device = X.device
        if sample_weight is None:
            self.sample_weight_is_uniform = True
            return torch.tensor(1, dtype=dtype, device=device)

        elif isinstance(sample_weight, numbers.Number):
            self.sample_weight_is_uniform = True
            return torch.tensor(sample_weight, dtype=dtype, device=device)

        else:
            with _validate_with_array_api(device):
                sample_weight = check_array(
                    sample_weight,
                    accept_sparse=False,
                    dtype=dtype,
                    force_all_finite=True,
                    ensure_2d=False,
                    allow_nd=False,
                    estimator=self.estimator,
                    input_name="sample_weight",
                )

            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            if sample_weight.shape != (n_samples,):
                raise ValueError(
                    "sample_weight.shape == {}, expected {}!".format(
                        sample_weight.shape, (n_samples,)
                    )
                )

            if (sample_weight == (first_sample_weight := sample_weight[0])).all():
                self.sample_weight_is_uniform = True
                return torch.tensor(first_sample_weight, dtype=dtype, device=device)

            return sample_weight

    def _check_init(self, init, X, copy=False):
        device = X.device
        with _validate_with_array_api(device):
            init = check_array(
                init,
                dtype=X.dtype,
                accept_sparse=False,
                copy=False,
                force_all_finite=True,
                ensure_2d=True,
                estimator=self.estimator,
                input_name="init",
            )
            self.estimator._validate_center_shape(X, init)
            return init


def _get_namespace(*arrays):
    return get_torch_array_api_namespace(), True


@contextlib.contextmanager
def _validate_with_array_api(device):
    def _asarray_with_order(array, dtype, order, copy=None, xp=None):
        if order is not None:
            raise ValueError(
                f"Got order={order}, but enforcing a specific order is not "
                "supported by this plugin."
            )
        return torch.asarray(array, dtype=dtype, copy=copy, device=device)

    # TODO: when https://github.com/scikit-learn/scikit-learn/issues/25000 is
    #  solved remove those hacks.
    with sklearn.config_context(
        array_api_dispatch=True,
        assume_finite=True  # workaround 1: disable force_all_finite
        # TODO: might not be needed anymore
        # workaround2 : monkey patch get_namespace and _asarray_with_order to force
        # torch array namespace
    ), override_attr_context(
        sklearn_validation,
        get_namespace=_get_namespace,
        _asarray_with_order=_asarray_with_order,
    ):
        yield


def _get_batch_properties(
    expected_bytes_per_sample, max_compute_buffer_bytes, dataset_n_samples
):
    batch_size = max_compute_buffer_bytes / expected_bytes_per_sample

    if batch_size < 1:
        raise RuntimeError("Buffer size is too small")

    batch_size = min(math.floor(batch_size), dataset_n_samples)
    n_batches = math.ceil(dataset_n_samples / batch_size)
    n_full_batches = n_batches - 1
    last_batch_size = ((dataset_n_samples - 1) % batch_size) + 1

    return batch_size, n_batches, n_full_batches, last_batch_size


def _min_over_pairwise_distance(
    X,  # IN    (n_samples, n_features)
    centroids,  # IN    (n_clusters, n_feautres)
    n_batches,  # PARAM int
    n_full_batches,  # PARAM int
    batch_size,  # PARAM int
    last_batch_size,  # PARAM int
    dist_to_nearest_centroid,  # OUT   (n_samples, n_clusters)
    assignments_idx,  # OUT   (n_samples,)
):
    """The result is returned in `dist_to_nearest_centroid` and `assignments_idx`
    arrays that are modified inplace"""
    # TODO: slice here so that pairwise_distance has a max size of 1GB
    batch_start_idx = batch_end_idx = 0
    for batch_idx in range(n_batches):
        if batch_idx == n_full_batches:
            batch_end_idx += last_batch_size
        else:
            batch_end_idx += batch_size

        batch_slice = slice(batch_start_idx, batch_end_idx)
        pairwise_distances = torch.cdist(X[batch_slice], centroids)
        torch.min(
            pairwise_distances,
            axis=1,
            keepdims=True,
            out=(dist_to_nearest_centroid[batch_slice], assignments_idx[batch_slice]),
        )
        del pairwise_distances
        # HACK: force synchronization to avoid memory overflow
        # Similar to torch.cuda.synchronize(X.device) but with device interoperability
        # for a negligible cost.
        assignments_idx[-1, -1].item()

        batch_start_idx += batch_size
