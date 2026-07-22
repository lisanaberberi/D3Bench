"""Module for Alibi Detect detectors."""

from typing import Any, Optional

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from alibi_detect import cd

from d3bench import utils


def _statistic_value(data: dict[str, Any]) -> Optional[Any]:
    """Return whichever of distance / test_stat / p_val alibi populated.

    Offline (batch) detectors report `distance` (a scalar for MMD/LSDD, a
    per-feature array for the univariate tests) alongside `p_val`. Online
    detectors instead report `test_stat` once past the ERT calibration
    window, leaving `distance`/`p_val` at None until then.
    """
    for key in ("distance", "test_stat", "p_val"):
        value = data.get(key)
        if value is not None:
            return value
    return None


# Concept Drift Online Detector Methods


class BaseUniOnlineTest(utils.BaseTestMethod, ABC):
    """Base class for online univariate drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.detector: Any
        self.drift: Any

    @property
    @abstractmethod
    def config(self) -> Any:
        """Property that returns the detector configuration."""

    @property
    @abstractmethod
    def detector_class(self) -> Any:
        """Property that returns the detector class."""

    max_samples: int | None = None          # None = full data
    _SEED = 31

    def _subsample(self, x: np.ndarray) -> np.ndarray:
        x = x[:, self._numeric_idx].astype(np.float32, copy=False)
        if self.max_samples is None or len(x) <= self.max_samples:
            return x
        rng = np.random.default_rng(self._SEED)
        return x[rng.choice(len(x), size=self.max_samples, replace=False)]

    def fit(self, x_reference: np.ndarray) -> None:
        # MMD/LSDD/CvM are continuous-only kernel/distance methods -- drop
        # categorical columns (a no-op on all-numeric datasets like
        # energy/occupancy) and narrow self.features to match, so per-feature
        # results (e.g. CVM online's per-feature statistic) stay aligned.
        self._numeric_idx = utils.numeric_column_indices(x_reference)
        self.features = [self.features[i] for i in self._numeric_idx]
        self.detector = self.detector_class(self._subsample(x_reference), **self.config)


    def test(self, x_test: np.ndarray) -> None: # to catch if drift fires early and the stream returns to normal
        self.drift_ever = False
        self.drift = None
        x_test = x_test[:, self._numeric_idx].astype(np.float32, copy=False)
        # detector.predict() is one TF forward pass per row; streaming the full split
        # (tens of thousands of rows) can take an hour+. Cap to a leading prefix --
        # not a random subsample -- so window/ERT-based online detectors still see a
        # contiguous, time-ordered stream.
        if self.max_samples is not None and len(x_test) > self.max_samples:
            x_test = x_test[: self.max_samples]
        for x in x_test:
            self.drift = self.detector.predict(x)
            if self.drift["data"]["is_drift"]:
                self.drift_ever = True

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {"drift": {feature: self.drift_ever for feature in self.features}}
        statistic = _statistic_value(self.drift["data"]) if self.drift is not None else None
        if statistic is not None:
            # MMD/LSDD online report one combined scalar for all features; CVM online
            # (shape (n_windows, n_features)) reports one value per feature -- flatten
            # and only index per feature when the shapes actually line up.
            statistic = np.asarray(statistic, dtype=float).reshape(-1)
            if statistic.size == len(self.features):
                result["statistic"] = {
                    feature: float(statistic[i]) for i, feature in enumerate(self.features)
                }
            else:
                result["statistic"] = {feature: float(statistic[0]) for feature in self.features}
        return result


class OnlineMaximumMeanDiscrepancy(BaseUniOnlineTest):
    """Online Maximum Mean Discrepancy"""

    detector_class = cd.MMDDriftOnline
    max_samples = 1000 
    config = {
        "ert": 100, # expected instances between false alarms (NOT a p-value),
        "window_size": 10,  # Expected run-time (ERT) in the absence of drift
        "backend": "tensorflow",
        "preprocess_fn": None,
        "x_ref_preprocessed": False,
        "kernel": None,
        "sigma": None,
        "n_bootstraps": 1000,
        "device": None,
        "verbose": True,
        "input_shape": None,
        "data_type": None,
    }


class OnlineLeastSquaresDensityDifference(BaseUniOnlineTest):
    """Online Least-Squares Density Difference"""

    detector_class = cd.LSDDDriftOnline
    max_samples = 1000 
    config = {
        "ert": 100, # expected instances between false alarms (NOT a p-value),
        "window_size": 10,  # Window size for the sliding test-window
        "backend": "tensorflow",
        "preprocess_fn": None,
        "x_ref_preprocessed": False,
        "sigma": None,
        "n_bootstraps": 1000,
        "n_kernel_centers": None,
        "lambda_rd_max": 0.2,
        "device": None,
        "verbose": True,
        "input_shape": None,
        "data_type": None,
    }


class OnlineCramerVonMisesTest(BaseUniOnlineTest):
    """Online Cramér-von Mises Test
    !This detector is multi-threaded, with Numba used to parallelise over the simulated streams.
    """

    detector_class = cd.CVMDriftOnline
    max_samples = 1000  # threshold calibration is O(n^2) in reference size
    config = {
        "ert": 100, # expected instances between false alarms (NOT a p-value),
        "window_sizes": [10],  # Window size for the sliding test-window
        "preprocess_fn": None,
        "x_ref_preprocessed": False,
        "n_bootstraps": 10000,
        "batch_size": 64,
        "n_features": None,
        "verbose": True,
        "input_shape": None,
        "data_type": None,
    }


class OnlineFisherExactTest(BaseUniOnlineTest):
    """Online Fisher Exact Test"""

    detector_class = cd.FETDriftOnline
    config = {
        "ert": 100, # expected instances between false alarms (NOT a p-value),
        "window_sizes": [10],  # Window size for the sliding test-window
        "preprocess_fn": None,
        "x_ref_preprocessed": False,
        "n_bootstraps": 10000,
        "t_max": None,
        "alternative": "greater",
        "lam": 0.99,
        "n_features": None,
        "verbose": True,
        "input_shape": None,
        "data_type": None,
    }


# Special Online Drift Detectors
# TODO: Implement these classes contexts and configurations


class BaseSpecialOnlineTests(utils.BaseTestMethod, ABC):
    """Base class for special offline drift detectors.
    These methods inherit from DriftConfigMixin, there is no base implementation
    in Alibi Detect and therefore a clear classification.
    """

    def fit(self, x_reference: np.ndarray) -> None:
        raise NotImplementedError("Method not implemented.")

    def test(self, x_test: np.ndarray) -> None:
        raise NotImplementedError("Method not implemented.")

    def result(self) -> dict[str, Any]:
        raise NotImplementedError("Method not implemented.")


# Batch Data Drift Univariate Detector Methods


class BaseUnivariateTest(utils.BaseTestMethod, ABC):
    """
    Base class for univariate drift detectors.
    

    drift_type
        Predict drift at the 'feature' or 'batch' level. For 'batch', the test statistics for
        each feature are aggregated using the Bonferroni or False Discovery Rate correction (if n_features>1).
    """

    #: Per-side sample cap. None = full data. Set on kernel detectors (MMD/LSDD)
    #: whose Gram matrix is O(n_ref * n_test) and OOMs on the full split.
    max_samples: int | None = None
    _SEED = 31

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.detector: Any
        self.drift: Any

    @property
    @abstractmethod
    def config(self) -> Any:
        """Property that returns the detector configuration."""

    @property
    @abstractmethod
    def detector_class(self) -> Any:
        """Property that returns the detector class."""

    def _subsample(self, x: np.ndarray) -> np.ndarray:
        x = x[:, self._numeric_idx].astype(np.float32, copy=False)
        if self.max_samples is None or len(x) <= self.max_samples:
            return x
        rng = np.random.default_rng(self._SEED)
        return x[rng.choice(len(x), size=self.max_samples, replace=False)]

    def fit(self, x_reference: np.ndarray) -> None:
        # KS/CvM/FET/MMD/LSDD are continuous-only -- drop categorical columns
        # (a no-op on all-numeric datasets like energy/occupancy) and narrow
        # self.features to match, so per-feature results stay aligned.
        # ChiSquareTest/MixedTypeTabularData need categorical columns *kept*
        # (encoded, not dropped) and override this in BaseMixedTypeTest below.
        self._numeric_idx = utils.numeric_column_indices(x_reference)
        self.features = [self.features[i] for i in self._numeric_idx]
        self.detector = self.detector_class(self._subsample(x_reference), **self.config)

    def test(self, x_test: np.ndarray) -> None:
        self.drift = self.detector.predict(self._subsample(x_test), drift_type="feature")

    def result(self) -> dict[str, Any]:
        is_drift = self.drift["data"]["is_drift"]
        result: dict[str, Any] = {
            "drift": {feature: bool(is_drift[i]) for i, feature in enumerate(self.features)}
        }
        statistic = _statistic_value(self.drift["data"])
        if statistic is not None:
            result["statistic"] = {
                feature: float(statistic[i]) for i, feature in enumerate(self.features)
            }
        return result


class BaseMixedTypeTest(BaseUnivariateTest):
    """Base for detectors that need categorical columns kept, not dropped.

    Unlike the continuous-only BaseUnivariateTest.fit (KS/CvM/FET/MMD/LSDD),
    ChiSquareTest and MixedTypeTabularData are specifically meant to test
    categorical features, so instead of dropping non-numeric columns, this
    label-encodes them to integers (fit on the reference split, reused as-is
    for testing; a category unseen in the reference is coded -1). A no-op on
    all-numeric datasets like energy/occupancy: no columns to encode, and
    every column is kept either way.
    """

    #: Whether to build an explicit `categories_per_feature` dict for the
    #: detector. ChiSquareDrift already treats every feature as categorical
    #: when it's None (matching today's energy/occupancy behaviour either
    #: way); TabularDrift means the *opposite* ("nothing is categorical")
    #: when it's None, so MixedTypeTabularData needs the dict built
    #: explicitly or it silently runs KS-only on every feature.
    infer_categories_per_feature: bool = False

    def _encode(self, x: np.ndarray, fit: bool) -> np.ndarray:
        encoded = x.astype(object).copy()
        if fit:
            self._categorical_idx = [
                i for i in range(x.shape[1]) if i not in utils.numeric_column_indices(x)
            ]
            self._category_maps: dict[int, dict[Any, int]] = {}
        for i in self._categorical_idx:
            if fit:
                codes, uniques = pd.factorize(x[:, i], sort=True)
                self._category_maps[i] = {value: code for code, value in enumerate(uniques)}
                encoded[:, i] = codes
            else:
                lookup = self._category_maps[i]
                encoded[:, i] = [lookup.get(value, -1) for value in x[:, i]]
        return encoded.astype(np.float64)

    def fit(self, x_reference: np.ndarray) -> None:
        encoded = self._encode(x_reference, fit=True)
        self._numeric_idx = list(range(encoded.shape[1]))  # keep every column
        config = dict(self.config)
        if self.infer_categories_per_feature and self._categorical_idx:
            config["categories_per_feature"] = {i: None for i in self._categorical_idx}
        self.detector = self.detector_class(self._subsample(encoded), **config)

    def test(self, x_test: np.ndarray) -> None:
        encoded = self._encode(x_test, fit=False)
        self.drift = self.detector.predict(self._subsample(encoded), drift_type="feature")


class ChiSquareTest(BaseMixedTypeTest):
    """Chi-square Test"""

    detector_class = cd.ChiSquareDrift
    config = {
        "p_val": 0.05,
        "categories_per_feature": None,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "correction": "bonferroni",
        "n_features": None,
        "input_shape": None,
        "data_type": None,
    }


class KolmogorovSmirnovTest(BaseUnivariateTest):
    """Kolmogorov-Smirnov test"""

    detector_class = cd.KSDrift
    config = {
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "correction": "bonferroni",
        "alternative": "two-sided",
        "n_features": None,
        "input_shape": None,
        "data_type": None,
    }


class CramerVonMisesTest(BaseUnivariateTest):
    """Cramér-von Mises Test"""

    detector_class = cd.CVMDrift
    config = {
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "correction": "bonferroni",
        "n_features": None,
        "input_shape": None,
        "data_type": None,
    }


class FisherExactTest(BaseUnivariateTest):
    """Fisher Exact Test"""

    detector_class = cd.FETDrift
    config = {
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "correction": "bonferroni",
        "alternative": "greater",
        "n_features": None,
        "input_shape": None,
        "data_type": None,
    }

class MixedTypeTabularData(BaseMixedTypeTest):
    """Mixed Type Tabular Data"""

    infer_categories_per_feature = True
    detector_class = cd.TabularDrift
    config = {
        "p_val": 0.05,
        "categories_per_feature": None,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "correction": "bonferroni",
        "alternative": "two-sided",
        "n_features": None,
        "input_shape": None,
        "data_type": None,
    }



class BaseMultivariateTest(BaseUnivariateTest):
    """MMD/LSDD: multivariate, single verdict, no drift_type."""

    def test(self, x_test: np.ndarray) -> None:
        self.drift = self.detector.predict(self._subsample(x_test))

    def result(self) -> dict[str, Any]:
        is_drift = bool(self.drift["data"]["is_drift"])
        result: dict[str, Any] = {"drift": {feature: is_drift for feature in self.features}}
        statistic = _statistic_value(self.drift["data"])
        if statistic is not None:
            result["statistic"] = {feature: float(statistic) for feature in self.features}
        return result



class MaximumMeanDiscrepancy(BaseMultivariateTest):
    """Maximum Mean Discrepancy"""

    detector_class = cd.MMDDrift
    max_samples = 1000 
    config = {
        "backend": "tensorflow",
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "kernel": None,
        "sigma": None,
        "configure_kernel_from_x_ref": True,
        "n_permutations": 100,
        "batch_size_permutations": 1000000,
        "device": None,
        "input_shape": None,
        "data_type": None,
    }


class LeastSquaresDensityDifference(BaseMultivariateTest):
    """Least-Squares Density Difference"""

    detector_class = cd.LSDDDrift
    max_samples = 1000 
    config = {
        "backend": "tensorflow",
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "sigma": None,
        "n_permutations": 100,
        "n_kernel_centers": None,
        "lambda_rd_max": 0.2,
        "device": None,
        "input_shape": None,
        "data_type": None,
    }

# Special Offline Drift Detectors
# TODO: Implement these classes contexts and configurations


class BaseSpecialOfflineTests(utils.BaseTestMethod, ABC):
    """Base class for special offline drift detectors.
    These methods inherit from DriftConfigMixin, there is no base implementation
    in Alibi Detect and therefore a clear classification.
    """

    def fit(self, x_reference: np.ndarray) -> None:
        raise NotImplementedError("Method not implemented.")

    def test(self, x_test: np.ndarray) -> None:
        raise NotImplementedError("Method not implemented.")

    def result(self) -> dict[str, Any]:
        raise NotImplementedError("Method not implemented.")


class LearnedKernelDriftDetection(BaseSpecialOfflineTests):
    """Learned Kernel Drift Detection"""

    detector_class = cd.LearnedKernelDrift
    config = {
        # "kernel": Callable, TODO: Implement API and config for kernel
        "backend": "tensorflow",
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "n_permutations": 100,
        "batch_size_permutations": 1000000,
        "var_reg": 1e-5,
        "reg_loss_fn": (lambda kernel: 0),
        "train_size": 0.75,
        "retrain_from_scratch": True,
        "optimizer": None,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "batch_size_predict": 32,
        "preprocess_batch_fn": None,
        "epochs": 3,
        "num_workers": 0,
        "verbose": 0,
        "train_kwargs": None,
        "device": None,
        "dataset": None,
        "dataloader": None,
        "input_shape": None,
        "data_type": None,
    }


class ClassifierDriftDetector(BaseSpecialOfflineTests):
    """Classifier Drift Detector"""

    detector_class = cd.ClassifierDrift
    config = {
        # "model": Callable,  TODO: Implement API and config for model
        "backend": "tensorflow",
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "preds_type": "probs",
        "binarize_preds": False,
        "reg_loss_fn": (lambda model: 0),
        "train_size": 0.75,
        "n_folds": None,
        "retrain_from_scratch": True,
        "seed": 0,
        "optimizer": None,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "preprocess_batch_fn": None,
        "epochs": 3,
        "verbose": 0,
        "train_kwargs": None,
        "device": None,
        "dataset": None,
        "dataloader": None,
        "input_shape": None,
        "use_calibration": False,
        "calibration_kwargs": None,
        "use_oob": False,
        "data_type": None,
    }


class SpotTheDiffDriftDetector(BaseSpecialOfflineTests):
    """Spot The Diff Drift Detector"""

    detector_class = cd.SpotTheDiffDrift
    config = {
        "backend": "tensorflow",
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_fn": None,
        "kernel": None,
        "n_diffs": 1,
        "initial_diffs": None,
        "l1_reg": 0.01,
        "binarize_preds": False,
        "train_size": 0.75,
        "n_folds": None,
        "retrain_from_scratch": True,
        "seed": 0,
        "optimizer": None,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "preprocess_batch_fn": None,
        "epochs": 3,
        "verbose": 0,
        "train_kwargs": None,
        "device": None,
        "dataset": None,
        "dataloader": None,
        "input_shape": None,
        "data_type": None,
    }


class ClassifierUncertaintyDriftDetector(BaseSpecialOfflineTests):
    """Classifier Uncertainty Drift Detector"""

    detector_class = cd.ClassifierUncertaintyDrift
    config = {
        # model: Callable, # TODO: Implement API and config for model
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "backend": None,
        "update_x_ref": None,
        "preds_type": "probs",
        "uncertainty_type": "entropy",
        "margin_width": 0.1,
        "batch_size": 32,
        "preprocess_batch_fn": None,
        "device": None,
        "tokenizer": None,
        "max_len": None,
        "input_shape": None,
        "data_type": None,
    }


class ContextAwareMaximumMeanDiscrepancy(BaseSpecialOfflineTests):
    """Context-Aware Maximum Mean Discrepancy"""

    detector_class = cd.ContextMMDDrift
    config = {
        # c_ref: np.ndarray, TODO: Implement API and config for context
        "backend": "tensorflow",
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_ref": None,
        "preprocess_fn": None,
        "x_kernel": None,
        "c_kernel": None,
        "n_permutations": 1000,
        "prop_c_held": 0.25,
        "n_folds": 5,
        "batch_size": 256,
        "device": None,
        "input_shape": None,
        "data_type": None,
        "verbose": False,
    }
