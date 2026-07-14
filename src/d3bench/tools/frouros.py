"""Module for Frouros detectors."""

from typing import Any, Optional

from abc import ABC, abstractmethod
from frouros.callbacks.batch import PermutationTestDistanceBased
from frouros.detectors import concept_drift, data_drift
import numpy as np

from d3bench import utils
from d3bench.config import SIGNIFICANCE_LEVEL
from frouros.utils.kernels import rbf_kernel

from scipy.stats import PermutationMethod

# TODO: Might be interesting to move all configurations to a toml file

# Settings for the permutation test that supplies a p-value (and thus an
# is-drifted decision) to the otherwise threshold-free distance-based batch
# detectors. `num_permutations` bounds the achievable p-value from below at
# 1 / (num_permutations + 1), so it must sit comfortably under
# SIGNIFICANCE_LEVEL: at 20 the floor is 0.0476 against a 0.05 threshold,
# leaving the test with effectively no resolution. 100 gives a floor of 0.0099.
# It also multiplies detector runtime by roughly this factor.
_PERMUTATION_TEST_KWARGS: dict[str, Any] = {
    "num_permutations": 100,
    "random_state": 31,
}
 
# Seed shared by every stochastic component in this module.
_SEED = 31
 
 
def _permutation_test() -> PermutationTestDistanceBased:
    """Build a *fresh* permutation-test callback.
 
    Never share one instance between detectors. `frouros.callbacks.BaseCallback`
    carries per-instance `detector` and `logs` state, and `compare()` returns
    that `logs` dict *by reference*. A module-level singleton therefore makes
    every per-feature detector alias the same dict, so all features report
    whichever p-value was computed last -- a silent, total false-negative.
    """
    return PermutationTestDistanceBased(**_PERMUTATION_TEST_KWARGS)



# Online Concept Drift Detection


class BaseOnlineCD(utils.BaseTestMethod, ABC):
    """Base class for online concept drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.detector = self.detector_class(self.config)
        self.features = features
        self.drift: bool

    @property
    @abstractmethod
    def config(self) -> Any:
        """Property that returns the detector configuration."""

    @property
    @abstractmethod
    def detector_class(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: np.ndarray) -> None:
        # Detector is trained one by one on the reference data
        # See:
        # https://frouros.readthedocs.io/en/latest/examples/concept_drift/DDM_advance.html#warm-up-phase
        # !!! only 1000 instances are used for training, very high time consumption
        for x in np.linalg.norm(x_reference[:1000], ord=2, axis=1):
            self.detector.update(value=x)

    def test(self, x_test: np.ndarray) -> None:
        # Only one feature is accepted
        for x in np.linalg.norm(x_test, ord=2, axis=1):
            self.detector.update(value=x)

    def result(self) -> dict[str, Any]:
        return {"drift": self.detector.status["drift"]}


class BayesianOnlineChangeDetection(BaseOnlineCD):
    """Bayesian Online Change Detection."""

    detector_class = concept_drift.BOCD
    config = concept_drift.BOCDConfig(
        model=None,
        hazard=1e-2,  # hazard rate
        min_num_instances=1000,  # instances to start looking changes
    )


class CumulativeSumControlChart(BaseOnlineCD):
    """Cumulative Sum Control Chart"""

    detector_class = concept_drift.CUSUM
    config = concept_drift.CUSUMConfig(
        delta=1e-3,  # Delta property
        lambda_=50,  # Threshold property
        min_num_instances=1000,  # instances to start looking changes
    )


class GeometricMovingAverage(BaseOnlineCD):
    """Geometric Moving Average"""

    detector_class = concept_drift.GeometricMovingAverage
    config = concept_drift.GeometricMovingAverageConfig(
        alpha=0.99,  # Forgetting factor value
        lambda_=1.0,  # Threshold property
        min_num_instances=1000,  # instances to start looking changes
    )


class PageHinkleyTest(BaseOnlineCD):
    """Page-Hinkley Test"""

    detector_class = concept_drift.PageHinkley
    config = concept_drift.PageHinkleyConfig(
        delta=1e-3,  # Delta property
        lambda_=50,  # Threshold property
        alpha=0.9999,  # Forgetting factor value
        min_num_instances=1000,  # instances to start looking changes
    )


class DriftDetectionMethod(BaseOnlineCD):
    """Drift Detection Method"""

    detector_class = concept_drift.DDM
    config = concept_drift.DDMConfig(
        warning_level=2.0,  # Warning level property
        drift_level=3.0,  # Drift level property
        min_num_instances=1000,  # instances to start looking changes
    )


class EWMAConceptDriftDetectionWarning(BaseOnlineCD):
    """EWMA Concept Drift Detection Warning"""

    detector_class = concept_drift.ECDDWT
    config = concept_drift.ECDDWTConfig(
        lambda_=0.2,  # Weight given to recent data compared to older data
        average_run_length=400,  # Expected time between false positive detections
        warning_level=0.5,  # Warning level property
        min_num_instances=1000,  # instances to start looking changes
    )


class EarlyDriftDetectionMethod(BaseOnlineCD):
    """Early Drift Detection Method"""

    detector_class = concept_drift.EDDM
    config = concept_drift.EDDMConfig(
        alpha=0.95,  # Warning zone value
        beta=0.9,  # Change zone value
        level=2.0,  # Drift level factor
        min_num_misclassified_instances=1000,  # instances to start looking changes
    )


class HoeffdingDriftDetectionMethodTestA(BaseOnlineCD):
    """Hoeffding's Drift Detection Method Test-A"""

    detector_class = concept_drift.HDDMA
    config = concept_drift.HDDMAConfig(
        alpha_d=0.001,  # Significance level for the drift
        alpha_w=0.005,  # Significance level for the warning
        two_sided_test=False,  # Two-sided test flag
        min_num_instances=1000,  # instances to start looking changes
    )


class HoeffdingDriftDetectionMethodTestW(BaseOnlineCD):
    """Hoeffding's Drift Detection Method Test-W"""

    detector_class = concept_drift.HDDMW
    config = concept_drift.HDDMWConfig(
        alpha_d=0.001,  # Significance level for the drift
        alpha_w=0.005,  # Significance level for the warning
        two_sided_test=False,  # Two-sided test flag
        lambda_=0.05,  # weight given to recent data compared to older data
        min_num_instances=1000,  # instances to start looking changes
    )


class ReactiveDriftDetectionMethod(BaseOnlineCD):
    """Reactive Drift Detection Method"""

    detector_class = concept_drift.RDDM
    config = concept_drift.RDDMConfig(
        warning_level=1.773,  # Warning level property
        drift_level=2.258,  # Drift level property
        max_concept_size=40000,  # Maximum number of instances to consider
        max_num_instances_warning=7000,  # Maximum number of instances warning level
        min_num_instances=100,  # instances to start looking changes
    )


class AdaptiveWindowing(BaseOnlineCD):
    """Adaptive Windowing"""

    detector_class = concept_drift.ADWIN
    config = concept_drift.ADWINConfig(
        clock=32,  # Clock property
        delta=0.002,  # Confidence value
        m=5,  # Amount of mem and closeness of cutpoints checked
        min_window_size=5,  # Minimum number of instances per window to start looking changes
        min_num_instances=1000,  # instances to start looking changes
    )


class KolmogorovSmirnovWindowing(BaseOnlineCD):
    """Kolmogorov-Smirnov Windowing detector."""

    detector_class = concept_drift.KSWIN
    config = concept_drift.KSWINConfig(
        alpha=1e-4,  # significance level
        seed=31,  # random seed
        min_num_instances=1000,  # instances to start looking changes
        num_test_instances=30,  # instances used by statistical test
    )


class StatisticalTestEqualProportionsDetection(BaseOnlineCD):
    """Statistical Test of Equal Proportions"""

    detector_class = concept_drift.STEPD
    config = concept_drift.STEPDConfig(
        alpha_d=0.003,  # Significance value for overall
        alpha_w=0.005,  # Significance value for last
        min_num_instances=1000,  # instances to start looking changes
    )


# Online Data Drift Detection


class OnlineMaximumMeanDiscrepancy(utils.BaseTestMethod):
    """Maximum Mean Discrepancy"""

    detector_class = data_drift.MMDStreaming
    config = {
        "window_size": 10,  # Window size value
        "kernel": rbf_kernel,  # Kernel function
        "chunk_size": 1000,  # Chunk size value
        "callbacks": None,  # Callbacks
    }

    def __init__(self, features: list[str]) -> None:
        self.detector = self.detector_class(**self.config)
        self.features = features
        self.distance = None

    def fit(self, x_reference: np.ndarray) -> None:
        self.detector.fit(X=x_reference)

    def test(self, x_test: np.ndarray) -> None:
        # Take only the first 10 instances for testing as window size is 10
        self.distance = [self.detector.update(value=x)[0] for x in x_test[:10]][-1]

    def result(self) -> dict[str, Any]:
        return {"distance": self.distance}


class IncrementalKolmogorovSmirnovTest(utils.BaseTestMethod):
    """Incremental Kolmogorov-Smirnov Test"""

    detector_class = data_drift.IncrementalKSTest
    config = {
        "window_size": 10,  # Window size value
        "callbacks": None,  # Callbacks
    }

    def __init__(self, features: list[str]) -> None:
        self.detectors = [self.detector_class(**self.config) for _ in features]
        self.features = features
        self.results: list[Any] = [None for _ in features]

    def fit(self, x_reference: np.ndarray) -> None:
        # Only one feature is accepted
        for i, _ in enumerate(self.features):
            # 1000 values otherwise "IndexError": invalid index to scalar variable.
            self.detectors[i].fit(X=x_reference[:1000, i])

    def test(self, x_test: np.ndarray) -> None:
        # Only one feature is accepted
        for i, _ in enumerate(self.features):
            res = [self.detectors[i].update(value=x)[0] for x in x_test[:10, i]][-1]
            self.results[i] = res

    def result(self) -> dict[str, Any]:
        return {
            "distances": [self.results[i].statistic for i, _ in enumerate(self.features)],
            "p_values": [self.results[i].p_value for i, _ in enumerate(self.features)],
            "drift": {
                feature: self.results[i].p_value < SIGNIFICANCE_LEVEL
                for i, feature in enumerate(self.features)
            },
        }


# Batch Concept Drift Detection


# Batch Data Drift Detection


class BaseBatchDD(utils.BaseTestMethod, ABC):
    """Base class for batch data drift detectors.

    One detector is built per feature. Subclasses declare their behaviour with
    three class attributes rather than having it inferred at runtime.
    """
 
    #: True for detectors whose `compare` returns a bare distance (EMD, MMD, PSI,
    #: KL, ...). A distance has no null distribution, so its p-value must come
    #: from a permutation-test callback. False for the statistical tests (KS,
    #: Welch, Mann-Whitney, ...), which carry a p-value on the result object.
    distance_based: bool = False
 
    #: Cap on samples per side; None uses the full split. Needed by detectors
    #: that are super-linear in n (see BatchMaximumMeanDiscrepancy).
    max_samples: Optional[int] = None
 
    #: Extra kwargs forwarded to compare() -> _compare() -> _statistical_test().
    #: REBIND in subclasses (`compare_kwargs = {...}`); never mutate in place --
    #: this is a class attribute shared by every subclass that doesn't override it.
    compare_kwargs: dict[str, Any] = {}
 
    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.detectors = [self._build_detector() for _ in features]
        self.results: list[Any] = []

    @property
    @abstractmethod
    def config(self) -> Any:
        """Property that returns the detector configuration."""
 
    @property
    @abstractmethod
    def detector_class(self) -> Any:
        """Property that returns the detector class."""
 
    def _build_detector(self) -> Any:
        """Instantiate one detector, with its own permutation-test callback if needed."""
        config = dict(self.config)
        if self.distance_based:
            config["callbacks"] = _permutation_test()
        return self.detector_class(**config)
 
    def _subsample(self, x: np.ndarray) -> np.ndarray:
        """Draw at most `max_samples` rows, deterministically.
 
        Reseeded on every call: `test()` runs 7x per method under the default
        scenario (3 runtime repetitions + 3 cpu-runtime + 1 functional), so a
        stateful rng would hand each pass a different draw and the reported
        functional result would come from the last one.
        """
        if self.max_samples is None or len(x) <= self.max_samples:
            return x
        rng = np.random.default_rng(seed=_SEED)
        return x[rng.choice(len(x), size=self.max_samples, replace=False)]
 
    def fit(self, x_reference: np.ndarray) -> None:
        x_reference = self._subsample(x_reference)
        for i, _ in enumerate(self.features):
            self.detectors[i].fit(X=x_reference[:, i])
 
    def test(self, x_test: np.ndarray) -> None:
        x_test = self._subsample(x_test)
        self.results = [
            self.detectors[i].compare(X=x_test[:, i], **self.compare_kwargs)
            for i, _ in enumerate(self.features)
        ]
 
    def _p_value(self, index: int) -> float:
        """Statistical tests carry their own p-value; distance-based detectors
        take theirs from the permutation-test callback log."""
        stat_result, callback_logs = self.results[index]
        if self.distance_based:
            return callback_logs[PermutationTestDistanceBased.__name__]["p_value"]
        return stat_result.p_value
 
    def result(self) -> dict[str, Any]:
        p_values = [self._p_value(i) for i, _ in enumerate(self.features)]
        return {
            "results": self.results,
            "p_values": p_values,
            "drift": {
                feature: p_value < SIGNIFICANCE_LEVEL
                for feature, p_value in zip(self.features, p_values)
            },
        }
 

class BhattacharyyaDistance(BaseBatchDD):
    """Bhattacharyya Distance"""

    detector_class = data_drift.BhattacharyyaDistance
    distance_based = True
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }


class EarthMoverDistance(BaseBatchDD):
    """Earth Mover's Distance"""

    detector_class = data_drift.EMD
    distance_based = True
    config: dict[str, Any] = {}



class EnergyDistance(BaseBatchDD):
    """Energy Distance"""

    detector_class = data_drift.EnergyDistance
    distance_based = True
    config: dict[str, Any] = {}



class HellingerDistance(BaseBatchDD):
    """Hellinger Distance"""

    detector_class = data_drift.HellingerDistance
    distance_based = True
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }



class HistogramIntersectionNormalizedComplement(BaseBatchDD):
    """Histogram Intersection Normalized Complement"""

    detector_class = data_drift.HINormalizedComplement
    distance_based = True
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }



class JensenShannonDivergenceDriftDetection(BaseBatchDD):
    """Jensen-Shannon Divergence Drift Detection"""

    detector_class = data_drift.JS
    distance_based = True
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }


class KullbackLeiblerDivergenceDriftDetection(BaseBatchDD):
    """Kullback-Leibler Divergence Drift Detection"""

    detector_class = data_drift.KL
    distance_based = True
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }



class BatchMaximumMeanDiscrepancy(BaseBatchDD):
    """Maximum Mean Discrepancy"""

    detector_class = data_drift.MMD
    distance_based = True
    config = {
        "kernel": rbf_kernel,  # Kernel function
        "chunk_size": 1000,  # Chunk size value
    }


    # The RBF kernel matrix between reference and test is O(n_reference *
    # n_test); combined with the 20-permutation test above, this is measured
    # to scale quadratically (~4.8s at n=1000 per side, ~102s at n=8000 --
    # extrapolating to Energy's real 35k/70k split gives multi-hour runtimes
    # per feature). Subsample both sides before computing it, which is the
    # standard mitigation for kernel two-sample tests on large samples.
    _max_samples = 1000


class PopulationStabilityIndex(BaseBatchDD):
    """Population Stability Index"""

    detector_class = data_drift.PSI
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
        "callbacks": _PERMUTATION_TEST,
    }


class AndersonDarlingTest(BaseBatchDD):
    """Anderson-Darling Test"""

    detector_class = data_drift.AndersonDarlingTest
    config = {
        "callbacks": None,
    }


class BaumgartnerWeissSchindlerTest(BaseBatchDD):
    """Baumgartner Weiss Schindler Test"""

    detector_class = data_drift.BWSTest
    config = {
        "callbacks": None,
    }
    
    # scipy's bws_test has no asymptotic null, so method=None falls back to
    # PermutationMethod(n_resamples=9999) *unbatched*, which materialises a
    # (9999, n_ref + n_test) float64 array -- ~21 GB at Occupancy's 16k/30k
    # split, i.e. an OOM. batch= bounds the working set; 999 resamples still
    # gives a p-value floor of 0.001.
    compare_kwargs = {"method": PermutationMethod(n_resamples=999, batch=50, random_state=31)}


class ChiSquareTest(BaseBatchDD):
    """Chi-square Test"""

    detector_class = data_drift.ChiSquareTest
    config = {
        "callbacks": None,
    }


class CramerVonMisesTest(BaseBatchDD):
    """Cramér-von Mises Test"""

    detector_class = data_drift.CVMTest
    config = {
        "callbacks": None,
    }


class KolmogorovSmirnovTest(BaseBatchDD):
    """Kolmogorov-Smirnov Test"""

    detector_class = data_drift.KSTest
    config = {
        "callbacks": None,
    }


class KuiperTest(BaseBatchDD):
    """Kuiper's Test"""

    detector_class = data_drift.KuiperTest
    config = {
        "callbacks": None,
    }


class MannWhitneyUTest(BaseBatchDD):
    """Mann-Whitney U-Test"""

    detector_class = data_drift.MannWhitneyUTest
    config = {
        "callbacks": None,
    }


class WelchTTest(BaseBatchDD):
    """Welch's T-Test"""

    detector_class = data_drift.WelchTTest
    config = {
        "callbacks": None,
    }
