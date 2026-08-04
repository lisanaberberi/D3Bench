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
    "num_permutations": 50,
    "random_state": 31,
}
 
# Seed shared by every stochastic component in this module.
_SEED = 31

# Cap on samples per side for permutation-heavy batch detectors that have no
# other reason to scale with n (see `max_samples` on BaseBatchDD). Set above
# every existing dataset's largest split side (Energy: 70_052, Occupancy:
# 30_464) so it is a strict no-op there -- reference/testing sizes, and thus
# every reported statistic, stay bit-identical. It only engages for splits
# larger than that, e.g. French Motor's ~514k/164k Region split, where these
# same detectors were measured to take 4-46s *per feature, per call* (and
# each method's fit+test pipeline runs ~10x over a scenario's criteria) --
# capping brings that down to well under a second up to ~8s (BWSTest).
_MAX_SAMPLES = 80_000


def _subsample_rows(x: np.ndarray, max_samples: int) -> np.ndarray:
    """Deterministically cap x's row count, preserving relative row order.

    Unlike BaseBatchDD._subsample (order-invariant batch statistics), this
    backs BaseOnlineCD.test()'s row-by-row streaming loop, where a detector
    like KSWIN or ADWIN is sensitive to sequence -- so sampled indices are
    sorted rather than left in random draw order. A no-op on every existing
    dataset (see _MAX_SAMPLES); only French Motor's testing side is larger.
    """
    if len(x) <= max_samples:
        return x
    rng = np.random.default_rng(seed=_SEED)
    idx = np.sort(rng.choice(len(x), size=max_samples, replace=False))
    return x[idx]
 
 
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
        # Whether drift fired at *any* point during the test pass. status["drift"]
        # is a per-update flag that flips back to False after a detection for
        # several of these detectors (CUSUM, Page-Hinkley, ECDDWT, HDDM-A/W --
        # confirmed by drift_index being set while the end-of-stream flag reads
        # False), so it is NOT a reliable end-of-stream verdict. Latch it here,
        # mirroring the river adapter's drift_ever, so `drift` and `drift_index`
        # always agree: fired-ever <=> drift_index is not None.
        self.drift_ever: bool = False
        # Stream index at which drift first fired (set in test()); None until
        # then. Paired with drift_ever above.
        self.drift_index: Optional[int] = None

    @property
    @abstractmethod
    def config(self) -> Any:
        """Property that returns the detector configuration."""

    @property
    @abstractmethod
    def detector_class(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: np.ndarray) -> None:
        # Skip detectors whose applicability doesn't match the (present/absent)
        # error stream -- e.g. KSWIN under a supervised run.
        self._guard_error_stream()
        # Supervised concept-drift path (drift_type == "concept"): the shared
        # classifier is already trained (d3bench.supervised), so fit has nothing
        # to do here -- the detector self-calibrates its baseline error rate on
        # the first `min_num_instances` of the TEST error stream (see test()),
        # exactly as Frouros's canonical Elec2/DDM example streams only the test
        # set. Deliberately NO warm-up on the reference error stream: fed tens
        # of thousands of reference instances, a ~25%-error Bernoulli stream's
        # ordinary fluctuation trips DDM/CUSUM/ADWIN/STEPD and latches
        # status["drift"] BEFORE any test data -- a warm-up artifact that pinned
        # the verdict True and drift_index at 0.
        if self.error_stream is not None:
            return
        # Unsupervised feature-norm path (covariate/prior). Detector is trained
        # one by one on the reference data. See:
        # https://frouros.readthedocs.io/en/latest/examples/concept_drift/DDM_advance.html#warm-up-phase
        # !!! only 1000 instances are used for training, very high time consumption
        #
        # Every online CD method feeds one detector the L2-norm of the whole
        # feature row (no per-feature indexing anywhere below), so dropping
        # categorical columns is safe for all of them -- a no-op on
        # all-numeric datasets like energy/occupancy.
        self._numeric_idx = utils.numeric_column_indices(
            x_reference, self.features, self.categorical_columns
        )
        if not self._numeric_idx:
            # e.g. drift_type == "prior": the only monitored column is
            # categorical. x_reference[:, []] is a valid but 0-column array,
            # and np.linalg.norm over it would silently produce a constant
            # zero stream -- every detector below would then see zero
            # variance forever, either trivially reporting "no drift" or
            # (observed with STEPD) tripping on the degenerate/zero-variance
            # input for reasons unrelated to real drift. Either way that's a
            # meaningless result dressed up as a real one, worse than
            # failing loudly: raise MethodNotApplicable instead, so
            # _try_report logs one short line instead of a full traceback.
            raise utils.MethodNotApplicable(
                "Frouros online concept-drift detectors have no numeric columns to "
                "monitor here -- they only support a single combined numeric feature "
                "vector (no per-column/categorical path), and every monitored column "
                "for this scenario is non-numeric."
            )
        x_reference = x_reference[:, self._numeric_idx].astype(np.float64)
        for x in np.linalg.norm(x_reference[:1000], ord=2, axis=1):
            self.detector.update(value=x)

    def test(self, x_test: np.ndarray) -> None:
        # Reset before each stream so drift_ever/drift_index describe *this*
        # test pass, not a stale value latched by an earlier one.
        self.drift_ever = False
        self.drift_index = None
        # Supervised concept-drift path: stream the shared classifier's testing
        # error over time (see fit above). status["drift"] can un-latch between
        # updates, so record the *first* firing (drift_index) and that it fired
        # at all (drift_ever) rather than trusting the end-of-stream flag.
        if self.error_stream is not None:
            for i, error in enumerate(self.error_stream.testing):
                self.detector.update(value=float(error))
                if self.detector.status["drift"]:
                    self.drift_ever = True
                    if self.drift_index is None:
                        self.drift_index = i
            return
        # Only one feature is accepted
        x_test = x_test[:, self._numeric_idx].astype(np.float64)
        # Each row is fed through a pure-Python per-instance update() loop with
        # no internal batching -- cheap per call on energy/occupancy's test
        # sizes, but e.g. KSWIN was measured at ~35s/call (and this whole
        # fit+test pipeline runs ~10x per method per scenario) on French
        # Motor's ~164k-row testing split. Capped the same way as the batch
        # detectors' _subsample, and just as much of a no-op there.
        self._record_sample_size("testing", min(len(x_test), _MAX_SAMPLES), len(x_test))
        x_test = _subsample_rows(x_test, _MAX_SAMPLES)
        for i, x in enumerate(np.linalg.norm(x_test, ord=2, axis=1)):
            self.detector.update(value=x)
            if self.detector.status["drift"]:
                self.drift_ever = True
                if self.drift_index is None:
                    self.drift_index = i

    def result(self) -> dict[str, Any]:
        # Online CD detectors expose no uniform test statistic across algorithms.
        # Report the verdict as drift_ever (fired at any point) rather than
        # status["drift"] (the end-of-stream flag, which un-latches after a
        # detection for CUSUM/Page-Hinkley/ECDDWT/HDDM and would report False
        # while drift_index is set -- a self-contradiction). This guarantees
        # drift <=> (drift_index is not None). No comparable D-value exists.
        return {"drift": self.drift_ever, "drift_index": self.drift_index}


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

    # KSWIN's test is a two-sample KS between windows of the monitored *values*
    # -- an assumption about the distribution of X, not classifier performance,
    # and degenerate on a two-valued 0/1 error stream. Excluded from the
    # supervised concept-drift run (see BaseTestMethod.error_stream_capable).
    error_stream_capable = False
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
        # RBF-kernel MMD is continuous-only -- drop categorical columns (a
        # no-op on all-numeric datasets like energy/occupancy). result() only
        # reports a single scalar (no per-feature dict), so no self.features
        # bookkeeping is needed to stay aligned.
        self._numeric_idx = utils.numeric_column_indices(
            x_reference, self.features, self.categorical_columns
        )
        if not self._numeric_idx:
            # e.g. drift_type == "prior": nothing numeric to kernelize.
            # frouros's own DimensionError below already catches this, but
            # raising MethodNotApplicable here first gives _try_report a
            # short, informative line instead of frouros's internal
            # "Dimensions of X (0)" traceback.
            raise utils.MethodNotApplicable(
                "Frouros's online MMD needs at least one numeric column to build a "
                "kernel from, and every monitored column for this scenario is "
                "non-numeric."
            )
        x_reference = x_reference[:, self._numeric_idx].astype(np.float64)
        self.detector.fit(X=x_reference)

    def test(self, x_test: np.ndarray) -> None:
        # Take only the first 10 instances for testing as window size is 10.
        # update() returns (None, {}) until the window fills, then
        # (DistanceResult(distance=...), {}); unwrap it to a plain float.
        x_test = x_test[:, self._numeric_idx].astype(np.float64)
        result = [self.detector.update(value=x)[0] for x in x_test[:10]][-1]
        self.distance = result.distance if result is not None else None

    def result(self) -> dict[str, Any]:
        return {"statistic": self.distance}


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
        # KS is continuous-only -- only fit detectors for numeric columns (a
        # no-op on all-numeric datasets like energy/occupancy); self.detectors
        # is already built one-per-feature in __init__ (before any data is
        # seen), so categorical positions are simply left unfit/unused rather
        # than resized away.
        self._numeric_idx = utils.numeric_column_indices(
            x_reference, self.features, self.categorical_columns
        )
        for i in self._numeric_idx:
            # 1000 values otherwise "IndexError": invalid index to scalar variable.
            self.detectors[i].fit(X=x_reference[:1000, i].astype(np.float64))

    def test(self, x_test: np.ndarray) -> None:
        for i in self._numeric_idx:
            res = [
                self.detectors[i].update(value=x)[0]
                for x in x_test[:10, i].astype(np.float64)
            ][-1]
            self.results[i] = res

    def result(self) -> dict[str, Any]:
        return {
            "statistic": {
                self.features[i]: self.results[i].statistic for i in self._numeric_idx
            },
            "drift": {
                self.features[i]: self.results[i].p_value < SIGNIFICANCE_LEVEL
                for i in self._numeric_idx
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

    #: False (default): continuous-only detectors (KS, Welch, Bhattacharyya, ...)
    #: drop categorical columns -- a no-op on all-numeric datasets like
    #: energy/occupancy. ChiSquareTest overrides this to True: it counts raw
    #: value frequencies, which is meaningful for categorical *and* continuous
    #: columns alike, so nothing needs to be dropped.
    keep_categorical_columns: bool = False

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.detectors = [self._build_detector() for _ in features]
        self.results: dict[int, Any] = {}

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

    def _subsample(self, x: np.ndarray, side: str) -> np.ndarray:
        """Draw at most `max_samples` rows, deterministically.

        Reseeded on every call: `test()` runs 7x per method under the default
        scenario (3 runtime repetitions + 3 cpu-runtime + 1 functional), so a
        stateful rng would hand each pass a different draw and the reported
        functional result would come from the last one.
        """
        if self.max_samples is None or len(x) <= self.max_samples:
            return x
        self._record_sample_size(side, self.max_samples, len(x))
        rng = np.random.default_rng(seed=_SEED)
        return x[rng.choice(len(x), size=self.max_samples, replace=False)]

    def _column(self, x: np.ndarray, i: int) -> np.ndarray:
        """One feature column, cast back to a real numeric dtype where safe.

        x is object-dtype (Frouros.preprocess preserves native per-column
        types rather than letting np.stack silently coerce everything to
        strings). Several detectors (JS, KL, MMD, BWS, Mann-Whitney, Welch-T)
        require an actual float array and raise on dtype=object even when
        every value is numeric, so cast explicitly here -- except for
        ChiSquareTest (keep_categorical_columns=True), whose categorical
        columns can hold raw strings that can't be cast to float.
        """
        column = x[:, i]
        return column if self.keep_categorical_columns else column.astype(np.float64)

    def fit(self, x_reference: np.ndarray) -> None:
        # self.detectors is already built one-per-feature in __init__ (before
        # any data is seen), so categorical positions are simply left
        # unfit/unused below rather than resized away.
        if self.keep_categorical_columns:
            self._numeric_idx = list(range(x_reference.shape[1]))
        else:
            self._numeric_idx = utils.numeric_column_indices(
                x_reference, self.features, self.categorical_columns
            )
        x_reference = self._subsample(x_reference, "reference")
        for i in self._numeric_idx:
            self.detectors[i].fit(X=self._column(x_reference, i))

    def test(self, x_test: np.ndarray) -> None:
        x_test = self._subsample(x_test, "testing")
        self.results = {
            i: self.detectors[i].compare(X=self._column(x_test, i), **self.compare_kwargs)
            for i in self._numeric_idx
        }

    def _p_value(self, index: int) -> float:
        """Statistical tests carry their own p-value; distance-based detectors
        take theirs from the permutation-test callback log."""
        stat_result, callback_logs = self.results[index]
        if self.distance_based:
            return callback_logs[PermutationTestDistanceBased.__name__]["p_value"]
        return stat_result.p_value

    def _statistic(self, index: int) -> float:
        """The D-value: the raw distance for distance-based detectors (PSI,
        EMD, ...), the test statistic for the statistical tests (KS, Welch,
        ...) -- distinct from `_p_value`, which is only used for the drift
        decision and, for distance-based detectors, comes from a separate
        permutation-test callback rather than this result object."""
        stat_result, _ = self.results[index]
        if self.distance_based:
            return stat_result.distance
        return stat_result.statistic

    def result(self) -> dict[str, Any]:
        return {
            "statistic": {self.features[i]: self._statistic(i) for i in self._numeric_idx},
            "drift": {
                self.features[i]: self._p_value(i) < SIGNIFICANCE_LEVEL
                for i in self._numeric_idx
            },
        }


class BhattacharyyaDistance(BaseBatchDD):
    """Bhattacharyya Distance"""

    detector_class = data_drift.BhattacharyyaDistance
    distance_based = True
    max_samples = _MAX_SAMPLES
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }


class EarthMoverDistance(BaseBatchDD):
    """Earth Mover's Distance"""

    detector_class = data_drift.EMD
    distance_based = True
    max_samples = _MAX_SAMPLES
    config: dict[str, Any] = {}



class EnergyDistance(BaseBatchDD):
    """Energy Distance"""

    detector_class = data_drift.EnergyDistance
    distance_based = True
    max_samples = _MAX_SAMPLES
    config: dict[str, Any] = {}



class HellingerDistance(BaseBatchDD):
    """Hellinger Distance"""

    detector_class = data_drift.HellingerDistance
    distance_based = True
    max_samples = _MAX_SAMPLES
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }



class HistogramIntersectionNormalizedComplement(BaseBatchDD):
    """Histogram Intersection Normalized Complement"""

    detector_class = data_drift.HINormalizedComplement
    distance_based = True
    max_samples = _MAX_SAMPLES
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }



class JensenShannonDivergenceDriftDetection(BaseBatchDD):
    """Jensen-Shannon Divergence Drift Detection"""

    detector_class = data_drift.JS
    distance_based = True
    max_samples = _MAX_SAMPLES
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }


class KullbackLeiblerDivergenceDriftDetection(BaseBatchDD):
    """Kullback-Leibler Divergence Drift Detection"""

    detector_class = data_drift.KL
    distance_based = True
    max_samples = _MAX_SAMPLES
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
    max_samples = 1000


class PopulationStabilityIndex(BaseBatchDD):
    """Population Stability Index"""

    detector_class = data_drift.PSI
    distance_based = True
    max_samples = _MAX_SAMPLES
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
    }



class AndersonDarlingTest(BaseBatchDD):
    """Anderson-Darling Test"""

    detector_class = data_drift.AndersonDarlingTest
    config = {
        "callbacks": None,
    }
    
    # NOTE: scipy's anderson_ksamp interpolates its p-value from a table covering
    # only [0.001, 0.25] and warns when it clamps. Both clamps fall outside the
    # 0.05 decision boundary, so the drift verdict is unaffected -- but a reported
    # p of 0.001 means "<= 0.001", which matters if these numbers reach a table.



class BaumgartnerWeissSchindlerTest(BaseBatchDD):
    """Baumgartner Weiss Schindler Test"""

    detector_class = data_drift.BWSTest
    max_samples = _MAX_SAMPLES
    config = {
        "callbacks": None,
    }

    # scipy's bws_test has no asymptotic null, so method=None falls back to
    # PermutationMethod(n_resamples=9999) *unbatched*, which materialises a
    # (9999, n_ref + n_test) float64 array -- ~21 GB at Occupancy's 16k/30k
    # split, i.e. an OOM. batch= bounds the working set; 999 resamples still
    # gives a p-value floor of 0.001. Still the single slowest detector per
    # feature even after batching (measured ~46s/feature on French Motor's
    # unsampled ~514k/164k split, vs ~8s/feature at the max_samples cap).
    compare_kwargs = {"method": PermutationMethod(n_resamples=999, batch=50, random_state=_SEED),
    }



class ChiSquareTest(BaseBatchDD):
    """Chi-square Test"""

    keep_categorical_columns = True
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
