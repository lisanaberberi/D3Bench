"""Module for River Detect detectors."""

from typing import Any, Optional

from abc import ABC, abstractmethod
import numpy as np
from river import drift

from d3bench import utils


# Online Supervised Concept Drift Detection


class BaseOnlineTest(utils.BaseTestMethod, ABC):
    """Base class for online univariate drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.detector = self.detector_class(**self.config)
        self.drift: Any
        # Stream index at which drift first fired (set in test()); None until
        # then. drift_detected is transient per update, so result() would
        # otherwise only recover *that* drift fired, not *when*.
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
        # error stream: KSWIN under a supervised run, or the binary
        # DDM/EDDM/HDDM (error_stream_only) under an unsupervised one.
        self._guard_error_stream()
        # Supervised concept-drift path (drift_type == "concept"): the shared
        # classifier is already trained (d3bench.supervised); the detector
        # self-calibrates on the TEST error stream (see test()). No warm-up on
        # the reference error stream -- see the note in frouros.py
        # BaseOnlineCD.fit (avoids latching the verdict on reference-stream
        # fluctuation before any test data is seen).
        if self.error_stream is not None:
            return
        # Unsupervised feature-norm path (covariate/prior). Detector is trained
        # one by one on the reference data. See:
        # https://frouros.readthedocs.io/en/latest/examples/concept_drift/DDM_advance.html#warm-up-phase
        # !!! only 1000 instances are used for training, very high time consumption
        for x in np.linalg.norm(x_reference[:1000], ord=2, axis=1):
            self.detector.update(x)

    def test(self, x_test: np.ndarray) -> None:
        self.drift_ever = False
        # Reset before each stream so drift_index is the first-firing offset
        # *within this test pass*, not a stale value from an earlier one.
        self.drift_index = None
        # Supervised concept-drift path: stream the classifier's testing error.
        if self.error_stream is not None:
            for i, error in enumerate(self.error_stream.testing):
                self.detector.update(float(error))
                if self.detector.drift_detected:
                    self.drift_ever = True
                    if self.drift_index is None:
                        self.drift_index = i
            return
        for i, x in enumerate(np.linalg.norm(x_test, ord=2, axis=1)):
            self.detector.update(x)
            if self.detector.drift_detected:
                self.drift_ever = True
                if self.drift_index is None:
                    self.drift_index = i

    def result(self) -> dict[str, Any]:
        # No uniform test statistic across river's online CD algorithms
        # (ADWIN, PageHinkley track different private state); report the verdict
        # plus drift_index, the stream offset it first fired at (see test()).
        # KSWIN overrides with its KS statistic.
        return {"drift": self.drift_ever, "drift_index": self.drift_index}

class AdaptiveWindowing(BaseOnlineTest):
    """Online Maximum Mean Discrepancy"""

    detector_class = drift.ADWIN
    config = {
        "delta": 0.002,
        "clock": 32,
        "max_buckets": 5,
        "min_window_length": 5,
        "grace_period": 10,
    }


class DriftDetectionMethod(BaseOnlineTest):
    """Drift Detection Method"""

    # Binary error-stream detector: 1 = misclassification. No unsupervised
    # feature-norm fallback (raises "math domain error" on unbounded input).
    error_stream_only = True
    detector_class = drift.binary.DDM
    config = {
        "warm_start": 30,
        "warning_threshold": 2.0,
        "drift_threshold": 3.0,
    }


class EarlyDriftDetectionMethod(BaseOnlineTest):
    """Early Drift Detection Method"""

    # Binary error-stream detector: 1 = misclassification. See DriftDetectionMethod.
    error_stream_only = True
    detector_class = drift.binary.EDDM
    config = {
        "warm_start": 30,
        "alpha": 0.95,
        "beta": 0.9,
    }


class HoeffdingDriftDetectionMethodTestA(BaseOnlineTest):
    """Hoeffding Drift Detection Method Test A"""

    # Binary error-stream detector: 1 = misclassification. See DriftDetectionMethod.
    error_stream_only = True
    detector_class = drift.binary.HDDM_A
    config = {
        "drift_confidence": 0.001,
        "warning_confidence": 0.005,
        "two_sided_test": False,
    }


class HoeffdingDriftDetectionMethodTestW(BaseOnlineTest):
    """Hoeffding Drift Detection Method Test W"""

    # Binary error-stream detector: 1 = misclassification. See DriftDetectionMethod.
    error_stream_only = True
    detector_class = drift.binary.HDDM_W
    config = {
        "drift_confidence": 0.001,
        "warning_confidence": 0.005,
        "lambda_val": 0.05,
        "two_sided_test": False,
    }


class OnlineKolmogorovSmirnov(BaseOnlineTest):
    """Online Kolmogorov-Smirnov"""

    # KS-windowing over the monitored values -- an assumption about the
    # distribution of X, degenerate on a two-valued 0/1 error stream. Excluded
    # from the supervised concept-drift run; inherited fit() raises
    # MethodNotApplicable before the overridden test() below is reached.
    error_stream_capable = False
    detector_class = drift.KSWIN
    config = {
        "alpha": 0.005,
        "window_size": 100,
        "stat_size": 30,
        "seed": None,
        "window": None,
    }

    def test(self, x_test: np.ndarray) -> None:
        self.drift_ever = False
        self.drift_index = None
        self.drift_p_value = None
        for i, x in enumerate(np.linalg.norm(x_test, ord=2, axis=1)):
            self.detector.update(x)
            if self.detector.drift_detected:
                self.drift_ever = True
                if self.drift_index is None:            # first firing only
                    self.drift_index = i
                if self.drift_p_value is None:          # first firing only
                    self.drift_p_value = self.detector.p_value

    def result(self) -> dict[str, Any]:
        # KSWIN's KS p-value captured at the moment drift first fired -- not the
        # stream-end window, which resets after each firing.
        result: dict[str, Any] = {"drift": self.drift_ever, "drift_index": self.drift_index}
        if self.drift_p_value is not None:
            result["statistic"] = self.drift_p_value
        return result


class PageHinkleyTest(BaseOnlineTest):
    """Page Hinkley Test"""

    detector_class = drift.PageHinkley
    config = {
        "min_instances": 30,
        "delta": 0.005,
        "threshold": 50.0,
        "alpha": 1 - 0.0001,
        "mode": "both",
    }


class PeriodicTrigger(BaseOnlineTest):
    """Periodic Trigger
    Changelog: 0.15.0 - 2023-01-29
    - Renamed `drift.PeriodicTrigger` to `drift.DummyDriftDetector` to clarify it is a naive baseline.
    """

    detector_class = drift.DummyDriftDetector
    config = {
        "trigger_method": "fixed",
        "t_0": 300,
        "w": 0,
        "dynamic_cloning": False,
        "seed": None,
    }
