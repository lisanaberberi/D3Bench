"""Module for Frouros detectors."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from menelaus import change_detection, concept_drift, data_drift

from d3bench import utils

# Change detection and Concept drift


class BaseStreamingDetector(utils.BaseTestMethod, ABC):
    """Base class for online concept drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.detector = self.detector_class(**self.config)
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

    def fit(self, x_reference: pd.DataFrame) -> None:
        for _, row in x_reference.iterrows():
            self.detector.update(X=row["consumption"])

    def test(self, x_test: pd.DataFrame) -> None:
        raise NotImplementedError

    def result(self) -> dict[str, Any]:
        raise NotImplementedError


class CumulativeSumControlChart(BaseStreamingDetector):
    """Cumulative Sum Control Chart"""

    detector_class = change_detection.CUSUM
    config = {
        "target": None,
        "sd_hat": None,
        "burn_in": 30,
        "delta": 0.005,
        "threshold": 5,
        "direction": None,
    }


class PageHinkleyTest(BaseStreamingDetector):
    """Page-Hinkley Test"""

    detector_class = change_detection.PageHinkley
    config = {
        "delta": 0.01,
        "threshold": 20,
        "burn_in": 30,
        "direction": "positive",
    }


class AdaptiveWindowing(BaseStreamingDetector):
    """Adaptive Windowing"""

    detector_class = change_detection.ADWIN
    config = {
        "delta": 0.002,
        "max_buckets": 5,
        "new_sample_thresh": 32,
        "window_size_thresh": 10,
        "subwindow_size_thresh": 5,
        "conservative_bound": False,
    }


class DriftDetectionMethod(BaseStreamingDetector):
    """Drift Detection Method"""

    detector_class = concept_drift.DDM
    config = {
        "n_threshold": 30,
        "warning_scale": 2,
        "drift_scale": 3,
    }


class EarlyDriftDetectionMethod(BaseStreamingDetector):
    """Early Drift Detection Method"""

    detector_class = concept_drift.EDDM
    config = {
        "n_threshold": 30,
        "warning_thresh": 0.95,
        "drift_thresh": 0.9,
    }


class LinearFourRates(BaseStreamingDetector):
    """Linear Four Rates"""

    detector_class = concept_drift.LinearFourRates
    config = {
        "time_decay_factor": 0.9,
        "warning_level": 0.05,
        "detect_level": 0.05,
        "burn_in": 50,
        "num_mc": 10000,
        "subsample": 1,
        "rates_tracked": ["tpr", "tnr", "ppv", "npv"],
        "parallelize": False,
        "round_val": 4,
    }


class StatisticalTestEqualProportionsDetection(BaseStreamingDetector):
    """Statistical Test of Equal Proportions Detection"""

    detector_class = concept_drift.STEPD
    config = {
        "window_size": 30,
        "alpha_warning": 0.05,
        "alpha_drift": 0.003,
    }


class MarginDensityDriftDetectionMethod(BaseStreamingDetector):
    """Margin Density Drift Detection Method"""

    detector_class = concept_drift.MD3
    config = {
        # clf,  TODO: classifier for which we are tracking drift
        "sensitivity": 2,
        "k": 10,
        "oracle_data_length_required": None,
    }


# Online Data Drift Detection


class StreamingKDQTreeDetectionMethod(utils.BaseTestMethod, ABC):
    """KDQ Tree Detection Method"""

    def __init__(self, features: list[str]) -> None:
        self.detector = data_drift.KdqTreeStreaming(**self.config)
        self.features = features
        self.drift: bool

    config = {
        "window_size": 10,
        "persistence": 0.05,
        "alpha": 0.01,
        "bootstrap_samples": 500,
        "count_ubound": 100,
        "cutpoint_proportion_lbound": 2e-10,
    }

    def fit(self, x_reference: pd.DataFrame) -> None:
        raise NotImplementedError

    def test(self, x_test: pd.DataFrame) -> None:
        raise NotImplementedError

    def result(self) -> dict[str, Any]:
        raise NotImplementedError


class PrincipalComponentsAnalysisChangeDetection(utils.BaseTestMethod, ABC):
    """Principal Components Analysis Change Detection"""

    def __init__(self, features: list[str]) -> None:
        self.detector = data_drift.PCACD(**self.config)
        self.features = features
        self.drift: bool

    config = {
        "window_size": 10,
        "ev_threshold": 0.99,
        "delta": 0.1,
        "divergence_metric": "kl",  # Uses Jensen-Shannon distance as metric
        "sample_period": 0.05,
        "online_scaling": True,
    }

    def fit(self, x_reference: pd.DataFrame) -> None:
        raise NotImplementedError

    def test(self, x_test: pd.DataFrame) -> None:
        raise NotImplementedError

    def result(self) -> dict[str, Any]:
        raise NotImplementedError


# Batch Concept Drift Detection


# Batch Data Drift Detection


class BaseHistogramDensityMethod(utils.BaseTestMethod, ABC):
    """Base class for online concept drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.detector = self.detector_class(**self.config)
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

    def fit(self, x_reference: pd.DataFrame) -> None:
        raise NotImplementedError

    def test(self, x_test: pd.DataFrame) -> None:
        raise NotImplementedError

    def result(self) -> dict[str, Any]:
        raise NotImplementedError


class KullbackLeiblerDivergenceDriftDetection(BaseHistogramDensityMethod):
    """Kullback-Leibler Divergence Drift Detection"""

    detector_class = data_drift.CDBD
    config = {
        "divergence": "KL",
        "detect_batch": 1,
        "statistic": "tstat",
        "significance": 0.05,
        "subsets": 5,
    }


class HellingerDistance(BaseHistogramDensityMethod):
    """Hellinger Distance"""

    detector_class = data_drift.HDDDM
    config = {
        "divergence": "H",
        "detect_batch": 1,
        "statistic": "tstat",
        "significance": 0.05,
        "subsets": 5,
    }


class BatchKDQTreeDetectionMethod(utils.BaseTestMethod, ABC):
    """KDQ Tree Detection Method"""

    def __init__(self, features: list[str]) -> None:
        self.detector = data_drift.KdqTreeBatch(**self.config)
        self.features = features
        self.drift: bool

    config = {
        "alpha": 0.01,
        "bootstrap_samples": 500,
        "count_ubound": 100,
        "cutpoint_proportion_lbound": 2e-10,
    }

    def fit(self, x_reference: pd.DataFrame) -> None:
        raise NotImplementedError

    def test(self, x_test: pd.DataFrame) -> None:
        raise NotImplementedError

    def result(self) -> dict[str, Any]:
        raise NotImplementedError
