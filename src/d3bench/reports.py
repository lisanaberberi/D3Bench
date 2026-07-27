"""Utility functions and classes for drift detection benchmark results.

This module provides the data classes needed to store, analyze, and compare
benchmark results across different drift detection methods and frameworks.
"""

import datetime as dt
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from d3bench import methods
from d3bench.config import Framework

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


class TestInformation(BaseModel):
    """Information about the test method used in the benchmark.

    This class is used to store information about the test method used in the
    benchmark. It includes the framework used, whether the benchmark was run
    on a VM, the number of repetitions, and the length of the training and
    test data.
    """

    framework: Framework = Field(..., description="Framework used")
    run_on_vm: bool = Field(..., description="Whether run on a VM")
    repetitions: int = Field(..., description="Number of repetitions")
    len_reference: int = Field(..., description="Length of the reference data")
    len_testing: int = Field(..., description="Length of the testing data")


class Stats(BaseModel):
    """Statistics computed from multiple benchmark runs.

    Note: It's tempting to calculate mean and standard deviation from the
    result vector and report these. However, this is not very useful. In
    a typical case, the lowest value gives a lower bound for how fast your
    machine can run the given code snippet; higher values in the result
    vector are typically not caused by variability in Python's speed, but
    by other processes interfering with your timing accuracy. So the min()
    of the result is probably the only number you should be interested in.

    After that, you should look at the entire vector and apply common sense
    rather than statistics.
    """

    avg: float = Field(..., description="Average value")
    max: float = Field(..., description="Maximum value")
    min: float = Field(..., description="Minimum value")

    @staticmethod
    def from_values(values: List[float]) -> "Stats":
        """Create a Stats object from a list of values."""
        return Stats(
            avg=sum(values) / len(values),
            max=max(values),
            min=min(values),
        )


class Report(BaseModel):
    """
    Base data class to store the results of the benchmark.
    This is the parent class for all specific report types.
    """

    method: str = Field(..., description="Method used in benchmark")
    method_class: str = Field(..., description="Method class used in benchmark")
    test_information: TestInformation = Field(..., description="Test information")
    time: dt.datetime = Field(default_factory=dt.datetime.now)
    runtime: Optional[Stats] = Field(None, description="Runtime statistics")
    cputime: Optional[Stats] = Field(None, description="CPU time statistics")
    memory: Optional[Stats] = Field(None, description="Memory usage statistics")
    functional: Optional[Dict[str, bool]] = Field(
        None, description="Whether drift was flagged on each monitored column"
    )
    statistic: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Per-column value each method compared against its threshold to reach the "
            "functional verdict above -- a p-value for classical hypothesis tests, a "
            "distance/divergence for distance-based methods. Semantics vary by method, "
            "mirroring the source library's own convention (e.g. Evidently's 'value', "
            "NannyML's 'value', Alibi-Detect's 'distance'/'test_stat'); this is the D-value "
            "column in the benchmark paper's tables."
        ),
    )
    drift_index: Optional[int] = Field(
        None,
        description=(
            "For streaming online-CD detectors only: the 0-based offset in the testing "
            "stream at which drift FIRST fired. The verdict latches (status['drift'] / "
            "drift_detected), so this is what tells a real detection apart from one that "
            "only fired deep into the stream -- a distinct scalar, deliberately NOT "
            "per-column and NOT folded into `statistic`. None for batch/data-drift "
            "methods (no per-instance stream) or when the detector never fired."
        ),
    )


class OnlineCDReport(Report):
    """
    Data class to store the results of a Online Supervised Concept Drift
    Detection benchmark.

    This report captures metrics specific to online supervised drift detection
    methods, focusing on detection speed, accuracy and false
    positives/negatives.
    """

    method: methods.OnlineCD = Field(..., description="Method used in benchmark")
    # `statistic` (on the base Report) now covers "method-specific stats".
    # TODO: these all need a ground-truth drift point/label to compute, which
    # the current Data/Dataset abstraction doesn't carry -- not implemented.
    # detection_delay: Optional[Stats] = Field(None, description="Samples until drift detection")
    # false_alarm_rate: Optional[float] = Field(None, description="False positives")
    # missed_detection_rate: Optional[float] = Field(None, description="False negatives")
    # f1_score: Optional[float] = Field(None, description="Harmonic mean of precision and recall")
    # adaptation_time: Optional[Stats] = Field(None, description="Time to adapt after drift")
    # auc_score: Optional[float] = Field(None, description="Area under ROC curve")


class OnlineDDReport(Report):
    """
    Data class to store the results of a Online Unsupervised Data Drift
    Detection benchmark.

    This report captures metrics specific to online unsupervised drift
    detection, focusing on statistical properties and computational
    efficiency.
    """

    method: methods.OnlineDD = Field(..., description="Method used in benchmark")
    # `statistic` (on the base Report) now covers "method-specific stats".
    # TODO: these all need a ground-truth drift point/label to compute, which
    # the current Data/Dataset abstraction doesn't carry -- not implemented.
    # detection_delay: Optional[Stats] = Field(None, description="Samples until drift detection")
    # false_alarm_rate: Optional[float] = Field(None, description="False positives")
    # statistical_power: Optional[float] = Field(None, description="Ability to detect drift")
    # processing_time_per_sample: Optional[Stats] = Field(None, description="Computational efficiency")


class BatchCDReport(Report):
    """
    Data class to store the results of a Batch Concept Drift Detection
    benchmark.

    This report captures metrics specific to batch concept drift methods,
    focusing on detection accuracy and characteristics of identified drift.
    """

    method: methods.BatchCD = Field(..., description="Method used in benchmark")
    # drift_detected -> `functional`, test_statistics -> `statistic` (both on the base Report).
    # TODO: these all need a ground-truth drift point/label to compute, which
    # the current Data/Dataset abstraction doesn't carry -- not implemented.
    # detection_accuracy: Optional[float] = Field(None, description="Correct detections")
    # drift_magnitude: Optional[float] = Field(None, description="Magnitude of the drift")
    # confidence_level: Optional[float] = Field(None, description="Confidence in detection")
    # drift_location: Optional[Union[int, List[int]]] = Field(None, description="Est. drift pos.")
    # testing_power: Optional[float] = Field(None, description="Statistical power")


class BatchDDReport(Report):
    """
    Data class to store the results of a Batch Data Drift Detection
    benchmark.

    This report captures metrics specific to batch data drift detection,
    focusing on statistical measures and feature-level drift analysis.
    """

    method: methods.BatchDD = Field(..., description="Method used in benchmark")
    # drift_detected -> `functional`; drift_score/feature_drift_scores/p_value/test_statistics
    # -> `statistic` (all on the base Report, per-column, standardized across tools).
    # TODO: neither effect_size nor a confidence_interval is computed by any tool
    # module today (most methods here are distances/divergences with no natural
    # effect-size or CI, e.g. PSI, Wasserstein, KL) -- not implemented.
    # effect_size: Optional[float] = Field(None, description="Magnitude of the effect")
    # confidence_interval: Optional[tuple] = Field(None, description="Confidence interval")
