"""Module for Nanny ML detectors."""

from abc import ABC, abstractmethod
from typing import Any

import nannyml as nml
import pandas as pd

from d3bench import utils


def _alert_per_feature(results: Any, method: str, features: list[str]) -> dict[str, bool]:
    """Return, per feature, whether any chunk of the tested window alerted.

    A feature is left out if this method was never applicable to it (e.g. a
    categorical method run against a continuous column), so it does not
    count towards the "functional" denominator for that column.

    Only the *analysis* period counts. NannyML returns both the chunks of the
    window passed to `calculate()` (period "analysis") and the chunks of the
    baseline passed to `fit()` (period "reference"); a reference chunk can
    alert against thresholds derived from its own period's spread, so an
    unfiltered `to_df()` reports the baseline's internal heterogeneity as
    drift in the tested window. `period` is NannyML's own label, identical for
    every dataset -- this is not a per-scenario setting.
    """
    df = results.filter(period="analysis").to_df()
    return {
        feature: bool(df[(feature, method, "alert")].any())
        for feature in features
        if (feature, method, "alert") in df.columns
    }


def _statistic_per_feature(results: Any, method: str, features: list[str]) -> dict[str, float]:
    """Return, per feature, the peak drift value seen across the tested window's chunks.

    NannyML reports one value per chunk, not a single number; take the chunk
    with the largest magnitude, the same "worst case over the window" choice
    `_alert_per_feature` makes for the boolean verdict -- and, for the same
    reason, over the analysis period only, so the D-value describes the tested
    window rather than the largest chunk of the baseline.
    """
    df = results.filter(period="analysis").to_df()
    return {
        feature: float(df[(feature, method, "value")].abs().max())
        for feature in features
        if (feature, method, "value") in df.columns
    }


# Univariate Continuous Data Drift Detection


class BaseUnivariateContinuous(utils.BaseTestMethod, ABC):
    """Base class for batch data drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.detector = nml.UnivariateDriftCalculator(
            column_names=features,
            chunk_number=None,
            timestamp_column_name="time",
            continuous_methods=[self.detector_reference],
        )
        self.results: Any = None

    @property
    @abstractmethod
    def detector_reference(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: pd.DataFrame) -> None:
        self.detector.fit(x_reference)

    def test(self, x_test: pd.DataFrame) -> None:
        self.results = self.detector.calculate(x_test)

    def result(self) -> dict[str, Any]:
        return {
            "drift": _alert_per_feature(self.results, self.detector_reference, self.features),
            "statistic": _statistic_per_feature(self.results, self.detector_reference, self.features),
        }


class JensenShannonDivergenceDriftDetection(BaseUnivariateContinuous):
    """Jensen-Shannon Divergence Drift Detection"""

    detector_reference = "jensen_shannon"


class WassersteinDistance(BaseUnivariateContinuous):
    """Wasserstein Distance"""

    detector_reference = "wasserstein"


class HellingerDistance(BaseUnivariateContinuous):
    """Hellinger Distance"""

    detector_reference = "hellinger"


class KolmogorovSmirnovTest(BaseUnivariateContinuous):
    """Kolmogorov-Smirnov Test"""

    detector_reference = "kolmogorov_smirnov"


# Univariate Categorical Data Drift Detection


class BaseUnivariateCategorical(utils.BaseTestMethod, ABC):
    """Base class for batch data drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.detector = nml.UnivariateDriftCalculator(
            column_names=features,
            chunk_number=None,
            timestamp_column_name="time",
            categorical_methods=[self.detector_reference],
        )
        self.results: Any = None

    @property
    @abstractmethod
    def detector_reference(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: pd.DataFrame) -> None:
        self.detector.fit(x_reference)

    def test(self, x_test: pd.DataFrame) -> None:
        self.results = self.detector.calculate(x_test)

    def result(self) -> dict[str, Any]:
        return {
            "drift": _alert_per_feature(self.results, self.detector_reference, self.features),
            "statistic": _statistic_per_feature(self.results, self.detector_reference, self.features),
        }


class JensenShannonDivergenceCategorical(BaseUnivariateCategorical):
    """Jensen-Shannon Divergence Drift Detection"""

    detector_reference = "jensen_shannon"


class HellingerDistanceCategorical(BaseUnivariateCategorical):
    """Hellinger Distance"""

    detector_reference = "hellinger"


class ChiSquareTest(BaseUnivariateCategorical):
    """Chi-Square Test"""

    detector_reference = "chi2"


class LInfinityDistance(BaseUnivariateCategorical):
    """L-Infinity Distance"""

    detector_reference = "l_infinity"
