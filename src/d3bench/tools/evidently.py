"""Module for Evidently detectors."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from evidently.future.metrics import ValueDrift
from evidently.future.report import Report

from d3bench import utils

from evidently.future.datasets import Dataset as EDataset
from evidently.future.datasets import DataDefinition as EDataDefinition

import numpy as np

logger = logging.getLogger(__name__)

# Tabular Univariate Data Drift Detection


class BaseTabularDetectors(utils.BaseTestMethod, ABC):
    """Base class for Evidently tabular detectors."""

    #: Whether this method operates on categorical columns rather than
    #: numeric ones. Read by Evidently.usable_features() (tools/__init__.py)
    #: to decide which of data.features this method's per-column Reports get
    #: built for -- before preprocess ever runs, so it can't inspect dtypes
    #: itself.
    categorical: bool = False

    def __init__(self, features: list[str]) -> None:
        self.reports = {
            f: Report([ValueDrift(column=f, method=self.detector_reference)], include_tests=True)
            for f in features
        }  # fmt: skip
        self._x_reference: pd.DataFrame
        self.results: dict[str, Any]

    @property
    @abstractmethod
    def detector_reference(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: pd.DataFrame) -> None:
        self._x_reference = x_reference
        raise NotImplementedError("Evidence does not provide fit method")

    def _run_reports(self, x_reference: Any, x_test: Any) -> dict[str, Any]:
        """Run each feature's report independently.

        Some methods (e.g. Epps-Singleton) reject a column outright based on
        its distribution (a near-constant binary column has IQR 0). Running
        every feature in one dict comprehension means one such column takes
        the whole method down; a feature that fails is logged and left out of
        the results instead, the same way NannyML skips inapplicable columns.
        """
        results = {}
        for f, report in self.reports.items():
            try:
                results[f] = report.run(x_reference, x_test)
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "%s: skipping column %r, report run failed",
                    type(self).__name__,
                    f,
                    exc_info=True,
                )
        return results

    def test(self, x_test: pd.DataFrame) -> None:
        x_reference = self._x_reference
        self.results = self._run_reports(x_reference, x_test)

    def result(self) -> dict[str, Any]:
        return {
            # Evidently's ValueDrift "value" is polymorphic: a p-value for
            # classical tests (KS, T-Test, ...), a distance/divergence for
            # the rest (PSI, Wasserstein, KL, ...) -- whatever it thresholds
            # against to reach the "tests" verdict below.
            "statistic": {f: self.results[f].dict()["metrics"][0]["value"]
                         for f in self.results}, # fmt: skip
            "drift": {
                f: self.results[f].dict()["tests"][0]["status"] == "FAIL"
                for f in self.results
            },
        }


class KolmogorovSmirnovTest(BaseTabularDetectors):
    """Kolmogorov-Smirnov Test"""

    detector_reference = "ks"


class ChiSquareTest(BaseTabularDetectors):
    """Chi-Square Test"""

    categorical = True
    detector_reference = "chisquare"


class ZTest(BaseTabularDetectors):
    """Z Test"""

    categorical = True
    detector_reference = "z"


class WassersteinDistance(BaseTabularDetectors):
    """Wasserstein Distance"""

    detector_reference = "wasserstein"


class KullbackLeiblerDivergenceDriftDetection(BaseTabularDetectors):
    """Kullback-Leibler Divergence Drift Detection"""

    detector_reference = "kl_div"


class PopulationStabilityIndex(BaseTabularDetectors):
    """Population Stability Index"""

    detector_reference = "psi"


class JensenShannonDivergenceDriftDetection(BaseTabularDetectors):
    """Jensen-Shannon Divergence Drift Detection"""

    detector_reference = "jensenshannon"


class AndersonDarlingTest(BaseTabularDetectors):
    """Anderson-Darling Test"""

    detector_reference = "anderson"


class FisherExactTest(BaseTabularDetectors):
    """Fisher Exact Test"""

    categorical = True
    detector_reference = "fisher_exact"


class CramerVonMisesTest(BaseTabularDetectors):
    """Cramér-von Mises Test"""

    detector_reference = "cramer_von_mises"


class GTest(BaseTabularDetectors):
    """G Test"""

    categorical = True
    detector_reference = "g_test"


class HellingerDistance(BaseTabularDetectors):
    """Hellinger Distance"""

    detector_reference = "hellinger"


class MannWhitneyUTest(BaseTabularDetectors):
    """Mann-Whitney U-Test"""

    detector_reference = "mannw"


class EnergyDistance(BaseTabularDetectors):
    """Energy Distance"""

    detector_reference = "ed"


class EppsSingletonTest(BaseTabularDetectors):
    """Epps-Singleton Test"""

    detector_reference = "es"


class TTest(BaseTabularDetectors):
    """T Test"""

    detector_reference = "t_test"


class SubsampledTabularDetectors(BaseTabularDetectors, ABC):
    """Base for Evidently methods whose cost blows up with sample size.

    Both known cases here re-run their full statistic once per resample:
    MMD builds an O(n^2) kernel Gram matrix (~88 GB at the Energy 35k/70k
    split -- swaps the machine rather than erroring); TVD runs evidently's
    1000-iteration permutation test, each iteration recomputing the
    statistic over the whole concatenated reference+testing array (~450s
    for a single column at Motor's ~514k/164k split, so ~4.5h across every
    categorical column x the benchmark's repeated runtime/cputime/memory
    criteria). Cap both sides to _MAX_SAMPLES, the standard mitigation for
    this kind of test.

    NOTE: subclasses see far fewer samples than the other Evidently
    detectors (which run on the full split), so their power and runtime are
    NOT comparable to the rest. Record _MAX_SAMPLES alongside the result.
    """

    _MAX_SAMPLES = 1000
    _SEED = 31

    def _subsample(self, dataset: EDataset, side: str) -> EDataset:
        df = dataset.as_dataframe()
        if len(df) > self._MAX_SAMPLES:
            self._record_sample_size(side, self._MAX_SAMPLES, len(df))
            df = df.sample(n=self._MAX_SAMPLES, random_state=self._SEED)
        # Reuse the numerical/categorical split preprocess() already
        # declared (dataset.data_definition) -- subsampling only changes
        # row count, not column kind, so re-deriving from raw pandas dtype
        # here (as this used to) would silently override any column
        # intentionally declared against its dtype (e.g. DataMotorPrior's
        # ClaimNb, forced categorical in Evidently._numeric_columns so
        # Chi-square/Z-Test/TVD see it despite being int64).
        definition = dataset.data_definition
        schema = EDataDefinition(
            numerical_columns=definition.numerical_columns,
            categorical_columns=definition.categorical_columns,
        )
        return EDataset.from_pandas(df, data_definition=schema)

    def fit(self, x_reference: EDataset) -> None:
        self._x_reference_subsampled = self._subsample(x_reference, "reference")
        raise NotImplementedError("Evidently does not provide fit method")

    def test(self, x_test: EDataset) -> None:
        x_reference = self._x_reference_subsampled
        x_test = self._subsample(x_test, "testing")
        self.results = self._run_reports(x_reference, x_test)


class EmpiricalMaximumMeanDiscrepancy(SubsampledTabularDetectors):
    """Empirical Maximum Mean Discrepancy"""

    detector_reference = "empirical_mmd"


class TotalVariationDistance(SubsampledTabularDetectors):
    """Total Variation Distance"""

    categorical = True
    detector_reference = "TVD"
