"""Module for Evidently detectors."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from evidently.future.metrics import ValueDrift
from evidently.future.report import Report

from d3bench import utils

from evidently.future.datasets import Dataset as EDataset
from evidently.future.datasets import DataDefinition as EDataDefinition

import numpy as np

# Tabular Univariate Data Drift Detection


class BaseTabularDetectors(utils.BaseTestMethod, ABC):
    """Base class for Evidently tabular detectors."""

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

    def test(self, x_test: pd.DataFrame) -> None:
        x_reference = self._x_reference
        self.results = {f: self.reports[f].run(x_reference, x_test)
                        for f in self.reports} # fmt: skip

    def result(self) -> dict[str, Any]:
        return {
            "p-values": {f: self.results[f].dict()["metrics"][0]["value"]
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

    detector_reference = "chisquare"


class ZTest(BaseTabularDetectors):
    """Z Test"""

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

    detector_reference = "fisher_exact"


class CramerVonMisesTest(BaseTabularDetectors):
    """Cramér-von Mises Test"""

    detector_reference = "cramer_von_mises"


class GTest(BaseTabularDetectors):
    """G Test"""

    detector_reference = "g-test"


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


class EmpiricalMaximumMeanDiscrepancy(BaseTabularDetectors):
    """Empirical Maximum Mean Discrepancy

    # detector_reference = "empirical_mmd"

    MMD's kernel Gram matrix is O(n^2): ~88 GB at the Energy 35k/70k split,
    which swaps the machine rather than erroring. Cap both sides to _MAX_SAMPLES,
    the standard mitigation for kernel two-sample tests on large samples.

    NOTE: this detector sees far fewer samples than the other Evidently
    detectors (which run on the full split), so its power and runtime are NOT
    comparable to theirs. Record _MAX_SAMPLES alongside the result.
    """

    detector_reference = "empirical_mmd"
    _MAX_SAMPLES = 1000
    _SEED = 31

    def _subsample(self, dataset: EDataset) -> EDataset:
        df = dataset.as_dataframe()                   
        if len(df) > self._MAX_SAMPLES:
            df = df.sample(n=self._MAX_SAMPLES, random_state=self._SEED)
        schema = EDataDefinition(numerical_columns=list(df.columns))
        return EDataset.from_pandas(df, data_definition=schema)

    def fit(self, x_reference: EDataset) -> None:
        # subsample the reference before stashing; name-mangled attr from the base
        self._BaseTabularDetectors_x_reference = self._subsample(x_reference)
        raise NotImplementedError("Evidently does not provide fit method")

    def test(self, x_test: EDataset) -> None:
        x_reference = self._BaseTabularDetectors_x_reference
        x_test = self._subsample(x_test)
        self.results = {f: self.reports[f].run(x_reference, x_test) for f in self.reports}


class TotalVariationDistance(BaseTabularDetectors):
    """Total Variation Distance"""

    detector_reference = "TVD"
