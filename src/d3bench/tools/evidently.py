"""Module for Evidently detectors."""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator, Optional

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

    #: Seed to pin numpy's *global* RNG around this method's run, or None to
    #: leave it alone. Only needed for empirical_mmd: evidently's shared
    #: permutation_test helper (legacy/calculations/stattests/utils.py) calls
    #: np.random.seed(0) itself, so TVD/KL/JS/PSI/chi-square/Z/G/Fisher are
    #: all reproducible, but mmd_stattest.py has its own permutation loop that
    #: seeds nothing and draws from the global RNG -- so consecutive runs on
    #: identical data returned different p-values and, near the 0.1 threshold,
    #: different verdicts. Evidently 0.7.21 (latest) exposes no seed on
    #: StatTest or _mmd_stattest, so pinning the global state around the call
    #: is the only fix available from this side.
    _global_rng_seed: Optional[int] = None

    def _run_reports(self, x_reference: Any, x_test: Any) -> dict[str, Any]:
        """Run each feature's report independently.

        Some methods (e.g. Epps-Singleton) reject a column outright based on
        its distribution (a near-constant binary column has IQR 0). Running
        every feature in one dict comprehension means one such column takes
        the whole method down; a feature that fails is logged and left out of
        the results instead, the same way NannyML skips inapplicable columns.
        """
        with self._pinned_global_rng():
            return self._run_reports_unseeded(x_reference, x_test)

    @contextmanager
    def _pinned_global_rng(self) -> Iterator[None]:
        """Pin numpy's global RNG for the duration, restoring it afterwards.

        Save/restore rather than a bare seed() so this cannot perturb any other
        detector's stochastic behaviour (e.g. a later adapter's own sampling).
        """
        if self._global_rng_seed is None:
            yield
            return
        state = np.random.get_state()
        np.random.seed(self._global_rng_seed)
        try:
            yield
        finally:
            np.random.set_state(state)

    def _run_reports_unseeded(self, x_reference: Any, x_test: Any) -> dict[str, Any]:
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
    """Empirical Maximum Mean Discrepancy.

    The one Evidently method that needs its RNG pinned from here -- see
    BaseTabularDetectors._global_rng_seed. Note that pinning makes the result
    *reproducible*, not precise: evidently draws only 100 permutations
    (mmd_stattest.mmd_pval), so the p-value carries a Monte-Carlo standard
    error of about 0.03 near its own 0.1 decision threshold, and a column
    whose true p-value sits in that band could legitimately fall either side.
    """

    detector_reference = "empirical_mmd"
    _global_rng_seed = 31


class TotalVariationDistance(SubsampledTabularDetectors):
    """Total Variation Distance"""

    categorical = True
    detector_reference = "TVD"
