"""Module to define the drift detection tools."""

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
from evidently.future.datasets import DataDefinition as EDataDefinition
from evidently.future.datasets import Dataset as EDataset
from pydantic import Field
from pydantic_settings import BaseSettings

import d3bench.tools.alibi as tools_alibi
import d3bench.tools.evidently as tools_evidently
import d3bench.tools.frouros as tools_frouros
import d3bench.tools.menelaus as tools_menelaus
import d3bench.tools.nannyml as tools_nannyml
import d3bench.tools.river as tools_river
from d3bench import methods
from d3bench.config import Framework
from d3bench.utils import Data

# pylint: disable=too-few-public-methods
# pylint: disable=line-too-long


logger = logging.getLogger(__name__)


class Options(BaseSettings):
    """Settings to instantiate a tool."""

    repetitions: int = Field(
        default=3,
        description="Number of repetitions for the benchmark.",
    )

    on_vm: bool = Field(
        default=False,
        description="Flag to run the benchmark on a VM.",
    )


class Tool(ABC):
    """Abstract class for drift detection tools."""

    # Abstract attributes to define by child classes
    name: Framework
    online_cd_methods: dict[methods.OnlineCD, Any]
    online_dd_methods: dict[methods.OnlineDD, Any]
    batch_cd_methods: dict[methods.BatchCD, Any]
    batch_dd_methods: dict[methods.BatchDD, Any]

    def __init__(self, data: Data, settings: Optional[Options] = None):
        self.settings = settings or Options()
        self.data = data

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> Any:
        """Preprocess the data before drift detection."""
        raise NotImplementedError

    def process(self, df: pd.DataFrame) -> Any:
        """Call to the preprocess method with a copy of the data."""
        return self.preprocess(df.copy())

    @property
    def _monitored_columns(self) -> list[str]:
        """Columns preprocess()/usable_features() operate on: the X features
        for covariate scenarios, or just the label column for prior-drift
        scenarios (data.target). Centralized here rather than in each
        adapter's preprocess() -- Frouros/River/Alibi-Detect stack every
        column physically present in the frame they're handed regardless of
        data.features, so routing the column selection once here (used by
        both usable_features() and reference_data/testing_data below) keeps
        every adapter correctly scoped with no per-adapter changes."""
        if self.data.drift_type == "prior":
            assert self.data.target is not None
            return [self.data.target]
        return self.data.features

    def usable_features(self, test: type) -> list[str]:
        """Return the subset of ``data.features`` a given method will see.

        Job builds each method's detector (one per feature, for tools that
        report per-column like Evidently) from this list *before*
        ``preprocess`` ever runs, so it must be derivable from raw dtypes
        alone. ``test`` is the detector class Job is about to instantiate
        (``benchmark.test``), so tools whose methods split by column kind
        (e.g. Evidently: continuous-only methods vs. categorical-only ones)
        can pick a different subset per method. Defaults to every monitored
        column (see ``_monitored_columns``), ignoring ``test``.
        """
        return self._monitored_columns

    @cached_property
    def reference_data(self) -> Any:
        """Return the reference data."""
        return self.preprocess(self.data.reference[self._monitored_columns + ["time"]].copy())

    @cached_property
    def testing_data(self) -> Any:
        """Return the testing data."""
        return self.preprocess(self.data.testing[self._monitored_columns + ["time"]].copy())


class Frouros(Tool):
    """Frouros drift detection tool."""

    name: Framework = "Frouros"
    online_cd_methods: dict[methods.OnlineCD, Any] = {
       # methods.OnlineCD.BAYESIAN_ONLINE_CHANGE_DETECTION: tools_frouros.BayesianOnlineChangeDetection,  # TODO: Very long evaluation times for 100k ref data, worked with a small siz of data, but not with the full dataset: O(n²) memory growth.
        methods.OnlineCD.CUMULATIVE_SUM_CONTROL_CHART: tools_frouros.CumulativeSumControlChart,
        methods.OnlineCD.GEOMETRIC_MOVING_AVERAGE: tools_frouros.GeometricMovingAverage,
        methods.OnlineCD.PAGE_HINKLEY_TEST: tools_frouros.PageHinkleyTest,
        methods.OnlineCD.DRIFT_DETECTION_METHOD: tools_frouros.DriftDetectionMethod,
        methods.OnlineCD.EWMA_CONCEPT_DRIFT_DETECTION_WARNING: tools_frouros.EWMAConceptDriftDetectionWarning,
        methods.OnlineCD.EARLY_DRIFT_DETECTION_METHOD: tools_frouros.EarlyDriftDetectionMethod,
        methods.OnlineCD.HOEFFDING_DRIFT_DETECTION_METHOD_TEST_A: tools_frouros.HoeffdingDriftDetectionMethodTestA,
        methods.OnlineCD.HOEFFDING_DRIFT_DETECTION_METHOD_TEST_W: tools_frouros.HoeffdingDriftDetectionMethodTestW,
        methods.OnlineCD.REACTIVE_DRIFT_DETECTION_METHOD: tools_frouros.ReactiveDriftDetectionMethod,
        methods.OnlineCD.ADAPTIVE_WINDOWING: tools_frouros.AdaptiveWindowing,
        methods.OnlineCD.ONLINE_KOLMOGOROV_SMIRNOV: tools_frouros.KolmogorovSmirnovWindowing,
        methods.OnlineCD.STATISTICAL_TEST_EQUAL_PROPORTIONS_DETECTION: tools_frouros.StatisticalTestEqualProportionsDetection,
    }
    online_dd_methods: dict[methods.OnlineDD, Any] = {
        methods.OnlineDD.ONLINE_MAXIMUM_MEAN_DISCREPANCY: tools_frouros.OnlineMaximumMeanDiscrepancy,
        methods.OnlineDD.INCREMENTAL_KOLMOGOROV_SMIRNOV_TEST: tools_frouros.IncrementalKolmogorovSmirnovTest,
    }
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {
        methods.BatchDD.BHATTACHARYYA_DISTANCE: tools_frouros.BhattacharyyaDistance,
        methods.BatchDD.EARTH_MOVER_DISTANCE: tools_frouros.EarthMoverDistance,
        methods.BatchDD.ENERGY_DISTANCE: tools_frouros.EnergyDistance,
        methods.BatchDD.HELLINGER_DISTANCE: tools_frouros.HellingerDistance,
        methods.BatchDD.HISTOGRAM_INTERSECTION_NORMALIZED_COMPLEMENT: tools_frouros.HistogramIntersectionNormalizedComplement,
        methods.BatchDD.JENSEN_SHANNON_DIVERGENCE_DRIFT_DETECTION: tools_frouros.JensenShannonDivergenceDriftDetection,
        methods.BatchDD.KULLBACK_LEIBLER_DIVERGENCE_DRIFT_DETECTION: tools_frouros.KullbackLeiblerDivergenceDriftDetection,
        methods.BatchDD.BATCH_MAXIMUM_MEAN_DISCREPANCY: tools_frouros.BatchMaximumMeanDiscrepancy,
        methods.BatchDD.POPULATION_STABILITY_INDEX: tools_frouros.PopulationStabilityIndex,
        methods.BatchDD.ANDERSON_DARLING_TEST: tools_frouros.AndersonDarlingTest,
        methods.BatchDD.BAUMGARTNER_WEISS_SCHINDLER_TEST: tools_frouros.BaumgartnerWeissSchindlerTest,
        methods.BatchDD.CHI_SQUARE_TEST: tools_frouros.ChiSquareTest,
        methods.BatchDD.CRAMER_VON_MISES_TEST: tools_frouros.CramerVonMisesTest,
        methods.BatchDD.KOLMOGOROV_SMIRNOV_TEST: tools_frouros.KolmogorovSmirnovTest,
        methods.BatchDD.KUIPER_TEST: tools_frouros.KuiperTest,
        methods.BatchDD.MANN_WHITNEY_U_TEST: tools_frouros.MannWhitneyUTest,
        methods.BatchDD.WELCH_T_TEST: tools_frouros.WelchTTest,
    }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns={"time"}, inplace=True)
        data = [df[feature].to_numpy() for feature in df.columns]
        # dtype=object preserves each column's native scalar type; np.stack would
        # coerce the *whole* array to string dtype the moment one column is
        # non-numeric (verified: even int columns turn into '1'), silently
        # corrupting numeric columns before frouros.py ever sees them. A no-op
        # for all-numeric datasets (energy/occupancy): astype(float32/float64)
        # downstream produces bit-identical values either way.
        return np.array(data, dtype=object).T


class Evidently(Tool):
    """Evidently drift detection tool."""

    name: Framework = "Evidently"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {
        methods.BatchDD.KOLMOGOROV_SMIRNOV_TEST: tools_evidently.KolmogorovSmirnovTest,
        methods.BatchDD.CHI_SQUARE_TEST: tools_evidently.ChiSquareTest,
        methods.BatchDD.Z_TEST: tools_evidently.ZTest,
        methods.BatchDD.WASSERSTEIN_DISTANCE: tools_evidently.WassersteinDistance,
        methods.BatchDD.KULLBACK_LEIBLER_DIVERGENCE_DRIFT_DETECTION: tools_evidently.KullbackLeiblerDivergenceDriftDetection,
        methods.BatchDD.POPULATION_STABILITY_INDEX: tools_evidently.PopulationStabilityIndex,
        methods.BatchDD.JENSEN_SHANNON_DIVERGENCE_DRIFT_DETECTION: tools_evidently.JensenShannonDivergenceDriftDetection,
        methods.BatchDD.ANDERSON_DARLING_TEST: tools_evidently.AndersonDarlingTest,
        # FISHER_EXACT_TEST: evidently's fisher_exact_stattest requires
        # reference_data and current_data to be the *same length*, which no
        # real drift scenario satisfies (reference/testing are different
        # splits by construction) -- always empty, so left disabled.
        # methods.BatchDD.FISHER_EXACT_TEST: tools_evidently.FisherExactTest,
        methods.BatchDD.CRAMER_VON_MISES_TEST: tools_evidently.CramerVonMisesTest,
        # G_TEST: evidently's g_stattest feeds raw (unnormalized) category
        # counts straight into scipy.stats.power_divergence, which requires
        # sum(reference) == sum(testing) -- fails on any scenario where the
        # two splits differ in size (i.e. essentially all of them here).
        # Always empty, so left disabled.
        # methods.BatchDD.G_TEST: tools_evidently.GTest,
        methods.BatchDD.HELLINGER_DISTANCE: tools_evidently.HellingerDistance,
        methods.BatchDD.MANN_WHITNEY_U_TEST: tools_evidently.MannWhitneyUTest,
        methods.BatchDD.ENERGY_DISTANCE: tools_evidently.EnergyDistance,
        methods.BatchDD.EPPS_SINGLETON_TEST: tools_evidently.EppsSingletonTest,
        methods.BatchDD.T_TEST: tools_evidently.TTest,
        methods.BatchDD.EMPIRICAL_MAXIMUM_MEAN_DISCREPANCY: tools_evidently.EmpiricalMaximumMeanDiscrepancy, # this is not categorical
        methods.BatchDD.TOTAL_VARIATION_DISTANCE: tools_evidently.TotalVariationDistance,
    }

    def usable_features(self, test: type) -> list[str]:
        """Split features by dtype: categorical methods only see non-numeric
        columns, continuous ones (the majority, see batch_dd_methods above)
        only see numeric columns -- mirrors the numerical/categorical split
        declared on the dataset itself in preprocess(). Dtype-driven only
        (e.g. DataMotorPrior's target is stored as explicit "claim"/
        "no_claim" labels, not relabeled-but-still-numeric values, so this
        needs no drift_type-specific override -- every tool that inspects
        this column, by whatever mechanism it uses, agrees it's categorical)."""
        numeric = set(self.data.reference.select_dtypes(include="number").columns)
        wants_categorical = getattr(test, "categorical", False)
        return [
            f for f in self._monitored_columns
            if (f not in numeric) == wants_categorical
        ]  # fmt: skip

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns={"time"}, inplace=True)
        # Declare each column's real dtype instead of blanket-labeling
        # everything numerical -- evidently's np.isinf() stats collection
        # crashes on a string column (e.g. motor's VehBrand) declared
        # numerical. Kept in sync with usable_features(), which Job uses to
        # build each method's per-feature detectors before preprocess runs;
        # a method never gets a Report for a column outside its own kind.
        numerical = list(df.select_dtypes(include="number").columns)
        categorical = [c for c in df.columns if c not in numerical]
        schema = EDataDefinition(numerical_columns=numerical, categorical_columns=categorical)
        return EDataset.from_pandas(df, data_definition=schema)


class NannyML(Tool):
    """NannyML drift detection tool."""

    name: Framework = "NannyML"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {
        methods.BatchDD.JENSEN_SHANNON_DIVERGENCE_DRIFT_DETECTION: tools_nannyml.JensenShannonDivergenceDriftDetection,
        methods.BatchDD.JENSEN_SHANNON_DIVERGENCE_CATEGORICAL: tools_nannyml.JensenShannonDivergenceCategorical,
        methods.BatchDD.WASSERSTEIN_DISTANCE: tools_nannyml.WassersteinDistance,
        methods.BatchDD.HELLINGER_DISTANCE: tools_nannyml.HellingerDistance,
        methods.BatchDD.HELLINGER_DISTANCE_CATEGORICAL: tools_nannyml.HellingerDistanceCategorical,
        methods.BatchDD.KOLMOGOROV_SMIRNOV_TEST: tools_nannyml.KolmogorovSmirnovTest,
        methods.BatchDD.CHI_SQUARE_TEST: tools_nannyml.ChiSquareTest,
        methods.BatchDD.L_INFINITY_DISTANCE: tools_nannyml.LInfinityDistance,
    }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # No preprocessing needed


class AlibiDetect(Tool):
    """Alibi-Detect drift detection tool."""

    name: Framework = "Alibi-Detect"
    online_cd_methods: dict[methods.OnlineCD, Any] = {
        # Online kernel detectors: OOM fixed via max_samples=1000 (alibi.py); test()
        # also caps the streamed x_test to the same leading 1000 rows (BaseUniOnlineTest.test),
        # since predict() is one TF forward pass per row and streaming tens of
        # thousands of rows could take an hour+.
        methods.OnlineCD.ONLINE_MAXIMUM_MEAN_DISCREPANCY: tools_alibi.OnlineMaximumMeanDiscrepancy,
        methods.OnlineCD.ONLINE_LEAST_SQUARES_DENSITY_DIFFERENCE: tools_alibi.OnlineLeastSquaresDensityDifference,
        methods.OnlineCD.ONLINE_CRAMER_VON_MISES_TEST: tools_alibi.OnlineCramerVonMisesTest,

        # FET is binary-only (x_ref must be 0/1); N/A for continuous data.
        # methods.OnlineCD.ONLINE_FISHER_EXACT_TEST: tools_alibi.OnlineFisherExactTest,  TODO: ValueError: The `x_ref` data must consist of only (0,1)'s or (False,True)'s for the FETDriftOnline detector.
    }
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {
        # FisherExact: binary-only, N/A for continuous data.
        # methods.BatchCD.FISHER_EXACT_TEST: tools_alibi.FisherExactTest,
        # The five below need a user-supplied model/kernel (see alibi.py stubs).
        # Scope decision: implement with a shared model spec across frameworks, or drop.
        # methods.BatchCD.LEARNED_KERNEL_DRIFT_DETECTION: 
        # methods.BatchCD.FISHER_EXACT_TEST: tools_alibi.FisherExactTest, TODO: ValueError: The `x_ref` data must consist of only (0,1)'s or (False,True)'s for the FETDrift detector.
        # methods.BatchCD.LEARNED_KERNEL_DRIFT_DETECTION: tools_alibi.LearnedKernelDriftDetection, TODO: Fix implementation
        # methods.BatchCD.CLASSIFIER_DRIFT_DETECTOR: tools_alibi.ClassifierDriftDetector, TODO: Fix implementation
        # methods.BatchCD.SPOT_DIFF_DRIFT_DETECTOR: tools_alibi.SpotTheDiffDriftDetector, TODO: Fix implementation
        # methods.BatchCD.CLASSIFIER_UNCERTAINTY_DRIFT_DETECTOR: tools_alibi.ClassifierUncertaintyDriftDetector, TODO: Fix implementation
        # methods.BatchCD.CONTEXT_AWARE_MAXIMUM_MEAN_DISCREPANCY: tools_alibi.ContextAwareMaximumMeanDiscrepancy, TODO: Fix implementation
        # BaseUnivariateTest), not concept drift -- it stays here only because
        # methods.BatchDD has no equivalent enum member yet.
        
    }

    batch_dd_methods: dict[methods.BatchDD, Any] = {
        # Moved from batch_cd_methods: univariate feature-distribution tests
        # (covariate drift, P(X)) per alibi.py BaseUnivariateTest, not concept drift.
        # MMD/LSDD are capped to 1000 samples/side (see alibi.py) -- kernel O(n^2).
        methods.BatchDD.CHI_SQUARE_TEST: tools_alibi.ChiSquareTest,
        methods.BatchDD.KOLMOGOROV_SMIRNOV_TEST: tools_alibi.KolmogorovSmirnovTest,
        methods.BatchDD.CRAMER_VON_MISES_TEST: tools_alibi.CramerVonMisesTest,
        methods.BatchDD.BATCH_MAXIMUM_MEAN_DISCREPANCY: tools_alibi.MaximumMeanDiscrepancy,
        methods.BatchDD.LEAST_SQUARES_DENSITY_DIFFERENCE: tools_alibi.LeastSquaresDensityDifference, 
        # NOTE: MixedTypeTabularData is a per-feature covariate-drift test (see alibi.py
        methods.BatchDD.MIXED_TYPE_TABULAR_DATA: tools_alibi.MixedTypeTabularData,
    }

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        df.drop(columns={"time"}, inplace=True)
        data = [df[feature].to_numpy() for feature in df.columns]
        # dtype=object preserves each column's native scalar type; np.stack would
        # coerce the *whole* array to string dtype the moment one column is
        # non-numeric (verified: even int columns turn into '1'), silently
        # corrupting numeric columns before alibi.py ever sees them. A no-op for
        # all-numeric datasets (energy/occupancy): astype(float32) downstream
        # produces bit-identical values either way.
        return np.array(data, dtype=object).T


class River(Tool):
    """River drift detection tool."""

    name: Framework = "River"
    online_cd_methods: dict[methods.OnlineCD, Any] = {
        methods.OnlineCD.ADAPTIVE_WINDOWING: tools_river.AdaptiveWindowing,
        # methods.OnlineCD.DRIFT_DETECTION_METHOD: tools_river.DriftDetectionMethod,  TODO: ValueError: math domain error
        # methods.OnlineCD.EARLY_DRIFT_DETECTION_METHOD: tools_river.EarlyDriftDetectionMethod,
        # methods.OnlineCD.HOEFFDING_DRIFT_DETECTION_METHOD_TEST_A: tools_river.HoeffdingDriftDetectionMethodTestA,
        # methods.OnlineCD.HOEFFDING_DRIFT_DETECTION_METHOD_TEST_W: tools_river.HoeffdingDriftDetectionMethodTestW,  TODO: river.drift.binary.HDDM_W expects a bounded 0/1 correctness stream; fed the L2-norm of raw covariate features (unbounded), it hangs rather than converging -- effectively never returns on Energy/Occupancy-sized data.
        methods.OnlineCD.ONLINE_KOLMOGOROV_SMIRNOV: tools_river.OnlineKolmogorovSmirnov,
        methods.OnlineCD.PAGE_HINKLEY_TEST: tools_river.PageHinkleyTest,
        methods.OnlineCD.PERIODIC_TRIGGER: tools_river.PeriodicTrigger,
    }
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns={"time"}, inplace=True)
        # Every River method is online_cd: one detector fed the L2-norm of the
        # whole feature row (no per-feature indexing anywhere in river.py), so
        # dropping non-numeric columns here is safe for every method River has.
        # np.stack coerces the *entire* array to string dtype the moment one
        # column is non-numeric (verified: even int columns turn into '1'),
        # which is silent on all-numeric datasets (energy/occupancy, where
        # this is a no-op) but corrupts np.linalg.norm on mixed data (motor).
        numeric_df = df.select_dtypes(include="number")
        dropped = set(df.columns) - set(numeric_df.columns)
        if dropped:
            logger.info(
                "River: dropping non-numeric columns %s from the concept-drift feature vector",
                sorted(dropped),
            )
        if numeric_df.empty:
            # e.g. drift_type == "prior": the only monitored column is
            # DataMotorPrior's categorical target. Silently returning a
            # 0-column array here would make np.linalg.norm produce a
            # constant zero stream below -- every detector would then
            # report "no drift" not because nothing shifted, but because
            # it was fed no signal at all. That's a false negative
            # disguised as a real result, worse than failing loudly: raise
            # instead, so _try_report's existing catch-all skips this
            # tool/method combination the same way it already does for any
            # other unsupported case, with a message that explains why
            # rather than a bare "need at least one array to stack".
            raise ValueError(
                "River has no numeric columns to monitor here -- its concept-drift "
                "methods only support a single combined numeric feature vector (no "
                "per-column/categorical path), and every monitored column for this "
                "scenario is non-numeric."
            )
        data = [numeric_df[feature].to_numpy() for feature in numeric_df.columns]
        return np.stack(data).T


class Menelaus(Tool):
    """Menelaus drift detection tool."""

    name: Framework = "Menelaus"
    online_cd_methods: dict[methods.OnlineCD, Any] = {
        methods.OnlineCD.CUMULATIVE_SUM_CONTROL_CHART: tools_menelaus.CumulativeSumControlChart,
    }
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # No preprocessing needed


class TorchDrift(Tool):
    """TorchDrift drift detection tool."""

    name: Framework = ""
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
