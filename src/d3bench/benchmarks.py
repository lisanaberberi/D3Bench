"""Module to run a benchmark to obtain Results."""

import logging
import time
import timeit
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Generator, Optional, Type, TypeAlias, Union

from memory_profiler import memory_usage
from pydantic import Field
from pydantic_settings import BaseSettings

from d3bench import reports, supervised
from d3bench.config import Criteria, Method, MethodFamily
from d3bench.reports import Report, TestInformation
from d3bench.tools import Tool
from d3bench.utils import BaseTestMethod, Data, MethodNotApplicable

# pylint: disable=too-few-public-methods
logger = logging.getLogger(__name__)
Test: TypeAlias = Type[BaseTestMethod]


class BaseBenchmark(ABC):
    """Base class to define benchmarks to obtain Results."""

    def __init__(self, method: Method, test: Test, tool: Tool) -> None:
        self.method = method
        self.test = test
        self.tool = tool

    @property
    def data(self) -> Data:
        """Return the data used in the benchmark."""
        return self.tool.data

    @property
    def repetitions(self) -> int:
        """Return the number of repetitions for the benchmark."""
        return self.tool.settings.repetitions

    @property
    def run_on_vm(self) -> bool:
        """Return whether the benchmark was run on a VM."""
        return self.tool.settings.on_vm

    def _fitted_job(self) -> "Job":
        """Build and fit a fresh detector.

        Each benchmark criterion (runtime, cputime, memory, results) calls this
        for its own independent detector, so no state leaks from one criterion's
        streaming into another's. Online-CD detectors accumulate state and their
        status["drift"]/drift_detected latches; previously all criteria shared
        one detector (constructed once and shallow-copied), so results depended
        on whether the timing passes had run first -- drift_index in particular
        collapsed to 0 once a timing pass had already latched the verdict.
        """
        job = Job(benchmark=self)
        try:
            job.fit()
        except NotImplementedError:
            # Some tools (notably Evidently) do the whole reference-vs-current
            # comparison in test() and expose no fit step -- tolerate that and
            # let test() run, exactly as the old construction-time _prepare_job
            # did. A method that also has no working test() then fails in
            # test() and is skipped by _try_report, as before.
            logger.debug("No train method for %s", self.method)
        return job

    @abstractmethod
    def get_results(self) -> dict[str, Any]:
        """Return the drift statistics of the job."""

    @abstractmethod
    def get_runtimes(self) -> reports.Stats:
        """Run the benchmark for the RUNTIME criterion."""

    @abstractmethod
    def get_cputimes(self) -> reports.Stats:
        """Run the benchmark for the CPUTIME criterion."""

    @abstractmethod
    def get_memories(self) -> reports.Stats:
        """Run the benchmark for the MEMORY criterion."""

    def _broadcast(self, value: Any, cast: Any) -> Optional[dict[str, Any]]:
        """Turn a `result()` entry into a per-column mapping.

        Each detector's `result()` exposes entries ("drift", "statistic")
        that are either a single scalar (methods that only look at the
        combined feature signal) or a dict keyed by column (methods that
        test each feature separately). A single scalar is broadcast to every
        monitored column so criteria that read this always get a per-column
        mapping.
        """
        if value is None:
            return None
        if isinstance(value, dict):
            return {column: cast(v) for column, v in value.items()}
        return {feature: cast(value) for feature in self.tool._monitored_columns}

    def get_functional(self) -> Optional[dict[str, bool]]:
        """Return whether drift was flagged on each monitored column."""
        return self._broadcast(self.get_results().get("drift"), bool)

    def get_statistic(self) -> Optional[dict[str, float]]:
        """Return the per-column value each method compared against its
        threshold to reach the functional verdict (see Report.statistic)."""
        return self._broadcast(self.get_results().get("statistic"), float)

    def get_drift_index(self) -> Optional[int]:
        """Return the testing-stream offset at which drift first fired, for the
        streaming online-CD detectors that record it (see river.py/frouros.py
        result()); None otherwise. A single scalar, deliberately NOT broadcast
        per column (unlike drift/statistic) and NOT folded into `statistic`."""
        index = self.get_results().get("drift_index")
        return int(index) if index is not None else None

    def report(self, criteria: set[Criteria]) -> Report:
        """Return the drift detection values."""
        return reports.Report(
            runtime=self.get_runtimes() if "runtime" in criteria else None,
            cputime=self.get_cputimes() if "cputime" in criteria else None,
            memory=self.get_memories() if "memory" in criteria else None,
            functional=self.get_functional() if "functional" in criteria else None,
            statistic=self.get_statistic() if "functional" in criteria else None,
            drift_index=self.get_drift_index() if "functional" in criteria else None,
            test_information=TestInformation(
                framework=self.tool.name,
                run_on_vm=self.run_on_vm,
                repetitions=self.repetitions,
                len_reference=len(self.data.reference),
                len_testing=len(self.data.testing),
            ),
            method=str(self.method),
            method_class=str(self.test),
        )


class Benchmark(BaseBenchmark):
    """Class to run a benchmark to obtain Results."""

    @cached_property
    def _results(self) -> dict[str, Any]:
        """Fit a fresh detector, run one test pass, and cache the result.

        functional, statistic and drift_index are all read from this single
        cached pass, on a detector that no other criterion has streamed -- so
        the three always describe the same clean run and drift_index no longer
        depends on whether a timing pass latched the verdict first (see
        _fitted_job). Caching also keeps the three getters from each re-running
        test().
        """
        job = self._fitted_job()
        job.test()
        return job.results

    def get_results(self) -> dict[str, Any]:
        """Return the drift statistics of the job (single cached test pass)."""
        return self._results

    def get_runtimes(self) -> reports.Stats:
        """
        Run the benchmark for the RUNTIME criterion.
        Measure elapsed time using wall-clock time in milliseconds.
        Includes waiting time for resources.
        """

        # Fresh detector for this criterion (see _fitted_job).
        # TODO: Future implementation for time training phase
        job = self._fitted_job()
        timer = timeit.Timer(job.test, timer=time.time)

        # Time runtimes measurements
        times = timer.repeat(self.repetitions, number=1)
        return reports.Stats.from_values(times)

    def get_cputimes(self) -> reports.Stats:
        """
        Run the benchmark for the CPUTIME criterion.
        Measures CPU resources consumed by the process (user and system)
        (exclude: waiting time for resources): time in ms
        """

        # Fresh detector for this criterion (see _fitted_job).
        # TODO: Future implementation for time training phase
        job = self._fitted_job()
        timer = timeit.Timer(job.test, timer=time.process_time)

        # Time runtimes measurements
        times = timer.repeat(self.repetitions, number=1)
        return reports.Stats.from_values(times)

    def get_memories(self) -> reports.Stats:
        """
        Run the benchmark for the MEMORY criterion.
        Measures RAM resources consumed by the process (user and system)
        """

        # Fresh detector for this criterion (see _fitted_job).
        # TODO: Future implementation for time training phase
        job = self._fitted_job()
        repeat = range(self.repetitions)
        rmem = [memory_usage(job.test) for _ in repeat]

        # Memory in run as the maximum memory used during the run
        mems = [max(mem) for mem in rmem]
        return reports.Stats.from_values(mems)


def _try_report(method: Method, test: Test, tool: Tool, criteria: set[Criteria]) -> Optional[Report]:
    """Run one method's benchmark, or return None if it fails.

    A single incompatible method/column combination (e.g. a statistical
    test whose preconditions the data doesn't satisfy) shouldn't take down
    every other method and framework in the same run.
    """
    try:
        return Benchmark(method, test, tool).report(criteria)
    except MethodNotApplicable as exc:
        logger.warning("Skipping %s/%s: not supported (%s)", tool.name, method, exc)
        return None
    except Exception:  # pylint: disable=broad-except
        logger.exception("Skipping %s/%s: benchmark failed", tool.name, method)
        return None


# Fixed iteration order, matching the sequence the four method dicts used to
# run in unconditionally (see git history) -- covariate/prior resolve to
# every family (config.DEFAULT_FAMILIES), so get_reports must still visit
# them in this exact order for those scenarios' JSON output to stay
# byte-identical; sorting `families` alphabetically instead would silently
# reorder every existing scenario's results (batch_* before online_*).
_FAMILY_ORDER: tuple[MethodFamily, ...] = ("online_cd", "online_dd", "batch_cd", "batch_dd")


def get_reports(
    tool: Tool, criteria: set[Criteria], families: frozenset[MethodFamily]
) -> Generator[Report, None, None]:
    """Return the drift detection values for methods in the given families.

    ``families`` is the scenario's resolved_families (see d3bench.scenario
    and config.DEFAULT_FAMILIES) -- restricts which of tool's four method
    dicts get run, so e.g. a concept-drift scenario isn't answered by a
    tool's covariate-only (batch_dd) methods.
    """
    for family in _FAMILY_ORDER:
        if family not in families:
            continue
        family_methods = tool.methods_by_family[family]
        if not family_methods:
            logger.warning(
                "%s: no %s methods -- contributes nothing to this scenario", tool.name, family
            )
            continue
        for method, test in family_methods.items():
            if report := _try_report(method, test, tool, criteria):
                yield report


class Job:
    """Class to run a benchmark job with the given parameters."""

    def __init__(self, benchmark: BaseBenchmark) -> None:
        self.benchmark = benchmark
        self.detector = benchmark.test(benchmark.tool.usable_features(benchmark.test))
        # Hand the detector the scenario's declared categorical set so its
        # fit() can call utils.numeric_column_indices with an explicit list
        # rather than inferring purely from dtype (which disagrees with the
        # other adapters on e.g. Elec2's digit-string `day`). Empty for every
        # dataset that declares none, so this is inert there.
        self.detector.categorical_columns = benchmark.tool.data.categorical_columns
        # For a concept-drift scenario, hand the detector the shared supervised
        # classifier's 0/1 error stream (see d3bench.supervised). Its presence
        # is what switches the online-CD adapters from their unsupervised
        # feature-norm path onto genuine P(y|X) error-stream monitoring; None
        # for covariate/prior leaves that path untouched. Only online_cd
        # methods run for "concept" (config.DEFAULT_FAMILIES), so no batch/
        # data-drift adapter ever reads this.
        data = benchmark.tool.data
        if data.drift_type == "concept":
            self.detector.error_stream = supervised.error_stream(data)

    def fit(self) -> None:
        """Run the benchmark with the given parameters."""
        # Call tool.reference_data to ensure preprocessing
        self.detector.fit(self.benchmark.tool.reference_data)  # Cached

    def test(self) -> None:
        """Run the benchmark with the given parameters."""
        # Call tool.testing_data to ensure preprocessing
        self.detector.test(self.benchmark.tool.testing_data)

    @property
    def results(self) -> dict[str, Any]:
        """Return the results of the benchmark."""
        return self.detector.result()
