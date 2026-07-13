"""Module to run a benchmark to obtain Results."""

import logging
import time
import timeit
from abc import ABC, abstractmethod
from copy import copy
from functools import cached_property
from typing import Any, Generator, Optional, Type, TypeAlias, Union

from memory_profiler import memory_usage
from pydantic import Field
from pydantic_settings import BaseSettings

from d3bench import reports
from d3bench.config import Criteria, Method
from d3bench.reports import Report, TestInformation
from d3bench.tools import Tool
from d3bench.utils import BaseTestMethod, Data

# pylint: disable=too-few-public-methods
logger = logging.getLogger(__name__)
Test: TypeAlias = Type[BaseTestMethod]


class BaseBenchmark(ABC):
    """Base class to define benchmarks to obtain Results."""

    def __init__(self, method: Method, test: Test, tool: Tool) -> None:
        self.method = method
        self.test = test
        self.tool = tool
        self.job = Job(benchmark=self)
        self._prepare_job()

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

    def _prepare_job(self) -> None:
        """Fit the job for the benchmark."""
        try:
            self.job.fit()
        except NotImplementedError:
            logger.debug("No train method for %s", self.method)

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

    def get_functional(self) -> Optional[dict[str, bool]]:
        """Return whether drift was flagged on each monitored column.

        Each detector's `result()` exposes a "drift" entry that is either a
        single bool (methods that only look at the combined feature signal)
        or a dict keyed by column (methods that test each feature
        separately). A single bool is broadcast to every monitored column so
        the "functional" criterion always yields a per-column mapping.
        """
        drift = self.get_results().get("drift")
        if drift is None:
            return None
        if isinstance(drift, dict):
            return {column: bool(value) for column, value in drift.items()}
        return {feature: bool(drift) for feature in self.data.features}

    def report(self, criteria: set[Criteria]) -> Report:
        """Return the drift detection values."""
        return reports.Report(
            runtime=self.get_runtimes() if "runtime" in criteria else None,
            cputime=self.get_cputimes() if "cputime" in criteria else None,
            memory=self.get_memories() if "memory" in criteria else None,
            functional=self.get_functional() if "functional" in criteria else None,
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

    def get_results(self) -> dict[str, Any]:
        """
        Return the drift statistics of the job.
        Run the drift detection for the data.
        """

        # Copy the job to avoid modifying the original job
        _job = self.job.copy()

        # Run the drift detection for the data
        _job.test()

        # Return the drift statistics
        return _job.results

    def get_runtimes(self) -> reports.Stats:
        """
        Run the benchmark for the RUNTIME criterion.
        Measure elapsed time using wall-clock time in milliseconds.
        Includes waiting time for resources.
        """

        # Create a runtime timer
        # TODO: Future implementation for time training phase
        timer = timeit.Timer(self.job.test, timer=time.time)

        # Time runtimes measurements
        times = timer.repeat(self.repetitions, number=1)
        return reports.Stats.from_values(times)

    def get_cputimes(self) -> reports.Stats:
        """
        Run the benchmark for the CPUTIME criterion.
        Measures CPU resources consumed by the process (user and system)
        (exclude: waiting time for resources): time in ms
        """

        # Create a runtime timer
        # TODO: Future implementation for time training phase
        timer = timeit.Timer(self.job.test, timer=time.process_time)

        # Time runtimes measurements
        times = timer.repeat(self.repetitions, number=1)
        return reports.Stats.from_values(times)

    def get_memories(self) -> reports.Stats:
        """
        Run the benchmark for the MEMORY criterion.
        Measures RAM resources consumed by the process (user and system)
        """

        # Run the drift detection for each building
        # TODO: Future implementation for time training phase
        repeat = range(self.repetitions)
        rmem = [memory_usage(self.job.test) for _ in repeat]

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
    except Exception:  # pylint: disable=broad-except
        logger.exception("Skipping %s/%s: benchmark failed", tool.name, method)
        return None


def get_reports(tool: Tool, criteria: set[Criteria]) -> Generator[Report, None, None]:
    """Return the drift detection values."""
    for method, test in tool.online_cd_methods.items():
        if report := _try_report(method, test, tool, criteria):
            yield report
    for method, test in tool.online_dd_methods.items():
        if report := _try_report(method, test, tool, criteria):
            yield report
    for method, test in tool.batch_cd_methods.items():
        if report := _try_report(method, test, tool, criteria):
            yield report
    for method, test in tool.batch_dd_methods.items():
        if report := _try_report(method, test, tool, criteria):
            yield report


class Job:
    """Class to run a benchmark job with the given parameters."""

    def __init__(self, benchmark: BaseBenchmark) -> None:
        self.benchmark = benchmark
        self.detector = benchmark.test(benchmark.tool.data.features)

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

    def copy(self) -> "Job":
        """Return a copy of the job."""
        job = copy(self)
        job.detector = copy(self.detector)
        return job
