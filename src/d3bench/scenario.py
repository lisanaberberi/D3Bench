"""Scenario abstraction: load a TOML-defined benchmarking scenario.

A *scenario* is the reproducible evaluation unit described in the
drift-benchmark methodology: a dataset selection with its ground-truth
split boundary, plus the frameworks and criteria to run against it (see
``scenarios/energy_covariate.toml`` and ``scenarios/occupancy_covariate.toml``).

This module only turns a TOML file into calls against the existing
abstractions (``Dataset``, ``Tool``, ``BaseTestMethod``, ``Benchmark``,
``Results``); it adds no new drift-detection logic. In particular, the
"functional" criterion -- whether each framework's tests flagged drift on
each monitored column -- is already computed generically by
``Benchmark.get_functional()`` for any ``Dataset``/``Tool`` pair, so a new
scenario needs no framework-specific or dataset-specific code here.
"""

import datetime as dt
import tomllib
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field

from d3bench import DATASET_CLASSES, TOOLS
from d3bench.config import Criteria, Datafile, Framework
from d3bench.datasets import Dataset
from d3bench.datasets import Options as DatasetOptions
from d3bench.results import Results
from d3bench.tools import Options as ToolOptions
from d3bench.tools import Tool
from d3bench.utils import Data

# pylint: disable=too-few-public-methods

# Scenario files use the paper's UPPER_SNAKE criteria names; map them onto
# the lowercase Criteria literals used by Benchmark/Report.
_CRITERIA_ALIASES: dict[str, Criteria] = {
    "FUNCTIONAL": "functional",
    "RUNTIME": "runtime",
    "CPU_RUNTIME": "cputime",
    "MEMORY": "memory",
}

_SPLIT_BOUNDARY_FORMAT = "%m-%d-%Y %H:%M"


class DataConfig(BaseModel):
    """The ``[data]`` table of a scenario TOML file."""

    dataset: Datafile
    path: Optional[Path] = None
    building_id: Optional[int] = None
    split_boundary: str = Field(..., description='Ground-truth split, e.g. "04-01-2020 00:00".')


class RunConfig(BaseModel):
    """The ``[run]`` table of a scenario TOML file."""

    tools: list[Framework]
    criteria: list[str]
    show_report: bool = True
    vm: bool = False

    @property
    def resolved_criteria(self) -> set[Criteria]:
        """Map the TOML's UPPER_SNAKE criteria names onto internal literals."""
        return {_CRITERIA_ALIASES.get(name, name.lower()) for name in self.criteria}


class Scenario(BaseModel):
    """A standardized, reproducible evaluation unit.

    Combines a data source, its ground-truth drift boundary, and the
    frameworks/criteria under test, as declared in a TOML file such as
    ``scenarios/energy_covariate.toml`` (Listing "energy-scenario").
    """

    data: DataConfig
    run: RunConfig

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "Scenario":
        """Parse a scenario definition from a TOML file."""
        with open(path, "rb") as toml_file:
            return cls.model_validate(tomllib.load(toml_file))

    def load_dataset(self) -> Dataset:
        """Instantiate the scenario's dataset with its ground-truth boundary."""
        dataset_cls = DATASET_CLASSES[self.data.dataset]
        boundary = dt.datetime.strptime(self.data.split_boundary, _SPLIT_BOUNDARY_FORMAT).date()
        settings = DatasetOptions(boundary=boundary)
        if self.data.building_id is not None:
            return dataset_cls(self.data.building_id, settings=settings, path=self.data.path)
        return dataset_cls(settings=settings, path=self.data.path)

    def load_tools(self, data: Data) -> list[Tool]:
        """Instantiate the scenario's tools (adapters) against the split data."""
        settings = ToolOptions(on_vm=self.run.vm)
        return [TOOLS[name](data, settings=settings) for name in self.run.tools]

    def run_benchmark(self) -> Results:
        """Prepare the scenario's data/tools and run the benchmark."""
        data = self.load_dataset().split_data()
        tool_instances = self.load_tools(data)
        return Results(tool_instances, self.run.resolved_criteria)
