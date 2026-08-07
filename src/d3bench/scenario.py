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

from pydantic import BaseModel, Field, field_validator, model_validator

from d3bench import DATASET_CLASSES, TOOLS
from d3bench.config import (
    DEFAULT_FAMILIES,
    Criteria,
    Datafile,
    Framework,
    MethodFamily,
    resolve_project_path,
)
from d3bench.datasets import Dataset
from d3bench.datasets import Options as DatasetOptions
from d3bench.results import Results
from d3bench.tools import Options as ToolOptions
from d3bench.tools import Tool
from d3bench.utils import Data, DriftType

# Datasets that configure their own reference/testing split internally (via
# Options defaults -- seed, target_claim_rate, ... -- or, for elec2, a fixed
# chronological split baked into DataElec2 itself) rather than a
# scenario-declared split_boundary/current_regions key.
_SELF_CONFIGURING_DATASETS: frozenset[Datafile] = frozenset({"motor_prior", "elec2", "elec2_injected"})

# pylint: disable=too-few-public-methods

# Scenario files use the paper's UPPER_SNAKE criteria names; map them onto
# the lowercase Criteria literals used by Benchmark/Report.
_CRITERIA_ALIASES: dict[str, Criteria] = {
    "FUNCTIONAL": "functional",
    "STATISTICS": "statistics",
    "RUNTIME": "runtime",
    "CPU_RUNTIME": "cputime",
    "MEMORY": "memory",
}

_SPLIT_BOUNDARY_FORMAT = "%m-%d-%Y %H:%M"


class DataConfig(BaseModel):
    """The ``[data]`` table of a scenario TOML file.

    Time-indexed datasets (energy, occupancy) split on ``split_boundary``.
    Cross-sectional datasets with no time axis (motor) split on
    ``current_regions`` instead -- exactly one of the two must be set,
    matching whichever ground-truth split the dataset's ``Dataset`` subclass
    implements (see ``d3bench.datasets``). Self-configuring datasets (e.g.
    motor_prior, see ``_SELF_CONFIGURING_DATASETS``) require neither -- they
    build their own split from ``Options`` defaults instead.
    """

    dataset: Datafile
    drift_type: DriftType
    path: Optional[Path] = None
    building_id: Optional[int] = None
    split_boundary: Optional[str] = Field(
        default=None, description='Ground-truth time split, e.g. "04-01-2020 00:00".'
    )
    current_regions: Optional[list[str]] = Field(
        default=None,
        description='Ground-truth group split for cross-sectional datasets, e.g. ["R82", "R93"].',
    )

    @field_validator("path")
    @classmethod
    def _anchor_path(cls, value: Optional[Path]) -> Optional[Path]:
        """A scenario's ``path`` is written project-root-relative (e.g.
        ``datafiles/elecNormNew.arff``), so resolve it like ``config
        .data_path`` does rather than against the process cwd."""
        return value if value is None else resolve_project_path(value)

    @model_validator(mode="after")
    def _check_split_is_set(self) -> "DataConfig":
        if self.dataset in _SELF_CONFIGURING_DATASETS:
            if self.split_boundary is not None or self.current_regions is not None:
                raise ValueError(
                    f"{self.dataset} self-configures its split via Options defaults; "
                    "split_boundary/current_regions must not be set in [data]"
                )
            return self
        if (self.split_boundary is None) == (self.current_regions is None):
            raise ValueError(
                "exactly one of split_boundary or current_regions must be set in [data]"
            )
        return self


class RunConfig(BaseModel):
    """The ``[run]`` table of a scenario TOML file."""

    tools: list[Framework]
    criteria: list[str]
    show_report: bool = True
    vm: bool = False
    method_families: Optional[list[str]] = Field(
        default=None,
        description=(
            "Override config.DEFAULT_FAMILIES[data.drift_type] -- which of each "
            'tool\'s online_cd/online_dd/batch_cd/batch_dd method dicts run, e.g. '
            '["online_cd", "batch_cd"]. Rarely needed: the drift_type default '
            "already picks the right families for covariate/prior/concept."
        ),
    )

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
        """Parse a scenario definition from a TOML file.

        A relative ``path`` is read from the cwd if it exists there and from
        the project root otherwise, so ``"scenarios/elec2_concept.toml"``
        works from a notebook in ``scripts/`` as well as from the root.
        """
        with open(resolve_project_path(path), "rb") as toml_file:
            return cls.model_validate(tomllib.load(toml_file))

    def load_dataset(self) -> Dataset:
        """Instantiate the scenario's dataset with its ground-truth split."""
        dataset_cls = DATASET_CLASSES[self.data.dataset]
        # Dataset.drift_type is declared on the class, so a misdeclared
        # scenario is caught here rather than after reading the datafile (the
        # energy CSV is ~100 MB). run_benchmark still re-checks the *built*
        # Data, which covers a subclass that overrides split_data.
        if dataset_cls.drift_type != self.data.drift_type:
            raise ValueError(
                f"scenario declares drift_type={self.data.drift_type!r} but dataset "
                f"{self.data.dataset!r} ({dataset_cls.__name__}) is a "
                f"{dataset_cls.drift_type!r} construction -- "
                "scenario file and Dataset subclass disagree"
            )
        if self.data.split_boundary is not None:
            boundary = dt.datetime.strptime(self.data.split_boundary, _SPLIT_BOUNDARY_FORMAT).date()
            settings = DatasetOptions(boundary=boundary)
        elif self.data.current_regions is not None:
            settings = DatasetOptions(current_regions=self.data.current_regions)
        else:
            # self-configuring dataset (e.g. motor_prior): Options defaults apply.
            settings = DatasetOptions()
        if self.data.building_id is not None:
            return dataset_cls(self.data.building_id, settings=settings, path=self.data.path)
        return dataset_cls(settings=settings, path=self.data.path)

    def load_tools(self, data: Data) -> list[Tool]:
        """Instantiate the scenario's tools (adapters) against the split data."""
        settings = ToolOptions(on_vm=self.run.vm)
        return [TOOLS[name](data, settings=settings) for name in self.run.tools]

    @property
    def resolved_families(self) -> frozenset[MethodFamily]:
        """Which method families (see config.MethodFamily) to run: [run]
        .method_families if the scenario overrides it, else
        config.DEFAULT_FAMILIES[data.drift_type]."""
        if self.run.method_families is not None:
            return frozenset(name.lower() for name in self.run.method_families)
        return DEFAULT_FAMILIES[self.data.drift_type]

    def run_benchmark(self) -> Results:
        """Prepare the scenario's data/tools and run the benchmark."""
        data = self.load_dataset().split_data()
        if data.drift_type != self.data.drift_type:
            raise ValueError(
                f"scenario declares drift_type={self.data.drift_type!r} but dataset "
                f"{self.data.dataset!r} constructed drift_type={data.drift_type!r} -- "
                "scenario file and Dataset subclass disagree"
            )
        tool_instances = self.load_tools(data)
        return Results(tool_instances, self.run.resolved_criteria, self.resolved_families)
