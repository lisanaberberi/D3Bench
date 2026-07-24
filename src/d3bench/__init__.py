"""
This module contains the configuration of the datasets and tools used in the
benchmarking process.
"""

from typing import Type

from d3bench import datasets, tools
from d3bench.config import Criteria, Datafile, Framework
from d3bench.datasets import Dataset
from d3bench.results import Results
from d3bench.tools import Tool

# pylint: disable=too-few-public-methods


# Dataset classes keyed by datafile name. Scenarios (see d3bench.scenario)
# instantiate these directly with per-scenario boundary/building_id/path;
# DATASETS below covers the plain-CLI path with fixed defaults.
DATASET_CLASSES: dict[Datafile, Type[Dataset]] = {
    "energy": datasets.DataEnergy,
    "occupancy": datasets.DataOccupancy,
    "motor": datasets.DataMotor,
    "motor_prior": datasets.DataMotorPrior,
    "elec2": datasets.DataElec2,
}

# Initialize the tools constant
TOOLS: dict[Framework, Type[Tool]] = {
    "Frouros": tools.Frouros,
    "Evidently": tools.Evidently,
    "NannyML": tools.NannyML,
    "Alibi-Detect": tools.AlibiDetect,
    "River": tools.River,
    "Menelaus": tools.Menelaus,
}

# Canonical scenario TOML per datafile -- the single source of truth for each
# dataset's split boundary/current_regions/building_id (see scenarios/*.toml).
# The plain `--datafile` CLI path (no --scenario) below builds DATASETS from
# these instead of duplicating the same split values as Python literals,
# which had drifted out of sync with the scenario files (e.g. "energy" here
# used to default to Options' boundary (2022-01-01) while
# scenarios/energy_covariate.toml has always split on 2020-04-01).
_CANONICAL_SCENARIOS: dict[Datafile, str] = {
    "energy": "scenarios/energy_covariate.toml",
    "occupancy": "scenarios/occupancy_covariate.toml",
    "motor": "scenarios/french_motor_covariate.toml",
    "motor_prior": "scenarios/french_motor_prior.toml",
    "elec2": "scenarios/elec2_concept.toml",
}


def _load_datasets() -> dict[Datafile, Dataset]:
    """Instantiate DATASETS from each dataset's canonical scenario TOML.

    Local import: d3bench.scenario itself does `from d3bench import
    DATASET_CLASSES, TOOLS`, so importing it at module scope (before those
    two names exist) would deadlock this circular import.
    """
    from d3bench.scenario import Scenario  # pylint: disable=import-outside-toplevel

    return {name: Scenario.from_toml(path).load_dataset() for name, path in _CANONICAL_SCENARIOS.items()}


# Initialize the datasets constant (plain-CLI path with fixed defaults).
DATASETS: dict[Datafile, Dataset] = _load_datasets()
