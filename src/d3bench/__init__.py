"""
This module contains the configuration of the datasets and tools used in the
benchmarking process.
"""

import datetime as dt
from typing import Type

from d3bench import datasets, tools
from d3bench.config import Criteria, Datafile, Framework
from d3bench.datasets import Dataset
from d3bench.datasets import Options as DatasetOptions
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
}

# Initialize the datasets constant.
# occupancy's own date range (2021-03-30 -- 2021-07-11) falls entirely
# before Options' default boundary (2022-01-01), which would otherwise
# leave the testing split empty; use the boundary reported in the paper
# (9 May 2021) instead.
# motor has no time axis at all -- it group-splits on Region instead of a
# boundary date (see datasets.DataMotor); R82/R93 (~24% of policies) are held
# out as the "current" (new-geography) set, the rest as "reference".
DATASETS: dict[Datafile, Dataset] = {
    "energy": DATASET_CLASSES["energy"](building_id=1),
    "occupancy": DATASET_CLASSES["occupancy"](
        settings=DatasetOptions(boundary=dt.date(2021, 5, 9))
    ),
    "motor": DATASET_CLASSES["motor"](
        settings=DatasetOptions(current_regions=["R82", "R93"])
    ),
    # motor_prior self-configures via Options defaults (seed/target_claim_rate) --
    # no boundary/region key needed, see datasets.DataMotorPrior.
    "motor_prior": DATASET_CLASSES["motor_prior"](),
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
