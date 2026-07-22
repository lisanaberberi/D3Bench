"""Configuration settings for the d3bench package."""

import os
from pathlib import Path
from typing import Literal, TypeAlias, Union

from d3bench import methods

# pylint: disable=too-few-public-methods


# Path where the data is stored
DATA_PATH = os.getenv("DATA_PATH", "datafiles")
data_path = Path(DATA_PATH)


# Path where the results are stored
RESULTS_PATH = os.getenv("RESULTS_PATH", "results")
results_path = Path(RESULTS_PATH)


# Define the types of drift detection methods
OnlineMethod: TypeAlias = Union[methods.OnlineCD, methods.OnlineDD]
BatchMethod: TypeAlias = Union[methods.BatchCD, methods.BatchDD]
Method: TypeAlias = Union[OnlineMethod, BatchMethod]


# Define the allowed frameworks for the benchmark
Framework: TypeAlias = Literal[
    "Frouros",
    "Evidently",
    "NannyML",
    "Alibi-Detect",
    "River",
    "Menelaus",
]

# Define the available datasets for the benchmark
Datafile: TypeAlias = Literal[
    "energy",
    "occupancy",
    "motor",
]


# Define the criteria evaluations for the benchmark
Criteria: TypeAlias = Literal[
    "functional",
    "statistics",
    "runtime",
    "cputime",
    "memory",
]

# Significance level used to turn a p-value into a drift/no-drift decision
# for the "functional" criterion (see Report.functional).
SIGNIFICANCE_LEVEL = 0.05
