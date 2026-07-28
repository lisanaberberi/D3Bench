"""Configuration settings for the d3bench package."""

import os
from pathlib import Path
from typing import Literal, TypeAlias, Union

from d3bench import methods
from d3bench.utils import DriftType

# pylint: disable=too-few-public-methods


# The D3Bench project root (this file is src/d3bench/config.py). The
# datafiles/, results/ and scenarios/ directories live next to it, not inside
# the package, and every path below used to be interpreted against the process
# cwd -- which broke `import d3bench` from anywhere but the project root (e.g.
# a notebook in scripts/, where a Jupyter kernel starts).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: Union[str, Path]) -> Path:
    """Resolve a benchmark path independently of the process cwd.

    Absolute paths pass through. A relative path that already resolves from
    the current working directory keeps that meaning, so a cwd-local override
    (or an explicit DATA_PATH/RESULTS_PATH) still wins; otherwise it is taken
    as relative to PROJECT_ROOT.
    """
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


# Path where the data is stored
DATA_PATH = os.getenv("DATA_PATH", "datafiles")
data_path = resolve_project_path(DATA_PATH)


# Path where the results are stored
RESULTS_PATH = os.getenv("RESULTS_PATH", "results")
results_path = resolve_project_path(RESULTS_PATH)


# Path where the scenario TOML files are stored
SCENARIOS_PATH = os.getenv("SCENARIOS_PATH", "scenarios")
scenarios_path = resolve_project_path(SCENARIOS_PATH)


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
    "motor_prior",
    "elec2",
    "elec2_injected",
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


# Names match the four Tool.*_methods attribute prefixes (see d3bench.tools.Tool).
MethodFamily: TypeAlias = Literal["online_cd", "online_dd", "batch_cd", "batch_dd"]

_ALL_FAMILIES: frozenset[MethodFamily] = frozenset({"online_cd", "online_dd", "batch_cd", "batch_dd"})

# Which method families run for each scenario drift_type (see d3bench.scenario
# .resolved_families; overridable per-scenario via [run].method_families).
#
# Only "concept" is actually restricted, to {online_cd, batch_cd}: those are
# the families a concept-drift scenario (e.g. elec2_concept.toml) should be
# answered by, since "concept" is the one drift_type "prior"'s own
# Tool._monitored_columns special-case doesn't already narrow to a single
# label column.
#
# "covariate" and "prior" stay unrestricted (every family) rather than
# narrowing to {online_dd, batch_dd} as the *_dd/*_cd naming might suggest:
# in this codebase the cd/dd split is partly just online-vs-batch, not a
# clean P(y|X)-vs-P(X) split -- e.g. Alibi-Detect's OnlineMaximumMeanDiscrepancy
# /OnlineLeastSquaresDensityDifference/OnlineCramerVonMisesTest and every one
# of River's online methods are filed under online_cd purely because no
# online_dd enum member exists for them yet (see the comments in
# d3bench.tools.AlibiDetect/River), even though they test P(X) like their
# batch_dd cousins. Restricting covariate/prior would silently drop those
# already-published results (all of River's contribution, most of Frouros's,
# some of Alibi-Detect's) for no semantic gain.
DEFAULT_FAMILIES: dict[DriftType, frozenset[MethodFamily]] = {
    "covariate": _ALL_FAMILIES,
    "prior": _ALL_FAMILIES,
    "concept": frozenset({"online_cd", "batch_cd"}),
}
