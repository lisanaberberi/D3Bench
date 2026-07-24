"""Utility functions and classes for the drift detection methods."""

import dataclasses as dc
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeAlias, Literal

import numpy as np
import pandas as pd
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
)
from rich_argparse import RichHelpFormatter

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


def _is_numeric_column(column: np.ndarray) -> bool:
    """Whether an object-dtype column (from Tool.preprocess) holds numeric values."""
    try:
        column.astype(np.float64)
        return True
    except (TypeError, ValueError):
        return False


def numeric_column_indices(
    x: np.ndarray, features: list[str], categorical_columns: list[str]
) -> list[int]:
    """Positions of x's (object-dtype, shape [n, n_features]) numeric-valued columns.

    Shared by tool adapters (alibi.py, frouros.py, ...) whose methods are
    continuous-only and need to drop categorical columns from a mixed-dtype
    dataset (e.g. French Motor Claims) -- a no-op (returns every index) on
    all-numeric datasets like energy/occupancy.

    ``features`` names x's columns in order (``features[i]`` is column ``i``);
    ``categorical_columns`` (from ``Data.categorical_columns``) is the explicit,
    declared categorical set. A declared column is excluded *by name* before its
    dtype is ever inspected -- this is the only mechanism that catches Elec2's
    ``day`` (an ARFF nominal {1..7} decoded to digit-strings like "2", which
    ``astype(np.float64)`` happily accepts, so the dtype probe alone would keep
    it and disagree with the other adapters). The dtype probe stays as a
    *fallback* for every undeclared column, so datasets that never declare a
    categorical set (energy/occupancy: all-numeric; French Motor: real string
    columns that already fail the float cast) are entirely unaffected.
    """
    declared = set(categorical_columns)
    return [
        i
        for i in range(x.shape[1])
        if features[i] not in declared and _is_numeric_column(x[:, i])
    ]


class BaseArguments(BaseSettings):
    """Base class for all scripts"""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Enable CLI formatter_class to work properly."""
        return (
            CliSettingsSource(
                settings_cls,
                formatter_class=RichHelpFormatter,
                cli_parse_args=True,
            ),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


class MethodNotApplicable(Exception):
    """A method/tool combination that cannot run on the current scenario's
    monitored columns (e.g. a continuous-only method with nothing numeric
    left to monitor, such as a "prior"-drift scenario whose only monitored
    column is categorical). Raised by tool adapters so ``_try_report`` can
    log one short line instead of a full traceback or library-internal
    error."""


class BaseTestMethod(ABC):
    """Base class for the test methods."""

    #: Declared categorical columns (from Data.categorical_columns), set by
    #: Job on the instance after construction (benchmarks.Job.__init__). Empty
    #: by default so an adapter that reads it before Job wires it -- or a tool
    #: whose data declares none -- falls back to pure dtype inference.
    #: numeric_column_indices consumes this to exclude e.g. Elec2's `day`.
    categorical_columns: list[str] = []

    @abstractmethod
    def __init__(self, features: list[str]) -> None:
        """Initialize the test method."""

    @abstractmethod
    def fit(self, x_reference: Any) -> None:
        """Run the test on the reference data."""

    @abstractmethod
    def test(self, x_test: Any) -> None:
        """Run the test on the test data."""

    @abstractmethod
    def result(self) -> dict[str, Any]:
        """Return the result of the test."""


# "concept" ships as scenarios/elec2_concept.toml. Unlike "prior" (which
# narrows Tool._monitored_columns to a single label column), "concept"
# selects *method families* instead -- see config.DEFAULT_FAMILIES and
# Scenario.resolved_families -- since there's no single "the concept-drift
# column" to narrow to; a concept scenario is answered by online_cd/batch_cd
# methods across every monitored column. OnlineCDReport's TODO below still
# stands: ground-truth-drift-point metrics (detection delay, false alarm
# rate, ...) aren't implemented, since Data/Dataset carries no drift-point
# label -- only which methods run is drift_type-aware so far.
DriftType: TypeAlias = Literal["covariate", "prior", "concept"]


@dc.dataclass
class Data:  # pylint: disable=missing-class-docstring
    features: list[str]
    reference: pd.DataFrame
    testing: pd.DataFrame
    drift_type: DriftType
    #: Label column name for non-covariate scenarios (e.g. DataMotorPrior's
    #: "ClaimNb"), already present in `reference`/`testing`. None for
    #: drift_type == "covariate".
    target: Optional[str] = None
    #: Columns to treat as categorical regardless of how they happen to be
    #: stored -- this OVERRIDES storage dtype. Motivating case: Elec2's `day`
    #: is an ARFF nominal attribute {1..7} that _read_dataset_file decodes to
    #: digit-strings ("2", ...), which cast cleanly to float, so a
    #: dtype/value probe would wrongly keep it as a continuous feature. Naming
    #: it here is the single source of truth that makes every continuous-only
    #: adapter (Frouros/Alibi via numeric_column_indices, River via
    #: preprocess) drop the *same* columns. Empty for datasets whose
    #: categorical columns are genuinely non-numeric (French Motor's VehBrand/
    #: VehGas) or that have none at all (energy/occupancy).
    categorical_columns: list[str] = dc.field(default_factory=list)

    @property
    def len_reference(self) -> int:
        """Return the number of samples in the reference data."""
        return self.reference.shape[0]

    @property
    def len_testing(self) -> int:
        """Return the number of samples in the testing data."""
        return self.testing.shape[0]
