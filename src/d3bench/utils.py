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


def numeric_column_indices(x: np.ndarray) -> list[int]:
    """Positions of x's (object-dtype, shape [n, n_features]) numeric-valued columns.

    Shared by tool adapters (alibi.py, frouros.py, ...) whose methods are
    continuous-only and need to drop categorical columns from a mixed-dtype
    dataset (e.g. French Motor Claims) -- a no-op (returns every index) on
    all-numeric datasets like energy/occupancy.
    """
    return [i for i in range(x.shape[1]) if _is_numeric_column(x[:, i])]


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


# "concept" is reserved for a future scenario (not yet implemented -- see
# OnlineCDReport's TODO below on why it needs more than a column selection).
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

    @property
    def len_reference(self) -> int:
        """Return the number of samples in the reference data."""
        return self.reference.shape[0]

    @property
    def len_testing(self) -> int:
        """Return the number of samples in the testing data."""
        return self.testing.shape[0]
