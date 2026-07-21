import datetime as dt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings

from d3bench import config
from d3bench.utils import Data

# pylint: disable=too-few-public-methods


class Options(BaseSettings):
    """Settings to instantiate a dataset."""

    data_start: dt.date = Field(
        default=dt.date(2019, 4, 1),
        description="Start date.",
    )
    data_end: dt.date = Field(
        default=dt.date(2022, 4, 1),
        description="End date.",
    )
    boundary: dt.date = Field(
        default=dt.date(2022, 1, 1),
        description="Boundary date. Used by time-indexed datasets (e.g. DataEnergy, DataOccupancy).",
    )
    current_regions: Optional[list[str]] = Field(
        default=None,
        description=(
            "Region codes held out as the 'current' (deployment) set for cross-sectional "
            "datasets with no time axis (e.g. DataMotor), which group-split instead of "
            "boundary-split."
        ),
    )


class Dataset(ABC):
    """Abstract class for the datasets."""

    file_name: str
    measure_columns: list[str]

    def __init__(self, settings: Optional[Options] = None, path: Optional[Union[str, Path]] = None):
        settings = settings or Options()
        self.df: pd.DataFrame = pd.read_csv(path or config.data_path / self.file_name, low_memory=False)
        self.df["time"] = self.preprocess_time()
        self.data_start = settings.data_start
        self.data_end = settings.data_end
        self.filter_data()  # Remove data out of limits
        self.boundary = settings.boundary
        self.current_regions = settings.current_regions

    @abstractmethod
    def preprocess_time(self) -> pd.DataFrame:
        """Returns the dataset with a datetime column."""

    def filter_data(self) -> None:
        """Remove columns and rows not used in the benchmark."""
        nan_rows = self.df[self.measure_columns].isna()
        self.df = self.df[~nan_rows.any(axis=1)]  # rm rows with nan
        self.df = self.df[self.df["time"] >= str(self.data_start)]
        self.df = self.df[self.df["time"] <= str(self.data_end)]

    def split_data(self) -> Data:
        """Return the dataset for the given building."""
        boundary_timestamp = pd.Timestamp(self.boundary)
        train_filter = self.df["time"] < boundary_timestamp
        return Data(
            features=self.measure_columns,
            reference=self.df[train_filter],
            testing=self.df[~train_filter],
        )


class DataEnergy(Dataset):
    """Class for the energy dataset."""

    file_name = "energy_data.csv"
    datetime_columns = ["year", "month", "day", "hour"]
    consumption_unit = "MWh"
    temperature_unit = "°C"  # TODO: Check if it is correct
    measure_columns = ["consumption", "temp_outside"]

    def __init__(self, building_id: int, *args, **kwds):
        super().__init__(*args, **kwds)
        self.df = self.df[self.df["ids"] == building_id]
        self.purpose_of_use = self.df["purpose_of_use"].iloc
        self.gross_area = self.df["gross_area"].iloc[0]
        self.floor_area = self.df["floor_area"].iloc[0]
        self.apartment_sector = self.df["apartment_sector"].iloc[0]
        self.total_volume = self.df["total_volume"].iloc[0]
        self.building_type = self.df["building_type"].iloc[0]
        self.building_id = self.df["ids"].iloc[0]
        self.df = self.df[self.measure_columns + ["time"]]

    def preprocess_time(self) -> pd.DataFrame:
        """Merge date and time columns to datetime column."""
        datetime = self.df[DataEnergy.datetime_columns]
        return pd.to_datetime(datetime)


class DataOccupancy(Dataset):
    """Class for the occupancy dataset."""

    file_name = "occupancy_data.csv"
    measure_columns = ["measured", "co2", "temperature"]

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.df = self.df[self.measure_columns + ["time"]]

    def preprocess_time(self) -> pd.DataFrame:
        """Parse the time column to a datetime column."""
        return pd.to_datetime(self.df["time"])


class DataMotor(Dataset):
    """Class for the French Motor Third-Party Liability Claims dataset (freMTPL2freq).

    Cross-sectional (one row per policy, no time axis) rather than time-indexed
    like DataEnergy/DataOccupancy, so covariate drift here is constructed by
    portfolio composition rather than temporal progression: policies whose
    ``Region`` is in ``settings.current_regions`` form the "current" set
    (simulating deployment of a model to a new geography), and every other
    Region forms the "reference" set. ``Region`` itself is excluded from
    ``measure_columns`` since it is the split key -- testing drift on it would
    be trivial by construction.

    measure_columns is the reference paper's own 9-dimensional feature vector
    (Noll/Salzmann/Wuthrich, "Case Study: French Motor Third-Party Liability
    Claims", datafiles/french-motor-paper.pdf, Sec. 2.1: "x_i = (Area_i,
    VehPower_i, VehAge_i, DrivAge_i, BonusMalus_i, VehBrand_i, VehGas_i,
    Density_i, Region_i)'", also the exact GLM1 formula fit in Listing 4)
    minus Region, which is this scenario's split key instead of a feature.
    Region-based covariate-drift splitting itself is *not* from that paper --
    its own train/test split is a plain random 90/10 sample (Listing 2); the
    Region split simulating deployment to a new geography is this benchmark's
    own construction, described in the drift-benchmark paper instead.
    """

    file_name = "freMTPL2freq.csv"
    measure_columns = [
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Density",
    ]

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        if not self.current_regions:
            raise ValueError(
                "DataMotor requires settings.current_regions (e.g. ['R82', 'R93']) -- "
                "there is no time-based boundary to fall back on for this dataset."
            )

    def preprocess_time(self) -> pd.Series:
        """No genuine time axis; Region grouping stands in for the split key."""
        return pd.Series(pd.NaT, index=self.df.index)

    def filter_data(self) -> None:
        """Drop rows with missing values in the tested columns (no date range to apply)."""
        nan_rows = self.df[self.measure_columns].isna()
        self.df = self.df[~nan_rows.any(axis=1)]

    def split_data(self) -> Data:
        """Split by Region membership instead of a time boundary.

        Region is only needed to compute the split; the returned frames carry
        just measure_columns (+ the unused "time" column every Tool.preprocess
        drops) -- Region, IDpol, ClaimNb, and Exposure are dropped here so they
        can't leak into tools that stack every column in the frame (River,
        Frouros), the same way DataEnergy/DataOccupancy pre-narrow to
        measure_columns in their own __init__.
        """
        current_filter = self.df["Region"].isin(self.current_regions)
        columns = self.measure_columns + ["time"]
        return Data(
            features=self.measure_columns,
            reference=self.df.loc[~current_filter, columns],
            testing=self.df.loc[current_filter, columns],
        )
