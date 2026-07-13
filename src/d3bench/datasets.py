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
        description="Boundary date.",
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
