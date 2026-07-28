import datetime as dt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings
from scipy.io import arff as scipy_arff

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
    seed: int = Field(
        default=31,
        description="Random seed for semi-synthetic/resampled constructions (e.g. DataMotorPrior).",
    )
    target_claim_rate: float = Field(
        default=0.15,
        description=(
            "Target share of policies with >=1 claim in DataMotorPrior's resampled 'current' "
            "set (natural population rate is ~5.0%, see datafiles/french-motor-paper.pdf Table 1)."
        ),
    )


def _read_dataset_file(path: Union[str, Path]) -> pd.DataFrame:
    """Load a dataset file into a DataFrame, dispatching on suffix.

    Every dataset but DataElec2 is a plain CSV; DataElec2's elecNormNew.arff
    needs scipy's ARFF reader instead, plus decoding its nominal attributes
    (day, class) from bytes to str -- loadarff returns those as bytes.
    """
    path = Path(path)
    if path.suffix == ".arff":
        raw, _meta = scipy_arff.loadarff(path)
        df = pd.DataFrame(raw)
        for column in df.select_dtypes(include="object"):
            df[column] = df[column].str.decode("utf-8")
        return df
    return pd.read_csv(path, low_memory=False)


class Dataset(ABC):
    """Abstract class for the datasets."""

    file_name: str
    measure_columns: list[str]
    #: Columns among measure_columns/target to treat as categorical regardless
    #: of storage dtype -- forwarded to Data.categorical_columns by split_data()
    #: and read by the continuous-only adapters. Empty on datasets whose
    #: categorical columns are genuinely non-numeric (French Motor) or absent
    #: (energy/occupancy); overridden only by DataElec2 (see there).
    categorical_columns: list[str] = []

    def __init__(self, settings: Optional[Options] = None, path: Optional[Union[str, Path]] = None):
        settings = settings or Options()
        self.df: pd.DataFrame = _read_dataset_file(path or config.data_path / self.file_name)
        self.df["time"] = self.preprocess_time()
        self.data_start = settings.data_start
        self.data_end = settings.data_end
        self.filter_data()  # Remove data out of limits
        self.boundary = settings.boundary
        self.current_regions = settings.current_regions
        self.seed = settings.seed
        self.target_claim_rate = settings.target_claim_rate

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
            drift_type="covariate",
            categorical_columns=self.categorical_columns,
        )


class DataEnergy(Dataset):
    """Class for the energy dataset."""

    file_name = "energy_data.csv"
    datetime_columns = ["year", "month", "day", "hour"]
    consumption_unit = "MWh"
    temperature_unit = "°C"  # Outdoor air temperature
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
            drift_type="covariate",
            categorical_columns=self.categorical_columns,
        )


def _random_half_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded, disjoint 50/50 row split -- stands in for DataMotor's Region
    grouping for constructions that must NOT carry any covariate-drift signal
    of their own (DataMotorPrior)."""
    half_a = df.sample(frac=0.5, random_state=seed)
    half_b = df.drop(half_a.index)
    return half_a, half_b


def _resample_to_rate(df: pd.DataFrame, positive: pd.Series, target_rate: float, seed: int) -> pd.DataFrame:
    """Return a subset of df's real rows (no duplication, no replacement) whose
    `positive` share is target_rate, holding P(X | positive) exactly fixed --
    every retained row is untouched; only the relative count of positive vs.
    negative rows changes. Downsamples whichever class is over-represented
    relative to the target ratio."""
    rng = np.random.default_rng(seed)
    pos_idx, neg_idx = df.index[positive], df.index[~positive]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    neg_needed = n_pos * (1 - target_rate) / target_rate
    if neg_needed <= n_neg:
        keep_pos, keep_neg = pos_idx.to_numpy(), rng.choice(neg_idx, size=int(neg_needed), replace=False)
    else:
        pos_needed = n_neg * target_rate / (1 - target_rate)
        if pos_needed > n_pos:
            raise ValueError(
                f"target_rate={target_rate} infeasible without replacement: "
                f"pool has {n_pos} positive / {n_neg} negative rows"
            )
        keep_pos, keep_neg = rng.choice(pos_idx, size=int(pos_needed), replace=False), neg_idx.to_numpy()
    return df.loc[np.concatenate([keep_pos, keep_neg])]


class DataMotorPrior(Dataset):
    """Prior-drift (target/label-drift) construction on freMTPL2freq.

    Both reference and current are disjoint random halves of the same
    portfolio (no group/region signal, unlike DataMotor's covariate split --
    so P(X) is unaffected by construction). Reference keeps its natural
    ClaimNb composition untouched; current is resampled (real rows only, no
    replacement/duplication) to shift P(ClaimNb) towards
    ``settings.target_claim_rate``, holding P(X | has_claim) exactly fixed
    since every retained row is reused unperturbed -- only the relative
    count of claim vs. no-claim rows changes. See
    datafiles/french-motor-paper.pdf Table 1 for the natural ~5.02% claim
    rate this shifts away from (default target: 15%).

    measure_columns/file_name mirror DataMotor; unlike DataMotor, ClaimNb is
    kept (it's the label here, not a leakage risk) and Region is dropped
    (no group-based split is used for this construction).
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
    target = "ClaimNb"

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        if not 0 < self.target_claim_rate < 1:
            raise ValueError("target_claim_rate must be in (0, 1)")

    def preprocess_time(self) -> pd.Series:
        """No genuine time axis; a random split stands in for the split key."""
        return pd.Series(pd.NaT, index=self.df.index)

    def filter_data(self) -> None:
        """Drop rows with missing values in the tested columns (no date range to apply)."""
        cols = self.measure_columns + [self.target]
        self.df = self.df[~self.df[cols].isna().any(axis=1)]

    def split_data(self) -> Data:
        """Random drift-free base split, then resample current's ClaimNb rate."""
        reference, candidate_pool = _random_half_split(self.df, self.seed)
        testing = _resample_to_rate(
            candidate_pool, candidate_pool[self.target] > 0, self.target_claim_rate, self.seed
        )
        columns = self.measure_columns + [self.target, "time"]
        reference = reference[columns].copy()
        testing = testing[columns].copy()
        # Store as explicit category labels, not raw counts -- this is what
        # was actually resampled on (ClaimNb > 0), and matters for
        # cross-tool fairness: Evidently classifies columns by pandas
        # dtype, but Frouros/Alibi-Detect classify by trying to cast the
        # actual VALUES to float, ignoring dtype metadata -- so a dtype
        # relabeling alone wouldn't make them agree. Only genuinely
        # non-numeric values make every tool's own classification
        # mechanism independently and consistently treat this column as
        # categorical, without needing a per-tool override anywhere.
        for frame in (reference, testing):
            frame[self.target] = np.where(frame[self.target] > 0, "claim", "no_claim")
        return Data(
            features=self.measure_columns,
            reference=reference,
            testing=testing,
            drift_type="prior",
            target=self.target,
            categorical_columns=self.categorical_columns,
        )


class DataElec2(Dataset):
    """Concept-drift construction on the Elec2 dataset (NSW electricity market,
    datafiles/elecNormNew.arff -- the classic Harries/"elecNormNew" release
    used throughout the concept-drift literature).

    Like DataMotor's Region split and DataMotorPrior's random half-split,
    there is no real time axis to boundary-split on: ``date``/``period`` are
    already min-max normalized to [0, 1] rather than real timestamps, so
    Options.boundary (a calendar date) cannot apply here either. Unlike
    those two, the file's row order *is* chronological, so this scenario
    self-configures (see d3bench.scenario._SELF_CONFIGURING_DATASETS) on a
    fixed first-70%/last-30% split by row order -- the standard train/test
    split used for this dataset in the concept-drift benchmark literature.
    The split stays chronological (not random) on purpose: Elec2's actual
    concept drift is a real regime change partway through the series (the
    NSW-Victoria interconnect altering the price/demand relationship), and a
    random split would spread that shift evenly across both halves, erasing
    the drift the sequence-sensitive online detectors are meant to catch.

    ``class`` (UP/DOWN, whether the NSW price moved up relative to a moving
    average) is the label, so it is excluded from measure_columns and
    exposed as ``target`` instead (mirroring DataMotorPrior) -- this lets
    "concept" drift_type be told apart from "covariate" via Data.target the
    same way "prior" already is (see Tool._monitored_columns). Note what
    this does and does not fix: it identifies the label, and
    Tool._monitored_columns now includes it for concept scenarios, but no
    detector yet consumes it as a label -- the online CD detectors
    (Frouros/River/Alibi-Detect) still reduce every monitored column
    (covariates and label alike) to a covariate-style stream (e.g. a
    feature-vector norm) rather than tracking a model's prediction error
    over time. Wiring up genuine P(y|X)-style supervised monitoring is a
    separate follow-up. ``date`` itself is dropped from measure_columns
    since it is only the chronological ordering key used to build the
    split, not a feature to test for drift.
    """

    file_name = "elecNormNew.arff"
    measure_columns = [
        "day",
        "period",
        "nswprice",
        "nswdemand",
        "vicprice",
        "vicdemand",
        "transfer",
    ]
    target = "class"
    #: Both are ARFF nominal attributes: `class` is UP/DOWN (genuinely
    #: non-numeric, so every adapter would classify it categorical anyway),
    #: but `day` is {1..7} decoded to digit-strings that cast cleanly to float
    #: -- declaring it here is what makes Frouros/Alibi drop it too, so they
    #: norm over the same 6 continuous columns River's preprocess keeps.
    categorical_columns = ["day", "class"]
    train_fraction = 0.7

    def preprocess_time(self) -> pd.Series:
        """No genuine timestamps; row order (already chronological) stands in."""
        return pd.Series(pd.NaT, index=self.df.index)

    def filter_data(self) -> None:
        """Drop rows with missing values in the tested columns (no date range to apply)."""
        cols = self.measure_columns + [self.target]
        nan_rows = self.df[cols].isna()
        self.df = self.df[~nan_rows.any(axis=1)]

    def split_data(self) -> Data:
        """First train_fraction of rows (chronological order) as reference, rest as testing."""
        split_idx = int(len(self.df) * self.train_fraction)
        columns = self.measure_columns + [self.target, "time"]
        return Data(
            features=self.measure_columns,
            reference=self.df.iloc[:split_idx][columns].copy(),
            testing=self.df.iloc[split_idx:][columns].copy(),
            drift_type="concept",
            target=self.target,
            categorical_columns=self.categorical_columns,
        )


class DataElec2Injected(DataElec2):
    """Semi-synthetic *positive control* for concept drift, built on Elec2.

    The unmodified Elec2 stream (DataElec2, scenarios/elec2_concept.toml)
    turns out to drift covariately rather than conceptually: its label is
    "did the NSW price move up against its own 24h moving average", a
    self-adjusting rule that stays valid as price *levels* drift, so P(y|X)
    holds while P(X) moves. That makes it a poor test of whether the
    concept-drift detectors work at all -- a null result there is
    indistinguishable from a broken pipeline.

    This subclass supplies the missing control: it injects a genuine,
    located P(y|X) change into the same stream and changes nothing else.
    At row ``k = len(slice) * inject_fraction`` the labelling rule flips for
    one region of feature space (rows with ``flip_feature`` above its
    median); everywhere else, and everywhere before ``k``, the original
    label stands. Two properties make the injected drift *pure* concept
    drift:

    * **No feature is ever written.** Only the label column is reassigned,
      so P(X) is bit-identical to DataElec2's on both sides of ``k`` and no
      covariate signal is added.
    * **The flip is class-balanced.** Equal numbers of UP->DOWN and
      DOWN->UP are flipped (seeded by ``inject_seed``), so the label counts
      after ``k`` -- and hence P(y) -- are unchanged. Flipping one
      direction only would inject prior drift alongside the concept drift
      and confound the two.

    The known onset is reported as ``Data.drift_point`` so a report can
    measure detection against ground truth instead of against a guessed
    split boundary. This is a labeled synthetic control, not a claim about
    real Elec2: it answers "do the detectors fire when P(y|X) genuinely
    changes?", which is what makes the unmodified scenario's covariate-only
    result trustworthy.

    Two placement constraints, both learned the hard way; neither is
    cosmetic, and "simplifying" either one back silently produces a control
    that validates nothing.

    **The injection runs on the post-market slice, not the whole frame.**
    Elec2's full series spans the NSW-Victoria market opening (``nem_row``,
    where the Victoria columns activate) and carries large covariate swings
    across it -- the very swings the unmodified scenario's covariate finding
    documents. Injected into the whole frame, the flip's error step
    (~6 percentage points) is buried under Elec2's own ~20pp error
    fluctuation, so every detector's *first* firing latches onto the
    pre-existing nonstationarity thousands of rows before the injected onset
    and the control is masked. Restricting to the homogeneous regime after
    the market opening (plus ``post_buffer`` rows to clear the activation
    transient) gives the detectors a stationary pre-flip baseline, so the
    injected change point is the first real change they see. The fix is to
    clean the baseline, not to enlarge the flip: tuning the injection
    upwards until detectors fire would only test whether they catch a shift
    made big enough to be unmissable, rather than a realistic one.

    **``inject_fraction`` must exceed ``train_fraction``** (both are
    fractions *of the slice*) -- see the guard in ``inject_concept_drift``.
    The concept path (d3bench.supervised) trains on the reference split and
    streams the *testing* errors to the detectors, so an onset placed before
    the split is off-screen: the detectors would watch a uniformly
    post-flip, stationary error signal with no change point in it, and any
    firing would be ordinary fluctuation. An obvious-looking 0.5 does
    exactly that, landing thousands of rows inside the reference window, and
    would additionally train the model on a label-contradictory mixture of
    both rules. That is the same failure mode that makes real Elec2's NEM
    event undetectable in this benchmark.
    """

    #: Absolute row of the NSW-Victoria market opening (verified: the
    #: Victoria columns activate here), and the rows skipped after it to
    #: clear the activation transient. Everything from nem_row + post_buffer
    #: on is the homogeneous regime this control is built in -- see the
    #: class docstring on why the whole frame does not work.
    nem_row = 17424
    post_buffer = 500
    #: Fraction *of the post-market slice* at which the labelling rule
    #: changes. 0.85 puts the onset ~halfway into the testing split, leaving
    #: the reference split entirely pre-flip. Must stay above
    #: train_fraction -- see the class docstring.
    inject_fraction = 0.85
    #: Feature whose upper half (above its median over the slice) is the
    #: region of feature space the rule flips in. Read, never written.
    flip_feature = "nswprice"
    #: Seed for choosing which rows flip, so the injected stream is fixed.
    inject_seed = 31

    def post_market_frame(self) -> pd.DataFrame:
        """Rows from the market opening (plus transient buffer) onward.

        Positional, matching how the analysis notebooks slice it: Elec2 has
        no missing values in the tested columns, so filter_data drops
        nothing and position still equals absolute row.
        """
        return self.df.iloc[self.nem_row + self.post_buffer :]

    def inject_concept_drift(self) -> tuple[pd.DataFrame, int]:
        """Return (the post-market slice with flipped labels, the injection row k).

        ``k`` is an offset into the returned slice, which is also the frame
        the reference/testing split is then cut from -- so it stays
        comparable to the detectors' stream offsets.
        """
        if self.inject_fraction <= self.train_fraction:
            raise ValueError(
                f"inject_fraction={self.inject_fraction} must be greater than "
                f"train_fraction={self.train_fraction}: an onset at or before the "
                "split falls in the reference window, which the concept-drift "
                "detectors never see (they stream testing errors only, see "
                "d3bench.supervised) -- the injected drift would be undetectable "
                "by construction. See DataElec2Injected's docstring."
            )

        df = self.post_market_frame().copy()
        drift_point = int(len(df) * self.inject_fraction)

        # Positional throughout (numpy, not .loc) so this holds regardless of
        # whether filter_data left a gappy index.
        values = df[self.flip_feature].to_numpy(dtype=float)
        region = (np.arange(len(df)) >= drift_point) & (values > np.median(values))
        region_rows = np.flatnonzero(region)

        labels = df[self.target].to_numpy(copy=True)
        ups = region_rows[labels[region_rows] == "UP"]
        downs = region_rows[labels[region_rows] == "DOWN"]
        # Class-balanced: flipping n each way leaves the post-k label counts
        # (and so P(y)) exactly as they were -- only P(y|X) moves.
        n_flip = min(len(ups), len(downs))
        rng = np.random.default_rng(self.inject_seed)
        labels[rng.choice(ups, size=n_flip, replace=False)] = "DOWN"
        labels[rng.choice(downs, size=n_flip, replace=False)] = "UP"

        # ASSERTION OF INTENT: the label column is the only one assigned to
        # here, so every feature column of `df` is still the one DataElec2
        # would have produced. Do not add feature edits to this method --
        # they would turn this control into a covariate+concept mixture.
        df[self.target] = labels
        return df, drift_point

    def split_data(self) -> Data:
        """Inject the label-rule flip, then split it exactly as DataElec2 does.

        The injected post-market slice is swapped in only for the duration of
        the super() call (rather than assigned onto self) so that the split
        stays a pure function of the file: injecting into an already-injected
        frame would flip a second, different set of rows. train_fraction then
        applies to the slice, so reference is entirely pre-flip and the
        testing stream carries the onset in its middle.
        """
        injected, drift_point = self.inject_concept_drift()
        original, self.df = self.df, injected
        try:
            data = super().split_data()
        finally:
            self.df = original
        data.drift_point = drift_point
        return data
