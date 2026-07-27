"""Shared supervised model for the concept-drift execution path.

A concept-drift scenario (``drift_type == "concept"``, e.g.
``scenarios/elec2_concept.toml``) asks whether P(y|X) changed -- which only a
*model's prediction error over time* can answer, not a statistic on the raw
feature stream. This module trains ONE shared classifier on the reference
split and exposes its 0/1 misclassification stream for both splits, so every
framework's concept-drift detector (Frouros DDM/EDDM/HDDM/..., River
ADWIN/Page-Hinkley/...) is compared on the *identical* error signal rather
than each adapter reducing the features to its own covariate-style stream.
This is the "genuine P(y|X)-style supervised monitoring" the DataElec2
docstring flags as a follow-up (see d3bench.datasets).

The classifier is deliberately fixed and simple -- a StandardScaler (numeric)
+ OneHotEncoder (categorical) + LogisticRegression pipeline, mirroring the
canonical Frouros Elec2/DDM example -- because its job is only to *produce a
comparable error stream*, not to be a competitive model; a shared, boring
spec keeps the benchmark measuring detectors, not models.

The detectors consume ``testing`` only: each self-calibrates its baseline
error rate on the first ``min_num_instances``/``warm_start`` of the test
stream and then detects, exactly as the canonical Frouros Elec2/DDM example
streams the test set alone. They are deliberately NOT warmed up on the
reference error stream -- fed tens of thousands of reference instances, a
Bernoulli error stream's ordinary fluctuation trips DDM/CUSUM/... and latches
the verdict before any test data, a warm-up artifact (see the fit() notes in
tools/frouros.py and tools/river.py).

``reference`` is still computed and exposed as a diagnostic baseline: the
model's out-of-fold (``cross_val_predict``) error rate on the reference split,
in the same generalization regime as ``testing`` (in-sample training error
would be optimistically low). Nothing consumes it for detection today; it is
retained so a reference-vs-test error-rate comparison is available.
"""

import dataclasses as dc
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from d3bench.utils import Data

# Shared with the tool adapters' own RNGs so the whole run is reproducible.
_SEED = 31
# Folds for the out-of-fold reference error stream. cross_val_predict uses a
# non-shuffled StratifiedKFold for classifiers by default, so this is
# deterministic without a seed.
_CV_FOLDS = 5


@dc.dataclass(frozen=True)
class ErrorStream:
    """A shared classifier's 0/1 misclassification stream for both splits.

    ``testing`` (model refit on the full reference split, scored on testing) is
    the signal every error-stream concept-drift detector consumes -- they
    self-calibrate on its leading window, then detect. ``reference`` is the
    out-of-fold baseline error rate (diagnostic only; not fed to detectors, see
    module docstring). Both are int arrays (1 = misclassified).
    """

    reference: np.ndarray
    testing: np.ndarray


# Cached per Data instance (keyed by id): every tool in a scenario shares the
# same Data object (see scenario.load_tools), and every concept detector then
# shares the same fitted model / error stream -- computed once instead of
# once per (tool, method, repetition). The dict lives only for the process
# running the benchmark, and Results keeps the Data alive, so id() is stable.
_CACHE: dict[int, ErrorStream] = {}


def _build_pipeline(data: Data) -> Pipeline:
    """StandardScaler(numeric) + OneHotEncoder(categorical) + LogisticRegression.

    Columns are split with the same declared-categorical set the adapters use
    (Data.categorical_columns), so e.g. Elec2's digit-string ``day`` is
    one-hot encoded rather than scaled as if it were continuous.
    """
    categorical = [f for f in data.features if f in data.categorical_columns]
    numeric = [f for f in data.features if f not in data.categorical_columns]
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", transformer),
            ("model", LogisticRegression(max_iter=1000, random_state=_SEED)),
        ]
    )


def _errors(y_true: Any, y_pred: Any) -> np.ndarray:
    """1 where the model misclassified, 0 where it was right."""
    return (np.asarray(y_true) != np.asarray(y_pred)).astype(int)


def error_stream(data: Data) -> ErrorStream:
    """Train the shared classifier on ``data``'s reference split and return its
    0/1 error stream for both splits. Cached per Data instance (see _CACHE)."""
    assert data.target is not None, "concept-drift error stream needs a label column"
    cached = _CACHE.get(id(data))
    if cached is not None:
        return cached

    x_reference = data.reference[data.features]
    y_reference = data.reference[data.target]
    x_testing = data.testing[data.features]
    y_testing = data.testing[data.target]

    pipeline = _build_pipeline(data)
    # Out-of-fold predictions for the reference baseline (see module docstring).
    reference_pred = cross_val_predict(pipeline, x_reference, y_reference, cv=_CV_FOLDS)
    # Refit on the full reference split to score the testing stream.
    pipeline.fit(x_reference, y_reference)
    testing_pred = pipeline.predict(x_testing)

    stream = ErrorStream(
        reference=_errors(y_reference, reference_pred),
        testing=_errors(y_testing, testing_pred),
    )
    _CACHE[id(data)] = stream
    return stream
