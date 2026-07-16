"""
Drift detection method definitions and categorization for the package.

This module provides a comprehensive taxonomy of drift detection methods,
organized by their characteristics, requirements, and use cases.
"""

from enum import StrEnum


# Online Supervised Concept Drift Detection
class OnlineCD(StrEnum):
    """
    Definition: These methods monitor prediction errors in a supervised
        learning model to detect when the relationship between input and
        output changes.
    Comparison Factors: Drift detection speed, false positive rate, memory
        efficiency.
    Use Cases: Model monitoring, fraud detection, real-time anomaly
        detection.
    """

    # Change Detection
    BAYESIAN_ONLINE_CHANGE_DETECTION = "Bayesian Online Change Detection"
    CUMULATIVE_SUM_CONTROL_CHART = "Cumulative Sum Control Chart"
    GEOMETRIC_MOVING_AVERAGE = "Geometric Moving Average"
    PRINCIPAL_COMPONENT_ANALYSIS_CONCEPT_DRIFT = "Principal Component Analysis for Concept Drift"
    PAGE_HINKLEY_TEST = "Page-Hinkley Test"

    # Statistical Process Control
    DRIFT_DETECTION_METHOD = "Drift Detection Method"
    EXPONENTIAL_CUMULATIVE_DRIFT_DETECTION = "Exponential Cumulative Drift Detection"
    EWMA_CONCEPT_DRIFT_DETECTION_WARNING = "EWMA Concept Drift Detection Warning"
    EARLY_DRIFT_DETECTION_METHOD = "Early Drift Detection Method"
    HOEFFDING_DRIFT_DETECTION_METHOD_TEST_A = "Hoeffding's Drift Detection Method Test-A"
    HOEFFDING_DRIFT_DETECTION_METHOD_TEST_W = "Hoeffding's Drift Detection Method Test-W"
    REACTIVE_DRIFT_DETECTION_METHOD = "Reactive Drift Detection Method"

    # Window Based
    ADAPTIVE_WINDOWING = "Adaptive Windowing"
    STATISTICAL_TEST_EQUAL_PROPORTIONS_DETECTION = "Statistical Test of Equal Proportions Detection"
    ONLINE_KOLMOGOROV_SMIRNOV = "Online Kolmogorov-Smirnov"
    ONLINE_ACCURACY_UPDATED_ENSEMBLE = "Online Accuracy Updated Ensemble"
    ONLINE_MAXIMUM_MEAN_DISCREPANCY = "Online Maximum Mean Discrepancy"
    ONLINE_CRAMER_VON_MISES_TEST = "Online Cramér-von Mises Test"
    ONLINE_FISHER_EXACT_TEST = "Online Fisher Exact Test"
    ONLINE_LEAST_SQUARES_DENSITY_DIFFERENCE = "Online Least-Squares Density Difference"

    # Other
    CHANGE_DETECTION_DATA_STREAMS = "Change Detection in Data Streams"
    PERIODIC_TRIGGER = "Periodic Trigger"
    LINEAR_FOUR_RATES = "Linear Four Rates"
    MARGIN_DENSITY_DRIFT_DETECTION_METHOD = "Margin Density Drift Detection Method"



# Online Unsupervised Data Drift Detection
class OnlineDD(StrEnum):
    """
    Definition: These methods detect distribution changes in input features
        without needing labels.
    Comparison Factors: Sensitivity to small distribution changes, robustness
        against noise.
    Use Cases: Feature monitoring, drift in unlabelled data, preemptive
        retraining triggers.
    """

    # Distance Based
    DRIFT_DETECTION_MODEL_OUTPUT = "Drift Detection Model Output"
    ONLINE_MAXIMUM_MEAN_DISCREPANCY = "Online Maximum Mean Discrepancy"
    UNSUPERVISED_DRIFT_DETECTOR = "Unsupervised Drift Detector"

    # Statistical Test
    LOG_LIKELIHOOD_RATIO_TEST = "Log-Likelihood Ratio Test"
    INCREMENTAL_KOLMOGOROV_SMIRNOV_TEST = "Incremental Kolmogorov-Smirnov Test"


# Batch Concept Drift Detection
class BatchCD(StrEnum):
    """
    Definition: These methods analyze model predictions periodically (not in
        real-time) to detect changes in the feature-label relationship.
    Comparison Factors: Detection lag, computational efficiency, precision.
    Use Cases: Periodic model validation, offline analysis of model decay.
    """

    # TODO: add subclassification

    CHI_SQUARE_TEST = "Chi-square Test"
    KOLMOGOROV_SMIRNOV_TEST = "Kolmogorov-Smirnov Test"
    CRAMER_VON_MISES_TEST = "Cramér-von Mises Test"
    FISHER_EXACT_TEST = "Fisher Exact Test"
    MAXIMUM_MEAN_DISCREPANCY = "Maximum Mean Discrepancy"
    LEAST_SQUARES_DENSITY_DIFFERENCE = "Least-Squares Density Difference"
    MIXED_TYPE_TABULAR_DATA = "Mixed-Type Tabular Data"


# Batch Data Drift Detection
class BatchDD(StrEnum):
    """
    Definition: These methods detect changes in feature distributions in large
        batches of data.
    Comparison Factors: Performance on high-dimensional data, ability to
        detect local vs. global drift.
    Use Cases: Data validation, detecting changes in training data before
        model retraining.
    """

    # TODO: NannyML classifies Data Drift into Univariate Continuous and Categorical
    # TODO: Maybe a new subclass for Univariate Continuous and Categorical?
    # See nannyml.py, repeated method names with different implementations
    # TODO: nannyml.JensenShannonDivergenceCategorical and HellingerDistanceCategorical
    # are implemented but not wired into NannyML.batch_dd_methods (tools/__init__.py):
    # they'd collide with the existing continuous JENSEN_SHANNON_DIVERGENCE_DRIFT_DETECTION
    # / HELLINGER_DISTANCE entries in that dict. Needs the continuous/categorical
    # taxonomy question above resolved first (e.g. separate _CATEGORICAL enum members,
    # or a single dispatching class keyed on column dtype) before registering them.
    # Also, the benchmark currently only exercises continuous data, so there's no
    # categorical column to route to them yet regardless.

    # Distance Based
    BHATTACHARYYA_DISTANCE = "Bhattacharyya Distance"
    EARTH_MOVER_DISTANCE = "Earth Mover's Distance"
    ENERGY_DISTANCE = "Energy Distance"
    WASSERSTEIN_DISTANCE = "Wasserstein Distance"
    HELLINGER_DISTANCE = "Hellinger Distance"
    HISTOGRAM_INTERSECTION_NORMALIZED_COMPLEMENT = "Histogram Intersection Normalized Complement"
    JENSEN_SHANNON_DIVERGENCE_DRIFT_DETECTION = "Jensen-Shannon Divergence Drift Detection"
    KULLBACK_LEIBLER_DIVERGENCE_DRIFT_DETECTION = "Kullback-Leibler Divergence Drift Detection"
    LEAST_SQUARES_DENSITY_DIFFERENCE = "Least-Squares Density Difference"
    BATCH_MAXIMUM_MEAN_DISCREPANCY = "Batch Maximum Mean Discrepancy"
    POPULATION_STABILITY_INDEX = "Population Stability Index"
    L_INFINITY_DISTANCE = "L-Infinity Distance"
    TOTAL_VARIATION_DISTANCE = "Total Variation Distance"

    # Statistical Test
    ANDERSON_DARLING_TEST = "Anderson-Darling Test"
    BAUMGARTNER_WEISS_SCHINDLER_TEST = "Baumgartner Weiss Schindler Test"
    CHI_SQUARE_TEST = "Chi-square Test"
    CRAMER_VON_MISES_TEST = "Cramér-von Mises Test"
    KOLMOGOROV_SMIRNOV_TEST = "Kolmogorov-Smirnov Test"
    KUIPER_TEST = "Kuiper's Test"
    MANN_WHITNEY_U_TEST = "Mann-Whitney U-Test"
    MC_DIARMID_DRIFT_DETECTION_METHOD_TEST_A = "Mc Diarmid Drift Detection Method Test-A"
    MC_DIARMID_DRIFT_DETECTION_METHOD_TEST_E = "Mc Diarmid Drift Detection Method Test-E"
    MC_DIARMID_DRIFT_DETECTION_METHOD_TEST_G = "Mc Diarmid Drift Detection Method Test-G"
    WELCH_T_TEST = "Welch's T-Test"

    # TODO: Unclassified
    FISHER_EXACT_TEST = "Fisher Exact Test"
    EPPS_SINGLETON_TEST = "EPPS-Singleton Test"
    EMPIRICAL_MAXIMUM_MEAN_DISCREPANCY = "Empirical Maximum Mean Discrepancy"
    G_TEST = "G-Test"
    T_TEST = "T-Test"
    Z_TEST = "Z-Test"
    
    # Mixed / composite (routes per-column by dtype)
    MIXED_TYPE_TABULAR_DATA = "Mixed-Type Tabular Data"
