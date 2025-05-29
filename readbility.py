"""Module provides utilities for calculating and analyzing readability metrics.

It includes:
- Pydantic models for raw and normalized readability metrics.
- Functions to compute, normalize, and compare readability metrics.
- Distance calculation methods for comparing readability features.
"""

# Use annotations for cleaner type hinting (requires Python 3.7+)
from __future__ import annotations

import logging
import math
import sys
from typing import Optional, Union

import numpy as np
import textstat
from numpy.linalg import norm as np_norm
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import StandardScaler

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Pydantic Models ---


class BaseReadabilityMetrics(BaseModel):
    """Base model for readability metrics.

    Defines common fields and validators. This can be inherited by Raw and
    Normalized metric models.
    """

    flesch_reading_ease: float = Field(..., description="Flesch Reading Ease score. Higher is easier.")
    flesch_kincaid_grade: float = Field(..., description="Flesch-Kincaid Grade Level.")
    smog_index: float = Field(..., description="SMOG Index. Estimates years of education needed.")
    gunning_fog: float = Field(..., description="Gunning Fog Index.")
    dale_chall: float = Field(..., description="Dale-Chall Readability Score.")
    automated_readability_index: float = Field(..., description="Automated Readability Index (ARI).")
    coleman_liau_index: float = Field(..., description="Coleman-Liau Index.")
    linsear_write_formula: float = Field(..., description="Linsear Write Formula.")
    difficult_words: int = Field(..., ge=0, description="Count of words considered difficult.")
    sentence_count: int = Field(..., ge=0, description="Total number of sentences.")
    avg_sentence_length: float = Field(..., ge=0.0, description="Average words per sentence.")
    syllable_count: int = Field(..., ge=0, description="Total number of syllables.")
    lexicon_count: int = Field(..., ge=0, description="Total words (punctuation removed).")

    @field_validator(
        "difficult_words",
        "sentence_count",
        "syllable_count",
        "lexicon_count",
        mode="before",
    )
    @classmethod
    def ensure_integer_counts(cls, value: float) -> int:
        """Ensure count metrics are integers. Textstat might return floats."""
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                msg = f"Count metric cannot be NaN or infinity, got {value}"
                raise ValueError(msg)
            return round(value)  # Pydantic will convert to int if field type is int
        if isinstance(value, int):
            return value
        msg = f"Count metric must be a number (int or float), got {type(value)}"
        raise TypeError(msg)

    @field_validator(
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "smog_index",
        "gunning_fog",
        "dale_chall",
        "automated_readability_index",
        "coleman_liau_index",
        "linsear_write_formula",
        "avg_sentence_length",
        mode="before",
    )
    @classmethod
    def ensure_float_scores(cls, value: float) -> float:
        """Ensure score metrics are floats."""
        if not isinstance(value, (int, float)):
            msg = f"Score metric must be a number, got {type(value)}"
            raise TypeError(msg)
        if math.isnan(value) or math.isinf(value):  # Check for NaN/inf before float conversion
            msg = f"Score metric cannot be NaN or infinity, got {value}"
            raise ValueError(msg)
        return float(value)


class ReadabilityMetricsRaw(BaseReadabilityMetrics):
    """Data model for storing raw readability metrics.

    Calculated for a given text. Inherits fields and validators from
    BaseReadabilityMetrics.
    """


class ReadabilityMetricsNormalized(BaseReadabilityMetrics):
    """Data model for storing normalized readability metrics.

    E.g., standardized. The values will be scaled, but the structure
    matches the raw metrics. Inherits fields and validators from
    BaseReadabilityMetrics.
    Note: ge=0 validation for counts might not apply strictly after
    normalization if normalization can result in negative values
    (e.g. standard scaler). We can override or remove those specific
    field constraints for normalized data if needed. For simplicity
    here,
    """

    difficult_words: float = Field(..., description="Normalized count of difficult words.")
    sentence_count: float = Field(..., description="Normalized count of sentences.")
    avg_sentence_length: float = Field(..., description="Normalized average words per sentence.")  # Already float
    syllable_count: float = Field(..., description="Normalized total number of syllables.")
    lexicon_count: float = Field(..., description="Normalized total words (punctuation removed).")


class MetricDifferencesRaw(BaseModel):
    """Data model for storing absolute differences.

    Between two sets of raw readability metrics. All difference values
    are non-negative floats.
    """

    flesch_reading_ease: float = Field(..., ge=0.0)
    flesch_kincaid_grade: float = Field(..., ge=0.0)
    smog_index: float = Field(..., ge=0.0)
    gunning_fog: float = Field(..., ge=0.0)
    dale_chall: float = Field(..., ge=0.0)
    automated_readability_index: float = Field(..., ge=0.0)
    coleman_liau_index: float = Field(..., ge=0.0)
    linsear_write_formula: float = Field(..., ge=0.0)
    difficult_words: float = Field(..., ge=0.0)
    sentence_count: float = Field(..., ge=0.0)
    avg_sentence_length: float = Field(..., ge=0.0)
    syllable_count: float = Field(..., ge=0.0)
    lexicon_count: float = Field(..., ge=0.0)


class ReadabilityAnalysisResult(BaseModel):
    """Holds the comprehensive results of a readability analysis between two texts."""

    student_text_raw_metrics: ReadabilityMetricsRaw
    model_text_raw_metrics: ReadabilityMetricsRaw
    raw_metric_differences: MetricDifferencesRaw
    student_text_normalized_metrics: Optional[ReadabilityMetricsNormalized] = None
    model_text_normalized_metrics: Optional[ReadabilityMetricsNormalized] = None
    euclidean_distance_normalized: Optional[float] = None
    manhattan_distance_normalized: Optional[float] = None
    # Optional: could include original texts if desired for the result package
    # student_text_input: str
    # model_text_input: str


# --- Functions ---


def get_readability_metrics(text: str) -> ReadabilityMetricsRaw:
    """Compute a set of raw readability metrics for a given text.

    Uses the `textstat` library.

    Args:
        text: The input string. An empty string or string with insufficient
              content might lead to errors or default/zero values from
              `textstat` functions.

    Returns:
        A ReadabilityMetricsRaw Pydantic model instance containing the
        calculated scores.

    Raises:
        TypeError: If the input `text` is not a string.
        Exception: Propagates exceptions from `textstat` if critical errors
                   occur (e.g., due to unsupported language if
                   `textstat.set_lang` was used incorrectly, or internal
                   errors for highly unusual inputs).

    """
    if not isinstance(text, str):
        logger.error("Input 'text' for readability metrics must be a string.")
        msg = "Input 'text' must be a string."
        raise TypeError(msg)

    if not text.strip():
        logger.warning("Input text is empty or whitespace only. Returning default zero metrics.")
        return ReadabilityMetricsRaw(
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            smog_index=0.0,
            gunning_fog=0.0,
            dale_chall=0.0,
            automated_readability_index=0.0,
            coleman_liau_index=0.0,
            linsear_write_formula=0.0,
            difficult_words=0,
            sentence_count=0,
            avg_sentence_length=0.0,
            syllable_count=0,
            lexicon_count=0,
        )
    try:
        # This comment is necessary for clarity. Do not remove the following line.
        raw_metrics_dict = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),  # type: ignore[attr-defined]
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),  # type: ignore[attr-defined]
            "smog_index": textstat.smog_index(text),  # type: ignore[attr-defined]
            "gunning_fog": textstat.gunning_fog(text),  # type: ignore[attr-defined]
            "dale_chall": textstat.dale_chall_readability_score(text),  # type: ignore[attr-defined]
            "automated_readability_index": textstat.automated_readability_index(text),  # type: ignore[attr-defined]
            "coleman_liau_index": textstat.coleman_liau_index(text),  # type: ignore[attr-defined]
            "linsear_write_formula": textstat.linsear_write_formula(text),  # type: ignore[attr-defined]
            "difficult_words": textstat.difficult_words(text),  # type: ignore[attr-defined]
            "sentence_count": textstat.sentence_count(text),  # type: ignore[attr-defined]
            "avg_sentence_length": textstat.avg_sentence_length(text),  # type: ignore[attr-defined]
            "syllable_count": textstat.syllable_count(text),  # type: ignore[attr-defined]
            "lexicon_count": textstat.lexicon_count(text, removepunct=True),  # type: ignore[attr-defined]
        }
        return ReadabilityMetricsRaw(**raw_metrics_dict)
    except Exception:  # Catch specific textstat errors if known, or generic Exception
        logger.exception(f"Error calculating readability metrics for text: '{text[:50]}...'")
        raise  # Re-raise the caught exception


def normalize_metrics(metrics_list: list[ReadabilityMetricsRaw]) -> list[ReadabilityMetricsNormalized]:
    """Standardize each readability feature across a list of texts."""
    if not metrics_list:
        logger.warning("normalize_metrics received an empty list. Returning empty list.")
        return []
    if not all(isinstance(m, ReadabilityMetricsRaw) for m in metrics_list):
        msg = "All items in metrics_list must be ReadabilityMetricsRaw instances."
        raise TypeError(msg)

    # Use the defined fields from the Pydantic model to ensure consistent order
    keys = list(ReadabilityMetricsRaw.model_fields.keys())
    try:
        mat = np.array([[getattr(m, key) for key in keys] for m in metrics_list], dtype=float)
    except Exception as e:
        logger.exception("Error converting metrics list to NumPy array")
        msg = "Could not convert metrics list to NumPy array. Check metric values."
        raise ValueError(msg) from e

    if mat.shape[0] == 0:  # Should have been caught by `if not metrics_list:`
        return []

    scaler = StandardScaler()
    if mat.shape[0] == 1:
        logger.warning("Normalizing metrics with only one sample. StandardScaler will produce all zeros.")
        mat_norm = np.zeros_like(mat, dtype=float)
    else:
        try:
            mat_norm = scaler.fit_transform(mat)
        except ValueError as e_scale:  # e.g., zero variance in a column
            log_msg = (
                f"Error during StandardScaler fit_transform: {e_scale}. "
                "This can happen if a feature has zero variance (all values are the same)."
            )
            logger.exception(log_msg)
            err_msg = f"StandardScaler failed, possibly due to zero variance in a feature: {e_scale}"
            raise ValueError(err_msg) from e_scale

    normalized_metrics_obj_list: list[ReadabilityMetricsNormalized] = []
    for row in mat_norm:
        norm_dict = dict(zip(keys, row, strict=True))
        try:
            normalized_metrics_obj_list.append(ReadabilityMetricsNormalized(**norm_dict))
        except Exception as e_pydantic:
            log_msg = (
                f"Error creating ReadabilityMetricsNormalized model from "
                f"normalized data: {norm_dict}. Error: {e_pydantic}"
            )
            logger.exception(log_msg)
            msg = "Failed to reconstruct Pydantic model for normalized metrics."
            raise ValueError(msg) from e_pydantic
    return normalized_metrics_obj_list


def calculate_euclidean_distance(
    metrics1: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
    metrics2: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
) -> float:
    """Compute the Euclidean (L2) distance between two feature vectors.

    Feature vectors are represented by ReadabilityMetrics Pydantic models.

    Args:
        metrics1: Readability metrics for the first text.
        metrics2: Readability metrics for the second text.

    Returns:
        The Euclidean distance as a float.

    """
    if not isinstance(metrics1, BaseReadabilityMetrics) or not isinstance(metrics2, BaseReadabilityMetrics):
        msg = "Inputs must be instances of a BaseReadabilityMetrics model."
        raise TypeError(msg)
    keys = list(BaseReadabilityMetrics.model_fields.keys())  # Ensure consistent order
    arr1 = np.array([getattr(metrics1, key) for key in keys], dtype=float)
    arr2 = np.array([getattr(metrics2, key) for key in keys], dtype=float)
    distance = np_norm(arr1 - arr2)
    return float(distance)


def calculate_manhattan_distance(
    metrics1: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
    metrics2: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
) -> float:
    """Compute the Manhattan (L1) distance between two feature vectors.

    Feature vectors are represented by ReadabilityMetrics Pydantic models.

    Args:
        metrics1: Readability metrics for the first text.
        metrics2: Readability metrics for the second text.

    Returns:
        The Manhattan distance as a float.

    """
    if not isinstance(metrics1, BaseReadabilityMetrics) or not isinstance(metrics2, BaseReadabilityMetrics):
        msg = "Inputs must be instances of a BaseReadabilityMetrics model."
        raise TypeError(msg)
    keys = list(BaseReadabilityMetrics.model_fields.keys())
    arr1 = np.array([getattr(metrics1, key) for key in keys], dtype=float)
    arr2 = np.array([getattr(metrics2, key) for key in keys], dtype=float)
    distance = np.abs(arr1 - arr2).sum()
    return float(distance)


def compare_raw_metrics_absolute_diff(
    metrics1: ReadabilityMetricsRaw,
    metrics2: ReadabilityMetricsRaw,
) -> MetricDifferencesRaw:
    """Compute absolute differences between raw scores.

    Scores are from two ReadabilityMetricsRaw objects. This is useful
    for direct interpretability of score differences.

    Args:
        metrics1: Raw readability metrics for the first text.
        metrics2: Raw readability metrics for the second text.

    Returns:
        A MetricDifferencesRaw Pydantic model instance containing the
        absolute differences.

    """
    if not isinstance(metrics1, ReadabilityMetricsRaw) or not isinstance(metrics2, ReadabilityMetricsRaw):
        msg = "Inputs must be ReadabilityMetricsRaw instances for raw comparison."
        raise TypeError(msg)
    m1_dict = metrics1.model_dump()
    m2_dict = metrics2.model_dump()
    differences_dict = {key: abs(m1_dict[key] - m2_dict[key]) for key in m1_dict}
    return MetricDifferencesRaw(**differences_dict)


# --- Main Analysis Function ---


def perform_readability_analysis(
    student_text: str,
    model_text: str,
    additional_texts_for_corpus: Optional[list[str]] = None,
) -> ReadabilityAnalysisResult:
    """Perform a readability analysis comparing a student text to a model text.

    Args:
        student_text: The text written by the student.
        model_text: The reference or model text.
        additional_texts_for_corpus: An optional list of other texts to include
                                     in the corpus for normalization. Normalization
                                     is more meaningful with a larger corpus.

    Returns:
        A ReadabilityAnalysisResult Pydantic model containing all raw metrics,
        differences, normalized metrics (if successful), and distances.

    """
    logger.info("Performing readability analysis...")

    # --- Compute Raw Metrics ---
    try:
        logger.debug("Computing raw metrics for student text...")
        student_metrics_raw = get_readability_metrics(student_text)
        logger.debug("Computing raw metrics for model text...")
        model_metrics_raw = get_readability_metrics(model_text)
    except Exception:  # Catch errors from get_readability_metrics
        logger.exception("Fatal error computing raw metrics:")
        # Depending on requirements, could return a partial result or re-raise
        raise  # Re-raise to indicate analysis could not complete

    # --- Show Raw Differences (for interpretability) ---
    logger.debug("Comparing raw metrics (student vs. model)...")
    raw_differences = compare_raw_metrics_absolute_diff(student_metrics_raw, model_metrics_raw)

    # --- Normalize Metrics Across a "Corpus" ---
    corpus_texts = [student_text, model_text]
    if additional_texts_for_corpus:
        corpus_texts.extend(additional_texts_for_corpus)

    corpus_raw_metrics: list[ReadabilityMetricsRaw] = []
    # Define default zero metrics once for reuse in case of errors
    default_zero_metrics = ReadabilityMetricsRaw(
        flesch_reading_ease=0.0, flesch_kincaid_grade=0.0, smog_index=0.0,
        gunning_fog=0.0, dale_chall=0.0, automated_readability_index=0.0,
        coleman_liau_index=0.0, linsear_write_formula=0.0, difficult_words=0,
        sentence_count=0, avg_sentence_length=0.0, syllable_count=0, lexicon_count=0
    )

    for i, text_item in enumerate(corpus_texts):
        try:
            # Pre-check for common invalid inputs not robustly handled by textstat or to avoid exceptions
            if not isinstance(text_item, str) or not text_item.strip():
                logger.warning(
                    f"Corpus text at index {i} is invalid (None, empty, or whitespace). "
                    f"Using default zero metrics. Text preview: '{str(text_item)[:50]}...'"
                )
                # get_readability_metrics already handles empty/whitespace by returning defaults,
                # but this explicit check handles None or other non-string types more gracefully
                # before they hit get_readability_metrics, which would raise TypeError for non-string.
                raw_metrics = default_zero_metrics
            else:
                raw_metrics = get_readability_metrics(text_item)
            corpus_raw_metrics.append(raw_metrics)
        except Exception as e:  # noqa: PERF203 # Individual text items can fail processing; loop must continue with placeholder.
            logger.exception(
                f"Could not get raw metrics for corpus text at index {i}: '{str(text_item)[:50]}...'. "
                f"Appending default zero metrics. Error: {e}",
            )
            corpus_raw_metrics.append(default_zero_metrics) # Ensure list correspondence

    norm_student_metrics: Optional[ReadabilityMetricsNormalized] = None
    norm_model_metrics: Optional[ReadabilityMetricsNormalized] = None
    euclidean_dist_norm: Optional[float] = None
    manhattan_dist_norm: Optional[float] = None
    meaningful_sample_size = 2
    # Need at least 2 samples for meaningful normalization by StandardScaler
    if len(corpus_raw_metrics) >= meaningful_sample_size:
        logger.info(f"Normalizing metrics across a corpus of {len(corpus_raw_metrics)} texts...")
        try:
            corpus_normalized_metrics: list[ReadabilityMetricsNormalized] = normalize_metrics(corpus_raw_metrics)
            # Check if normalization returned expected number of items
            if len(corpus_normalized_metrics) == len(corpus_raw_metrics):
                norm_student_metrics = corpus_normalized_metrics[0]
                norm_model_metrics = corpus_normalized_metrics[1]

                # --- Compute Distances on Normalized Features ---
                logger.debug("Computing distances on normalized features (student vs. model)...")
                euclidean_dist_norm = calculate_euclidean_distance(norm_student_metrics, norm_model_metrics)
                manhattan_dist_norm = calculate_manhattan_distance(norm_student_metrics, norm_model_metrics)
            else:
                logger.error(
                    "Normalization returned a different number of items than input.Skipping normalized metrics.",
                )
        except ValueError as e_norm_value:  # Catch errors from normalize_metrics (e.g., zero variance)
            logger.warning(
                f"Failed to normalize metrics due to ValueError: {e_norm_value}. Normalized distances will be None.",
                exc_info=True,
            )
        except Exception:  # Catch any other unexpected errors during normalization
            logger.exception(
                "Unexpected error during metric normalization. Normalized distances will be None.",
            )
    else:
        logger.warning(
            f"Corpus size is {len(corpus_raw_metrics)}, which is less than 2."
            " Skipping normalization and normalized distances.",
        )

    return ReadabilityAnalysisResult(
        student_text_raw_metrics=student_metrics_raw,
        model_text_raw_metrics=model_metrics_raw,
        raw_metric_differences=raw_differences,
        student_text_normalized_metrics=norm_student_metrics,
        model_text_normalized_metrics=norm_model_metrics,
        euclidean_distance_normalized=euclidean_dist_norm,
        manhattan_distance_normalized=manhattan_dist_norm,
    )


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Starting readability metrics processing example.")

    # Example texts: student vs. model essay
    student_text_main = (
        "Education is the passport to the future, for tomorrow belongs to "
        "those who prepare for it today. The journey of learning is lifelong."
    )
    model_text_main = (
        "The future belongs to those who prepare for it today; "
        "education is their passport. Learning is a continuous voyage."
    )
    # Add more texts to make normalization meaningful
    additional_corpus_texts = [
        "This is a very simple and short text. It has few words. Reading is easy.",
        "Conversely, academic papers often employ complex sentence structures and specialized vocabulary, "
        "resulting in higher readability scores indicating greater difficulty.",
        "Another example to provide variance for the scaler.",
    ]

    try:
        analysis_results = perform_readability_analysis(
            student_text_main,
            model_text_main,
            additional_texts_for_corpus=additional_corpus_texts,
        )
    except Exception as e_analysis:
        logger.critical(f"Readability analysis failed critically: {e_analysis}", exc_info=True)
        sys.exit(1)

    # --- Output Results from the analysis_results object ---
    print("\n=== Raw Readability Metrics & Differences (Student vs. Model) ===")
    print(f"{'Metric':<30} | {'Student (Raw)':>15} | {'Model (Raw)':>12} | {'Abs Diff':>10}")
    print("-" * 73)
    for key in ReadabilityMetricsRaw.model_fields:  # Iterate based on Pydantic model fields
        s_val = getattr(analysis_results.student_text_raw_metrics, key)
        m_val = getattr(analysis_results.model_text_raw_metrics, key)
        d_val = getattr(analysis_results.raw_metric_differences, key)

        fmt = "{:.0f}" if key in {"difficult_words", "sentence_count", "syllable_count", "lexicon_count"} else "{:.2f}"

        line_to_print = (
            f"{key.replace('_', ' ').title():<30} | "
            f"{fmt.format(s_val):>15} | "
            f"{fmt.format(m_val):>12} | "
            f"{fmt.format(d_val):>10}"
        )
        print(line_to_print)

    if analysis_results.student_text_normalized_metrics:
        print("\n=== Normalized Readability Metrics (Example: Student Text) ===")
        print(f"{'Metric':<30} | {'Normalized Value':>18}")
        print("-" * 53)
        for key in ReadabilityMetricsNormalized.model_fields:
            norm_val = getattr(analysis_results.student_text_normalized_metrics, key)
            print(f"{key.replace('_', ' ').title():<30} | {norm_val:>18.4f}")
    else:
        print("\n=== Normalized Readability Metrics could not be computed. ===")

    if (
        analysis_results.euclidean_distance_normalized is not None
        and analysis_results.manhattan_distance_normalized is not None
    ):
        print("\n=== Distances on Normalized Readability Features (Student vs. Model) ===")
        print(f"Euclidean Distance (Normalized): {analysis_results.euclidean_distance_normalized:.4f}")
        print(f"Manhattan Distance (Normalized): {analysis_results.manhattan_distance_normalized:.4f}")
    else:
        print("\n=== Distances on Normalized Readability Features could not be computed. ===")

    # Note: In an essay-scoring pipeline, these distances (raw or normalized)
    # can be valuable features themselves, combined with other text features,
    # or used to assess stylistic similarity/dissimilarity to model essays.
    # Normalization is crucial if these distances are fed into algorithms sensitive to feature scales.

    logger.info("Readability metrics processing example finished.")
