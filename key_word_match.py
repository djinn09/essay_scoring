"""Provides comprehensive text similarity calculation utilities.

This module offers a wide array of text similarity and distance metrics,
including:
- Basic string/sequence metrics (Levenshtein, Jaro-Winkler, LCS, Q-gram).
- Character n-gram similarities (Cosine, Jaccard).
- Advanced fuzzy matching via RapidFuzz and optionally FuzzyWuzzy.
- NLP-based metrics like BLEU score (adapted from machine translation) and BM25
  (from information retrieval).
- TF-IDF vector-based comparisons (Cosine, Euclidean, Manhattan, Jaccard, Hamming, Minkowski).

Key Components:
- `SimilarityCalculator`: The main orchestrator class that computes a configurable
  set of metrics for single or multiple text pairs. It can utilize parallel
  processing for efficiency with multiple pairs.
- `TFIDFCalculator`: A helper class for TF-IDF vectorization and related metric
  calculations, configurable via `TfidfConfig`.
- `BleuScorer`: A helper class for calculating BLEU scores with preprocessing.
- Preprocessing Utilities: Cached functions for tokenization, lemmatization,
  stemming, and stopword removal, leveraging NLTK. Includes mechanisms to
  ensure necessary NLTK data is available.
- Metric Instances: Globally initialized instances of various similarity algorithms
  from the `similarity` library and others for reuse.

The module is designed with configurability (via Pydantic models like
`SimilarityCalculatorConfig`) and efficiency (through caching and parallel
processing) in mind.
"""

# Use annotations for cleaner type hinting (requires Python 3.7+)
from __future__ import annotations

import difflib
import logging
import math
import os  # Added for os.cpu_count()
import string
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Union

# Similarity library components
from similarity.cosine import Cosine
from similarity.jaccard import Jaccard
from similarity.jarowinkler import JaroWinkler
from similarity.metric_lcs import MetricLCS
from similarity.normalized_levenshtein import NormalizedLevenshtein
from similarity.qgram import QGram

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
# --- Pydantic Import ---
from pydantic import BaseModel, Field, field_validator

from app_types import SimilarityCalculatorConfig, SimilarityMetrics, TfidfConfig

# Attempt NLTK imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError:
    msg = "NLTK library not found. Please install it: pip install nltk"
    raise ImportError(msg) from None

# Other core libraries
from rapidfuzz import fuzz as rapidfuzz_fuzz
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

# Optional libraries
try:
    from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz

    _fuzzywuzzy_available = True
except ImportError:
    _fuzzywuzzy_available = False
    warnings.warn("fuzzywuzzy library not found. UQRatio/UWRatio will be unavailable.", ImportWarning, stacklevel=2)

try:
    from rank_bm25 import BM25L as BM25

    _bm25_available = True
except ImportError:
    _bm25_available = False
    warnings.warn("rank_bm25 library not found. BM25 metric will be unavailable.", ImportWarning, stacklevel=2)


# --- Similarity Metric Instances (Globally Initialized for Reuse) ---
# These are generally stateless or thread-safe, allowing global instantiation.
NORMALIZED_LEVENSHTEIN = NormalizedLevenshtein()
JARO_WINKLER = JaroWinkler()
METRIC_LCS = MetricLCS()
QGRAM_2 = QGram(2)  # For 2-gram distance
QGRAM_3 = QGram(3)  # For 3-gram distance
QGRAM_4 = QGram(4)  # For 4-gram distance
SIM_COSINE_CHAR = Cosine(2)  # Cosine similarity on character 2-grams
SIM_JACCARD_CHAR = Jaccard(2)  # Jaccard similarity on character 2-grams


# --- NLTK Data Handling ---
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet.zip/wordnet/",
    "omw-1.4": "corpora/omw-1.4.zip/omw-1.4/",  # Often needed with wordnet
    "stopwords": "corpora/stopwords",
}
_NLTK_DATA_DOWNLOADED = dict.fromkeys(_NLTK_RESOURCES, False)


def _ensure_nltk_data(resource_name: str, download_dir: Optional[str] = None) -> bool:
    """Check if a given NLTK resource is available and downloads it if not.

    Uses a module-level dictionary `_NLTK_DATA_DOWNLOADED` to track downloaded
    resources within the current session to avoid redundant checks or download attempts.

    Args:
        resource_name: The name of the NLTK resource (e.g., "punkt", "wordnet").
        download_dir: Optional custom directory to download NLTK data.

    Returns:
        True if the resource is available or successfully downloaded, False otherwise.

    """
    if resource_name not in _NLTK_RESOURCES:
        warnings.warn(f"Attempting to ensure unknown NLTK resource: {resource_name}", RuntimeWarning, stacklevel=2)
        return False
    if _NLTK_DATA_DOWNLOADED.get(resource_name, False):  # Check if already confirmed as available in this session.
        return True

    try:
        # Attempt to find the resource using NLTK's data find mechanism.
        # This checks if the resource is already installed in one of NLTK's known data paths.
        nltk.data.find(_NLTK_RESOURCES[resource_name])
        _NLTK_DATA_DOWNLOADED[resource_name] = True  # Mark as available for the current session.
        logging.debug("NLTK data '%s' found locally.", resource_name)
    except LookupError:  # Raised by nltk.data.find if the resource is not found.
        logging.info(f"NLTK data '{resource_name}' not found locally. Attempting download...")
        try:
            # Attempt to download the specified NLTK resource.
            # `quiet=True` suppresses interactive prompts if run in a non-interactive environment.
            nltk.download(resource_name, download_dir=download_dir, quiet=True)
            # After attempting download, verify by trying to find it again.
            # This ensures the download was successful and the resource is now accessible.
            nltk.data.find(_NLTK_RESOURCES[resource_name])
            _NLTK_DATA_DOWNLOADED[resource_name] = True  # Mark as available post-download.
            logging.info(f"NLTK data '{resource_name}' downloaded and verified successfully.")
            return True
        except Exception as e:  # Catch any exception during download or re-verification.
            # Log a warning if download or subsequent verification fails.
            # This indicates that features dependent on this resource might not work.
            warnings.warn(
                f"Failed to download or verify NLTK data '{resource_name}'. "
                f"Dependent features might fail or be impaired. Error: {e}",
                RuntimeWarning,
                stacklevel=2,  # Show warning pointing to the caller of _ensure_nltk_data
            )
            return False  # Download/verification failed.
    return True  # Resource was found initially or successfully downloaded.


# --- Preprocessing Setup ---
@lru_cache(maxsize=1)  # Cache the result as stopwords list doesn't change often
def get_default_stopwords() -> set[str]:
    """Lazily loads and returns the default set of English stopwords from NLTK.

    Ensures the 'stopwords' resource is downloaded if necessary.
    Returns an empty set if NLTK data cannot be loaded.
    """
    return set(stopwords.words("english")) if _ensure_nltk_data("stopwords") else set()


@lru_cache(maxsize=1)  # Cache the lemmatizer instance
def get_default_lemmatizer() -> WordNetLemmatizer:
    """Lazily loads and returns a WordNetLemmatizer instance.

    Ensures 'wordnet' and 'omw-1.4' NLTK resources are downloaded.
    Warns if data is missing, as lemmatization quality will be affected.
    """
    if _ensure_nltk_data("wordnet") and _ensure_nltk_data("omw-1.4"):
        return WordNetLemmatizer()
    # Warn if essential NLTK data for lemmatization is missing
    warnings.warn(
        "WordNet or OMW-1.4 NLTK data not found or failed to download. "
        "Lemmatization might not work correctly or might be impaired.",
        RuntimeWarning,
        stacklevel=2,
    )
    return WordNetLemmatizer()  # Return instance; it might error later if data is truly inaccessible


@lru_cache(maxsize=1)  # Cache the stemmer instance
def get_default_stemmer() -> PorterStemmer:
    """Lazily loads and returns a PorterStemmer instance."""
    return PorterStemmer()


# Translation table for efficient punctuation removal
REMOVE_PUNCTUATION_MAP = str.maketrans("", "", string.punctuation)

# --- Cached Preprocessing Functions ---


@lru_cache(maxsize=1024)  # Cache results of preprocessing for efficiency
def preprocess_text_base(text: str) -> str:
    """Text preprocessing, convert to lowercase and remove punctuation.

    Args:
        text: The input string.

    Returns:
        The preprocessed string. Returns an empty string for non-string inputs.

    """
    if not isinstance(text, str):
        return ""  # Handle non-string inputs gracefully
    return text.lower().translate(REMOVE_PUNCTUATION_MAP)


@lru_cache(maxsize=1024)  # Cache tokenization results
def tokenize_text(text: str) -> tuple[str, ...]:
    """Tokenize text using NLTK's word_tokenize after basic preprocessing.

    Ensures 'punkt' NLTK resource is available. Falls back to simple split on error.

    Args:
        text: The input string.

    Returns:
        A tuple of tokens. Returns an empty tuple for non-string inputs.

    """
    if not isinstance(text, str):
        return ()
    _ensure_nltk_data("punkt")  # Ensure tokenizer data is available
    cleaned_text = ""  # Initialize to handle potential errors before assignment
    try:
        cleaned_text = preprocess_text_base(text)  # Apply basic cleaning (lowercase, punctuation removal)
        # Use NLTK's recommended tokenizer for word splitting.
        return tuple(word_tokenize(cleaned_text))
    except Exception as e:  # Catch any exception during tokenization.
        # Log the failure and fallback to a simple space-based split for robustness.
        # This might happen if 'punkt' data is corrupted or an unusual text causes issues.
        logging.debug(
            f"NLTK word_tokenize failed for text: '{text[:50]}...'. Error: {e}. Falling back to simple split.",
            exc_info=True,  # Log full traceback for debugging.
        )
        # Fallback: split the already cleaned text by whitespace.
        return tuple(cleaned_text.split())


@lru_cache(maxsize=1024)  # Cache lemmatization results
def lemmatize_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Lemmatize a tuple of tokens using the default WordNetLemmatizer.

    Returns original tokens if lemmatization fails (e.g., missing NLTK data).

    Args:
        tokens: A tuple of string tokens.

    Returns:
        A tuple of lemmatized tokens.

    """
    lemmatizer = get_default_lemmatizer()  # Get cached lemmatizer instance.
    try:
        # Apply lemmatization to each token in the input tuple.
        return tuple(lemmatizer.lemmatize(token) for token in tokens)
    except Exception as e:  # Catch potential errors during lemmatization.
        # Log if lemmatization fails, which can happen if NLTK data (WordNet, OMW) is missing or corrupt.
        logging.debug(
            f"Lemmatization failed for tokens: {tokens[:5]}... Error: {e}. "
            "Ensure WordNet/OMW NLTK data is downloaded correctly.",
            exc_info=True,
        )
        return tokens  # Return original tokens as a fallback if lemmatization fails.


@lru_cache(maxsize=1024)  # Cache stemming results
def stem_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Stems a tuple of tokens using the default PorterStemmer.

    Args:
        tokens: A tuple of string tokens.

    Returns:
        A tuple of stemmed tokens.

    """
    stemmer = get_default_stemmer()
    return tuple(stemmer.stem(token) for token in tokens)


@lru_cache(maxsize=1024)  # Cache stopword filtering results
def filter_stopwords(tokens: tuple[str, ...], stop_words: Optional[frozenset[str]] = None) -> tuple[str, ...]:
    """Filter stopwords from a tuple of tokens. Also filters out non-alphanumeric tokens.

    Args:
        tokens: A tuple of string tokens.
        stop_words: An optional frozenset of stopwords. If None, uses default English stopwords.

    Returns:
        A tuple of tokens with stopwords and non-alphanumeric tokens removed.

    """
    # Use provided stopwords or fetch default ones (frozenset for hashability if used as dict key)
    sw = stop_words if stop_words is not None else frozenset(get_default_stopwords())
    # Filter tokens that are not in stopwords and are alphanumeric
    return tuple(token for token in tokens if token not in sw and token.isalnum())


# --- Pydantic Models for Configuration and Results ---


class BleuResult(BaseModel):
    """Data model for storing BLEU (Bilingual Evaluation Understudy) scoring results."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall BLEU score, typically BLEU-N (e.g., BLEU-4).",
    )
    cumulative_ngram_scores: Optional[dict[int, float]] = Field(
        default=None,
        description="Cumulative BLEU scores for n-grams up to max_n.",
    )

    @field_validator("cumulative_ngram_scores")
    @classmethod
    def check_ngram_scores(cls, v: Optional[dict[int, float]]) -> Optional[dict[int, float]]:
        """Validate that all cumulative n-gram scores are within the [0.0, 1.0] range."""
        if v is not None:
            for score_val in v.values():
                if not (0.0 <= score_val <= 1.0):
                    msg = "Cumulative n-gram scores must be between 0.0 and 1.0"
                    raise ValueError(msg)
        return v


# --- TFIDFCalculator Class ---
class TFIDFCalculator:
    """Computes TF-IDF vectors and derived similarity/distance metrics.

    Uses a configurable TfidfVectorizer from scikit-learn.
    """

    def __init__(
        self,
        *,
        use_lemmatization: bool,
        use_stopwords: bool,
        stop_words: set[str],  # Expects a set from parent
        tfidf_config: TfidfConfig,  # Expects a Pydantic config model
        **tfidf_kwargs: Any,  # For any other TfidfVectorizer params  # noqa: ANN401
    ) -> None:
        """Initialize the TFIDFCalculator with specified settings.

        Args:
            use_lemmatization (bool): If True, applies lemmatization during tokenization.
            use_stopwords (bool): If True, filters out stopwords during tokenization.
            stop_words (set[str]): The set of stopwords to use, if stopword filtering is enabled.
            tfidf_config (TfidfConfig): Configuration for the TfidfVectorizer, including token pattern,
            n-gram range, max and min document frequency.
            **tfidf_kwargs (Any): Additional keyword arguments for the TfidfVectorizer.

        Notes:
            Initializes a TfidfVectorizer with custom tokenization and preprocessing
            steps based on the provided configuration.

        """
        self.use_lemmatization = use_lemmatization
        self.use_stopwords = use_stopwords
        self.frozen_stop_words = frozenset(stop_words)  # Store as frozenset for tokenizer
        self._tokenizer = self._build_tokenizer()  # Custom tokenizer pipeline

        # Initialize TfidfVectorizer using parameters from TfidfConfig and other args
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenizer,  # Use custom pipeline if defined
            token_pattern=tfidf_config.token_pattern
            if not self._tokenizer
            else None,  # Use pattern only if no custom tokenizer
            stop_words=list(self.frozen_stop_words)
            if self.use_stopwords and not self._tokenizer
            else None,  # SKLearn handles if no tokenizer
            ngram_range=tfidf_config.ngram_range,
            max_df=tfidf_config.max_df,
            min_df=tfidf_config.min_df,
            # Pass through any other sklearn TfidfVectorizer arguments (e.g., norm, use_idf, smooth_idf).
            **tfidf_kwargs,
        )
        logging.debug(
            "TFIDFCalculator initialized with lemmatization: %s, stopwords: %s, TF-IDF config: %s.",
            self.use_lemmatization,
            self.use_stopwords,
        )

    def _build_tokenizer(self) -> Optional[Callable[[str], list[str]]]:
        """Build a custom tokenizer function if lemmatization or custom stopword handling is enabled.

        If not, returns None, letting TfidfVectorizer use its internal tokenization.
        """
        # Only build custom tokenizer if specific preprocessing is needed beyond sklearn's defaults
        if not self.use_lemmatization and not self.use_stopwords:
            return None  # Let TfidfVectorizer use its token_pattern

        # Closure to capture current instance's settings
        # Frozenset is used for stopwords as it's hashable for potential caching in filter_stopwords
        current_stop_words = self.frozen_stop_words

        def tokenizer_func(text: str) -> list[str]:
            """Tokenize text, optionally lemmatizing and filtering stopwords."""
            tokens = tokenize_text(text)  # Base tokenization (cached)
            if self.use_lemmatization:
                tokens = lemmatize_tokens(tokens)  # Lemmatize (cached)
            if self.use_stopwords:  # Filter after potential lemmatization
                tokens = filter_stopwords(tokens, current_stop_words)  # Filter (cached)
            return list(tokens)  # TfidfVectorizer expects a list of strings

        return tokenizer_func

    def fit_transform(self, texts: Sequence[str]) -> csr_matrix:
        """Fits the TfidfVectorizer to the provided texts and transforms them into TF-IDF matrix.

        Returns a Compressed Sparse Row (CSR) matrix.
        """
        try:
            # fit_transform directly returns a csr_matrix
            return csr_matrix(self.vectorizer.fit_transform(texts))
        except Exception:
            logging.debug("TF-IDF fit_transform failed for input texts.", exc_info=True)
            # Return an empty sparse matrix with the correct number of rows (documents)
            # but zero columns (features), indicating no vocabulary was learned or texts were problematic.
            return csr_matrix((len(texts), 0), dtype=float)

    def calculate_metrics_pairwise(self, text1: str, text2: str) -> dict[str, Optional[float]]:
        """Calculate TF-IDF based similarity and distance metrics for a single pair of texts.

        This method fits the TF-IDF vectorizer specifically to the given pair of texts
        and then computes various metrics. It handles cases where no features are
        extracted (e.g., if both texts are empty or contain only stopwords after preprocessing).

        Args:
            text1 (str): The first text string.
            text2 (str): The second text string.

        Returns:
            dict[str, Optional[float]]: A dictionary where keys are metric names
            (e.g., "tfidf_cosine_similarity") and values are the calculated scores
            (float) or None if a calculation failed.

        """
        metrics: dict[str, Optional[float]] = {}  # Initialize dictionary to store results.
        try:
            # Fit and transform the current pair of texts into a TF-IDF matrix.
            # This means the TF-IDF vocabulary is specific to this pair.
            tfidf_matrix = self.fit_transform([text1, text2])

            # Check if the TF-IDF matrix has any features (columns).
            # If shape[1] is 0, it means no terms were found/kept (e.g., texts were empty,
            # all words were stopwords, or did not meet min_df/max_df criteria).
            if tfidf_matrix.shape[1] == 0:
                logging.debug(
                    "TF-IDF found no features for texts (vocab may be empty after processing): '%s...' vs '%s...'",
                    text1[:50],
                    text2[:50],
                )
                # Define behavior for no features:
                # Cosine similarity is undefined (or 0 if vectors are zero, 1 if both truly identical empty strings).
                # Distances can be tricky: Jaccard/Hamming on empty sets is 0. Euclidean/Manhattan on zero vectors is 0.
                # If one is empty and other not, distances could be max or based on norm of non-empty.
                # For simplicity, if no features, assume maximal difference for distances, minimal/zero for similarity,
                # unless both texts were effectively empty.
                are_both_effectively_empty = not text1.strip() and not text2.strip()
                metrics.update(
                    {
                        "tfidf_cosine_similarity": 1.0 if are_both_effectively_empty else 0.0,
                        "tfidf_jaccard_distance": 0.0
                        if are_both_effectively_empty
                        else 1.0,  # Jaccard distance of empty sets is 0.
                        "tfidf_euclidean_distance": 0.0
                        if are_both_effectively_empty
                        else float("inf"),  # Or some large number
                        "tfidf_manhattan_distance": 0.0
                        if are_both_effectively_empty
                        else float("inf"),  # Or some large number
                        "tfidf_minkowski_distance": 0.0
                        if are_both_effectively_empty
                        else float("inf"),  # Or some large number
                        "tfidf_hamming_distance": 0.0 if are_both_effectively_empty else 1.0,  # Hamming distance
                    },
                )
                return metrics  # Early exit if no features.

            # --- Calculate Cosine Similarity ---
            # `cosine_similarity` returns a matrix; [0, 1] gives similarity between text1 and text2.
            cos_sim = cosine_similarity(tfidf_matrix)[0, 1]
            metrics["tfidf_cosine_similarity"] = (
                float(cos_sim) if not math.isnan(cos_sim) else (1.0 if not text1.strip() and not text2.strip() else 0.0)
            )

            # --- Calculate Distances ---
            # Some distance metrics in sklearn require dense arrays.
            dense_matrix = tfidf_matrix.toarray()
            metrics["tfidf_euclidean_distance"] = float(pairwise_distances(dense_matrix, metric="euclidean")[0, 1])
            metrics["tfidf_manhattan_distance"] = float(pairwise_distances(dense_matrix, metric="manhattan")[0, 1])
            metrics["tfidf_minkowski_distance"] = float(pairwise_distances(dense_matrix, metric="minkowski")[0, 1])

            # --- Jaccard & Hamming Distances on Binarized TF-IDF vectors ---
            # These metrics operate on the presence/absence of terms rather than their TF-IDF weights.
            binary_presence_matrix = (dense_matrix > 0).astype(bool)  # Convert TF-IDF to binary (term present/absent).

            # Calculate Jaccard distance if at least one text has some terms present.
            if binary_presence_matrix[0].any() or binary_presence_matrix[1].any():
                j_dist = pairwise_distances(binary_presence_matrix, metric="jaccard")[0, 1]
                # Jaccard distance can be NaN if both vectors are all zeros after binarization
                # (e.g. if TF-IDF scores were all zero, though unlikely if shape[1]>0).
                # If NaN, it implies no common features and no features in one or both, treat as max distance (1.0).
                # If both binary vectors are all zero (no terms), Jaccard distance is conventionally 0.
                metrics["tfidf_jaccard_distance"] = float(j_dist) if not math.isnan(j_dist) else 1.0
                # Hamming distance (fraction of positions where term presence differs).
                metrics["tfidf_hamming_distance"] = float(
                    pairwise_distances(binary_presence_matrix.astype(int), metric="hamming")[0, 1],
                )
            else:  # Both binarized vectors are all zeros.
                metrics["tfidf_jaccard_distance"] = 0.0  # Jaccard distance of two empty sets is 0.
                metrics["tfidf_hamming_distance"] = 0.0  # No differing bits if both are all zeros.

        except Exception as e:  # Catch any other error during TF-IDF metric calculation.
            logging.debug(
                f"Error calculating TF-IDF metrics for '{text1[:50]}...' vs '{text2[:50]}...': {e}", exc_info=True,
            )
            # Ensure all TF-IDF metric keys are present in the output, defaulting to None on error.
            for k_tfidf in [
                "tfidf_cosine_similarity",
                "tfidf_euclidean_distance",
                "tfidf_manhattan_distance",
                "tfidf_jaccard_distance",
                "tfidf_hamming_distance",
                "tfidf_minkowski_distance",
            ]:
                metrics.setdefault(k_tfidf, None)
        return metrics


# --- BleuScorer Class ---
class BleuScorer:
    """Computes BLEU (Bilingual Evaluation Understudy) score, adapted for similarity.

    Uses NLTK's sentence_bleu. Allows custom preprocessing.
    """

    def __init__(
        self,
        stop_words: set[str],  # Expects a set from parent
        lemmatizer: Optional[WordNetLemmatizer],  # Can be None
        smoothing_function: Optional[Callable],  # e.g., SmoothingFunction().method1
    ) -> None:
        """Initialize the BleuScorer with specified preprocessing settings.

        Args:
            stop_words (set[str]): A set of stop words to exclude during preprocessing.
            lemmatizer (Optional[WordNetLemmatizer]): An instance of WordNetLemmatizer for lemmatization, if desired.
            smoothing_function (Optional[Callable]): A smoothing function for BLEU score calculation.
            Defaults to NLTK's method1.

        """
        self.lemmatizer = lemmatizer
        self.frozen_stop_words = frozenset(stop_words)  # For caching preprocessing
        self.smoothing = smoothing_function or SmoothingFunction().method1  # Default NLTK smoothing
        logging.debug("BleuScorer initialized.")

    @lru_cache(maxsize=1024)  # Cache preprocessed text based on its content
    def _preprocess_bleu_text(self, text: str) -> tuple[str, ...]:
        """Preprocesses text for BLEU: tokenize, lemmatize (optional), filter stopwords."""
        tokens = tokenize_text(text)  # Base tokenization (cached)
        if self.lemmatizer:
            tokens = lemmatize_tokens(tokens)  # Lemmatize (cached)
        # BLEU conventionally might not remove stopwords or uses specific recipes.
        # Here, we apply the general stopword list for consistency.
        return filter_stopwords(tokens, self.frozen_stop_words)  # Filter (cached)

    def _calculate_bleu(
        self,
        ref_tokens_list: list[list[str]],  # list of tokenized reference sentences
        hyp_tokens: list[str],  # Tokenized hypothesis sentence
        weights: tuple[float, ...],  # N-gram weights, e.g., (0.25, 0.25, 0.25, 0.25) for BLEU-4
    ) -> float:
        """Calculate the BLEU score using NLTK, with error handling."""
        if not hyp_tokens or not any(ref_tokens_list):  # Must have hypothesis and at least one reference
            return 0.0
        try:
            return sentence_bleu(
                references=ref_tokens_list,
                hypothesis=hyp_tokens,
                weights=weights,
                smoothing_function=self.smoothing,
            )
        except ZeroDivisionError:  # Can occur with very short texts / no common n-grams
            logging.exception(
                "BLEU calculation resulted in ZeroDivisionError (likely short hypothesis/reference or no overlap).",
            )
            return 0.0
        except Exception:
            logging.exception("Unexpected error during NLTK sentence_bleu calculation.")
            return 0.0

    def score_all_ngrams(self, references: Union[str, Sequence[str]], hypothesis: str, max_n: int = 4) -> BleuResult:
        """Compute cumulative BLEU scores from BLEU-1 to BLEU-max_n.

        The main 'score' in the returned BleuResult is typically the BLEU-max_n score.
        """
        ref_list = [references] if isinstance(references, str) else list(references)
        # Handle empty inputs early
        if not hypothesis.strip() or not ref_list or not any(r.strip() for r in ref_list):
            return BleuResult(score=0.0, cumulative_ngram_scores=dict.fromkeys(range(1, max_n + 1), 0.0))

        hyp_tokens = list(self._preprocess_bleu_text(hypothesis))
        ref_tokens_list_processed = [list(self._preprocess_bleu_text(ref)) for ref in ref_list]

        # Filter out any reference lists that became empty after preprocessing
        ref_tokens_list_valid = [r_list for r_list in ref_tokens_list_processed if r_list]
        if not ref_tokens_list_valid:  # All references became empty
            return BleuResult(score=0.0, cumulative_ngram_scores=dict.fromkeys(range(1, max_n + 1), 0.0))

        cumulative_scores: dict[int, float] = {}
        for n_val in range(1, max_n + 1):
            # Standard cumulative weights: (1/n, 1/n, ..., 1/n) for n components
            current_weights = tuple(1.0 / n_val for _ in range(n_val))
            # Ensure weights tuple has length of at least max_n for sentence_bleu if it expects that,
            # or adjust how weights are passed. NLTK sentence_bleu uses len(weights) as N.
            # So, current_weights is fine.
            ngram_score = self._calculate_bleu(ref_tokens_list_valid, hyp_tokens, current_weights)
            cumulative_scores[n_val] = ngram_score

        # The primary 'score' is often the highest n-gram score (e.g., BLEU-4 if max_n=4)
        overall_bleu_score = cumulative_scores.get(max_n, 0.0)
        return BleuResult(score=overall_bleu_score, cumulative_ngram_scores=cumulative_scores)


# --- BM25 Calculation Wrapper ---
def calculate_bm25(reference: str, hypothesis: str) -> Optional[float]:
    """Calculate BM25 relevance score between a reference (document) and a hypothesis (query).

    Returns None if the rank_bm25 library is unavailable or an error occurs.
    """
    if not _bm25_available:
        return None
    try:
        # BM25 expects a corpus of tokenized documents and a tokenized query.
        # Here, the "corpus" is just the single reference document.
        tokenized_corpus: list[list[str]] = [list(tokenize_text(reference))]
        tokenized_query: list[str] = list(tokenize_text(hypothesis))

        # Handle cases where tokenization results in empty lists
        if not tokenized_corpus[0] or not tokenized_query:
            return 0.0  # No common terms possible if one is empty

        bm25_calculator = BM25(tokenized_corpus)
        # Get scores for the query against the corpus (which has only one document)
        doc_scores = bm25_calculator.get_scores(tokenized_query)
        # doc_scores will be a list/array with one score corresponding to the reference
        return float(doc_scores[0]) if doc_scores is not None and len(doc_scores) > 0 else 0.0
    except Exception:
        logging.exception(f"BM25 calculation failed for '{reference[:50]}...' vs '{hypothesis[:50]}...'")
        return None


class SimilarityCalculator:
    """Orchestrates the calculation of a comprehensive set of text similarity metrics.

    Initializes and uses specialized calculators (TFIDFCalculator, BleuScorer) and
    various direct similarity functions. Supports parallel processing for multiple pairs.
    """

    def __init__(self, config: Optional[SimilarityCalculatorConfig] = None) -> None:
        """Initialize a SimilarityCalculator instance with the provided configuration.

        Args:
            config (Optional[SimilarityCalculatorConfig]): The configuration object to use.
            If None, uses the default config.

        Notes:
            The configuration object is used to control the behavior of the calculators and similarity functions used.
            The current implementation does not support passing custom stopwords or lemmatization settings
            for the Bleu scorer.For parallel processing, the config is pickled and sent to workers,
            so any custom stopwords or lemmatization settings must be picklable.

        """
        logging.info("Initializing SimilarityCalculator...")
        cfg = config or SimilarityCalculatorConfig()  # Use provided config or default

        self.use_lemmatization = cfg.use_lemmatization
        self.use_stopwords = cfg.use_stopwords

        # Combine custom stopwords with default if applicable
        _provided_sw_list = cfg.custom_stop_words if cfg.custom_stop_words is not None else []
        # If custom_stop_words is an empty list, we should use defaults.
        # If it's non-empty, we use only those (or combine, depending on desired logic).
        # Current logic: if custom_stop_words is provided (even empty), it overrides defaults.
        # To combine: self.stop_words = get_default_stopwords().union(set(_provided_sw_list))
        self.stop_words: set[str] = (
            set(_provided_sw_list) if cfg.custom_stop_words is not None else get_default_stopwords()
        )

        # Ensure necessary NLTK data is available based on configuration
        _ensure_nltk_data("punkt")  # Always needed for tokenization
        if self.use_stopwords:
            _ensure_nltk_data("stopwords")
        if self.use_lemmatization:
            _ensure_nltk_data("wordnet")
            _ensure_nltk_data("omw-1.4")

        self.lemmatizer = get_default_lemmatizer() if self.use_lemmatization else None

        # Initialize sub-calculators
        # Note: bleu_smoothing_function is not easily part of picklable config for workers.
        # Workers will use default smoothing unless a picklable strategy (e.g., name-based) is implemented.
        self.bleu_scorer = BleuScorer(
            stop_words=self.stop_words,
            lemmatizer=self.lemmatizer,
            smoothing_function=None,  # Default smoothing for main instance
        )
        self.tfidf_calculator = TFIDFCalculator(
            use_lemmatization=self.use_lemmatization,
            use_stopwords=self.use_stopwords,
            stop_words=self.stop_words,  # Pass the final set of stopwords
            tfidf_config=cfg.tfidf_config,  # Pass the Pydantic TfidfConfig model
        )
        logging.debug("SimilarityCalculator initialized successfully with config: %s", cfg.model_dump_json())

    def calculate_single_pair(self, text1: str, text2: str) -> SimilarityMetrics:
        """Calculate all configured similarity metrics for a single pair of texts.

        Args:
            text1: The first text string.
            text2: The second text string.

        Returns:
            A SimilarityMetrics Pydantic model instance containing the calculated scores.
            Returns a default (mostly None) SimilarityMetrics model for invalid inputs.

        """
        if not isinstance(text1, str) or not isinstance(text2, str):
            logging.warning("Invalid input types for similarity calculation. Both texts must be strings.")
            return SimilarityMetrics()  # Return default model with Nones

        raw_results: dict[str, Optional[float]] = {}  # Dictionary to accumulate raw scores
        s1_lower, s2_lower = text1.lower(), text2.lower()  # Common preprocessed versions

        # --- 1. Basic String / Sequence Metrics ---
        try:
            seq_matcher = difflib.SequenceMatcher(None, s1_lower, s2_lower, autojunk=False)
            raw_results["ratio"] = seq_matcher.ratio()
            raw_results["normalized_levenshtein"] = NORMALIZED_LEVENSHTEIN.similarity(s1_lower, s2_lower)
            raw_results["jaro_winkler"] = JARO_WINKLER.similarity(s1_lower, s2_lower)
            raw_results["metric_lcs_similarity"] = 1.0 - METRIC_LCS.distance(
                s1_lower,
                s2_lower,
            )  # Convert distance to similarity
            raw_results["qgram2_distance"] = QGRAM_2.distance(s1_lower, s2_lower)  # QGram returns distance
            raw_results["qgram3_distance"] = QGRAM_3.distance(s1_lower, s2_lower)
            raw_results["qgram4_similarity"] = QGRAM_4.distance(s1_lower, s2_lower)
            raw_results["cosine_char_2gram"] = SIM_COSINE_CHAR.similarity(s1_lower, s2_lower)
            raw_results["jaccard_char_2gram"] = SIM_JACCARD_CHAR.similarity(s1_lower, s2_lower)
        except Exception:
            logging.debug("Error during basic string similarity calculation.", exc_info=True)

        # --- 2. RapidFuzz Metrics ---
        try:
            raw_results["rfuzz_ratio"] = rapidfuzz_fuzz.ratio(s1_lower, s2_lower) / 100.0
            raw_results["rfuzz_partial_ratio"] = rapidfuzz_fuzz.partial_ratio(s1_lower, s2_lower) / 100.0
            raw_results["rfuzz_token_set_ratio"] = rapidfuzz_fuzz.token_set_ratio(s1_lower, s2_lower) / 100.0
            raw_results["rfuzz_token_sort_ratio"] = rapidfuzz_fuzz.token_sort_ratio(s1_lower, s2_lower) / 100.0
            raw_results["rfuzz_wratio"] = (
                rapidfuzz_fuzz.WRatio(s1_lower, s2_lower) / 100.0
            )  # Note: WRatio can exceed 100
            raw_results["rfuzz_qratio"] = rapidfuzz_fuzz.QRatio(s1_lower, s2_lower) / 100.0
        except Exception:
            logging.debug("Error during RapidFuzz calculation.", exc_info=True)

        # --- 3. FuzzyWuzzy Metrics (if available) ---
        if _fuzzywuzzy_available:
            try:
                raw_results["fz_uqratio"] = fuzzywuzzy_fuzz.UQRatio(s1_lower, s2_lower) / 100.0
                raw_results["fz_uwratio"] = fuzzywuzzy_fuzz.UWRatio(s1_lower, s2_lower) / 100.0
            except Exception:
                logging.debug("Error during FuzzyWuzzy calculation.", exc_info=True)

        # --- 4. BLEU Score ---
        try:
            # Use original case texts for BLEU, as preprocessing is handled by BleuScorer
            bleu_result_model = self.bleu_scorer.score_all_ngrams(text1, text2)
            raw_results["bleu_score"] = bleu_result_model.score
            # Optionally add cumulative scores to raw_results if desired for SimilarityMetrics model
            # if bleu_result_model.cumulative_ngram_scores:
            #     for n_val, score_val in bleu_result_model.cumulative_ngram_scores.items():
            #         raw_results[f"bleu_{n_val}_cumulative"] = score_val
        except Exception:
            logging.debug("Error calculating BLEU score.", exc_info=True)

        # --- 5. BM25 Score ---
        # calculate_bm25 handles its own errors and library check
        raw_results["bm25"] = calculate_bm25(text1, text2)

        # --- 6. TF-IDF Metrics ---
        # tfidf_calculator.calculate_metrics_pairwise handles its own errors
        raw_results.update(self.tfidf_calculator.calculate_metrics_pairwise(text1, text2))

        # Clean up NaN values to None before Pydantic model creation for consistency
        final_results_cleaned = {
            k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in raw_results.items()
        }
        # Create and return the Pydantic model.
        # Pydantic will validate the data and ignore any extra keys not defined in SimilarityMetrics
        # if the model's Config has `extra = 'ignore'` (default is 'ignore').
        return SimilarityMetrics(**final_results_cleaned)

    def calculate_multiple_pairs(
        self,
        text_pairs: Iterable[tuple[str, str]],
        max_workers: Optional[int] = None,
    ) -> list[SimilarityMetrics]:
        """Calculate similarity metrics for multiple text pairs in parallel using ProcessPoolExecutor.

        Args:
            text_pairs: An iterable of (text1, text2) tuples.
            max_workers: Maximum number of worker processes. Defaults to system's CPU count.

        Returns:
            A list of SimilarityMetrics model instances, in the same order as input pairs.
            If a worker fails for a pair, a default SimilarityMetrics model is returned for that pair.

        """
        text_pairs_list = list(text_pairs)  # Ensure it's a list for indexing
        if not text_pairs_list:
            return []

        # Prepare a picklable configuration dictionary for worker processes.
        # This involves extracting relevant parameters from the current instance.
        default_tfidf_config_for_worker = TfidfConfig()  # For fallback default values
        vectorizer_params = self.tfidf_calculator.vectorizer.get_params(
            deep=False,
        )  # Get actual params from current vectorizer

        picklable_config = SimilarityCalculatorConfig(
            use_lemmatization=self.use_lemmatization,
            use_stopwords=self.use_stopwords,
            custom_stop_words=list(self.stop_words),  # Convert set to list for JSON/pickling
            tfidf_config=TfidfConfig(  # Reconstruct TfidfConfig with current or default values
                token_pattern=vectorizer_params.get("token_pattern") or default_tfidf_config_for_worker.token_pattern,
                ngram_range=vectorizer_params.get("ngram_range", default_tfidf_config_for_worker.ngram_range),
                max_df=vectorizer_params.get("max_df", default_tfidf_config_for_worker.max_df),
                min_df=vectorizer_params.get("min_df", default_tfidf_config_for_worker.min_df),
            ),
        )
        # Convert the Pydantic config model to a dictionary for sending to workers
        picklable_config_dict = picklable_config.model_dump()

        futures_map: dict[Any, int] = {}  # Maps Future instances to original pair index
        results_unordered: dict[int, SimilarityMetrics] = {}  # Stores results indexed by original pair index

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Log actual number of workers, which might differ from max_workers request
                logging.info(
                    f"Submitting {len(text_pairs_list)} tasks to ProcessPoolExecutor with {max_workers} worker(s).",
                )
                # Submit tasks to the executor
                for i, (text1, text2) in enumerate(text_pairs_list):
                    future = executor.submit(_worker_calculate_single_pair, picklable_config_dict, text1, text2)
                    futures_map[future] = i  # Store index to reorder results later

                # Retrieve results as they complete
                for future in as_completed(futures_map):
                    index = futures_map[future]
                    try:
                        result_model = future.result()  # Get SimilarityMetrics model from worker
                        results_unordered[index] = result_model
                    except Exception:
                        # Log error from worker and provide a default result for that pair
                        logging.exception(f"Worker process error for pair index {index}")
                        results_unordered[index] = SimilarityMetrics()  # Default empty model on error
        except Exception:
            # Catch errors related to ProcessPoolExecutor itself (e.g., setup, shutdown)
            logging.exception("Error occurred in ProcessPoolExecutor management")
            # Ensure results list has the correct size even if processing failed midway, filling with defaults
            for i in range(len(text_pairs_list)):
                if i not in results_unordered:
                    results_unordered[i] = SimilarityMetrics()

        # Reconstruct results in the original order of the input pairs
        ordered_results = [results_unordered.get(i, SimilarityMetrics()) for i in range(len(text_pairs_list))]
        logging.info(f"Finished processing {len(ordered_results)} pairs in parallel.")
        return ordered_results


# --- Worker Function for Parallel Execution ---
# This function must be defined at the top level of the module to be picklable by multiprocessing.
def _worker_calculate_single_pair(config_dict: dict[str, Any], text1: str, text2: str) -> SimilarityMetrics:
    """Worker function executed by each process in the ProcessPoolExecutor.

    It re-initializes a SimilarityCalculator using the provided configuration dictionary
    and calculates metrics for a single text pair.

    Args:
        config_dict: A dictionary representation of SimilarityCalculatorConfig.
        text1: The first text string.
        text2: The second text string.

    Returns:
        A SimilarityMetrics Pydantic model instance.

    """
    try:
        # Reconstruct the Pydantic configuration model from the dictionary
        # This ensures the worker's calculator is configured consistently.
        worker_sim_config = SimilarityCalculatorConfig(**config_dict)
    except Exception:
        # Log if config reconstruction fails and use a default config as a fallback
        logging.exception(
            "Worker failed to reconstruct SimilarityCalculatorConfig from dict. Using default config.",
        )
        worker_sim_config = SimilarityCalculatorConfig()  # Fallback

    # Create a new SimilarityCalculator instance within the worker process
    calculator = SimilarityCalculator(config=worker_sim_config)
    # Calculate and return metrics for the given pair
    return calculator.calculate_single_pair(text1, text2)


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for the example execution
    logging.basicConfig(
        level=logging.INFO,  # set to INFO for general operational messages
        format="%(asctime)s - %(levelname)s - %(processName)s - %(module)s - %(message)s",
    )
    # For more detailed NLTK download/TFIDF/BLEU messages, set the module's logger to DEBUG
    # logging.getLogger(__name__).setLevel(logging.DEBUG)

    # Initialize SimilarityCalculator with a Pydantic configuration object
    main_app_config = SimilarityCalculatorConfig(
        use_lemmatization=True,
        use_stopwords=True,
        custom_stop_words=["example", "custom"],  # Example: add custom stop words
        tfidf_config=TfidfConfig(min_df=1, ngram_range=(1, 1)),  # Default TF-IDF, min_df=1
    )
    calculator = SimilarityCalculator(config=main_app_config)

    # Example texts for demonstration
    original_text = "The quick brown fox jumps over the lazy dog, an iconic pangram."
    compare_text_similar = "A fast brown fox leaped over a sleepy dog; this is a well-known sentence."
    compare_text_different = "This sentence is completely unrelated and discusses apples and oranges."
    empty_text = ""  # Edge case: empty text

    print("\n--- Single Pair Calculation Example ---")
    metrics1 = calculator.calculate_single_pair(original_text, compare_text_similar)
    print("\nSimilarity (Original vs. Similar):")
    # Use .model_dump() for easy printing, excluding None and default values for brevity
    for k, v_metric in metrics1.model_dump(exclude_none=True, exclude_defaults=True).items():
        print(f"  {k}: {v_metric:.4f}" if isinstance(v_metric, float) else f"  {k}: {v_metric}")

    metrics2 = calculator.calculate_single_pair(original_text, compare_text_different)
    print("\nSimilarity (Original vs. Different) (Key Metrics):")
    # Print a few key metrics, handling None with a fallback for display
    print(f"  Ratio: {metrics2.ratio or 0.0:.4f}")
    print(f"  BLEU Score: {metrics2.bleu_score or 0.0:.4f}")
    print(f"  TF-IDF Cosine Similarity: {metrics2.tfidf_cosine_similarity or 0.0:.4f}")

    metrics_empty = calculator.calculate_single_pair(original_text, empty_text)
    print("\nSimilarity (Original vs. Empty Text) (Key Metrics):")
    print(f"  Ratio: {metrics_empty.ratio or 0.0:.4f}")
    print(f"  BLEU Score: {metrics_empty.bleu_score or 0.0:.4f}")  # Should be 0.0 for empty hypothesis
    print(f"  TF-IDF Cosine Similarity: {metrics_empty.tfidf_cosine_similarity or 0.0:.4f}")

    print("\n--- Parallel Multi-Pair Calculation Example ---")
    text_pairs_for_parallel = [
        (original_text, compare_text_similar),
        ("The cat sat on the mat.", "A feline was resting upon a rug."),
        (original_text, compare_text_different),
        ("This is a short example text.", "Another short one for testing."),
        (original_text, empty_text),  # Include edge case in parallel processing
        ("Final test pair, quite distinct.", "The beginning of something new perhaps."),
    ]

    # Determine number of workers, e.g., based on CPU count or a fixed number
    # Be mindful of resource usage; for I/O bound tasks more workers might be fine,
    # for CPU bound (like this), os.cpu_count() is a good starting point.
    num_workers = min(4, os.cpu_count() or 1) if os.cpu_count() else 2  # Defensive check for os.cpu_count()

    parallel_results_list = calculator.calculate_multiple_pairs(text_pairs_for_parallel, max_workers=num_workers)

    print(f"\nParallel Calculation Results ({len(parallel_results_list)} pairs processed):")
    for i, result_model_instance in enumerate(parallel_results_list):
        p1_text, p2_text = text_pairs_for_parallel[i]
        print(f"\nPair {i + 1}: '{p1_text[:30]}...' vs '{p2_text[:30]}...'")
        # Print a subset of metrics for brevity, handling None values
        print(f"  Ratio: {result_model_instance.ratio or 0.0:.4f}")
        print(f"  BLEU Score: {result_model_instance.bleu_score or 0.0:.4f}")
        print(f"  TF-IDF Cosine Similarity: {result_model_instance.tfidf_cosine_similarity or 0.0:.4f}")

    print("\n--- Example Script Finished ---")
