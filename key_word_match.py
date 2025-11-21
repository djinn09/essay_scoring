"""
Provides comprehensive text similarity calculation utilities.

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

from __future__ import annotations

import difflib
import logging
import math
import os
import string
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Union

from rapidfuzz import fuzz as rapidfuzz_fuzz
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from pydantic import BaseModel, Field, field_validator

from similarity.cosine import Cosine
from similarity.jaccard import Jaccard
from similarity.jarowinkler import JaroWinkler
from similarity.metric_lcs import MetricLCS
from similarity.normalized_levenshtein import NormalizedLevenshtein
from similarity.qgram import QGram

from app_types import SimilarityCalculatorConfig, SimilarityMetrics, TfidfConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

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

# Optional libraries
try:
    from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz

    _fuzzywuzzy_available = True
except ImportError:
    _fuzzywuzzy_available = False

try:
    from rank_bm25 import BM25L as BM25

    _bm25_available = True
except ImportError:
    _bm25_available = False


# --- Similarity Metric Instances (Globally Initialized for Reuse) ---
NORMALIZED_LEVENSHTEIN = NormalizedLevenshtein()
JARO_WINKLER = JaroWinkler()
METRIC_LCS = MetricLCS()
QGRAM_2 = QGram(2)
QGRAM_3 = QGram(3)
QGRAM_4 = QGram(4)
SIM_COSINE_CHAR = Cosine(2)
SIM_JACCARD_CHAR = Jaccard(2)


# --- NLTK Data Handling ---
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet.zip/wordnet/",
    "omw-1.4": "corpora/omw-1.4.zip/omw-1.4/",
    "stopwords": "corpora/stopwords",
}
_NLTK_DATA_DOWNLOADED = dict.fromkeys(_NLTK_RESOURCES, False)


def _ensure_nltk_data(resource_name: str, download_dir: Optional[str] = None) -> bool:
    """
    Check if a given NLTK resource is available and downloads it if not.

    Uses a module-level dictionary `_NLTK_DATA_DOWNLOADED` to track downloaded
    resources within the current session to avoid redundant checks or download attempts.

    Args:
        resource_name (str): The name of the NLTK resource (e.g., "punkt", "wordnet").
        download_dir (Optional[str]): Optional custom directory to download NLTK data.

    Returns:
        bool: True if the resource is available or successfully downloaded, False otherwise.
    """
    if resource_name not in _NLTK_RESOURCES:
        warnings.warn(f"Attempting to ensure unknown NLTK resource: {resource_name}", RuntimeWarning, stacklevel=2)
        return False
    if _NLTK_DATA_DOWNLOADED.get(resource_name, False):
        return True

    try:
        nltk.data.find(_NLTK_RESOURCES[resource_name])
        _NLTK_DATA_DOWNLOADED[resource_name] = True
        logging.debug("NLTK data '%s' found locally.", resource_name)
    except LookupError:
        logging.info(f"NLTK data '{resource_name}' not found locally. Attempting download...")
        try:
            nltk.download(resource_name, download_dir=download_dir, quiet=True)
            nltk.data.find(_NLTK_RESOURCES[resource_name])
            _NLTK_DATA_DOWNLOADED[resource_name] = True
            logging.info(f"NLTK data '{resource_name}' downloaded and verified successfully.")
            return True
        except Exception as e:
            warnings.warn(
                f"Failed to download or verify NLTK data '{resource_name}'. "
                f"Dependent features might fail or be impaired. Error: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
    return True


# --- Preprocessing Setup ---
@lru_cache(maxsize=1)
def get_default_stopwords() -> set[str]:
    """
    Lazily loads and returns the default set of English stopwords from NLTK.

    Returns:
        set[str]: A set of English stopwords, or an empty set if unavailable.
    """
    return set(stopwords.words("english")) if _ensure_nltk_data("stopwords") else set()


@lru_cache(maxsize=1)
def get_default_lemmatizer() -> WordNetLemmatizer:
    """
    Lazily loads and returns a WordNetLemmatizer instance.

    Returns:
        WordNetLemmatizer: An instance of WordNetLemmatizer.
    """
    if _ensure_nltk_data("wordnet") and _ensure_nltk_data("omw-1.4"):
        return WordNetLemmatizer()
    warnings.warn(
        "WordNet or OMW-1.4 NLTK data not found or failed to download. "
        "Lemmatization might not work correctly or might be impaired.",
        RuntimeWarning,
        stacklevel=2,
    )
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def get_default_stemmer() -> PorterStemmer:
    """
    Lazily loads and returns a PorterStemmer instance.

    Returns:
        PorterStemmer: An instance of PorterStemmer.
    """
    return PorterStemmer()


REMOVE_PUNCTUATION_MAP = str.maketrans("", "", string.punctuation)


# --- Cached Preprocessing Functions ---


@lru_cache(maxsize=1024)
def preprocess_text_base(text: str) -> str:
    """
    Text preprocessing, convert to lowercase and remove punctuation.

    Args:
        text (str): The input string.

    Returns:
        str: The preprocessed string. Returns an empty string for non-string inputs.
    """
    if not isinstance(text, str):
        return ""
    return text.lower().translate(REMOVE_PUNCTUATION_MAP)


@lru_cache(maxsize=1024)
def tokenize_text(text: str) -> tuple[str, ...]:
    """
    Tokenize text using NLTK's word_tokenize after basic preprocessing.

    Args:
        text (str): The input string.

    Returns:
        tuple[str, ...]: A tuple of tokens. Returns an empty tuple for non-string inputs.
    """
    if not isinstance(text, str):
        return ()
    _ensure_nltk_data("punkt")
    cleaned_text = ""
    try:
        cleaned_text = preprocess_text_base(text)
        return tuple(word_tokenize(cleaned_text))
    except Exception as e:
        logging.debug(
            f"NLTK word_tokenize failed for text: '{text[:50]}...'. Error: {e}. Falling back to simple split.",
            exc_info=True,
        )
        return tuple(cleaned_text.split())


@lru_cache(maxsize=1024)
def lemmatize_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """
    Lemmatize a tuple of tokens using the default WordNetLemmatizer.

    Args:
        tokens (tuple[str, ...]): A tuple of string tokens.

    Returns:
        tuple[str, ...]: A tuple of lemmatized tokens.
    """
    lemmatizer = get_default_lemmatizer()
    try:
        return tuple(lemmatizer.lemmatize(token) for token in tokens)
    except Exception as e:
        logging.debug(
            f"Lemmatization failed for tokens: {tokens[:5]}... Error: {e}. "
            "Ensure WordNet/OMW NLTK data is downloaded correctly.",
            exc_info=True,
        )
        return tokens


@lru_cache(maxsize=1024)
def stem_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """
    Stems a tuple of tokens using the default PorterStemmer.

    Args:
        tokens (tuple[str, ...]): A tuple of string tokens.

    Returns:
        tuple[str, ...]: A tuple of stemmed tokens.
    """
    stemmer = get_default_stemmer()
    return tuple(stemmer.stem(token) for token in tokens)


@lru_cache(maxsize=1024)
def filter_stopwords(tokens: tuple[str, ...], stop_words: Optional[frozenset[str]] = None) -> tuple[str, ...]:
    """
    Filter stopwords from a tuple of tokens. Also filters out non-alphanumeric tokens.

    Args:
        tokens (tuple[str, ...]): A tuple of string tokens.
        stop_words (Optional[frozenset[str]]): An optional frozenset of stopwords.

    Returns:
        tuple[str, ...]: A tuple of tokens with stopwords and non-alphanumeric tokens removed.
    """
    sw = stop_words if stop_words is not None else frozenset(get_default_stopwords())
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
    """
    Computes TF-IDF vectors and derived similarity/distance metrics.

    Uses a configurable TfidfVectorizer from scikit-learn.
    """

    def __init__(
        self,
        *,
        use_lemmatization: bool,
        use_stopwords: bool,
        stop_words: set[str],
        tfidf_config: TfidfConfig,
        **tfidf_kwargs: Any,
    ) -> None:
        """
        Initialize the TFIDFCalculator with specified settings.

        Args:
            use_lemmatization (bool): If True, applies lemmatization during tokenization.
            use_stopwords (bool): If True, filters out stopwords during tokenization.
            stop_words (set[str]): The set of stopwords to use, if stopword filtering is enabled.
            tfidf_config (TfidfConfig): Configuration for the TfidfVectorizer.
            **tfidf_kwargs (Any): Additional keyword arguments for the TfidfVectorizer.
        """
        self.use_lemmatization = use_lemmatization
        self.use_stopwords = use_stopwords
        self.frozen_stop_words = frozenset(stop_words)
        self._tokenizer = self._build_tokenizer()

        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenizer,
            token_pattern=tfidf_config.token_pattern if not self._tokenizer else None,
            stop_words=list(self.frozen_stop_words) if self.use_stopwords and not self._tokenizer else None,
            ngram_range=tfidf_config.ngram_range,
            max_df=tfidf_config.max_df,
            min_df=tfidf_config.min_df,
            **tfidf_kwargs,
        )
        logging.debug(
            "TFIDFCalculator initialized with lemmatization: %s, stopwords: %s, TF-IDF config: %s.",
            self.use_lemmatization,
            self.use_stopwords,
            self.vectorizer.get_params(),
        )

    def _build_tokenizer(self) -> Optional[Callable[[str], list[str]]]:
        """
        Build a custom tokenizer function if lemmatization or custom stopword handling is enabled.

        Returns:
            Optional[Callable[[str], list[str]]]: A tokenizer function or None.
        """
        if not self.use_lemmatization and not self.use_stopwords:
            return None

        current_stop_words = self.frozen_stop_words

        def tokenizer_func(text: str) -> list[str]:
            """Tokenize text, optionally lemmatizing and filtering stopwords."""
            tokens = tokenize_text(text)
            if self.use_lemmatization:
                tokens = lemmatize_tokens(tokens)
            if self.use_stopwords:
                tokens = filter_stopwords(tokens, current_stop_words)
            return list(tokens)

        return tokenizer_func

    def fit_transform(self, texts: Sequence[str]) -> csr_matrix:
        """
        Fits the TfidfVectorizer to the provided texts and transforms them into TF-IDF matrix.

        Args:
            texts (Sequence[str]): Sequence of input texts.

        Returns:
            csr_matrix: A Compressed Sparse Row (CSR) matrix.
        """
        try:
            return csr_matrix(self.vectorizer.fit_transform(texts))
        except Exception:
            logging.debug("TF-IDF fit_transform failed for input texts.", exc_info=True)
            return csr_matrix((len(texts), 0), dtype=float)

    def calculate_metrics_pairwise(self, text1: str, text2: str) -> dict[str, Optional[float]]:
        """
        Calculate TF-IDF based similarity and distance metrics for a single pair of texts.

        Args:
            text1 (str): The first text string.
            text2 (str): The second text string.

        Returns:
            dict[str, Optional[float]]: A dictionary where keys are metric names
            and values are the calculated scores or None.
        """
        metrics: dict[str, Optional[float]] = {}
        try:
            tfidf_matrix = self.fit_transform([text1, text2])

            if tfidf_matrix.shape[1] == 0:
                are_both_effectively_empty = not text1.strip() and not text2.strip()
                metrics.update(
                    {
                        "tfidf_cosine_similarity": 1.0 if are_both_effectively_empty else 0.0,
                        "tfidf_jaccard_distance": 0.0 if are_both_effectively_empty else 1.0,
                        "tfidf_euclidean_distance": 0.0 if are_both_effectively_empty else float("inf"),
                        "tfidf_manhattan_distance": 0.0 if are_both_effectively_empty else float("inf"),
                        "tfidf_minkowski_distance": 0.0 if are_both_effectively_empty else float("inf"),
                        "tfidf_hamming_distance": 0.0 if are_both_effectively_empty else 1.0,
                    },
                )
                return metrics

            cos_sim = cosine_similarity(tfidf_matrix)[0, 1]
            metrics["tfidf_cosine_similarity"] = (
                float(cos_sim) if not math.isnan(cos_sim) else (1.0 if not text1.strip() and not text2.strip() else 0.0)
            )

            dense_matrix = tfidf_matrix.toarray()
            metrics["tfidf_euclidean_distance"] = float(pairwise_distances(dense_matrix, metric="euclidean")[0, 1])
            metrics["tfidf_manhattan_distance"] = float(pairwise_distances(dense_matrix, metric="manhattan")[0, 1])
            metrics["tfidf_minkowski_distance"] = float(pairwise_distances(dense_matrix, metric="minkowski")[0, 1])

            binary_presence_matrix = (dense_matrix > 0).astype(bool)

            if binary_presence_matrix[0].any() or binary_presence_matrix[1].any():
                j_dist = pairwise_distances(binary_presence_matrix, metric="jaccard")[0, 1]
                metrics["tfidf_jaccard_distance"] = float(j_dist) if not math.isnan(j_dist) else 1.0
                metrics["tfidf_hamming_distance"] = float(
                    pairwise_distances(binary_presence_matrix.astype(int), metric="hamming")[0, 1],
                )
            else:
                metrics["tfidf_jaccard_distance"] = 0.0
                metrics["tfidf_hamming_distance"] = 0.0

        except Exception as e:
            logging.debug(
                f"Error calculating TF-IDF metrics for '{text1[:50]}...' vs '{text2[:50]}...': {e}",
                exc_info=True,
            )
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
    """
    Computes BLEU (Bilingual Evaluation Understudy) score, adapted for similarity.

    Uses NLTK's sentence_bleu. Allows custom preprocessing.
    """

    def __init__(
        self,
        stop_words: set[str],
        lemmatizer: Optional[WordNetLemmatizer],
        smoothing_function: Optional[Callable],
    ) -> None:
        """
        Initialize the BleuScorer with specified preprocessing settings.

        Args:
            stop_words (set[str]): A set of stop words to exclude during preprocessing.
            lemmatizer (Optional[WordNetLemmatizer]): An instance of WordNetLemmatizer for lemmatization, if desired.
            smoothing_function (Optional[Callable]): A smoothing function for BLEU score calculation.
        """
        self.lemmatizer = lemmatizer
        self.frozen_stop_words = frozenset(stop_words)
        self.smoothing = smoothing_function or SmoothingFunction().method1
        logging.debug("BleuScorer initialized.")

    @staticmethod
    @lru_cache(maxsize=1024)
    def _preprocess_bleu_text(
        text: str,
        lemmatizer: Optional[WordNetLemmatizer],
        stop_words: frozenset[str],
    ) -> tuple[str, ...]:
        """Preprocesses text for BLEU: tokenize, lemmatize (optional), filter stopwords."""
        tokens = tokenize_text(text)
        if lemmatizer:
            tokens = lemmatize_tokens(tokens)
        return filter_stopwords(tokens, stop_words)

    def _calculate_bleu(
        self,
        ref_tokens_list: list[list[str]],
        hyp_tokens: list[str],
        weights: tuple[float, ...],
    ) -> float:
        """Calculate the BLEU score using NLTK, with error handling."""
        if not hyp_tokens or not any(ref_tokens_list):
            return 0.0
        try:
            return sentence_bleu(
                references=ref_tokens_list,
                hypothesis=hyp_tokens,
                weights=weights,
                smoothing_function=self.smoothing,
            )
        except ZeroDivisionError:
            logging.exception(
                "BLEU calculation resulted in ZeroDivisionError (likely short hypothesis/reference or no overlap).",
            )
            return 0.0
        except Exception:
            logging.exception("Unexpected error during NLTK sentence_bleu calculation.")
            return 0.0

    def score_all_ngrams(self, references: Union[str, Sequence[str]], hypothesis: str, max_n: int = 4) -> BleuResult:
        """
        Compute cumulative BLEU scores from BLEU-1 to BLEU-max_n.

        Args:
            references (Union[str, Sequence[str]]): The reference text(s).
            hypothesis (str): The hypothesis text.
            max_n (int): The maximum n-gram order to compute.

        Returns:
            BleuResult: An object containing the BLEU scores.
        """
        ref_list = [references] if isinstance(references, str) else list(references)
        if not hypothesis.strip() or not ref_list or not any(r.strip() for r in ref_list):
            return BleuResult(score=0.0, cumulative_ngram_scores=dict.fromkeys(range(1, max_n + 1), 0.0))

        hyp_tokens = list(
            BleuScorer._preprocess_bleu_text(hypothesis, self.lemmatizer, self.frozen_stop_words),
        )
        ref_tokens_list_processed = [
            list(BleuScorer._preprocess_bleu_text(ref, self.lemmatizer, self.frozen_stop_words)) for ref in ref_list
        ]

        ref_tokens_list_valid = [r_list for r_list in ref_tokens_list_processed if r_list]
        if not ref_tokens_list_valid:
            return BleuResult(score=0.0, cumulative_ngram_scores=dict.fromkeys(range(1, max_n + 1), 0.0))

        cumulative_scores: dict[int, float] = {}
        for n_val in range(1, max_n + 1):
            current_weights = tuple(1.0 / n_val for _ in range(n_val))
            ngram_score = self._calculate_bleu(ref_tokens_list_valid, hyp_tokens, current_weights)
            cumulative_scores[n_val] = ngram_score

        overall_bleu_score = cumulative_scores.get(max_n, 0.0)
        return BleuResult(score=overall_bleu_score, cumulative_ngram_scores=cumulative_scores)


# --- BM25 Calculation Wrapper ---
def calculate_bm25(reference: str, hypothesis: str) -> Optional[float]:
    """
    Calculate BM25 relevance score between a reference (document) and a hypothesis (query).

    Args:
        reference (str): The reference text.
        hypothesis (str): The hypothesis text.

    Returns:
        Optional[float]: The BM25 score, or None if unavailable/error.
    """
    if not _bm25_available:
        return None
    try:
        tokenized_corpus: list[list[str]] = [list(tokenize_text(reference))]
        tokenized_query: list[str] = list(tokenize_text(hypothesis))

        if not tokenized_corpus[0] or not tokenized_query:
            return 0.0

        bm25_calculator = BM25(tokenized_corpus)
        doc_scores = bm25_calculator.get_scores(tokenized_query)
        return float(doc_scores[0]) if doc_scores is not None and len(doc_scores) > 0 else 0.0
    except Exception:
        logging.exception(f"BM25 calculation failed for '{reference[:50]}...' vs '{hypothesis[:50]}...'")
        return None


class SimilarityCalculator:
    """
    Orchestrates the calculation of a comprehensive set of text similarity metrics.

    Initializes and uses specialized calculators (TFIDFCalculator, BleuScorer) and
    various direct similarity functions. Supports parallel processing for multiple pairs.
    """

    def __init__(self, config: Optional[SimilarityCalculatorConfig] = None) -> None:
        """
        Initialize a SimilarityCalculator instance with the provided configuration.

        Args:
            config (Optional[SimilarityCalculatorConfig]): The configuration object to use.
            If None, uses the default config.
        """
        logging.info("Initializing SimilarityCalculator...")
        cfg = config or SimilarityCalculatorConfig()

        self.use_lemmatization = cfg.use_lemmatization
        self.use_stopwords = cfg.use_stopwords

        _provided_sw_list = cfg.custom_stop_words if cfg.custom_stop_words is not None else []
        self.stop_words: set[str] = (
            set(_provided_sw_list) if cfg.custom_stop_words is not None else get_default_stopwords()
        )

        _ensure_nltk_data("punkt")
        if self.use_stopwords:
            _ensure_nltk_data("stopwords")
        if self.use_lemmatization:
            _ensure_nltk_data("wordnet")
            _ensure_nltk_data("omw-1.4")

        self.lemmatizer = get_default_lemmatizer() if self.use_lemmatization else None

        self.bleu_scorer = BleuScorer(
            stop_words=self.stop_words,
            lemmatizer=self.lemmatizer,
            smoothing_function=None,
        )
        self.tfidf_calculator = TFIDFCalculator(
            use_lemmatization=self.use_lemmatization,
            use_stopwords=self.use_stopwords,
            stop_words=self.stop_words,
            tfidf_config=cfg.tfidf_config,
        )
        logging.debug("SimilarityCalculator initialized successfully with config: %s", cfg.model_dump_json())

    def calculate_single_pair(self, text1: str, text2: str) -> SimilarityMetrics:
        """
        Calculate all configured similarity metrics for a single pair of texts.

        Args:
            text1 (str): The first text string.
            text2 (str): The second text string.

        Returns:
            SimilarityMetrics: A Pydantic model instance containing the calculated scores.
        """
        if not isinstance(text1, str) or not isinstance(text2, str):
            logging.warning("Invalid input types for similarity calculation. Both texts must be strings.")
            return SimilarityMetrics()

        raw_results: dict[str, Optional[float]] = {}
        s1_lower, s2_lower = text1.lower(), text2.lower()

        # --- 1. Basic String / Sequence Metrics ---
        try:
            seq_matcher = difflib.SequenceMatcher(None, s1_lower, s2_lower, autojunk=False)
            raw_results["ratio"] = seq_matcher.ratio()
            raw_results["normalized_levenshtein"] = NORMALIZED_LEVENSHTEIN.similarity(s1_lower, s2_lower)
            raw_results["jaro_winkler"] = JARO_WINKLER.similarity(s1_lower, s2_lower)
            raw_results["metric_lcs_similarity"] = 1.0 - METRIC_LCS.distance(
                s1_lower,
                s2_lower,
            )
            raw_results["qgram2_distance"] = QGRAM_2.distance(s1_lower, s2_lower)
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
            raw_results["rfuzz_wratio"] = rapidfuzz_fuzz.WRatio(s1_lower, s2_lower) / 100.0
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
            bleu_result_model = self.bleu_scorer.score_all_ngrams(text1, text2)
            raw_results["bleu_score"] = bleu_result_model.score
        except Exception:
            logging.debug("Error calculating BLEU score.", exc_info=True)

        # --- 5. BM25 Score ---
        raw_results["bm25"] = calculate_bm25(text1, text2)

        # --- 6. TF-IDF Metrics ---
        raw_results.update(self.tfidf_calculator.calculate_metrics_pairwise(text1, text2))

        final_results_cleaned = {
            k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in raw_results.items()
        }
        return SimilarityMetrics(**final_results_cleaned)

    def calculate_multiple_pairs(
        self,
        text_pairs: Iterable[tuple[str, str]],
        max_workers: Optional[int] = None,
    ) -> list[SimilarityMetrics]:
        """
        Calculate similarity metrics for multiple text pairs in parallel using ProcessPoolExecutor.

        Args:
            text_pairs (Iterable[tuple[str, str]]): An iterable of (text1, text2) tuples.
            max_workers (Optional[int]): Maximum number of worker processes. Defaults to system's CPU count.

        Returns:
            list[SimilarityMetrics]: A list of SimilarityMetrics model instances, in the same order as input pairs.
        """
        text_pairs_list = list(text_pairs)
        if not text_pairs_list:
            return []

        default_tfidf_config_for_worker = TfidfConfig()
        vectorizer_params = self.tfidf_calculator.vectorizer.get_params(deep=False)

        picklable_config = SimilarityCalculatorConfig(
            use_lemmatization=self.use_lemmatization,
            use_stopwords=self.use_stopwords,
            custom_stop_words=list(self.stop_words),
            tfidf_config=TfidfConfig(
                token_pattern=vectorizer_params.get("token_pattern") or default_tfidf_config_for_worker.token_pattern,
                ngram_range=vectorizer_params.get("ngram_range", default_tfidf_config_for_worker.ngram_range),
                max_df=vectorizer_params.get("max_df", default_tfidf_config_for_worker.max_df),
                min_df=vectorizer_params.get("min_df", default_tfidf_config_for_worker.min_df),
            ),
        )
        picklable_config_dict = picklable_config.model_dump()

        futures_map: dict[Any, int] = {}
        results_unordered: dict[int, SimilarityMetrics] = {}

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                logging.info(
                    f"Submitting {len(text_pairs_list)} tasks to ProcessPoolExecutor with {max_workers} worker(s).",
                )
                for i, (text1, text2) in enumerate(text_pairs_list):
                    future = executor.submit(_worker_calculate_single_pair, picklable_config_dict, text1, text2)
                    futures_map[future] = i

                for future in as_completed(futures_map):
                    index = futures_map[future]
                    try:
                        result_model = future.result()
                        results_unordered[index] = result_model
                    except Exception:
                        logging.exception(f"Worker process error for pair index {index}")
                        results_unordered[index] = SimilarityMetrics()
        except Exception:
            logging.exception("Error occurred in ProcessPoolExecutor management")
            for i in range(len(text_pairs_list)):
                if i not in results_unordered:
                    results_unordered[i] = SimilarityMetrics()

        ordered_results = [results_unordered.get(i, SimilarityMetrics()) for i in range(len(text_pairs_list))]
        logging.info(f"Finished processing {len(ordered_results)} pairs in parallel.")
        return ordered_results


# --- Worker Function for Parallel Execution ---
def _worker_calculate_single_pair(config_dict: dict[str, Any], text1: str, text2: str) -> SimilarityMetrics:
    """
    Worker function executed by each process in the ProcessPoolExecutor.

    It re-initializes a SimilarityCalculator using the provided configuration dictionary
    and calculates metrics for a single text pair.

    Args:
        config_dict (dict[str, Any]): A dictionary representation of SimilarityCalculatorConfig.
        text1 (str): The first text string.
        text2 (str): The second text string.

    Returns:
        SimilarityMetrics: A SimilarityMetrics Pydantic model instance.
    """
    try:
        worker_sim_config = SimilarityCalculatorConfig(**config_dict)
    except Exception:
        logging.exception(
            "Worker failed to reconstruct SimilarityCalculatorConfig from dict. Using default config.",
        )
        worker_sim_config = SimilarityCalculatorConfig()

    calculator = SimilarityCalculator(config=worker_sim_config)
    return calculator.calculate_single_pair(text1, text2)


# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(processName)s - %(module)s - %(message)s",
    )

    main_app_config = SimilarityCalculatorConfig(
        use_lemmatization=True,
        use_stopwords=True,
        custom_stop_words=["example", "custom"],
        tfidf_config=TfidfConfig(min_df=1, ngram_range=(1, 1)),
    )
    calculator = SimilarityCalculator(config=main_app_config)

    original_text = "The quick brown fox jumps over the lazy dog, an iconic pangram."
    compare_text_similar = "A fast brown fox leaped over a sleepy dog; this is a well-known sentence."
    compare_text_different = "This sentence is completely unrelated and discusses apples and oranges."
    empty_text = ""

    print("\n--- Single Pair Calculation Example ---")
    metrics1 = calculator.calculate_single_pair(original_text, compare_text_similar)
    print("\nSimilarity (Original vs. Similar):")
    for k, v_metric in metrics1.model_dump(exclude_none=True, exclude_defaults=True).items():
        print(f"  {k}: {v_metric:.4f}" if isinstance(v_metric, float) else f"  {k}: {v_metric}")

    metrics2 = calculator.calculate_single_pair(original_text, compare_text_different)
    print("\nSimilarity (Original vs. Different) (Key Metrics):")
    print(f"  Ratio: {metrics2.ratio or 0.0:.4f}")
    print(f"  BLEU Score: {metrics2.bleu_score or 0.0:.4f}")
    print(f"  TF-IDF Cosine Similarity: {metrics2.tfidf_cosine_similarity or 0.0:.4f}")

    metrics_empty = calculator.calculate_single_pair(original_text, empty_text)
    print("\nSimilarity (Original vs. Empty Text) (Key Metrics):")
    print(f"  Ratio: {metrics_empty.ratio or 0.0:.4f}")
    print(f"  BLEU Score: {metrics_empty.bleu_score or 0.0:.4f}")
    print(f"  TF-IDF Cosine Similarity: {metrics_empty.tfidf_cosine_similarity or 0.0:.4f}")

    print("\n--- Parallel Multi-Pair Calculation Example ---")
    text_pairs_for_parallel = [
        (original_text, compare_text_similar),
        ("The cat sat on the mat.", "A feline was resting upon a rug."),
        (original_text, compare_text_different),
        ("This is a short example text.", "Another short one for testing."),
        (original_text, empty_text),
        ("Final test pair, quite distinct.", "The beginning of something new perhaps."),
    ]

    num_workers = min(4, os.cpu_count() or 1) if os.cpu_count() else 2

    parallel_results_list = calculator.calculate_multiple_pairs(text_pairs_for_parallel, max_workers=num_workers)

    print(f"\nParallel Calculation Results ({len(parallel_results_list)} pairs processed):")
    for i, result_model_instance in enumerate(parallel_results_list):
        p1_text, p2_text = text_pairs_for_parallel[i]
        print(f"\nPair {i + 1}: '{p1_text[:30]}...' vs '{p2_text[:30]}...'")
        print(f"  Ratio: {result_model_instance.ratio or 0.0:.4f}")
        print(f"  BLEU Score: {result_model_instance.bleu_score or 0.0:.4f}")
        print(f"  TF-IDF Cosine Similarity: {result_model_instance.tfidf_cosine_similarity or 0.0:.4f}")

    print("\n--- Example Script Finished ---")
