"""
Module for matching keywords from one paragraph to another and scoring.

Provides a KeywordMatcher class with configurable preprocessing and
keyword extraction methods (including POS tagging). Calculates two scores:
1. Keyword Coverage: Proportion of keywords from A found in B.
2. Vocabulary Cosine Similarity: Cosine similarity based on shared non-stop words.
Uses rich logging.

**IMPORTANT:** This version assumes necessary NLTK data ('punkt', 'stopwords',
'wordnet', 'omw-1.4', 'averaged_perceptron_tagger') has been manually
downloaded beforehand. Run the following in your Python environment if needed:

>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
>>> nltk.download('averaged_perceptron_tagger')
"""

from __future__ import annotations

import logging
import string
from functools import lru_cache
from typing import Any, Optional

from app_types import KeywordMatcherConfig, KeywordMatcherScore, MatcherScores

# Attempt to import necessary libraries and provide guidance
try:
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError as e:
    missing_lib = "nltk"
    error_message = f"Missing {missing_lib}. Please install it (e.g., `pip install {missing_lib}`). Original error: {e}"
    raise ImportError(error_message) from e


# Attempt to import rich
try:
    from rich.logging import RichHandler

    _rich_available = True
except ImportError:
    _rich_available = False


# --- Global Resources ---
_LEMMA_INIT_FAILED = False
_GLOBAL_LEMMATIZER: Optional[WordNetLemmatizer] = None
try:
    _GLOBAL_LEMMATIZER = WordNetLemmatizer()
    _ = _GLOBAL_LEMMATIZER.lemmatize("tests")
except LookupError as e:
    print(f"[ERROR] NLTK LookupError initializing WordNetLemmatizer: {e}")
    print("        Ensure 'wordnet' and 'omw-1.4' NLTK data are downloaded.")
    _LEMMA_INIT_FAILED = True
except Exception as e:
    print(f"[ERROR] Unexpected error initializing WordNetLemmatizer: {e}")
    _LEMMA_INIT_FAILED = True

PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
DEFAULT_ALLOWED_POS_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}
logger = logging.getLogger(__name__)
GOOD_KEYWORD_COVERAGE = 0.5
GOOD_VOCAB_COSINE = 0.5
BAD_VOCAB_COSINE = 0.1


class KeywordMatcher:
    """
    Matches keywords from paragraph A in paragraph B and provides scores.

    Calculates two primary scores:
    1.  **Keyword Coverage**: The proportion of unique keywords extracted from a primary
        text (Paragraph A) that are found in a secondary text (Paragraph B). Keyword
        extraction can be configured to use Part-of-Speech (POS) tagging and/or
        lemmatization.
    2.  **Vocabulary Cosine Similarity**: The cosine similarity between the binary
        vectors representing the presence or absence of non-stopword vocabulary
        terms in Paragraph A and Paragraph B.

    Preprocessing steps include lowercasing, punctuation removal, tokenization,
    stopword removal, and optional lemmatization.
    """

    def __init__(
        self,
        config: Optional[KeywordMatcherConfig] = None,
    ) -> None:
        """
        Initialize the KeywordMatcher with specified or default configuration.

        Args:
            config (Optional[KeywordMatcherConfig]): Configuration settings for the matcher.
                If None, `KeywordMatcherConfig` defaults are used. This config controls
                lemmatization, POS tagging, allowed POS tags, and custom stopwords.
        """
        if not _rich_available:
            logger.warning("Module 'rich' for enhanced logging not found. Logging will use standard format.")

        logger.warning(
            "[bold yellow]Initializing KeywordMatcher. Ensure required NLTK data "
            "(punkt, stopwords, wordnet, omw-1.4, averaged_perceptron_tagger) is downloaded![/bold yellow]",
        )

        self.config = config or KeywordMatcherConfig()

        actual_allowed_pos_tags = self.config.allowed_pos_tags
        if self.config.use_pos_tagging and not self.config.allowed_pos_tags:
            logger.info(
                "POS tagging is enabled, but no specific POS tags were provided. "
                "Using default allowed_pos_tags (common nouns & adjectives).",
            )
            actual_allowed_pos_tags = DEFAULT_ALLOWED_POS_TAGS
        self.resolved_allowed_pos_tags: Optional[set[str]] = actual_allowed_pos_tags

        self.effective_use_lemmatization = self.config.use_lemmatization and not _LEMMA_INIT_FAILED
        if self.config.use_lemmatization and _LEMMA_INIT_FAILED:
            logger.error(
                "Lemmatization was requested in config, but the global WordNetLemmatizer failed to initialize "
                "(likely due to missing NLTK data: 'wordnet', 'omw-1.4'). "
                "Lemmatization will be DISABLED for keyword coverage scoring.",
            )

        try:
            nltk_stopwords = set(stopwords.words("english"))
            self._stopwords_loaded = True
        except LookupError:
            logger.exception(
                "Failed to load NLTK English stopwords (NLTK data 'stopwords' likely missing). "
                "Proceeding with only custom stopwords (if any) or no stopwords.",
            )
            nltk_stopwords = set()
            self._stopwords_loaded = False
        except Exception:
            logger.exception("An unexpected error occurred while loading NLTK stopwords.")
            nltk_stopwords = set()
            self._stopwords_loaded = False

        self.stop_words = nltk_stopwords.union(self.config.custom_stop_words or set())
        if not self.stop_words and self._stopwords_loaded:
            logger.info("NLTK stopwords loaded, but custom stop words resulted in an empty final set. This is unusual.")
        elif not self.stop_words and not self._stopwords_loaded:
            logger.warning(
                "No stopwords are defined (neither NLTK default nor custom). Stopword filtering will have no effect.",
            )

        log_message = (
            f"KeywordMatcher initialized with settings: "
            f"Lemmatization (for coverage score): {self.effective_use_lemmatization}, "
            f"POS Tagging (for keyword extraction): {self.config.use_pos_tagging}, "
            f"NLTK Stopwords Loaded: {self._stopwords_loaded}."
        )
        if self.config.use_pos_tagging:
            log_message += f" Allowed POS Tags: {self.resolved_allowed_pos_tags or 'None (defaulting or error)'}"
        logger.info(log_message)

    @staticmethod
    @lru_cache(maxsize=128)
    def _preprocess_text(text: str, stop_words_set: frozenset[str]) -> list[str]:
        """
        Perform basic preprocessing on a text string.

        Steps:
        1. Converts text to lowercase.
        2. Removes punctuation using `PUNCTUATION_TABLE`.
        3. Tokenizes the text using `nltk.word_tokenize`.
        4. Filters out tokens that are not alphanumeric or are in the `stop_words_set`.

        Args:
            text (str): The input text string.
            stop_words_set (frozenset[str]): The set of stopwords to use.

        Returns:
            list[str]: A list of processed (cleaned, tokenized, filtered) tokens.
                       Returns an empty list for invalid input or if NLTK data (e.g., 'punkt') is missing.
        """
        if not isinstance(text, str) or not text.strip():
            logger.debug("Preprocessing received empty or invalid text. Returning empty token list.")
            return []
        try:
            cleaned_text = text.lower().translate(PUNCTUATION_TABLE)
            tokens = word_tokenize(cleaned_text)
            processed_tokens = [token for token in tokens if token.isalnum() and token not in stop_words_set]
            logger.debug(f"Preprocessing result for text '{text[:30]}...': {len(processed_tokens)} tokens returned.")
            return processed_tokens
        except LookupError:
            logger.exception(
                f"NLTK LookupError during preprocessing (likely 'punkt' data missing for word_tokenize). "
                f"Text: '{text[:50]}...'",
            )
            return []
        except Exception:
            logger.exception(f"Unexpected error during basic preprocessing of text: '{text[:50]}...'.")
            return []

    @staticmethod
    @lru_cache(maxsize=128)
    def _normalize_tokens(
        tokens: tuple[str, ...],
        *,
        use_lemmatization: bool,
        lemmatizer: Optional[WordNetLemmatizer],
    ) -> tuple[str, ...]:
        """
        Normalize a tuple of tokens, primarily by lemmatization if enabled and available.

        Args:
            tokens (tuple[str, ...]): A tuple of string tokens.
            use_lemmatization (bool): Flag indicating if lemmatization should be applied.
            lemmatizer (Optional[WordNetLemmatizer]): The lemmatizer instance to use.

        Returns:
            tuple[str, ...]: A tuple of normalized (lemmatized) tokens. Returns the original
                             tokens if lemmatization is disabled or the lemmatizer is not functional.
        """
        if not use_lemmatization or lemmatizer is None:
            return tokens
        try:
            normalized = tuple(lemmatizer.lemmatize(token) for token in tokens)
            logger.debug(f"Lemmatized {len(tokens)} tokens successfully.")
            return normalized
        except LookupError:
            logger.exception(
                "NLTK LookupError during lemmatization (likely 'wordnet' or "
                f"'omw-1.4' data missing). Tokens: {tokens[:10]}",
            )
            return tokens
        except Exception:
            logger.exception(f"Unexpected error during lemmatization of {len(tokens)} tokens.")
            return tokens

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_pos_tags(tokens: tuple[str, ...]) -> list[tuple[str, str]]:
        """
        Perform Part-of-Speech (POS) tagging on a tuple of tokens.

        Args:
            tokens (tuple[str, ...]): A tuple of string tokens.

        Returns:
            list[tuple[str, str]]: A list of (token, POS_tag) tuples. Returns an empty list
                                   if input tokens are empty or if POS tagging fails (e.g.,
                                   NLTK 'averaged_perceptron_tagger' data missing).
        """
        if not tokens:
            return []
        try:
            tags = pos_tag(list(tokens))
            logger.debug(f"POS tagged {len(tokens)} tokens successfully.")
            return tags
        except LookupError:
            logger.exception(
                f"NLTK LookupError during POS tagging (likely 'averaged_perceptron_tagger' data missing). "
                f"Tokens: {tokens[:10]}",
            )
            return []
        except Exception:
            logger.exception(f"Unexpected error during POS tagging of {len(tokens)} tokens.")
            return []

    def _extract_keywords_from_a(self, paragraph_a: str) -> set[str]:
        """
        Extract a set of unique keywords from Paragraph A based on the matcher's configuration.

        The process involves:
        1. Basic preprocessing (`KeywordMatcher._preprocess_text`).
        2. If POS tagging is enabled: POS tag tokens (`KeywordMatcher._get_pos_tags`),
           filter by `resolved_allowed_pos_tags`.
        3. Normalization (lemmatization if enabled using `KeywordMatcher._normalize_tokens`)
           of the resulting tokens.

        Args:
            paragraph_a (str): The text of Paragraph A from which to extract keywords.

        Returns:
            set[str]: A set of unique extracted keywords. Returns an empty set if no
                      keywords can be extracted (e.g., empty input, processing errors).
        """
        processed_tokens_list = KeywordMatcher._preprocess_text(paragraph_a, frozenset(self.stop_words))
        if not processed_tokens_list:
            logger.warning("Keyword extraction from Paragraph A failed: Preprocessing returned no tokens.")
            return set()

        processed_tokens_tuple = tuple(processed_tokens_list)
        keywords: set[str] = set()

        if self.config.use_pos_tagging:
            if not self.resolved_allowed_pos_tags:
                logger.error(
                    "POS tagging is enabled, but no allowed POS tags are set. "
                    "Cannot extract POS-based keywords from Paragraph A.",
                )
            else:
                tagged_tokens = KeywordMatcher._get_pos_tags(processed_tokens_tuple)
                if tagged_tokens:
                    pos_filtered_tokens = [word for word, tag in tagged_tokens if tag in self.resolved_allowed_pos_tags]
                    if pos_filtered_tokens:
                        keywords = set(
                            KeywordMatcher._normalize_tokens(
                                tuple(pos_filtered_tokens),
                                use_lemmatization=self.effective_use_lemmatization,
                                lemmatizer=_GLOBAL_LEMMATIZER,
                            ),
                        )
                    else:
                        logger.warning("No tokens from Paragraph A matched the allowed POS tags after POS tagging.")
                else:
                    logger.error("Keyword extraction from Paragraph A failed: POS tagging returned no results.")
        else:
            keywords = set(
                KeywordMatcher._normalize_tokens(
                    processed_tokens_tuple,
                    use_lemmatization=self.effective_use_lemmatization,
                    lemmatizer=_GLOBAL_LEMMATIZER,
                ),
            )

        if not keywords:
            logger.warning("No keywords were extracted from Paragraph A after all processing steps.")
        else:
            logger.debug(f"Successfully extracted {len(keywords)} unique keywords from Paragraph A.")
        return keywords

    def find_matches_and_score(self, paragraph_a: str, paragraph_b: str) -> MatcherScores:
        """
        Calculate keyword coverage and vocabulary cosine similarity between two paragraphs.

        Args:
            paragraph_a (str): The primary text from which keywords are extracted (source).
            paragraph_b (str): The secondary text in which to find the keywords (target).

        Returns:
            MatcherScores: A Pydantic model containing:
                - `matched_keywords`: List of keywords from A found in B.
                - `keywords_matcher_result`: A nested model with scores.
        """
        logger.info("Starting keyword matching and scoring for Paragraph A vs. Paragraph B.")
        logger.debug(f"Paragraph A (first 60 chars): '{paragraph_a[:60]}...'")
        logger.debug(f"Paragraph B (first 60 chars): '{paragraph_b[:60]}...'")

        vocab_cosine_score = self._calculate_vocab_cosine(paragraph_a, paragraph_b)

        keywords_a_set = self._extract_keywords_from_a(paragraph_a)
        keywords_from_a_count_val = len(keywords_a_set)

        matched_keywords_list: list[str] = []
        matched_keyword_count_val: int = 0
        keyword_coverage_score_val: float = 0.0

        if not keywords_a_set:
            logger.warning(
                "No keywords extracted from Paragraph A. Keyword coverage score will be 0.",
            )
        else:
            coverage_components = self._calculate_keyword_coverage_components(keywords_a_set, paragraph_b)
            matched_keywords_list = coverage_components["matched_keywords"]
            matched_keyword_count_val = coverage_components["matched_keyword_count"]
            keyword_coverage_score_val = coverage_components["keyword_coverage_score"]

        keyword_scores = KeywordMatcherScore(
            keywords_from_a_count=keywords_from_a_count_val,
            matched_keyword_count=matched_keyword_count_val,
            keyword_coverage_score=keyword_coverage_score_val,
            vocabulary_cosine_similarity=vocab_cosine_score,
        )

        return MatcherScores(
            matched_keywords=matched_keywords_list,
            keywords_matcher_result=keyword_scores,
        )

    def _calculate_vocab_cosine(self, paragraph_a: str, paragraph_b: str) -> float:
        """
        Calculate the cosine similarity between the vocabularies of two paragraphs.

        Vocabularies are derived from preprocessed tokens (lowercase, no punctuation, no stopwords).
        Similarity is based on binary vectors representing word presence in each paragraph's vocabulary.

        Args:
            paragraph_a (str): The first paragraph.
            paragraph_b (str): The second paragraph.

        Returns:
            float: The cosine similarity score (0.0 to 1.0). Returns 0.0 if both paragraphs
                   are empty after preprocessing or if there's no shared vocabulary.
        """
        stop_words_fs = frozenset(self.stop_words)
        processed_tokens_a_list = KeywordMatcher._preprocess_text(paragraph_a, stop_words_fs)
        processed_tokens_b_list = KeywordMatcher._preprocess_text(paragraph_b, stop_words_fs)

        if not processed_tokens_a_list and not processed_tokens_b_list:
            logger.warning("Both paragraphs are empty after preprocessing. Vocabulary cosine similarity is 0.")
            return 0.0

        set_a = set(processed_tokens_a_list)
        set_b = set(processed_tokens_b_list)

        combined_vocabulary = set_a.union(set_b)
        if not combined_vocabulary:
            logger.debug(
                "No combined vocabulary for cosine similarity (texts likely empty after processing).",
            )
            return 0.0

        v_a = [1 if word in set_a else 0 for word in combined_vocabulary]
        v_b = [1 if word in set_b else 0 for word in combined_vocabulary]

        dot_product = sum(a_val * b_val for a_val, b_val in zip(v_a, v_b, strict=False))
        norm_a = sum(val_a**2 for val_a in v_a) ** 0.5
        norm_b = sum(val_b**2 for val_b in v_b) ** 0.5
        denominator = norm_a * norm_b

        return dot_product / denominator if denominator > 0 else 0.0

    def _calculate_keyword_coverage_components(
        self,
        keywords_a_set: set[str],
        paragraph_b: str,
    ) -> dict[str, Any]:
        """
        Calculate keyword coverage by checking how many keywords from Paragraph A are in Paragraph B.

        Args:
            keywords_a_set (set[str]): A set of unique, normalized keywords extracted from Paragraph A.
            paragraph_b (str): The raw text of Paragraph B.

        Returns:
            dict[str, Any]: A dictionary containing matched keywords, count, and score.
        """
        processed_tokens_b_list = KeywordMatcher._preprocess_text(paragraph_b, frozenset(self.stop_words))
        if not processed_tokens_b_list:
            logger.warning("Paragraph B is empty after preprocessing. Keyword coverage score is 0.")
            return {"matched_keywords": [], "matched_keyword_count": 0, "keyword_coverage_score": 0.0}

        normalized_tokens_b_set = set(
            KeywordMatcher._normalize_tokens(
                tuple(processed_tokens_b_list),
                use_lemmatization=self.effective_use_lemmatization,
                lemmatizer=_GLOBAL_LEMMATIZER,
            ),
        )

        matched_keywords_set = keywords_a_set.intersection(normalized_tokens_b_set)
        num_matches = len(matched_keywords_set)

        coverage_score = num_matches / len(keywords_a_set) if keywords_a_set else 0.0

        return {
            "matched_keywords": sorted(matched_keywords_set),
            "matched_keyword_count": num_matches,
            "keyword_coverage_score": coverage_score,
        }


# --- Example Usage ---
if __name__ == "__main__":
    logging.root.handlers.clear()
    LOG_LEVEL = logging.DEBUG
    logging.root.setLevel(LOG_LEVEL)
    rich_console_available = False

    if _rich_available:
        rich_handler = RichHandler(level=LOG_LEVEL, show_path=False, rich_tracebacks=True, markup=True)
        logging.root.addHandler(rich_handler)
        try:
            from rich.console import Console

            console = Console()
            separator = lambda: console.print("-" * 80, style="dim")
            rich_console_available = True
        except ImportError:
            separator = lambda: print("-" * 80)
        logger.info("Keyword Matching Example [bold green](Rich logging)[/bold green]")
    else:
        logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        separator = lambda: print("-" * 80)
        logger.info("Keyword Matching Example (standard logging - install 'rich' for better output)")

    para_a = """
    Natural Language Processing (NLP) is a fascinating subfield of artificial intelligence.
    Key techniques include tokenization, lemmatization, and part-of-speech tagging.
    These methods help computers understand human language. We love NLP.
    """
    para_b = """
    Understanding language with computers often involves NLP methods. For example,
    lemmatization reduces words to their base form. Artificial intelligence
    is advancing rapidly, especially in language analysis. The dog barked.
    """
    para_c = "Astrophysics studies distant galaxies. Minimal overlap is expected."
    para_empty = ""

    def print_match_results(scenario_name: str, results: MatcherScores) -> None:
        """Print the results of keyword matching for a given scenario."""
        logger.info(f"[bold cyan]>>> {scenario_name} Results:[/bold cyan]")
        scores = results.keywords_matcher_result
        logger.info(f"  Total Keywords from A (for coverage): {scores.keywords_from_a_count}")
        logger.info(f"  Matched Keywords Count: {scores.matched_keyword_count}")

        cov_score = scores.keyword_coverage_score
        cov_color = "green" if cov_score > GOOD_KEYWORD_COVERAGE else "yellow" if cov_score > 0 else "red"
        logger.info(f"  Keyword Coverage: [{cov_color}]{cov_score:.4f}[/{cov_color}]")

        cos_score = scores.vocabulary_cosine_similarity
        cos_color = "green" if cos_score > GOOD_VOCAB_COSINE else "yellow" if cos_score > BAD_VOCAB_COSINE else "red"
        logger.info(f"  Vocabulary Cosine Score: [{cos_color}]{cos_score:.4f}[/{cos_color}]")

        if results.matched_keywords:
            logger.info(f"  Matched Keywords (for coverage): {results.matched_keywords}")
        else:
            logger.info("  Matched Keywords (for coverage): None")
        separator()

    matcher_default = KeywordMatcher()
    results1 = matcher_default.find_matches_and_score(para_a, para_b)
    print_match_results("Scenario 1: Default Settings", results1)

    config_pos = KeywordMatcherConfig(use_pos_tagging=True)
    matcher_pos = KeywordMatcher(config=config_pos)
    results2 = matcher_pos.find_matches_and_score(para_a, para_b)
    print_match_results("Scenario 2: Using POS Tagging", results2)

    config_simple = KeywordMatcherConfig(use_lemmatization=False, use_pos_tagging=False)
    matcher_simple = KeywordMatcher(config=config_simple)
    results3 = matcher_simple.find_matches_and_score(para_a, para_b)
    print_match_results("Scenario 3: No Lemmatization or POS", results3)

    results4 = matcher_default.find_matches_and_score(para_a, para_c)
    print_match_results("Scenario 4: Matching Dissimilar Paragraphs", results4)

    results5a = matcher_default.find_matches_and_score(para_a, para_empty)
    print_match_results("Scenario 5a: Matching A vs Empty", results5a)

    results5b = matcher_default.find_matches_and_score(para_empty, para_b)
    print_match_results("Scenario 5b: Matching Empty vs B", results5b)

    logger.info("Keyword Matching Example Finished.")
