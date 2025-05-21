"""Module for matching keywords from one paragraph to another and scoring.

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
    # No immediate raise here; allow the script to run with standard logging if rich is missing
    # The example usage section will handle the ImportError for rich.console if needed


# --- Global Resources ---
_LEMMA_INIT_FAILED = False
_GLOBAL_LEMMATIZER: Optional[WordNetLemmatizer] = None
try:
    _GLOBAL_LEMMATIZER = WordNetLemmatizer()
    _ = _GLOBAL_LEMMATIZER.lemmatize("tests")  # Check if it works
except LookupError as e:
    print(f"[ERROR] NLTK LookupError initializing WordNetLemmatizer: {e}")
    print("        Ensure 'wordnet' and 'omw-1.4' NLTK data are downloaded.")
    _LEMMA_INIT_FAILED = True
except Exception as e:  # pylint: disable=broad-except
    print(f"[ERROR] Unexpected error initializing WordNetLemmatizer: {e}")
    _LEMMA_INIT_FAILED = True

PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
DEFAULT_ALLOWED_POS_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}
logger = logging.getLogger(__name__)
GOOD_KEYWORD_COVERAGE = 0.5
GOOD_VOCAB_COSINE = 0.5
BAD_VOCAB_COSINE = 0.1


class KeywordMatcher:
    """Matches keywords from paragraph A in paragraph B and provides scores.

    Calculates two scores:
    1. Keyword Coverage: Proportion of unique keywords extracted from A
       (based on config) found in processed B.
    2. Vocabulary Cosine Similarity: Cosine similarity of binary vectors
       representing the presence/absence of non-stop words in A and B.

    **Note:** Requires NLTK data ('punkt', 'stopwords', 'wordnet', 'omw-1.4',
    'averaged_perceptron_tagger') to be pre-downloaded. See module docstring.
    """

    def __init__(
        self,
        config: Optional[KeywordMatcherConfig] = None,
    ) -> None:
        """Initialize the KeywordMatcher.

        Args:
            config: Configuration settings for the matcher. If None, defaults are used.

        """
        if not _rich_available:
            logger.warning("Module 'rich' not found. Logging will use standard format.")

        logger.warning(
            "[bold yellow]Initializing KeywordMatcher. Ensure required NLTK data is downloaded![/bold yellow]",
        )

        self.config = config or KeywordMatcherConfig()

        actual_allowed_pos_tags = self.config.allowed_pos_tags
        if self.config.use_pos_tagging and not self.config.allowed_pos_tags:
            logger.info("POS tagging enabled, using default allowed_pos_tags (nouns & adjectives).")
            actual_allowed_pos_tags = DEFAULT_ALLOWED_POS_TAGS
        self.resolved_allowed_pos_tags: Optional[set[str]] = actual_allowed_pos_tags

        self.effective_use_lemmatization = self.config.use_lemmatization and not _LEMMA_INIT_FAILED
        if self.config.use_lemmatization and _LEMMA_INIT_FAILED:
            logger.error(
                "Lemmatization requested, but lemmatizer failed to initialize. "
                "Lemmatization is DISABLED for coverage score.",
            )

        try:
            nltk_stopwords = set(stopwords.words("english"))
            self._stopwords_loaded = True
        except LookupError:
            logger.exception(
                "Failed to load NLTK stopwords (data likely missing 'stopwords'). Proceeding without NLTK stopwords.",
            )
            nltk_stopwords = set()
            self._stopwords_loaded = False
        except Exception:  # pylint: disable=broad-except
            logger.exception("An unexpected error occurred loading stopwords:")
            nltk_stopwords = set()
            self._stopwords_loaded = False

        self.stop_words = nltk_stopwords.union(self.config.custom_stop_words or set())
        if not self.stop_words:
            logger.warning("No stopwords defined.")

        log_message = (
            f"KeywordMatcher initialized. "
            f"Lemmatization (for coverage): {self.effective_use_lemmatization}, "
            f"POS Tagging (for coverage): {self.config.use_pos_tagging}, "
            f"NLTK Stopwords loaded: {self._stopwords_loaded}. "
        )
        if self.config.use_pos_tagging:
            log_message += f"Allowed POS: {self.resolved_allowed_pos_tags}"
        logger.info(log_message)

    @lru_cache(maxsize=128)
    def _preprocess_text(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            logger.debug("Preprocessing empty or invalid text. Returning empty list.")
            return []
        try:
            cleaned_text = text.lower().translate(PUNCTUATION_TABLE)
            tokens = word_tokenize(cleaned_text)
            processed_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
            logger.debug(f"Preprocessing result for '{text[:30]}...': {len(processed_tokens)} tokens.")
            return processed_tokens
        except LookupError:
            logger.exception("NLTK LookupError (likely 'punkt' missing). Returning empty token list.")
            return []
        except Exception:  # pylint: disable=broad-except
            logger.exception(f"Unexpected error during basic preprocessing of text: '{text[:50]}...'")
            return []

    @lru_cache(maxsize=128)
    def _normalize_tokens(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        if not self.effective_use_lemmatization or _GLOBAL_LEMMATIZER is None:
            return tokens
        try:
            normalized = tuple(_GLOBAL_LEMMATIZER.lemmatize(token) for token in tokens)
            logger.debug(f"Lemmatized {len(tokens)} tokens.")
            return normalized
        except LookupError:
            logger.exception("NLTK LookupError (likely 'wordnet'/'omw-1.4' missing). Un-normalized tokens returned.")
            return tokens
        except Exception:  # pylint: disable=broad-except
            logger.exception(f"Unexpected error during lemmatization of {len(tokens)} tokens.")
            return tokens

    @lru_cache(maxsize=128)
    def _get_pos_tags(self, tokens: tuple[str, ...]) -> list[tuple[str, str]]:
        if not tokens:
            return []
        try:
            tags = pos_tag(list(tokens))
            logger.debug(f"POS tagged {len(tokens)} tokens.")
            return tags
        except LookupError:
            logger.exception("NLTK LookupError (likely 'averaged_perceptron_tagger' missing). Cannot POS tag.")
            return []
        except Exception:  # pylint: disable=broad-except
            logger.exception(f"Unexpected error during POS tagging of {len(tokens)} tokens.")
            return []

    def _extract_keywords_from_a(self, paragraph_a: str) -> set[str]:
        processed_tokens = self._preprocess_text(paragraph_a)
        if not processed_tokens:
            logger.warning("Keyword extraction failed: Preprocessing returned no tokens for Paragraph A.")
            return set()

        processed_tokens_tuple = tuple(processed_tokens)
        keywords: set[str] = set()

        if self.config.use_pos_tagging:
            if not self.resolved_allowed_pos_tags:
                logger.error("POS tagging requested but no allowed tags set. Cannot extract POS-based keywords.")
            else:
                tagged_tokens = self._get_pos_tags(processed_tokens_tuple)
                if tagged_tokens:
                    pos_filtered_tokens = [word for word, tag in tagged_tokens if tag in self.resolved_allowed_pos_tags]
                    if pos_filtered_tokens:
                        keywords = set(self._normalize_tokens(tuple(pos_filtered_tokens)))
                    else:
                        logger.warning("No tokens matched allowed POS tags in Paragraph A.")
                else:
                    logger.error("Keyword extraction failed: POS tagging returned no results for Paragraph A.")
        else:
            keywords = set(self._normalize_tokens(processed_tokens_tuple))

        if not keywords:
            logger.warning("No keywords extracted from Paragraph A after processing.")
        else:
            logger.debug(f"Extracted {len(keywords)} keywords from Paragraph A.")
        return keywords

    def find_matches_and_score(self, paragraph_a: str, paragraph_b: str) -> MatcherScores:
        """Find keywords and calculate keyword coverage and vocabulary cosine scores."""
        logger.info("Attempting to find matches and score from Paragraph A in Paragraph B.")
        logger.debug(f"Paragraph A (start): '{paragraph_a[:60]}...'")
        logger.debug(f"Paragraph B (start): '{paragraph_b[:60]}...'")

        vocab_cosine_score = self._calculate_vocab_cosine(paragraph_a, paragraph_b)
        keywords_a_set = self._extract_keywords_from_a(paragraph_a)
        keywords_from_a_count_val = len(keywords_a_set)

        # Default values for coverage components if no keywords from A
        matched_keywords_list: list[str] = []
        matched_keyword_count_val: int = 0
        keyword_coverage_score_val: float = 0.0

        if not keywords_a_set:
            logger.warning("Could not extract any keywords from Paragraph A for coverage score. Coverage is 0.")
        else:
            coverage_result_dict = self._calculate_keyword_coverage_components(keywords_a_set, paragraph_b)
            matched_keywords_list = coverage_result_dict["matched_keywords"]
            matched_keyword_count_val = coverage_result_dict["matched_keyword_count"]
            keyword_coverage_score_val = coverage_result_dict["keyword_coverage_score"]

        # Construct the nested KeywordMatcherScore model
        scores_component = KeywordMatcherScore(
            keywords_from_a_count=keywords_from_a_count_val,
            matched_keyword_count=matched_keyword_count_val,
            keyword_coverage_score=keyword_coverage_score_val,
            vocabulary_cosine_similarity=vocab_cosine_score,
        )

        # Construct the main MatcherScores model
        return MatcherScores(
            matched_keywords=matched_keywords_list,
            keywords_matcher_result=scores_component,  # Assign the nested model
        )

    def _calculate_vocab_cosine(self, paragraph_a: str, paragraph_b: str) -> float:
        processed_tokens_a_list = self._preprocess_text(paragraph_a)
        processed_tokens_b_list = self._preprocess_text(paragraph_b)

        if not processed_tokens_a_list and not processed_tokens_b_list:
            logger.warning("Both paragraphs empty after preprocessing. Vocab cosine score is 0.")
            return 0.0

        set_a, set_b = set(processed_tokens_a_list), set(processed_tokens_b_list)
        r_vector = set_a.union(set_b)
        if not r_vector:
            logger.debug("No common vocabulary for vocab cosine.")
            return 0.0

        v_a = [1 if word in set_a else 0 for word in r_vector]
        v_b = [1 if word in set_b else 0 for word in r_vector]

        dot_product = sum(a * b for a, b in zip(v_a, v_b, strict=False))
        norm_a = sum(v_a) ** 0.5
        norm_b = sum(v_b) ** 0.5
        denominator = norm_a * norm_b

        return dot_product / denominator if denominator > 0 else 0.0

    def _calculate_keyword_coverage_components(
        self,
        keywords_a_set: set[str],
        paragraph_b: str,
    ) -> dict[str, Any]:
        processed_tokens_b_list = self._preprocess_text(paragraph_b)
        if not processed_tokens_b_list:
            logger.warning("Paragraph B empty after preprocessing. Coverage score is 0.")
            return {"matched_keywords": [], "matched_keyword_count": 0, "keyword_coverage_score": 0.0}

        normalized_tokens_b_set = set(self._normalize_tokens(tuple(processed_tokens_b_list)))
        matched_keywords_set = keywords_a_set.intersection(normalized_tokens_b_set)
        num_matches = len(matched_keywords_set)
        coverage_score = num_matches / len(keywords_a_set) if keywords_a_set else 0.0  # Denominator check

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
        logger.info("Keyword Matching Example [bold green](Rich logging enabled)[/bold green]")
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
        # Access scores via the nested model
        scores = results.keywords_matcher_result
        logger.info(f"  Total Keywords from A (for coverage): {scores.keywords_from_a_count}")
        logger.info(f"  Matched Keywords Count: {scores.matched_keyword_count}")

        cov_score = scores.keyword_coverage_score
        cov_color = "green" if cov_score > GOOD_KEYWORD_COVERAGE else "yellow" if cov_score > 0 else "red"
        logger.info(f"  Keyword Coverage Score: [{cov_color}]{cov_score:.4f}[/{cov_color}]")

        cos_score = scores.vocabulary_cosine_similarity
        cos_color = "green" if cos_score > GOOD_VOCAB_COSINE else "yellow" if cos_score > BAD_VOCAB_COSINE else "red"
        logger.info(f"  Vocabulary Cosine Score: [{cos_color}]{cos_score:.4f}[/{cos_color}]")

        # matched_keywords is still directly on MatcherScores
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
