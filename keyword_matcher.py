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

    **Note:** This class relies on NLTK for various NLP tasks. Ensure necessary
    NLTK data resources (e.g., 'punkt' for tokenization, 'stopwords',
    'wordnet' & 'omw-1.4' for lemmatization, 'averaged_perceptron_tagger' for POS
    tagging) are pre-downloaded. Refer to the module docstring for download commands.
    """

    def __init__(
        self,
        config: Optional[KeywordMatcherConfig] = None,
    ) -> None:
        """Initialize the KeywordMatcher with specified or default configuration.

        Args:
            config (Optional[KeywordMatcherConfig]): Configuration settings for the matcher.
                If None, `KeywordMatcherConfig` defaults are used. This config controls
                lemmatization, POS tagging, allowed POS tags, and custom stopwords.

        """
        if not _rich_available:  # Check if rich library is available for enhanced logging.
            logger.warning("Module 'rich' for enhanced logging not found. Logging will use standard format.")

        # This warning is important as NLTK data is a common setup issue.
        logger.warning(
            "[bold yellow]Initializing KeywordMatcher. Ensure required NLTK data "
            "(punkt, stopwords, wordnet, omw-1.4, averaged_perceptron_tagger) is downloaded![/bold yellow]",
        )

        self.config = config or KeywordMatcherConfig()  # Use provided config or default.

        # Determine the set of POS tags to be used if POS tagging is enabled.
        # If POS tagging is on but no specific tags are provided, use defaults.
        actual_allowed_pos_tags = self.config.allowed_pos_tags
        if self.config.use_pos_tagging and not self.config.allowed_pos_tags:
            logger.info(
                "POS tagging is enabled, but no specific POS tags were provided. "
                "Using default allowed_pos_tags (common nouns & adjectives).",
            )
            actual_allowed_pos_tags = DEFAULT_ALLOWED_POS_TAGS
        self.resolved_allowed_pos_tags: Optional[set[str]] = actual_allowed_pos_tags

        # Determine if lemmatization can be effectively used, based on config and global lemmatizer status.
        self.effective_use_lemmatization = self.config.use_lemmatization and not _LEMMA_INIT_FAILED
        if self.config.use_lemmatization and _LEMMA_INIT_FAILED:
            logger.error(
                "Lemmatization was requested in config, but the global WordNetLemmatizer failed to initialize "
                "(likely due to missing NLTK data: 'wordnet', 'omw-1.4'). "
                "Lemmatization will be DISABLED for keyword coverage scoring.",
            )

        # Load NLTK's English stopwords and combine with any custom stopwords from config.
        try:
            nltk_stopwords = set(stopwords.words("english"))
            self._stopwords_loaded = True  # Flag indicating NLTK stopwords were loaded.
        except LookupError:  # NLTK's 'stopwords' resource might not be downloaded.
            logger.exception(
                "Failed to load NLTK English stopwords (NLTK data 'stopwords' likely missing). "
                "Proceeding with only custom stopwords (if any) or no stopwords.",
            )
            nltk_stopwords = set()
            self._stopwords_loaded = False
        except Exception:  # Catch any other unexpected error during stopword loading.
            logger.exception("An unexpected error occurred while loading NLTK stopwords.")
            nltk_stopwords = set()
            self._stopwords_loaded = False

        # Combine NLTK stopwords with custom stopwords provided in the configuration.
        self.stop_words = nltk_stopwords.union(self.config.custom_stop_words or set())
        if not self.stop_words and self._stopwords_loaded:  # NLTK loaded but custom resulted in empty
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
        if self.config.use_pos_tagging:  # Log allowed POS tags only if POS tagging is active.
            log_message += f" Allowed POS Tags: {self.resolved_allowed_pos_tags or 'None (defaulting or error)'}"
        logger.info(log_message)

    @lru_cache(maxsize=128)  # Cache results for previously seen texts.
    def _preprocess_text(self, text: str) -> list[str]:
        """Perform basic preprocessing on a text string.

        Steps:
        1. Converts text to lowercase.
        2. Removes punctuation using `PUNCTUATION_TABLE`.
        3. Tokenizes the text using `nltk.word_tokenize`.
        4. Filters out tokens that are not alphanumeric or are in the `self.stop_words` list.

        Args:
            text (str): The input text string.

        Returns:
            list[str]: A list of processed (cleaned, tokenized, filtered) tokens.
                       Returns an empty list for invalid input or if NLTK data (e.g., 'punkt') is missing.

        """
        if not isinstance(text, str) or not text.strip():  # Handle empty or non-string input.
            logger.debug("Preprocessing received empty or invalid text. Returning empty token list.")
            return []
        try:
            # Step 1 & 2: Lowercase and remove punctuation.
            cleaned_text = text.lower().translate(PUNCTUATION_TABLE)
            # Step 3: Tokenize. Requires 'punkt' NLTK data.
            tokens = word_tokenize(cleaned_text)
            # Step 4: Filter out stopwords and non-alphanumeric tokens.
            processed_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
            logger.debug(f"Preprocessing result for text '{text[:30]}...': {len(processed_tokens)} tokens returned.")
            return processed_tokens
        except LookupError:  # NLTK's 'punkt' tokenizer data might be missing.
            logger.exception(
                f"NLTK LookupError during preprocessing (likely 'punkt' data missing for word_tokenize). "
                f"Text: '{text[:50]}...'",
            )
            return []  # Return empty list on critical NLTK data error.
        except Exception:  # Catch any other unexpected errors.
            logger.exception(f"Unexpected error during basic preprocessing of text: '{text[:50]}...'.")
            return []

    @lru_cache(maxsize=128)  # Cache results for previously seen token tuples.
    def _normalize_tokens(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize a tuple of tokens, primarily by lemmatization if enabled and available.

        Args:
            tokens (tuple[str, ...]): A tuple of string tokens.

        Returns:
            tuple[str, ...]: A tuple of normalized (lemmatized) tokens. Returns the original
                             tokens if lemmatization is disabled or the lemmatizer is not functional.

        """
        # Check if lemmatization should be applied and if the global lemmatizer is available.
        if not self.effective_use_lemmatization or _GLOBAL_LEMMATIZER is None:
            return tokens  # Return original tokens if no lemmatization.
        try:
            # Lemmatize each token. Requires 'wordnet' and 'omw-1.4' NLTK data.
            normalized = tuple(_GLOBAL_LEMMATIZER.lemmatize(token) for token in tokens)
            logger.debug(f"Lemmatized {len(tokens)} tokens successfully.")
            return normalized
        except LookupError:  # NLTK data for lemmatization might be missing.
            logger.exception(
                f"NLTK LookupError during lemmatization (likely 'wordnet' or 'omw-1.4' data missing). Tokens: {tokens[:10]}",
            )
            return tokens  # Return original tokens if lemmatization fails.
        except Exception:  # Catch any other unexpected errors.
            logger.exception(f"Unexpected error during lemmatization of {len(tokens)} tokens.")
            return tokens

    @lru_cache(maxsize=128)  # Cache results for previously seen token tuples.
    def _get_pos_tags(self, tokens: tuple[str, ...]) -> list[tuple[str, str]]:
        """Perform Part-of-Speech (POS) tagging on a tuple of tokens.

        Args:
            tokens (tuple[str, ...]): A tuple of string tokens.

        Returns:
            list[tuple[str, str]]: A list of (token, POS_tag) tuples. Returns an empty list
                                   if input tokens are empty or if POS tagging fails (e.g.,
                                   NLTK 'averaged_perceptron_tagger' data missing).

        """
        if not tokens:  # No tokens to tag.
            return []
        try:
            # Perform POS tagging using NLTK's default tagger. Requires 'averaged_perceptron_tagger'.
            tags = pos_tag(list(tokens))  # pos_tag expects a list.
            logger.debug(f"POS tagged {len(tokens)} tokens successfully.")
            return tags
        except LookupError:  # NLTK data for POS tagging might be missing.
            logger.exception(
                f"NLTK LookupError during POS tagging (likely 'averaged_perceptron_tagger' data missing). "
                f"Tokens: {tokens[:10]}",
            )
            return []  # Return empty list on critical NLTK data error.
        except Exception:  # Catch any other unexpected errors.
            logger.exception(f"Unexpected error during POS tagging of {len(tokens)} tokens.")
            return []

    def _extract_keywords_from_a(self, paragraph_a: str) -> set[str]:
        """Extract a set of unique keywords from Paragraph A based on the matcher's configuration.

        The process involves:
        1. Basic preprocessing (`_preprocess_text`).
        2. If POS tagging is enabled: POS tag tokens, filter by `resolved_allowed_pos_tags`.
        3. Normalization (lemmatization if enabled) of the resulting tokens.

        Args:
            paragraph_a (str): The text of Paragraph A from which to extract keywords.

        Returns:
            set[str]: A set of unique extracted keywords. Returns an empty set if no
                      keywords can be extracted (e.g., empty input, processing errors).

        """
        # Step 1: Perform initial preprocessing (lowercase, punctuation, tokenize, stopwords).
        processed_tokens_list = self._preprocess_text(paragraph_a)
        if not processed_tokens_list:
            logger.warning("Keyword extraction from Paragraph A failed: Preprocessing returned no tokens.")
            return set()  # No tokens to process further.

        processed_tokens_tuple = tuple(processed_tokens_list)  # Convert to tuple for caching in subsequent steps.
        keywords: set[str] = set()

        # Step 2: Optional POS tagging for keyword filtering.
        if self.config.use_pos_tagging:
            if not self.resolved_allowed_pos_tags:  # Should not happen if __init__ logic is correct.
                logger.error(
                    "POS tagging is enabled, but no allowed POS tags are set. "
                    "Cannot extract POS-based keywords from Paragraph A.",
                )
            else:
                tagged_tokens = self._get_pos_tags(processed_tokens_tuple)
                if tagged_tokens:
                    # Filter tokens based on whether their POS tag is in the allowed set.
                    pos_filtered_tokens = [word for word, tag in tagged_tokens if tag in self.resolved_allowed_pos_tags]
                    if pos_filtered_tokens:
                        # Step 3 (for POS path): Normalize (lemmatize) the POS-filtered tokens.
                        keywords = set(self._normalize_tokens(tuple(pos_filtered_tokens)))
                    else:
                        logger.warning("No tokens from Paragraph A matched the allowed POS tags after POS tagging.")
                else:  # POS tagging itself failed (e.g., missing NLTK data).
                    logger.error("Keyword extraction from Paragraph A failed: POS tagging returned no results.")
        else:
            # Step 3 (for non-POS path): Directly normalize (lemmatize) the preprocessed tokens.
            keywords = set(self._normalize_tokens(processed_tokens_tuple))

        # Final logging based on outcome.
        if not keywords:
            logger.warning("No keywords were extracted from Paragraph A after all processing steps.")
        else:
            logger.debug(f"Successfully extracted {len(keywords)} unique keywords from Paragraph A.")
        return keywords

    def find_matches_and_score(self, paragraph_a: str, paragraph_b: str) -> MatcherScores:
        """Calculate keyword coverage and vocabulary cosine similarity between two paragraphs.

        Args:
            paragraph_a (str): The primary text from which keywords are extracted (source).
            paragraph_b (str): The secondary text in which to find the keywords (target).

        Returns:
            MatcherScores: A Pydantic model containing:
                - `matched_keywords`: List of keywords from A found in B.
                - `keywords_matcher_result`: A nested model with:
                    - `keywords_from_a_count`: Total unique keywords from A.
                    - `matched_keyword_count`: Number of A's keywords found in B.
                    - `keyword_coverage_score`: Proportion of A's keywords in B.
                    - `vocabulary_cosine_similarity`: Cosine similarity of A and B's vocabularies.

        """
        logger.info("Starting keyword matching and scoring for Paragraph A vs. Paragraph B.")
        logger.debug(f"Paragraph A (first 60 chars): '{paragraph_a[:60]}...'")
        logger.debug(f"Paragraph B (first 60 chars): '{paragraph_b[:60]}...'")

        # Calculate vocabulary cosine similarity first (independent of keyword extraction method for A).
        vocab_cosine_score = self._calculate_vocab_cosine(paragraph_a, paragraph_b)

        # Extract keywords from Paragraph A based on configuration.
        keywords_a_set = self._extract_keywords_from_a(paragraph_a)
        keywords_from_a_count_val = len(keywords_a_set)

        # Initialize coverage components; will be updated if keywords_a_set is not empty.
        matched_keywords_list: list[str] = []
        matched_keyword_count_val: int = 0
        keyword_coverage_score_val: float = 0.0

        if not keywords_a_set:
            logger.warning(
                "No keywords extracted from Paragraph A. Keyword coverage score will be 0.",
            )
        else:
            # Calculate keyword coverage components if keywords were extracted from A.
            coverage_components = self._calculate_keyword_coverage_components(keywords_a_set, paragraph_b)
            matched_keywords_list = coverage_components["matched_keywords"]
            matched_keyword_count_val = coverage_components["matched_keyword_count"]
            keyword_coverage_score_val = coverage_components["keyword_coverage_score"]

        # Construct the nested KeywordMatcherScore Pydantic model.
        keyword_scores = KeywordMatcherScore(
            keywords_from_a_count=keywords_from_a_count_val,
            matched_keyword_count=matched_keyword_count_val,
            keyword_coverage_score=keyword_coverage_score_val,
            vocabulary_cosine_similarity=vocab_cosine_score,
        )

        # Construct and return the main MatcherScores Pydantic model.
        return MatcherScores(
            matched_keywords=matched_keywords_list,
            keywords_matcher_result=keyword_scores,
        )

    def _calculate_vocab_cosine(self, paragraph_a: str, paragraph_b: str) -> float:
        """Calculate the cosine similarity between the vocabularies of two paragraphs.

        Vocabularies are derived from preprocessed tokens (lowercase, no punctuation, no stopwords).
        Similarity is based on binary vectors representing word presence in each paragraph's vocabulary.

        Args:
            paragraph_a (str): The first paragraph.
            paragraph_b (str): The second paragraph.

        Returns:
            float: The cosine similarity score (0.0 to 1.0). Returns 0.0 if both paragraphs
                   are empty after preprocessing or if there's no shared vocabulary.

        """
        # Preprocess both paragraphs to get lists of filtered tokens.
        processed_tokens_a_list = self._preprocess_text(paragraph_a)
        processed_tokens_b_list = self._preprocess_text(paragraph_b)

        # If both paragraphs result in empty token lists (e.g., they were empty or all stopwords).
        if not processed_tokens_a_list and not processed_tokens_b_list:
            logger.warning("Both paragraphs are empty after preprocessing. Vocabulary cosine similarity is 0.")
            return 0.0  # Or 1.0 if empty sets are considered perfectly similar; 0.0 is common.

        # Create sets of unique tokens for each paragraph.
        set_a = set(processed_tokens_a_list)
        set_b = set(processed_tokens_b_list)

        # Create a combined vocabulary (union of unique tokens from both sets).
        combined_vocabulary = set_a.union(set_b)
        if not combined_vocabulary:  # Should only happen if both set_a and set_b are empty.
            logger.debug(
                "No combined vocabulary found for cosine similarity calculation (both texts likely empty after processing).",
            )
            return 0.0  # Cosine is undefined or 0 for two empty sets.

        # Create binary vectors representing word presence in each paragraph's vocabulary.
        v_a = [1 if word in set_a else 0 for word in combined_vocabulary]
        v_b = [1 if word in set_b else 0 for word in combined_vocabulary]

        # Calculate dot product and vector norms (magnitudes).
        dot_product = sum(a_val * b_val for a_val, b_val in zip(v_a, v_b, strict=False))  # Python 3.10+ for strict
        norm_a = sum(val_a**2 for val_a in v_a) ** 0.5  # Magnitude of vector A
        norm_b = sum(val_b**2 for val_b in v_b) ** 0.5  # Magnitude of vector B
        denominator = norm_a * norm_b

        # Calculate cosine similarity. If denominator is zero (one or both vectors are zero-vectors), score is 0.
        return dot_product / denominator if denominator > 0 else 0.0

    def _calculate_keyword_coverage_components(
        self,
        keywords_a_set: set[str],  # Pre-extracted and normalized keywords from Paragraph A.
        paragraph_b: str,  # Raw text of Paragraph B.
    ) -> dict[str, Any]:
        """Calculate keyword coverage components by checking how many keywords from Paragraph A are present in Paragraph B.

        Args:
            keywords_a_set (set[str]): A set of unique, normalized keywords extracted from Paragraph A.
            paragraph_b (str): The raw text of Paragraph B.

        Returns:
            dict[str, Any]: A dictionary containing:
                - "matched_keywords": A sorted list of keywords from `keywords_a_set` found in Paragraph B.
                - "matched_keyword_count": The number of such matched keywords.
                - "keyword_coverage_score": The proportion of `keywords_a_set` found in Paragraph B.

        """
        # Preprocess Paragraph B (lowercase, punctuation, tokenize, stopwords).
        processed_tokens_b_list = self._preprocess_text(paragraph_b)
        if not processed_tokens_b_list:  # If Paragraph B is empty after preprocessing.
            logger.warning("Paragraph B is empty after preprocessing. Keyword coverage score is 0.")
            return {"matched_keywords": [], "matched_keyword_count": 0, "keyword_coverage_score": 0.0}

        # Normalize (lemmatize if enabled) tokens of Paragraph B to match normalization of keywords_a_set.
        normalized_tokens_b_set = set(self._normalize_tokens(tuple(processed_tokens_b_list)))

        # Find common keywords by intersecting the set of A's keywords with B's normalized tokens.
        matched_keywords_set = keywords_a_set.intersection(normalized_tokens_b_set)
        num_matches = len(matched_keywords_set)

        # Calculate coverage score: (number of matched keywords) / (total keywords from A).
        # Handle division by zero if keywords_a_set was empty (though typically checked before calling).
        coverage_score = num_matches / len(keywords_a_set) if keywords_a_set else 0.0

        return {
            "matched_keywords": sorted(matched_keywords_set),  # Return sorted list for consistent output.
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
