"""Defines Pydantic data models used throughout the essay grading application.

This module centralizes the definitions of data structures, ensuring type safety
and clear contracts between different parts of the application. Models include:
- Input structures (e.g., `Essay`, `SinglePairAnalysisInput`).
- Configuration objects (e.g., `TfidfConfig`, `SimilarityCalculatorConfig`, `KeywordMatcherConfig`).
- Result containers for various analyses (e.g., `SimilarityMetrics`, `GraphSimilarityOutput`,
  `PlagiarismScore`, `KeywordMatcherScore`, `SinglePairAnalysisResult`).
- The main output model (`EssayScores`) that aggregates all calculated scores.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Essay(BaseModel):
    """Represents an essay and its corresponding reference text(s).

    This model is typically used as an input to scoring functions.
    The `text` and `reference` can be single strings or lists of strings
    if multiple reference texts are applicable or if the essay is segmented.
    """

    text: str | list[str] = Field(
        ..., description="The essay text to be scored. Can be a single string or a list of strings (e.g., paragraphs).",
    )
    reference: str | list[str] = Field(
        ..., description="The reference text(s) to compare against. Can be a single string or a list of strings.",
    )


class SimilarityMetrics(BaseModel):
    """Defines a comprehensive set of similarity and distance metrics that can be calculated between two texts.

    All fields are optional and default to 0.0, as some metrics might fail,
    not be applicable for certain inputs, or simply not be requested for a given analysis.
    The descriptions aim to clarify the origin or nature of each metric.
    """

    # Basic String / Sequence Metrics (often applied on lowercase, tokenized, or raw strings)
    ratio: Optional[float] = Field(
        default=0.0,
        description="Similarity score from difflib.SequenceMatcher.ratio(). Measures likeness of sequences.",
    )
    normalized_levenshtein: Optional[float] = Field(
        default=0.0,
        description="Normalized Levenshtein similarity. 1.0 for identical strings, 0.0 for completely different.",
    )
    jaro_winkler: Optional[float] = Field(
        default=0.0, description="Jaro-Winkler similarity, emphasizes prefix matches. Score from 0.0 to 1.0.",
    )
    metric_lcs_similarity: Optional[float] = Field(
        default=0.0,
        description="Similarity based on the Longest Common Subsequence (LCS). Score often normalized.",
    )
    # Q-gram distances measure dissimilarity; lower values mean more similar.
    qgram2_distance: Optional[float] = Field(
        default=0.0,
        description="Q-gram distance using character bigrams (n=2). Lower values indicate more similarity.",
    )
    qgram3_distance: Optional[float] = Field(
        default=0.0,
        description="Q-gram distance using character trigrams (n=3). Lower values indicate more similarity.",
    )
    qgram4_distance: Optional[float] = Field(
        default=0.0,
        description="Q-gram distance using character 4-grams (n=4). Lower values indicate more similarity.",
    )

    # Character n-gram based similarity metrics
    cosine_char_2gram: Optional[float] = Field(
        default=0.0,
        description="Cosine similarity calculated on character bigram (2-gram) vectors.",
    )
    jaccard_char_2gram: Optional[float] = Field(
        default=0.0,
        description="Jaccard similarity calculated on sets of character bigrams (2-grams).",
    )

    # RapidFuzz Metrics (optimized fuzzy string matching)
    # Scores are typically 0-100 from RapidFuzz, often normalized to 0-1 by dividing by 100.0.
    rfuzz_ratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz simple ratio (equivalent to Levenshtein ratio). Normalized to 0-1.",
    )
    rfuzz_partial_ratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz partial ratio, for finding the best matching substring. Normalized to 0-1.",
    )
    rfuzz_token_set_ratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz token set ratio, compares sets of tokens ignoring order and duplicates. Normalized to 0-1.",  # noqa: E501
    )
    rfuzz_token_sort_ratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz token sort ratio, compares sorted tokens. Normalized to 0-1.",
    )
    rfuzz_wratio: Optional[float] = Field(
        default=0.0,
        description=(
            "RapidFuzz weighted ratio, a more advanced ratio considering various factors. "
            "Normalized to 0-1 (can sometimes exceed 1 before normalization if original > 100)."
        ),
    )
    rfuzz_qratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz quick ratio, a faster version of simple ratio. Normalized to 0-1.",
    )

    # FuzzyWuzzy Metrics (older library, similar to RapidFuzz but potentially slower)
    # Scores are typically 0-100, often normalized to 0-1.
    fz_uqratio: Optional[float] = Field(
        default=0.0,
        description="FuzzyWuzzy UQRatio (unicode quick ratio). Normalized to 0-1.",
    )
    fz_uwratio: Optional[float] = Field(
        default=0.0,
        description="FuzzyWuzzy UWRatio (unicode weighted ratio). Normalized to 0-1.",
    )

    # BLEU Score (Bilingual Evaluation Understudy)
    # Primarily a machine translation metric, adapted here for text similarity.
    bleu_score: Optional[float] = Field(
        default=0.0,
        description="BLEU score, typically BLEU-4, indicating n-gram overlap. Ranges from 0 to 1.",
    )

    # BM25 Score (Okapi BM25)
    # A ranking function used in information retrieval, adapted here for similarity. Higher is more similar.
    bm25: Optional[float] = Field(
        default=0.0,
        description="BM25 relevance score. Typically non-negative; magnitude depends on corpus statistics.",
    )

    # TF-IDF Based Metrics (vector space model comparisons)
    tfidf_cosine_similarity: Optional[float] = Field(
        default=0.0,
        description=(
            "Cosine similarity between TF-IDF vectors of the texts. Ranges from -1 to 1 (usually 0 to 1 for TF-IDF)."
        ),
    )
    tfidf_euclidean_distance: Optional[float] = Field(
        default=0.0,
        description="Euclidean distance between TF-IDF vectors. Non-negative; 0 for identical vectors.",
    )
    tfidf_manhattan_distance: Optional[float] = Field(
        default=0.0,
        description="Manhattan (L1) distance between TF-IDF vectors. Non-negative.",
    )
    tfidf_jaccard_distance: Optional[float] = Field(
        default=0.0,
        description="Jaccard distance between binarized TF-IDF vectors (presence/absence of terms). Ranges 0 to 1.",
    )
    tfidf_hamming_distance: Optional[float] = Field(
        default=0.0,
        description="Hamming distance between binarized TF-IDF vectors (proportion of differing terms). Ranges 0 to 1.",
    )
    tfidf_minkowski_distance: Optional[float] = Field(
        default=0.0,
        description=(
            "Minkowski distance between TF-IDF vectors (generalized form of Euclidean/Manhattan). Non-negative."
        ),
    )


class TfidfConfig(BaseModel):
    """Configuration for the `TfidfVectorizer` from scikit-learn.

    This model allows specifying parameters like token pattern, n-gram range,
    and document frequency thresholds (max_df, min_df) to customize TF-IDF vectorization.
    """

    token_pattern: str = Field(
        default=r"(?u)\b\w\w+\b",  # Default scikit-learn pattern: unicode words of 2+ chars.
        description="Regular expression denoting what constitutes a 'token'.",
    )
    ngram_range: tuple[int, int] = Field(
        default=(1, 1),  # Default: unigrams only.
        description=(
            "The lower and upper boundary of the range of n-values for different n-grams to be extracted. "
            "E.g., (1, 2) means unigrams and bigrams."
        ),
    )
    max_df: float = Field(
        default=1.0,  # Default: no upper limit on document frequency.
        ge=0.0,  # max_df must be between 0.0 and 1.0 (if float) or >= 1 (if int).
        le=1.0,
        description=(
            "When building the vocabulary, ignore terms that have a document frequency strictly higher "
            "than the given threshold (corpus-specific stop words). Expressed as a proportion (0.0 to 1.0) "
            "or absolute count."
        ),
    )
    min_df: int = Field(
        default=1,  # Default: term must appear in at least one document.
        ge=0,
        description=(
            "When building the vocabulary, ignore terms that have a document frequency strictly lower "
            "than the given threshold. Expressed as an absolute count."
        ),
    )


# --- Main SimilarityCalculator Class ---
class SimilarityCalculatorConfig(BaseModel):
    """Configuration for the `SimilarityCalculator` class.

    This model allows setting preferences for text preprocessing steps like
    lemmatization and stop-word removal, as well as providing a nested
    configuration for TF-IDF vectorization.
    """

    use_lemmatization: bool = Field(
        default=True,
        description="If True, texts will be lemmatized before TF-IDF vectorization and some other comparisons.",
    )
    use_stopwords: bool = Field(
        default=True,
        description="If True, common stop words will be removed from texts before TF-IDF and some other comparisons.",
    )
    custom_stop_words: Optional[list[str]] = Field(
        default=None,
        description="An optional list of custom stop words to be added to the default set or used exclusively.",
    )
    tfidf_config: TfidfConfig = Field(
        default_factory=TfidfConfig,  # Provides default TfidfConfig if not specified.
        description="Nested configuration object for TF-IDF vectorization parameters.",
    )
    # bleu_smoothing_function_name: Optional[str] = Field(default=None,
    #     description="Name of the BLEU smoothing function to use, if applicable.")


class SinglePairAnalysisInput(BaseModel):
    """Input parameters for analyzing a single pair of texts (e.g., one model answer vs. one student text).

    This is used by functions like `run_single_pair_text_analysis`.
    """

    model_answer: str = Field(..., description="The model or reference text string.")
    student_text: str = Field(
        ..., description="The student-provided text string to be analyzed against the model answer.",
    )
    plagiarism_k: int = Field(
        default=3,
        gt=0,
        description="K-gram size for plagiarism detection (Smith-Waterman variant).",
    )
    plagiarism_window_radius: int = Field(
        default=50,
        gt=0,
        description="Window radius around k-gram matches for applying Smith-Waterman algorithm.",
    )
    # Note: Additional parameters specific to other metrics calculated in single_pair_analysis
    # could be added here if they need to be configurable at this level.


class GraphSimilarityOutput(BaseModel):
    """Output structure for graph-based text similarity calculation.

    Contains the similarity score (typically subgraph density), details about
    the common subgraph, and an optional message.
    """

    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The calculated graph similarity score, often based on subgraph density.",
    )
    subgraph_nodes: int = Field(default=0, ge=0, description="Number of nodes in the common subgraph.")
    subgraph_edges: int = Field(default=0, ge=0, description="Number of edges in the common subgraph.")
    subgraph_density: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Density of the common subgraph. May be None if not applicable (e.g., too few nodes).",
    )
    message: Optional[str] = Field(
        default=None, description="An optional message providing context or details about the calculation.",
    )


class PlagiarismScore(BaseModel):
    """Represent the result of a plagiarism detection analysis."""

    overlap_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized plagiarism score, typically representing the degree of overlap or similarity, scaled to 0-1."
        ),
    )


class OverlapCoefficient(BaseModel):
    """Represent the overlap coefficient calculated between two sets of tokens.

    Formula: |A intersect B| / min(|A|, |B|)
    """

    coefficient: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The calculated overlap coefficient, ranging from 0.0 to 1.0.",
    )


class SorensenDiceCoefficient(BaseModel):
    """Represent the Sørensen-Dice coefficient (or Dice score) calculated between two sets of tokens.

    Formula: 2 * |A intersect B| / (|A| + |B|)
    """

    coefficient: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The calculated Sørensen-Dice coefficient, ranging from 0.0 to 1.0.",
    )


class CharEqualityScore(BaseModel):
    """Represent a character-by-character equality score, often with decaying weights to emphasize initial matches."""

    score: float = Field(
        default=0.0,
        ge=0.0,
        description="Calculated char equality score. Max value depends on string length & weighting.",
    )


class SemanticGraphSimilarity(BaseModel):
    """Represent similarity scores derived from comparing semantic graphs (e.g., from spaCy dependency parses)."""

    similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall similarity score, often an average or combination of node and edge similarities.",
    )
    nodes_jaccard: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity of the node sets of the two graphs.",
    )
    edges_jaccard: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity of the edge sets of the two graphs.",
    )


class SinglePairAnalysisResult(BaseModel):
    """Aggregates all analysis results for a single pair of texts (student vs. model).

    Each field is optional, allowing for flexibility if some analyses are skipped or fail.
    """

    graph_similarity: Optional[GraphSimilarityOutput] = Field(
        default=None, description="Graph-based similarity scores.",
    )
    plagiarism_score: Optional[PlagiarismScore] = Field(default=None, description="Plagiarism detection score.")
    overlap_coefficient: Optional[OverlapCoefficient] = Field(default=None, description="Overlap coefficient score.")
    dice_coefficient: Optional[SorensenDiceCoefficient] = Field(
        default=None, description="Sørensen-Dice coefficient score.",
    )
    char_equality_score: Optional[CharEqualityScore] = Field(
        default=None, description="Character-by-character equality score.",
    )
    semantic_graph_similarity: Optional[SemanticGraphSimilarity] = Field(
        default=None,
        description="Semantic graph similarity score (e.g., from spaCy). Active if spaCy components are used.",
    )


class KeywordMatcherConfig(BaseModel):
    """Configure settings for the `KeywordMatcher` class."""

    use_lemmatization: bool = Field(
        default=True,
        description=(
            "If True, keywords and text are lemmatized before matching. Relies on a globally available lemmatizer."
        ),
    )
    use_pos_tagging: bool = Field(
        default=False,
        description="If True, keywords are extracted based on allowed Part-of-Speech (POS) tags from the source text.",
    )
    allowed_pos_tags: Optional[set[str]] = Field(
        default=None,  # If None and use_pos_tagging is True, a default set (e.g., nouns, adjectives) might be used.
        description=(
            "Set of NLTK POS tags to consider for keyword extraction if `use_pos_tagging` is True. "
            "Example: {'NN', 'NNS', 'JJ'}."
        ),
    )
    custom_stop_words: Optional[set[str]] = Field(
        default=None,
        description="A set of custom stop words to be added to (or replace) the default NLTK English stop word list.",
    )

    model_config = ConfigDict(extra="forbid")  # Disallow extra fields not defined in the model.


class KeywordMatcherScore(BaseModel):
    """Contain scores and counts related to keyword matching between two texts."""

    keywords_from_a_count: int = Field(
        default=0,
        ge=0,
        description="Total number of unique keywords extracted from the primary text (text A or model text).",
    )
    matched_keyword_count: int = Field(
        default=0,
        ge=0,
        description="Number of unique keywords from text A that were found in text B (student text).",
    )
    keyword_coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Proportion of keywords from text A found in text B (matched_keyword_count / keywords_from_a_count)."
        ),
    )
    vocabulary_cosine_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between the (optionally non-stopword) vocabularies of the two texts.",
    )


class MatcherScores(BaseModel):  # This seems to be a wrapper, perhaps for future expansion or specific use context.
    """Aggregate detailed results from the keyword matching process."""

    matched_keywords: list[str] = Field(
        default_factory=list,
        description="A list of specific keywords from the primary text (A) that were found in the secondary text (B).",
    )
    keywords_matcher_result: KeywordMatcherScore = Field(
        ..., description="The core scores and counts from keyword matching.",
    )

    model_config = ConfigDict(extra="forbid")


class EssayScores(BaseModel):
    """The main data model for aggregating all calculated scores for an essay.

    This model brings together semantic scores, various similarity metrics,
    text-specific analysis results (like plagiarism and graph similarity),
    keyword matching scores, and POS-based scores.
    """

    semantic_score: Optional[float] = Field(
        default=0.0, description="Overall semantic similarity score, possibly an aggregation or primary model output.",
    )
    similarity_metrics: SimilarityMetrics = Field(
        ..., description="A collection of diverse similarity metrics calculated between the essay and reference text.",
    )
    text_score: SinglePairAnalysisResult = Field(
        ...,
        description="Detailed analysis results for the essay against a reference, including plagiarism and graph metrics.",  # noqa: E501
    )
    keyword_matcher: KeywordMatcherScore = Field(..., description="Scores related to keyword matching and coverage.")
    pos_score: Optional[float] = Field(
        default=0.0, description="Score based on Part-of-Speech (POS) tag patterns or similarity, if calculated.",
    )
