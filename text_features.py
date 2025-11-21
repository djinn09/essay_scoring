"""
Text Feature Extraction and Analysis Module.

This module provides a comprehensive suite of tools for extracting various
features from textual data, primarily aimed at text comparison, similarity
assessment, and plagiarism detection in educational contexts (e.g., comparing
student answers to model answers).

Key Capabilities:
- Tokenization: Includes basic tokenization and utilities for preparing text for
  further analysis.
- Word Vectorization: Uses `CountVectorizer` to create word-document matrices.
- Graph-based Features:
    - Builds word co-occurrence graphs based on cosine similarity of word vectors
      derived from the corpus.
    - Calculates similarity between texts based on the density of their common
      words' subgraph within the corpus graph.
- Plagiarism Detection:
    - Implements a fast variant of the Smith-Waterman algorithm using k-gram
      indexing and windowed application to identify local sequence similarities.
- Lexical and Syntactic Features:
    - Calculates overlap coefficients and Sørensen-Dice coefficients.
    - Provides a character-by-character equality score with geometric decay.
- Clustering-based Lexical Features:
    - Extracts features based on hierarchical clustering of TF-IDF vectors of texts.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import warnings
from collections import defaultdict
from typing import Any, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.sparse import csr_matrix as ScipyCSRMatrix
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity

from app_types import (
    CharEqualityScore,
    GraphSimilarityOutput,
    OverlapCoefficient,
    PlagiarismScore,
    SemanticGraphSimilarity,
    SinglePairAnalysisInput,
    SinglePairAnalysisResult,
    SorensenDiceCoefficient,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.feature_extraction.text",
)


# --- Pydantic Models ---


class WordVectorCreationResult(BaseModel):
    """
    Holds the result of word vector creation using CountVectorizer.

    This model stores the sparse word-document matrix and the vocabulary list
    generated during the vectorization process.
    """

    word_matrix_csr_scipy: Optional[ScipyCSRMatrix] = Field(
        default=None,
        description=(
            "Word-document matrix in SciPy CSR (Compressed Sparse Row) format. "
            "This matrix typically has documents as rows and words (vocabulary) as columns. "
        ),
    )
    words_vocabulary: list[str] = Field(
        default_factory=list,
        description="List of unique words (features) identified by the vectorizer, forming the vocabulary.",
    )

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True


class GraphMetrics(BaseModel):
    """
    Represents basic metrics of a graph.

    Used to summarize the size and density of graphs, such as the corpus graph
    or subgraphs generated during analysis.
    """

    nodes: int = Field(..., description="Number of nodes in the graph.", ge=0)
    edges: int = Field(..., description="Number of edges in the graph.", ge=0)
    density: Optional[float] = Field(
        default=None,
        description="Density of the graph. Calculated as 2*E / (N*(N-1)) for undirected graphs. Ranges from 0 to 1.",
        ge=0.0,
        le=1.0,
    )


class SmithWatermanParams(BaseModel):
    """
    Parameters for the Smith-Waterman local alignment algorithm variant.

    This model encapsulates all necessary inputs for performing a Smith-Waterman
    alignment on specific windows within two tokenized texts.
    """

    t1: list[str] = Field(..., description="First text represented as a list of tokens.")
    t2: list[str] = Field(..., description="Second text represented as a list of tokens.")
    sm: dict[str, dict[str, float]] = Field(
        ...,
        description=(
            "Scoring matrix for token matches and mismatches. "
            "Example: sm['word1']['word2'] gives the score for aligning 'word1' with 'word2'."
        ),
    )
    gap_penalty: float = Field(
        ...,
        description="Penalty for introducing a gap in the alignment (should be negative or zero).",
    )
    win1: tuple[int, int] = Field(
        ...,
        description="Tuple representing the (start_index, end_index) of the window in the first text (t1).",
    )
    win2: tuple[int, int] = Field(
        ...,
        description="Tuple representing the (start_index, end_index) of the window in the second text (t2).",
    )
    mismatch_score: float = Field(default=-1.0, description="Score for a mismatch during alignment.")


class LexicalClusterFeature(BaseModel):
    """
    Stores lexical and clustering-based features for a single text (e.g., a student answer).

    These features are derived from comparing the text against a set of model/reference texts
    using TF-IDF and hierarchical clustering.
    """

    coph_min: float = Field(..., description="Minimum cophenetic distance to model answers.")
    coph_mean: float = Field(..., description="Mean cophenetic distance to model answers.")
    coph_max: float = Field(..., description="Maximum cophenetic distance to model answers.")
    cluster_label: int = Field(..., description="Label of the cluster the text belongs to.")
    cluster_size: int = Field(..., description="Number of texts in the assigned cluster.")
    is_outlier: int = Field(
        ...,
        description="Boolean flag (0 or 1) indicating if the text is an outlier (cluster size is 1).",
        ge=0,
        le=1,
    )
    silhouette: float = Field(
        ...,
        description=(
            "Silhouette score for the text, measuring how similar it is to its own cluster "
            "compared to other clusters. Ranges from -1 to 1."
        ),
    )


class LexicalFeaturesAnalysis(BaseModel):
    """Container for a list of lexical and clustering features for all analyzed student answers."""

    student_features: list[LexicalClusterFeature] = Field(
        default_factory=list,
        description="A list, where each item contains the lexical/clustering features for one student answer.",
    )


class SmithWatermanConfig(BaseModel):
    """Configuration parameters for the Smith-Waterman algorithm variant used for plagiarism detection."""

    k: int = Field(
        default=3,
        gt=0,
        description="The k-gram size (number of consecutive tokens) used for indexing and finding initial matches.",
    )
    window_radius: int = Field(
        default=50,
        gt=0,
        description=(
            "Radius (number of tokens) around a k-gram match to define the window "
            "for applying the full Smith-Waterman alignment."
        ),
    )
    match_score: float = Field(default=1.0, description="Score awarded when two tokens match during alignment.")
    mismatch_score: float = Field(default=0.0, description="Score awarded (or penalty) when two tokens mismatch.")
    gap_penalty: float = Field(default=-1.0, description="Penalty for introducing a gap (indel) in the alignment.")


class FullTextAnalysisInput(BaseModel):
    """
    Input parameters for the comprehensive text analysis pipeline (`run_full_text_analysis`).

    This model gathers all necessary data and configuration to analyze a collection
    of student texts against a collection of model answers.
    """

    model_answers: list[str] = Field(..., description="A list of model/reference answer strings.")
    student_texts: list[str] = Field(..., description="A list of student-provided text strings to be analyzed.")

    plagiarism_k: int = Field(
        default=3,
        gt=0,
        description="K-gram size for the plagiarism detection algorithm.",
    )
    plagiarism_window_radius: int = Field(
        default=50,
        gt=0,
        description="Window radius for applying Smith-Waterman around k-gram matches.",
    )

    lexical_linkage_method: str = Field(
        default="average",
        description="Linkage method for hierarchical clustering (e.g., 'average', 'ward', 'complete').",
    )
    lexical_distance_metric: str = Field(
        default="sqeuclidean",
        description="Distance metric for pdist (e.g., 'cosine', 'euclidean', 'sqeuclidean').",
    )
    lexical_cluster_dist_thresh: float = Field(
        default=0.5,
        gt=0,
        description="Distance threshold for forming flat clusters from the linkage matrix.",
    )


class FullTextAnalysisResult(BaseModel):
    """
    Stores the comprehensive results from analyzing a set of student texts against model answers.

    This includes metrics for the corpus graph, lexical features for each student,
    and aggregated per-student similarity scores relative to the model answers.
    """

    corpus_graph_metrics: Optional[GraphMetrics] = Field(
        default=None,
        description="Metrics (nodes, edges, density) of the word co-occurrence graph built from the entire corpus.",
    )
    student_lexical_features: Optional[LexicalFeaturesAnalysis] = Field(
        default=None,
        description="Lexical and clustering-based features extracted for each student answer.",
    )
    per_student_analysis: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "A list of dictionaries, where each dictionary contains aggregated similarity scores "
            "for one student text compared against all model answers."
        ),
    )


MODULE_MODELS = [
    WordVectorCreationResult,
    GraphMetrics,
    LexicalClusterFeature,
    LexicalFeaturesAnalysis,
    SmithWatermanConfig,
    FullTextAnalysisInput,
    FullTextAnalysisResult,
    SmithWatermanParams,
]

for model_cls in MODULE_MODELS:
    model_cls.model_rebuild(force=True)


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenizes a text string by converting to lowercase, removing punctuation, and splitting by whitespace.

    Args:
        text (str): The input text string.

    Returns:
        list[str]: A list of processed tokens. Returns an empty list if input is not a string.
    """
    if not isinstance(text, str):
        logger.warning("simple received non-string input: %s. Returning empty list.", type(text))
        return []
    text_lower = text.lower()
    text_no_punct = re.sub(r"[^\w\s]", "", text_lower)
    return [token for token in text_no_punct.split() if token]


def create_word_vectors(texts: list[str]) -> WordVectorCreationResult:
    """
    Create a word-document matrix from a list of texts using CountVectorizer.

    The function uses `simple_tokenize` for tokenization. The resulting matrix
    is a sparse representation (CSR format) of word counts per document.

    Args:
        texts (list[str]): A list of text strings (documents).

    Returns:
        WordVectorCreationResult: A Pydantic model containing the sparse word-document
                                  matrix (`word_matrix_csr_scipy`) and the list of
                                  vocabulary words (`words_vocabulary`).
    """
    vectorizer = CountVectorizer(
        tokenizer=simple_tokenize,
        token_pattern=None,
    )
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not valid_texts:
        logger.warning("create_word_vectors: No valid (non-empty, string) texts provided. Returning empty result.")
        return WordVectorCreationResult()

    try:
        word_matrix: ScipyCSRMatrix = vectorizer.fit_transform(valid_texts)
        words: list[str] = vectorizer.get_feature_names_out().tolist()
        return WordVectorCreationResult(word_matrix_csr_scipy=word_matrix, words_vocabulary=words)
    except Exception:
        logger.exception("Error in create_word_vectors during vectorization.")
        return WordVectorCreationResult()


def build_graph_efficiently(word_matrix_result: WordVectorCreationResult) -> nx.Graph:
    """
    Build a word co-occurrence graph based on cosine similarity of word vectors.

    Nodes in the graph are words from the vocabulary. An edge exists between two
    words if their cosine similarity (based on their occurrences across documents)
    exceeds a defined threshold. Edge weights represent this similarity.

    Args:
        word_matrix_result (WordVectorCreationResult): The output from `create_word_vectors`.

    Returns:
        nx.Graph: A NetworkX graph where nodes are words and edges are weighted by
                  their cosine similarity.
    """
    if word_matrix_result.word_matrix_csr_scipy is None or not word_matrix_result.words_vocabulary:
        logger.warning("build_graph_efficiently: word_matrix or words vocabulary is empty. Returning empty graph.")
        return nx.Graph()

    word_matrix = word_matrix_result.word_matrix_csr_scipy
    words = word_matrix_result.words_vocabulary

    start_time = time.time()
    logger.info(f"Building graph for {len(words)} words...")

    try:
        word_vectors: np.ndarray = word_matrix.T.toarray()
    except MemoryError:
        logger.exception(
            "MemoryError converting sparse matrix to dense for graph building. "
            "Vocabulary size: %s.",
            len(words),
        )
        return nx.Graph()

    if word_vectors.shape[0] == 0:
        logger.warning("Word vectors are empty (no words in vocabulary). Returning empty graph.")
        return nx.Graph()

    similarity_matrix = cosine_similarity(word_vectors)
    logger.debug(f"  Similarity matrix calculation took {time.time() - start_time:.2f}s")

    num_words = len(words)
    graph = nx.Graph()
    graph.add_nodes_from(words)

    similarity_threshold = 0.01
    edges_added_count = 0

    for i in range(num_words):
        for j in range(i + 1, num_words):
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                graph.add_edge(words[i], words[j], weight=similarity)
                edges_added_count += 1

    logger.info(
        f"Graph building finished. Total time: {time.time() - start_time:.2f}s. "
        f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}",
    )
    return graph


def calculate_graph_similarity(graph: nx.Graph, text1: str, text2: str) -> GraphSimilarityOutput:
    """
    Calculate similarity between two texts based on the density of their common words' subgraph.

    The method identifies common words between `text1` and `text2` that also exist
    in the provided `graph`. A subgraph is induced from these common words. The density
    of this subgraph is used as the similarity score.

    Args:
        graph (nx.Graph): The global word co-occurrence graph (corpus graph).
        text1 (str): The first text string.
        text2 (str): The second text string.

    Returns:
        GraphSimilarityOutput: A Pydantic model containing the similarity score (subgraph density),
                               number of nodes and edges in the subgraph, and a message.
    """
    words1 = set(simple_tokenize(text1))
    words2 = set(simple_tokenize(text2))

    common_words_in_graph = list(words1.intersection(words2).intersection(graph.nodes))

    if not common_words_in_graph:
        logger.debug(
            "No common words found in the graph for the given texts (text1: '%s...', text2: '%s...').",
            text1[:30],
            text2[:30],
        )
        return GraphSimilarityOutput(
            similarity_score=0.0,
            subgraph_nodes=0,
            subgraph_edges=0,
            message="No common words in graph.",
        )

    subgraph = graph.subgraph(common_words_in_graph)
    sub_nodes = subgraph.number_of_nodes()
    sub_edges = subgraph.number_of_edges()

    graph_sim_score = 0.0
    sub_density_val: Optional[float] = None
    message: Optional[str] = None

    min_sub_nodes_for_density = 2
    if sub_nodes < min_sub_nodes_for_density:
        message = f"Subgraph has < {min_sub_nodes_for_density} nodes ({sub_nodes}), so density is considered 0."
    else:
        try:
            current_density = nx.density(subgraph)
            if np.isnan(current_density):
                logger.debug(
                    "Subgraph density calculated as NaN, treating as 0.",
                )
                sub_density_val = 0.0
            else:
                sub_density_val = float(current_density)
            graph_sim_score = sub_density_val
            message = f"Subgraph successfully created: {sub_nodes} nodes, {sub_edges} edges."
        except ZeroDivisionError:
            message = "Density calculation failed due to ZeroDivisionError (unexpected for N >= 2)."
            sub_density_val = 0.0

    logger.debug(
        f"Graph similarity for texts ('{text1[:30]}...', '{text2[:30]}...'): score={graph_sim_score:.4f}, {message}",
    )
    return GraphSimilarityOutput(
        similarity_score=graph_sim_score,
        subgraph_nodes=sub_nodes,
        subgraph_edges=sub_edges,
        subgraph_density=sub_density_val,
        message=message,
    )


def preprocess_sw(text: str, *, lowercase: bool = True, remove_punct: bool = True) -> list[str]:
    """
    Preprocesses text for Smith-Waterman algorithm by lowercasing, removing punctuation, and splitting into tokens.

    Args:
        text (str): The input text string.
        lowercase (bool): Whether to convert text to lowercase. Defaults to True.
        remove_punct (bool): Whether to remove punctuation. Defaults to True.

    Returns:
        list[str]: A list of processed tokens. Returns an empty list if input is not a string.
    """
    if not isinstance(text, str):
        logger.warning("preprocess_sw received non-string input: %s. Returning empty list.", type(text))
        return []
    processed_text = text
    if lowercase:
        processed_text = processed_text.lower()
    if remove_punct:
        processed_text = re.sub(r"[^\w\s]", " ", processed_text)
    return [token for token in processed_text.split() if token]


def build_ngram_index(tokens: list[str], k: int) -> dict[tuple[str, ...], list[int]]:
    """
    Build an index mapping each k-gram in a list of tokens to all its starting positions.

    Args:
        tokens (list[str]): A list of tokens representing a text.
        k (int): The size of the k-grams to index.

    Returns:
        dict[tuple[str, ...], list[int]]: A dictionary mapping k-grams to lists of starting indices.
    """
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    if k <= 0 or not tokens or len(tokens) < k:
        logger.debug(
            "Invalid input for build_ngram_index (k=%s, len(words)=%s). Returning empty index.",
            k,
            len(tokens),
        )
        return dict(index)

    for i in range(len(tokens) - k + 1):
        kgram = tuple(tokens[i : i + k])
        index[kgram].append(i)
    return dict(index)


def smith_waterman_window(params: SmithWatermanParams) -> float:
    """
    Perform the Smith-Waterman local alignment algorithm on specified windows of two token lists.

    Args:
        params (SmithWatermanParams): A Pydantic model containing the parameters.

    Returns:
        float: The maximum alignment score found within the specified windows.
    """
    t1, t2, sm, gap_penalty, win1, win2 = params.t1, params.t2, params.sm, params.gap_penalty, params.win1, params.win2

    a1 = t1[win1[0] : win1[1]]
    a2 = t2[win2[0] : win2[1]]
    n, m = len(a1), len(a2)

    if n == 0 or m == 0:
        return 0.0

    h_matrix = np.zeros((n + 1, m + 1), dtype=float)
    max_score_in_window = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_val = sm.get(a1[i - 1], {}).get(a2[j - 1], params.mismatch_score)

            score_diag = h_matrix[i - 1, j - 1] + match_val
            score_up = h_matrix[i - 1, j] + gap_penalty
            score_left = h_matrix[i, j - 1] + gap_penalty

            h_matrix[i, j] = max(0, score_diag, score_up, score_left)
            max_score_in_window = max(max_score_in_window, h_matrix[i, j])

    return max_score_in_window


def compute_plagiarism_score_fast(
    text1: str,
    text2: str,
    config: SmithWatermanConfig,
) -> PlagiarismScore:
    """
    Compute a plagiarism score using a k-gram indexed, windowed Smith-Waterman algorithm.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.
        config (SmithWatermanConfig): Configuration for the Smith-Waterman algorithm.

    Returns:
        PlagiarismScore: A Pydantic model containing the normalized `overlap_percentage`.
    """
    t1_tokens = preprocess_sw(text1)
    t2_tokens = preprocess_sw(text2)

    if not t1_tokens or not t2_tokens:
        logger.debug("One or both texts are empty after preprocessing. Plagiarism score is 0.")
        return PlagiarismScore(overlap_percentage=0.0)

    unique_words = set(t1_tokens) | set(t2_tokens)
    scoring_matrix = {
        w1: {w2: (config.match_score if w1 == w2 else config.mismatch_score) for w2 in unique_words}
        for w1 in unique_words
    }

    index_t2 = build_ngram_index(t2_tokens, config.k)
    if not index_t2:
        logger.debug("K-gram index for text2 is empty. Plagiarism score is 0.")
        return PlagiarismScore(overlap_percentage=0.0)

    max_overall_sw_score = 0.0

    for i in range(len(t1_tokens) - config.k + 1):
        current_kgram = tuple(t1_tokens[i : i + config.k])
        if current_kgram in index_t2:
            for j_start_pos_t2 in index_t2[current_kgram]:
                w1_start = max(0, i - config.window_radius)
                w1_end = min(len(t1_tokens), i + config.k + config.window_radius)
                w2_start = max(0, j_start_pos_t2 - config.window_radius)
                w2_end = min(len(t2_tokens), j_start_pos_t2 + config.k + config.window_radius)

                smith_waterman_params = SmithWatermanParams(
                    t1=t1_tokens,
                    t2=t2_tokens,
                    sm=scoring_matrix,
                    gap_penalty=config.gap_penalty,
                    win1=(w1_start, w1_end),
                    win2=(w2_start, w2_end),
                    mismatch_score=config.mismatch_score,
                )
                window_score = smith_waterman_window(smith_waterman_params)
                max_overall_sw_score = max(max_overall_sw_score, window_score)

    denominator = min(len(t1_tokens), len(t2_tokens)) * config.match_score
    normalized_score = 0.0 if denominator == 0 else max_overall_sw_score / denominator
    normalized_score = min(max(normalized_score, 0.0), 1.0)

    return PlagiarismScore(overlap_percentage=normalized_score)


def calculate_overlap_coefficient(text1: str, text2: str) -> OverlapCoefficient:
    """
    Calculate the overlap coefficient between two texts based on their token sets.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.

    Returns:
        OverlapCoefficient: Pydantic model containing the calculated coefficient.
    """
    set1 = set(simple_tokenize(text1))
    set2 = set(simple_tokenize(text2))
    intersection_len = len(set1.intersection(set2))
    min_set_len = min(len(set1), len(set2))
    coefficient = (intersection_len / min_set_len) if min_set_len > 0 else 0.0
    return OverlapCoefficient(coefficient=coefficient)


def calculate_sorensen_dice_coefficient(text1: str, text2: str) -> SorensenDiceCoefficient:
    """
    Calculate the Sørensen-Dice coefficient (or Dice score) between two texts based on their token sets.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.

    Returns:
        SorensenDiceCoefficient: Pydantic model containing the calculated coefficient.
    """
    set1 = set(simple_tokenize(text1))
    set2 = set(simple_tokenize(text2))
    intersection_len = len(set1.intersection(set2))
    sum_of_set_lengths = len(set1) + len(set2)
    coefficient = (2 * intersection_len / sum_of_set_lengths) if sum_of_set_lengths > 0 else 0.0
    return SorensenDiceCoefficient(coefficient=coefficient)


def get_char_by_char_equality_optimized(s1_in: Optional[str], s2_in: Optional[str]) -> CharEqualityScore:
    """
    Compare two strings character by character, applying a geometrically decaying weight for matches.

    Args:
        s1_in (Optional[str]): The first input string.
        s2_in (Optional[str]): The second input string.

    Returns:
        CharEqualityScore: Pydantic model containing the calculated score.
    """
    if s1_in is None or s2_in is None:
        logger.debug("One or both input strings are None for char_by_char_equality. Score is 0.")
        return CharEqualityScore(score=0.0)

    s1, s2 = str(s1_in), str(s2_in)
    min_len = min(len(s1), len(s2))
    total_score = 0.0
    current_weight = 1.0

    for i in range(min_len):
        if s1[i] == s2[i]:
            total_score += current_weight
        current_weight *= 0.5

    return CharEqualityScore(score=total_score)


def create_semantic_graph_spacy(text: str, spacy_nlp_model: Any) -> Optional[nx.Graph]:
    """
    Create a semantic graph from text using spaCy's dependency parse.

    Args:
        text (str): The input text string.
        spacy_nlp_model (Any): An initialized spaCy language model.

    Returns:
        Optional[nx.Graph]: A NetworkX graph representing the semantic structure.
    """
    if spacy_nlp_model is None:
        logger.warning("spaCy model (spacy_nlp_model) not loaded. Cannot create semantic graph.")
        return None
    doc = spacy_nlp_model(text)
    graph = nx.Graph()
    for token in doc:
        graph.add_node(token.i, text=token.text, lemma=token.lemma_, pos=token.pos_)
        for child in token.children:
            graph.add_edge(token.i, child.i, label=child.dep_)
    return graph


def calculate_semantic_graph_similarity_spacy(
    graph1: Optional[nx.Graph],
    graph2: Optional[nx.Graph],
) -> SemanticGraphSimilarity:
    """
    Calculate similarity between two semantic graphs based on Jaccard index of nodes and edges.

    Args:
        graph1 (Optional[nx.Graph]): The first semantic graph.
        graph2 (Optional[nx.Graph]): The second semantic graph.

    Returns:
        SemanticGraphSimilarity: Pydantic model containing the overall similarity.
    """
    if graph1 is None or graph2 is None or graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
        logger.debug("One or both semantic graphs are None or empty. Similarity is 0.")
        return SemanticGraphSimilarity(similarity=0.0, nodes_jaccard=0.0, edges_jaccard=0.0)

    nodes1, nodes2 = set(graph1.nodes), set(graph2.nodes)
    edges1, edges2 = set(graph1.edges), set(graph2.edges)

    intersection_nodes_len = len(nodes1.intersection(nodes2))
    union_nodes_len = len(nodes1.union(nodes2))
    nodes_jaccard = intersection_nodes_len / union_nodes_len if union_nodes_len > 0 else 0.0

    intersection_edges_len = len(edges1.intersection(edges2))
    union_edges_len = len(edges1.union(edges2))
    edges_jaccard = intersection_edges_len / union_edges_len if union_edges_len > 0 else 0.0

    overall_similarity = (nodes_jaccard + edges_jaccard) / 2.0

    return SemanticGraphSimilarity(
        similarity=overall_similarity,
        nodes_jaccard=nodes_jaccard,
        edges_jaccard=edges_jaccard,
    )


def preprocess_tfidf(text: str, *, lowercase: bool = True, remove_punct: bool = True) -> str:
    """
    Prepare text for TF-IDF vectorization by lowercasing and removing punctuation.

    Args:
        text (str): The input text string.
        lowercase (bool): Whether to convert the text to lowercase. Defaults to True.
        remove_punct (bool): Whether to remove punctuation. Defaults to True.

    Returns:
        str: The processed text string. Returns an empty string if input is not a string.
    """
    if not isinstance(text, str):
        logger.warning("preprocess_tfidf received non-string input: %s. Returning empty string.", type(text))
        return ""
    processed_text = text
    if lowercase:
        processed_text = processed_text.lower()
    if remove_punct:
        processed_text = re.sub(r"[^\w\s]", " ", processed_text)
    return processed_text.strip()


def extract_lexical_features(
    model_answers: list[str],
    student_answers: list[str],
    linkage_method: str = "average",
    distance_metric: str = "sqeuclidean",
    cluster_dist_thresh: float = 0.5,
) -> LexicalFeaturesAnalysis:
    """
    Extract lexical and clustering-based features for student answers relative to model answers.

    Args:
        model_answers (list[str]): A list of model/reference answer strings.
        student_answers (list[str]): A list of student answer strings.
        linkage_method (str): The linkage method to use for hierarchical clustering.
        distance_metric (str): The distance metric to use for `scipy.spatial.distance.pdist`.
        cluster_dist_thresh (float): The distance threshold used by `scipy.cluster.hierarchy.fcluster`.

    Returns:
        LexicalFeaturesAnalysis: A Pydantic model containing a list of `LexicalClusterFeature` objects.
    """
    if not model_answers or not student_answers:
        logger.warning(
            "extract_lexical_features: model_answers or student_answers list is empty. Returning empty features.",
        )
        return LexicalFeaturesAnalysis(student_features=[])

    all_texts_processed = [preprocess_tfidf(t) for t in model_answers + student_answers]
    num_models = len(model_answers)
    num_total_texts = len(all_texts_processed)

    vectorizer = TfidfVectorizer()
    tfidf_matrix: np.ndarray
    try:
        if not any(all_texts_processed):
            logger.warning(
                "All texts are empty after preprocessing in extract_lexical_features. Cannot compute TF-IDF.",
            )
            return LexicalFeaturesAnalysis(student_features=[])

        tfidf_matrix = vectorizer.fit_transform(all_texts_processed).toarray()
    except ValueError:
        logger.exception(
            "TF-IDF Vectorization error in extract_lexical_features. "
            "This can happen if texts are empty or contain only stopwords after preprocessing.",
        )
        return LexicalFeaturesAnalysis(student_features=[])

    min_samples_for_pdist = 2
    if tfidf_matrix.shape[0] < min_samples_for_pdist or tfidf_matrix.shape[1] == 0:
        logger.warning(
            f"Not enough samples ({tfidf_matrix.shape[0]}) or features ({tfidf_matrix.shape[1]}) "
            "for pdist. Cannot proceed with clustering-based lexical features.",
        )
        error_feature = LexicalClusterFeature(
            coph_min=0.0,
            coph_mean=0.0,
            coph_max=0.0,
            cluster_label=-1,
            cluster_size=0,
            is_outlier=1,
            silhouette=0.0,
        )
        return LexicalFeaturesAnalysis(student_features=[error_feature for _ in student_answers])

    pairwise_dist_matrix_condensed: np.ndarray = pdist(
        tfidf_matrix,
        metric=distance_metric,  # type: ignore[arg-type]
    )

    linkage_matrix: np.ndarray
    try:
        linkage_matrix = linkage(pairwise_dist_matrix_condensed, method=linkage_method)
    except ValueError:
        logger.exception(
            f"Linkage error in extract_lexical_features (TF-IDF matrix shape: {tfidf_matrix.shape}).",
        )
        error_feature = LexicalClusterFeature(
            coph_min=0.0,
            coph_mean=0.0,
            coph_max=0.0,
            cluster_label=-1,
            cluster_size=0,
            is_outlier=1,
            silhouette=0.0,
        )
        return LexicalFeaturesAnalysis(student_features=[error_feature for _ in student_answers])

    try:
        _cophenetic_corr_coeff, cophenetic_distances_condensed = cophenet(
            linkage_matrix,
            pairwise_dist_matrix_condensed,
        )
    except Exception:
        logger.exception("Cophenet calculation error. Using zero matrix for cophenetic distances.")
        cophenetic_distances_condensed = np.zeros_like(pairwise_dist_matrix_condensed)

    cophenetic_dist_matrix_square: np.ndarray = squareform(cophenetic_distances_condensed)

    cluster_labels: np.ndarray = fcluster(linkage_matrix, t=cluster_dist_thresh, criterion="distance")

    silhouette_vals: np.ndarray
    num_unique_labels = len(np.unique(cluster_labels))
    if num_unique_labels > 1 and num_unique_labels < num_total_texts:
        silhouette_vals = silhouette_samples(tfidf_matrix, cluster_labels, metric=distance_metric)
    else:
        logger.debug(
            f"Cannot compute meaningful silhouette scores (num_unique_labels={num_unique_labels}, "
            f"num_total_texts={num_total_texts}). Setting silhouette scores to 0.",
        )
        silhouette_vals = np.zeros(num_total_texts)

    student_feature_list: list[LexicalClusterFeature] = []
    for idx, _ in enumerate(student_answers):
        student_global_idx = num_models + idx

        student_coph_to_models: np.ndarray = cophenetic_dist_matrix_square[student_global_idx, :num_models]

        student_cluster_label_val: int = cluster_labels[student_global_idx].item()
        student_cluster_size_val: int = int((cluster_labels == student_cluster_label_val).sum())

        features_for_student = LexicalClusterFeature(
            coph_min=float(student_coph_to_models.min()) if student_coph_to_models.size > 0 else 0.0,
            coph_mean=float(student_coph_to_models.mean()) if student_coph_to_models.size > 0 else 0.0,
            coph_max=float(student_coph_to_models.max()) if student_coph_to_models.size > 0 else 0.0,
            cluster_label=student_cluster_label_val,
            cluster_size=student_cluster_size_val,
            is_outlier=int(student_cluster_size_val == 1),
            silhouette=float(silhouette_vals[student_global_idx].item()),
        )
        student_feature_list.append(features_for_student)

    return LexicalFeaturesAnalysis(student_features=student_feature_list)


def run_single_pair_text_analysis(
    inputs: SinglePairAnalysisInput,
    existing_graph: Optional[nx.Graph] = None,
) -> SinglePairAnalysisResult:
    """
    Analyzes a single model answer against a single student answer for various similarity metrics.

    Args:
        inputs (SinglePairAnalysisInput): A Pydantic model containing the model answer string,
                                          student text string, and parameters for plagiarism detection.
        existing_graph (Optional[nx.Graph]): An optional pre-built word co-occurrence graph.

    Returns:
        SinglePairAnalysisResult: A Pydantic model containing all computed similarity metrics.
    """
    logger.info(
        f"Starting single pair analysis for student text (first 30 chars): '{inputs.student_text[:30]}...' "
        f"and model answer (first 30 chars): '{inputs.model_answer[:30]}...'",
    )
    results = SinglePairAnalysisResult()

    graph_to_use: Optional[nx.Graph] = existing_graph
    if graph_to_use is None:
        logger.debug("No existing graph provided for single pair analysis; attempting to build a local one.")
        pair_word_vecs = create_word_vectors([inputs.model_answer, inputs.student_text])
        if pair_word_vecs.word_matrix_csr_scipy is not None and pair_word_vecs.words_vocabulary:
            graph_to_use = build_graph_efficiently(pair_word_vecs)
        else:
            logger.warning("Could not build local graph for single pair analysis (word vector creation failed).")

    if graph_to_use and graph_to_use.number_of_nodes() > 0:
        results.graph_similarity = calculate_graph_similarity(
            graph_to_use,
            inputs.student_text,
            inputs.model_answer,
        )
    else:
        logger.info("Graph for single pair analysis is empty or could not be built; skipping graph similarity.")
        results.graph_similarity = GraphSimilarityOutput(
            similarity_score=0.0,
            subgraph_nodes=0,
            subgraph_edges=0,
            message="Graph not available or empty for this pair.",
        )

    sw_config = SmithWatermanConfig(
        k=inputs.plagiarism_k,
        window_radius=inputs.plagiarism_window_radius,
    )
    results.plagiarism_score = compute_plagiarism_score_fast(
        inputs.student_text,
        inputs.model_answer,
        config=sw_config,
    )

    results.overlap_coefficient = calculate_overlap_coefficient(inputs.student_text, inputs.model_answer)
    results.dice_coefficient = calculate_sorensen_dice_coefficient(inputs.student_text, inputs.model_answer)
    results.char_equality_score = get_char_by_char_equality_optimized(inputs.student_text, inputs.model_answer)

    logger.info(f"Single pair analysis finished for student '{inputs.student_text[:30]}...'.")
    return results


def run_full_text_analysis(
    inputs: FullTextAnalysisInput,
) -> tuple[FullTextAnalysisResult, Optional[nx.Graph]]:
    """
    Orchestrates a comprehensive text analysis pipeline.

    Args:
        inputs (FullTextAnalysisInput): A Pydantic model containing lists of model answers
                                        and student texts.

    Returns:
        tuple[FullTextAnalysisResult, Optional[nx.Graph]]:
            - FullTextAnalysisResult: A Pydantic model containing all computed metrics and features.
            - Optional[nx.Graph]: The generated corpus graph if successfully built, otherwise None.
    """
    logger.info(
        "Starting full text analysis pipeline for %d student texts and %d model answers.",
        len(inputs.student_texts),
        len(inputs.model_answers),
    )
    results = FullTextAnalysisResult()
    corpus_graph: Optional[nx.Graph] = None

    all_corpus_texts_for_graph = inputs.model_answers + inputs.student_texts
    logger.info("Creating word vectors for corpus graph construction...")
    word_vec_result = create_word_vectors(all_corpus_texts_for_graph)

    if word_vec_result.word_matrix_csr_scipy is not None and word_vec_result.words_vocabulary:
        logger.info("Building corpus graph...")
        corpus_graph = build_graph_efficiently(word_vec_result)
        if corpus_graph and corpus_graph.number_of_nodes() > 0:
            results.corpus_graph_metrics = GraphMetrics(
                nodes=corpus_graph.number_of_nodes(),
                edges=corpus_graph.number_of_edges(),
                density=nx.density(corpus_graph) if corpus_graph.number_of_nodes() > 1 else 0.0,
            )
            logger.info(
                "Corpus graph built successfully: %d nodes, %d edges.",
                results.corpus_graph_metrics.nodes,
                results.corpus_graph_metrics.edges,
            )
        else:
            logger.warning("Corpus graph construction resulted in an empty or invalid graph.")
            corpus_graph = None
    else:
        logger.warning("Corpus graph could not be built: word vector creation failed or yielded empty results.")

    try:
        logger.info("Extracting lexical and clustering features for student answers...")
        lexical_features_result = extract_lexical_features(
            model_answers=inputs.model_answers,
            student_answers=inputs.student_texts,
            linkage_method=inputs.lexical_linkage_method,
            distance_metric=inputs.lexical_distance_metric,
            cluster_dist_thresh=inputs.lexical_cluster_dist_thresh,
        )
        results.student_lexical_features = lexical_features_result
        logger.info(
            "Lexical features extracted for %d students.",
            len(lexical_features_result.student_features) if lexical_features_result else 0,
        )
    except Exception:
        logger.exception("Error extracting lexical features.")
        results.student_lexical_features = None

    per_student_results_list: list[dict[str, Any]] = []
    logger.info("Starting per-student analysis against model answers...")

    for student_idx, student_text_item in enumerate(inputs.student_texts):
        student_specific_analysis_dict: dict[str, Any] = {"student_text_index": student_idx}

        graph_sims_to_models: list[float] = []
        plagiarism_scores_to_models: list[float] = []
        overlap_coeffs_to_models: list[float] = []
        dice_coeffs_to_models: list[float] = []
        char_eq_scores_to_models: list[float] = []

        for _, model_text_item in enumerate(inputs.model_answers):
            single_pair_input_params = SinglePairAnalysisInput(
                model_answer=model_text_item,
                student_text=student_text_item,
                plagiarism_k=inputs.plagiarism_k,
                plagiarism_window_radius=inputs.plagiarism_window_radius,
            )
            pair_analysis_result = run_single_pair_text_analysis(
                single_pair_input_params,
                existing_graph=corpus_graph,
            )

            if pair_analysis_result.graph_similarity:
                graph_sims_to_models.append(pair_analysis_result.graph_similarity.similarity_score)
            if pair_analysis_result.plagiarism_score:
                plagiarism_scores_to_models.append(pair_analysis_result.plagiarism_score.overlap_percentage)
            if pair_analysis_result.overlap_coefficient:
                overlap_coeffs_to_models.append(pair_analysis_result.overlap_coefficient.coefficient)
            if pair_analysis_result.dice_coefficient:
                dice_coeffs_to_models.append(pair_analysis_result.dice_coefficient.coefficient)
            if pair_analysis_result.char_equality_score:
                char_eq_scores_to_models.append(pair_analysis_result.char_equality_score.score)

        student_specific_analysis_dict["graph_similarity_to_model_avg"] = (
            np.mean(graph_sims_to_models).item() if graph_sims_to_models else None
        )
        student_specific_analysis_dict["plagiarism_score_to_model_max"] = (
            np.max(plagiarism_scores_to_models).item() if plagiarism_scores_to_models else None
        )
        student_specific_analysis_dict["overlap_coefficient_to_model_avg"] = (
            np.mean(overlap_coeffs_to_models).item() if overlap_coeffs_to_models else None
        )
        student_specific_analysis_dict["dice_coefficient_to_model_avg"] = (
            np.mean(dice_coeffs_to_models).item() if dice_coeffs_to_models else None
        )
        student_specific_analysis_dict["char_equality_to_model_avg"] = (
            np.mean(char_eq_scores_to_models).item() if char_eq_scores_to_models else None
        )
        per_student_results_list.append(student_specific_analysis_dict)
        logger.debug(f"Finished analysis for student index {student_idx}.")

    results.per_student_analysis = per_student_results_list
    logger.info("Full text analysis pipeline finished successfully.")
    return results, corpus_graph


if __name__ == "__main__":
    logger.info("Starting example script for text analysis...")

    model_answers_main = [
        "Education is the passport to the future, for tomorrow belongs to those who prepare for it today.",
        "The future belongs to those who prepare for it today; education is their passport.",
        "Effective learning strategies involve consistent practice and active engagement with the material.",
    ]
    student_texts_main = [
        "Tomorrow belongs to those who plan ahead; learning opens doors to tomorrow. Education is indeed key.",
        "Education is key to the future because those who learn early succeed. Preparation is important.",
        "Cooking recipes differ from studying methods for tomorrow's success. I like to bake cakes.",
        "The future belongs to those who prepare for it today; education is their passport. "
        "I agree with this statement.",
        "To succeed, one must prepare. Education is that preparation for what lies ahead in the future.",
        "",
        "   ",
    ]

    analysis_input_data = FullTextAnalysisInput(
        model_answers=model_answers_main,
        student_texts=student_texts_main,
        plagiarism_k=4,
        lexical_cluster_dist_thresh=0.6,
    )

    analysis_output: FullTextAnalysisResult
    main_corpus_graph: Optional[nx.Graph] = None
    try:
        analysis_output, main_corpus_graph = run_full_text_analysis(analysis_input_data)
    except Exception as e_main_analysis:
        logger.critical("Main analysis pipeline failed: %s", e_main_analysis, exc_info=True)
        sys.exit(1)

    print("\n--- Full Analysis Results ---")

    if analysis_output.corpus_graph_metrics:
        print("\nCorpus Graph Metrics:")
        print(f"  Nodes: {analysis_output.corpus_graph_metrics.nodes}")
        print(f"  Edges: {analysis_output.corpus_graph_metrics.edges}")
        print(
            f"  Density: {analysis_output.corpus_graph_metrics.density:.4f}"
            if analysis_output.corpus_graph_metrics.density is not None
            else "N/A",
        )

    if analysis_output.student_lexical_features:
        print("\nLexical/Clustering Features for Students:")
        for i, features in enumerate(analysis_output.student_lexical_features.student_features):
            print(f"  Student {i + 1} (Text: '{student_texts_main[i][:30]}...'):")
            print(f"    Cophenetic Mean to Models: {features.coph_mean:.4f}")
            print(f"    Cluster Label: {features.cluster_label}, Size: {features.cluster_size}")
            print(f"    Is Outlier: {'Yes' if features.is_outlier else 'No'}")
            print(f"    Silhouette Score: {features.silhouette:.4f}")

    print("\nPer-Student Similarity to Model Answers (Averages/Max):")
    for i, student_res in enumerate(analysis_output.per_student_analysis):
        print(
            f"  Student {i + 1} (Index {student_res.get('student_text_index')},"
            f" Text: '{student_texts_main[i][:30]}...'):",
        )
        gs_avg = student_res.get("graph_similarity_to_model_avg")
        ps_max = student_res.get("plagiarism_score_to_model_max")
        oc_avg = student_res.get("overlap_coefficient_to_model_avg")
        dc_avg = student_res.get("dice_coefficient_to_model_avg")
        ce_avg = student_res.get("char_equality_to_model_avg")

        print(f"    Avg Graph Similarity to Models: {gs_avg:.4f}" if gs_avg is not None else "N/A")
        print(f"    Max Plagiarism Score to Models: {ps_max:.4f}" if ps_max is not None else "N/A")
        print(f"    Avg Overlap Coefficient to Models: {oc_avg:.4f}" if oc_avg is not None else "N/A")
        print(f"    Avg Dice Coefficient to Models: {dc_avg:.4f}" if dc_avg is not None else "N/A")
        print(f"    Avg Char Equality to Models: {ce_avg:.4f}" if ce_avg is not None else "N/A")

    print("\n--- Single Pair Analysis Example ---")
    single_model = "The quick brown fox jumps over the lazy dog."
    single_student = "A fast, dark-colored fox leaps above a sleepy canine."

    single_pair_input_data = SinglePairAnalysisInput(
        model_answer=single_model,
        student_text=single_student,
        plagiarism_k=3,
        plagiarism_window_radius=20,
    )

    single_pair_result = run_single_pair_text_analysis(single_pair_input_data, existing_graph=main_corpus_graph)

    print(f"Results for single pair ('{single_model[:20]}...' vs '{single_student[:20]}...'):")
    if single_pair_result.graph_similarity:
        print(
            f"  Graph Similarity: {single_pair_result.graph_similarity.similarity_score:.4f} "
            f"(Subgraph Nodes: {single_pair_result.graph_similarity.subgraph_nodes}, "
            f"Message: {single_pair_result.graph_similarity.message})",
        )
    if single_pair_result.plagiarism_score:
        print(f"  Plagiarism Score: {single_pair_result.plagiarism_score.overlap_percentage:.4f}")
    if single_pair_result.overlap_coefficient:
        print(f"  Overlap Coefficient: {single_pair_result.overlap_coefficient.coefficient:.4f}")
    if single_pair_result.dice_coefficient:
        print(f"  Dice Coefficient: {single_pair_result.dice_coefficient.coefficient:.4f}")
    if single_pair_result.char_equality_score:
        print(f"  Char Equality Score: {single_pair_result.char_equality_score.score:.4f}")

    s1 = model_answers_main[0]
    s2 = student_texts_main[3]

    print(f"\n--- Individual Metric Examples ('{s1[:20]}...' vs '{s2[:20]}...') ---")
    individual_pair_input = SinglePairAnalysisInput(model_answer=s1, student_text=s2)
    individual_pair_results = run_single_pair_text_analysis(individual_pair_input, existing_graph=main_corpus_graph)

    if individual_pair_results.graph_similarity:
        print(f"Graph Similarity: {individual_pair_results.graph_similarity.similarity_score:.4f}")
    if individual_pair_results.plagiarism_score:
        print(f"Fast Plagiarism Overlap: {individual_pair_results.plagiarism_score.overlap_percentage:.2%}")
    if individual_pair_results.overlap_coefficient:
        print(f"Overlap Coefficient: {individual_pair_results.overlap_coefficient.coefficient:.4f}")
    if individual_pair_results.dice_coefficient:
        print(f"Sørensen-Dice Coefficient: {individual_pair_results.dice_coefficient.coefficient:.4f}")
    if individual_pair_results.char_equality_score:
        print(f"Character by Character Equality: {individual_pair_results.char_equality_score.score:.4f}")

    logger.info("Example script finished.")
