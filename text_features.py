# Use annotations for cleaner type hinting (requires Python 3.7+)
from __future__ import annotations

import logging
import re
import sys
import time
import warnings
from collections import defaultdict
from typing import Any, Optional  # Added Union earlier, now just using specific types

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field  # Pydantic imports
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.sparse import csr_matrix as ScipyCSRMatrix  # noqa: N812, TC002
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

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.feature_extraction.text",
)  # For get_feature_names_out


# --- Pydantic Models ---


class WordVectorCreationResult(BaseModel):
    """Result of word vector creation using CountVectorizer."""

    word_matrix_csr_scipy: Optional[ScipyCSRMatrix] = Field(
        default=None,
        description="Word-document matrix (CSR format). Scipy sparse matrix not directly serializable by Pydantic.",
    )
    words_vocabulary: list[str] = Field(default_factory=list, description="List of unique words in the vocabulary.")

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True  # To allow scipy.sparse.csr_matrix


class GraphMetrics(BaseModel):
    """Basic metrics for a graph."""

    nodes: int = Field(..., ge=0)
    edges: int = Field(..., ge=0)
    density: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SmithWatermanParams(BaseModel):
    """Parameter for the Smith-Waterman algorithm variant.

    Attributes:
        t1 (list[str]): First text as a list of tokens.
        t2 (list[str]): Second text as a list of tokens.
        sm (dict[str, dict[str, float]]): Scoring matrix for Smith-Waterman.
        gap_penalty (float): Gap penalty for Smith-Waterman.
        win1 (tuple[int, int]): Window boundaries for the first text.
        win2 (tuple[int, int]): Window boundaries for the second text.

    """

    t1: list[str]
    t2: list[str]
    sm: dict[str, dict[str, float]]
    gap_penalty: float
    win1: tuple[int, int]
    win2: tuple[int, int]


class LexicalClusterFeature(BaseModel):
    """Lexical and clustering features for a single student answer."""

    # cosine_min: float # Commented out in original
    # cosine_mean: float
    # cosine_max: float
    coph_min: float
    coph_mean: float
    coph_max: float
    cluster_label: int
    cluster_size: int
    is_outlier: int = Field(..., ge=0, le=1)  # bool as int
    silhouette: float
    # index: int # This was student_idx, might not be needed in the feature set itself
    # text: str # Original student text, might be too large for feature set


class LexicalFeaturesAnalysis(BaseModel):
    """List of lexical/clustering features for all student answers."""

    student_features: list[LexicalClusterFeature]


class SmithWatermanConfig(BaseModel):
    """Configuration for the Smith-Waterman algorithm variant."""

    k: int = Field(default=3, gt=0, description="The k-gram size for indexing.")
    window_radius: int = Field(default=50, gt=0, description="Radius around k-gram matches to apply Smith-Waterman.")
    match_score: float = Field(default=1.0, description="Score for matching tokens in Smith-Waterman.")
    mismatch_score: float = Field(default=0.0, description="Score for mismatching tokens.")
    gap_penalty: float = Field(default=-1.0, description="Penalty for gaps.")


class FullTextAnalysisInput(BaseModel):
    """Input for the comprehensive text analysis pipeline."""

    model_answers: list[str]
    student_texts: list[str]
    # Parameters for various methods could be added here if they need to be configurable
    plagiarism_k: int = Field(default=3, gt=0)
    plagiarism_window_radius: int = Field(default=50, gt=0)
    # Note: Smith-Waterman match_score, mismatch_score, gap_penalty will use defaults
    # from SmithWatermanConfig unless explicitly added here and passed through.
    lexical_linkage_method: str = Field(default="average")
    lexical_distance_metric: str = Field(default="sqeuclidean")  # Note: sklearn pdist uses 'sqeuclidean'
    lexical_cluster_dist_thresh: float = Field(default=0.5, gt=0)


class FullTextAnalysisResult(BaseModel):
    """Comprehensive results from analyzing student texts against model answers."""

    corpus_graph_metrics: Optional[GraphMetrics] = None
    student_lexical_features: Optional[LexicalFeaturesAnalysis] = None  # For all students
    # Storing per-pair results in a list of dicts or a more structured list of models
    per_student_analysis: list[dict[str, Any]] = Field(default_factory=list)
    # Example of what might go into per_student_analysis dict for each student:
    # {
    #     "student_text_index": int,
    #     "graph_similarity_to_model_avg": Optional[float],
    #     "plagiarism_score_to_model_max": Optional[float],
    #     "overlap_coefficient_to_model_avg": Optional[float],
    #     "dice_coefficient_to_model_avg": Optional[float],
    #     "char_equality_to_model_avg": Optional[float],
    #     "semantic_graph_similarity_to_model_avg": Optional[float] # if spaCy part is active
    # }


# --- Rebuild Pydantic models to resolve forward references and complex types ---
# This is particularly important for types like ScipyCSRMatrix with arbitrary_types_allowed.
MODULE_MODELS = [
    WordVectorCreationResult,
    GraphMetrics,
    # GraphSimilarityOutput,
    # PlagiarismScore,
    # OverlapCoefficient,
    # SorensenDiceCoefficient,
    # CharEqualityScore,
    # SemanticGraphSimilarity,
    LexicalClusterFeature,
    LexicalFeaturesAnalysis,
    SmithWatermanConfig,
    FullTextAnalysisInput,
    # SinglePairAnalysisInput,  # Added new model
    # SinglePairAnalysisResult,  # Added new model
    FullTextAnalysisResult,
    SmithWatermanParams,
]

for model_cls in MODULE_MODELS:
    model_cls.model_rebuild(force=True)


# --- spaCy SRL (Semantic Role Labeling) Section - Commented Out as in Original ---
# try:
#     import spacy # Moved import here
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     logger.info("Downloading 'en_core_web_sm' model for spaCy...")
#     try:
#         spacy.cli.download("en_core_web_sm")
#         nlp = spacy.load("en_core_web_sm")
#     except Exception as e_spacy_download:
#         logger.error("Failed to download/load spaCy model: %s", e_spacy_download)
#         nlp = None # Ensure nlp is defined
# except ImportError:
#     logger.info("spaCy library not installed. SRL features will be unavailable.")
#     nlp = None


# # Define two example sentences
# sentence1 = "The black cat sat calmly on the mat."
# sentence2 = "A white kitten sits near the mat."
# sentence3 = "The dog chased the cat quickly."
# sentence4 = "The cat was chased by the dog." # Passive voice, different structure

# # Process the sentences with spaCy
# if nlp:
#     doc1 = nlp(sentence1)
#     doc2 = nlp(sentence2)
#     doc3 = nlp(sentence3)
#     doc4 = nlp(sentence4)


# # Extract predicate-argument structures (improved version)
# def extract_predicate_arguments_improved(doc):
#     """
#     Extracts verbs as predicates and their key syntactic arguments (subj, obj).
#     Returns a dictionary mapping predicate lemma to a list of (dep_relation, argument_lemma) tuples.
#     """
#     predicate_arguments = defaultdict(list)
#     for token in doc:
#         # Consider all verbs as potential predicates
#         if token.pos_ == "VERB":
#             predicate_lemma = token.lemma_
#             arguments = []
#             for child in token.children:
#                 # Focus on core grammatical relations
#                 # nsubj: nominal subject, dobj: direct object,
#                 # nsubjpass: nominal subject (passive), auxpass: passive auxiliary (identifies passive)
#                 # We could add more like 'pobj' for prepositional objects if needed.
#                 if child.dep_ in ("nsubj", "dobj", "nsubjpass", "agent", "attr", "acomp", "xcomp"): # Added agent for passive, attr/acomp/xcomp for linking verbs/complements  # noqa: E501
#                      # Store the dependency relation and the argument's lemma
#                      arguments.append((child.dep_, child.lemma_))

#             # Handle passive voice slightly better (agent often attached to auxpass or verb)
#             if any(c.dep_ == 'auxpass' for c in token.children):
#                  for child in token.children:
#                      if child.dep_ == 'agent':
#                          for grand_child in child.children:
#                              if grand_child.dep_ == 'pobj': # Object of the 'by' preposition in agent phrase
#                                  arguments.append(('agent_pobj', grand_child.lemma_))


#             if arguments: # Only add if we found relevant arguments
#                 predicate_arguments[predicate_lemma].extend(arguments)

#     # Convert defaultdict back to dict for clarity if desired (optional)
#     return dict(predicate_arguments)


# # Calculate SRL similarity (improved version)
# def srl_similarity_improved(doc1, doc2):
#     srl_similarity_score = 0.0
#     match_count = 0

#     srl1 = extract_predicate_arguments_improved(doc1)
#     srl2 = extract_predicate_arguments_improved(doc2)

#     # Use predicate lemmas as keys
#     predicates1 = set(srl1.keys())
#     predicates2 = set(srl2.keys())

#     common_predicates = predicates1.intersection(predicates2)

#     if not common_predicates:
#         return 0.0

#     for predicate_lemma in common_predicates:
#         # Get argument lists for this common predicate
#         # srl1[predicate_lemma] is like [('nsubj', 'cat'), ('prep', 'on'), ...] but only has key deps
#         args1_list = srl1[predicate_lemma] # List of (dep, lemma) tuples
#         args2_list = srl2[predicate_lemma]

#         # Create sets of (dep, lemma) for easier comparison
#         args1_set = set(args1_list)
#         args2_set = set(args2_list)

#         # --- Similarity Calculation ---
#         # Option 1: Simple Jaccard on (dep, lemma) pairs
#         intersection_args = len(args1_set.intersection(args2_set))
#         union_args = len(args1_set.union(args2_set))
#         if union_args > 0:
#             similarity = intersection_args / union_args
#         else:
#             # Both empty -> perfect match? Or 0? Let's say 1.
#             similarity = 1.0 if not args1_list and not args2_list else 0.0

#         # Option 2 (More nuanced): Score based on lemma matches, boosted by role match
#         # similarity = 0
#         # arg1_lemmas = {lemma for dep, lemma in args1_list}
#         # arg2_lemmas = {lemma for dep, lemma in args2_list}
#         # common_lemmas = arg1_lemmas.intersection(arg2_lemmas)
#         # total_unique_lemmas = len(arg1_lemmas.union(arg2_lemmas))
#         # if total_unique_lemmas > 0:
#         #      base_sim = len(common_lemmas) / total_unique_lemmas
#         #      role_bonus = 0
#         #      matches_with_role = 0
#         #      for dep1, lemma1 in args1_list:
#         #           for dep2, lemma2 in args2_list:
#         #                if lemma1 == lemma2: # Lemma match
#         #                     if dep1 == dep2: # Role match
#         #                          matches_with_role += 1
#         #      # Simple bonus (could be more sophisticated)
#         #      if len(common_lemmas) > 0:
#         #           role_bonus = (matches_with_role / len(common_lemmas)) * 0.2 # Small bonus for role match
#         #      similarity = base_sim + role_bonus
#         # else:
#         #      similarity = 1.0 if not args1_list and not args2_list else 0.0
#         # similarity = min(similarity, 1.0) # Ensure score doesn't exceed 1

#         srl_similarity_score += similarity
#         match_count += 1


#     # Normalize score: Average similarity over the number of *matching* predicates
#     if match_count > 0:
#          normalized_score = srl_similarity_score / match_count
#     else:
#          normalized_score = 0.0 # No common predicates found

#     # Alternative normalization: Divide by total unique predicates?
#     # total_predicates = len(predicates1.union(predicates2))
#     # if total_predicates > 0:
#     #      normalized_score = srl_similarity_score / total_predicates
#     # else:
#     #      normalized_score = 1.0 if not predicates1 and not predicates2 else 0.0


#     return normalized_score

# if nlp:
#     # Calculate SRL similarity between the sentences
#     srl_sim_12 = srl_similarity_improved(doc1, doc2)
#     srl_sim_34 = srl_similarity_improved(doc3, doc4)
#     srl_sim_13 = srl_similarity_improved(doc1, doc3)

#     print(f"SRL Similarity (Sentence 1 vs 2): {srl_sim_12:.4f}")
# Expect higher score due to lemma match + passive handling
#     print(f"SRL Similarity (Sentence 3 vs 4): {srl_sim_34:.4f}")
#     print(f"SRL Similarity (Sentence 1 vs 3): {srl_sim_13:.4f}") # Expect lower score

#     # Optional: Print extracted structures to debug
#     print("\nExtracted SRL for Sentence 1:", extract_predicate_arguments_improved(doc1))
#     print("Extracted SRL for Sentence 2:", extract_predicate_arguments_improved(doc2))
#     print("Extracted SRL for Sentence 3:", extract_predicate_arguments_improved(doc3))
#     print("Extracted SRL for Sentence 4:", extract_predicate_arguments_improved(doc4))


def simple_tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, and split text into tokens."""
    if not isinstance(text, str):
        logger.warning("simple_tokenize received non-string input: %s. Returning empty list.", type(text))
        return []
    text_lower = text.lower()
    text_no_punct = re.sub(r"[^\w\s]", "", text_lower)
    return [token for token in text_no_punct.split() if token]


def create_word_vectors(texts: list[str]) -> WordVectorCreationResult:
    """Create word-document matrix using CountVectorizer.

    Args:
        texts: A list of text strings.

    Returns:
        WordVectorCreationResult containing the sparse matrix and vocabulary words.

    """
    # Use the tokenizer in CountVectorizer
    vectorizer = CountVectorizer(
        tokenizer=simple_tokenize,
        token_pattern=None,  # token_pattern=None when tokenizer is provided
    )
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not valid_texts:
        logger.warning("create_word_vectors: No valid texts provided. Returning empty result.")
        return WordVectorCreationResult()

    try:
        word_matrix: ScipyCSRMatrix = vectorizer.fit_transform(valid_texts)
        words: list[str] = vectorizer.get_feature_names_out().tolist()  # Convert numpy array to list
        return WordVectorCreationResult(word_matrix_csr_scipy=word_matrix, words_vocabulary=words)
    except Exception:  # Catching generic Exception
        logger.exception("Error in create_word_vectors:")  # Log the actual exception e
        return WordVectorCreationResult()  # Return empty on error


def build_graph_efficiently(word_matrix_result: WordVectorCreationResult) -> nx.Graph:
    """Build a word co-occurrence graph based on cosine similarity of word vectors.

    Args:
        word_matrix_result: Output from create_word_vectors.

    Returns:
        A networkx.Graph where nodes are words and edges are weighted by similarity.

    """
    if word_matrix_result.word_matrix_csr_scipy is None or not word_matrix_result.words_vocabulary:
        logger.warning("build_graph_efficiently: word_matrix or words vocabulary is empty. Returning empty graph.")
        return nx.Graph()

    word_matrix = word_matrix_result.word_matrix_csr_scipy
    words = word_matrix_result.words_vocabulary

    start_time = time.time()
    logger.info(f"Building graph for {len(words)} words...")

    # Word vectors (columns of word_matrix are word vectors)
    # Convert to dense array for cosine_similarity. Be cautious with very large vocabularies.
    try:
        # word_matrix is ScipyCSRMatrix, .T is ScipyCSCMatrix, both have .toarray()
        word_vectors: np.ndarray = word_matrix.T.toarray()
    except MemoryError:
        logger.exception(
            "MemoryError converting sparse matrix to dense for graph building. Vocab size: %s.",
            len(words),
        )
        # Potentially fallback to an approximate method or raise
        raise

    if word_vectors.shape[0] == 0:  # No words means no vectors
        logger.warning("Word vectors are empty. Returning empty graph.")
        return nx.Graph()

    similarity_matrix = cosine_similarity(word_vectors)
    logger.debug(f"  Similarity matrix calculation took {time.time() - start_time:.2f}s")

    graph_build_start = time.time()
    num_words = len(words)
    graph = nx.Graph()
    graph.add_nodes_from(words)

    similarity_threshold = 0.01  # Connect words with at least this similarity
    edges_added_count = 0

    # Iterate efficiently through the upper triangle of the similarity matrix
    for i in range(num_words):
        for j in range(i + 1, num_words):
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                graph.add_edge(words[i], words[j], weight=similarity)
                edges_added_count += 1

    logger.debug(f"  Graph construction (added {edges_added_count} edges) took {time.time() - graph_build_start:.2f}s")
    logger.info(
        f"Graph building finished. Total time: {time.time() - start_time:.2f}s. "
        f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}",
    )
    return graph


def calculate_graph_similarity(graph: nx.Graph, text1: str, text2: str) -> GraphSimilarityOutput:
    """Calculate similarity between two texts based on the density of their common words' subgraph.

    Args:
        graph: The global word co-occurrence graph.
        text1: The first text string.
        text2: The second text string.

    Returns:
        GraphSimilarityOutput containing the similarity score and subgraph metrics.

    """
    words1 = set(simple_tokenize(text1))
    words2 = set(simple_tokenize(text2))

    common_words_in_graph = list(words1.intersection(words2).intersection(graph.nodes))

    if not common_words_in_graph:
        logger.debug("No common words found in the graph for the given texts.")
        return GraphSimilarityOutput(
            similarity_score=0.0,
            subgraph_nodes=0,
            subgraph_edges=0,
            message="No common words in graph.",
        )

    # Create subgraph from common words that are present in the main graph
    subgraph = graph.subgraph(common_words_in_graph)
    sub_nodes = subgraph.number_of_nodes()
    sub_edges = subgraph.number_of_edges()

    # Density is typically defined for graphs with at least 2 nodes.
    # nx.density handles graphs with 0 or 1 node (returns 0 or NaN that we can catch).
    graph_sim_score = 0.0
    sub_density = None
    message = None
    min_sub_nodes = 2  # Minimum nodes for density calculation
    if sub_nodes < min_sub_nodes:  # Density undefined or 0 for graphs with < 2 nodes
        message = f"Subgraph has < 2 nodes ({sub_nodes}), density considered 0."
        # graph_sim_score remains 0.0
    else:
        try:
            sub_density = nx.density(subgraph)
            if np.isnan(sub_density):  # nx.density can return NaN for isolated nodes
                logger.debug("Subgraph density is NaN (likely isolated nodes), treating as 0.")
                sub_density = 0.0
            graph_sim_score = sub_density  # Use density as the similarity score
            message = f"Subgraph: {sub_nodes} nodes, {sub_edges} edges."
        except ZeroDivisionError:  # Should be caught by sub_nodes < 2, but as a safeguard
            message = "Density calculation failed (ZeroDivisionError)."
            # graph_sim_score remains 0.0

    logger.debug(f"Graph similarity: score={graph_sim_score:.4f}, {message}")
    return GraphSimilarityOutput(
        similarity_score=graph_sim_score,
        subgraph_nodes=sub_nodes,
        subgraph_edges=sub_edges,
        subgraph_density=sub_density,
        message=message,
    )


# --- Smith-Waterman and Other Similarity Functions ---


def preprocess_sw(text: str, *, lowercase: bool = True, remove_punct: bool = True) -> list[str]:
    """Preprocess text for Smith-Waterman: lowercase, remove punctuation, split."""
    if not isinstance(text, str):
        return []
    processed_text = text
    if lowercase:
        processed_text = processed_text.lower()
    if remove_punct:
        processed_text = re.sub(r"[^\w\s]", " ", processed_text)  # Replace with space to handle word boundaries
    return [token for token in processed_text.split() if token]  # Filter empty strings


def build_ngram_index(tokens: list[str], k: int) -> dict[tuple[str, ...], list[int]]:
    """Map each k-gram in tokens to a list of its starting positions."""
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    if k <= 0 or not tokens or len(tokens) < k:
        return dict(index)
    for i in range(len(tokens) - k + 1):
        key = tuple(tokens[i : i + k])
        index[key].append(i)
    return dict(index)


def smith_waterman_window(params: SmithWatermanParams) -> float:
    """Run Smith-Waterman on specified windows of t1 and t2, return max score in H matrix."""
    t1, t2, sm, gap_penalty, win1, win2 = params.t1, params.t2, params.sm, params.gap_penalty, params.win1, params.win2

    a1, a2 = t1[win1[0] : win1[1]], t2[win2[0] : win2[1]]
    n, m = len(a1), len(a2)
    if n == 0 or m == 0:
        return 0.0

    h_matrix = np.zeros((n + 1, m + 1), dtype=float)  # H matrix
    max_score_in_window = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Default to mismatch_score if words not in sm (should not happen if sm is built from unique words)
            match_val = sm.get(a1[i - 1], {}).get(a2[j - 1], -1.0)  # Assuming -1 for unknown mismatch

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
    """Compute a plagiarism score using a fast Smith-Waterman variant with k-gram indexing.

    Args:
        text1: The first text string.
        text2: The second text string.
        config: Configuration for the Smith-Waterman algorithm.

    Returns:
        PlagiarismScore model with the normalized overlap percentage.

    """
    t1_tokens = preprocess_sw(text1)
    t2_tokens = preprocess_sw(text2)

    if not t1_tokens or not t2_tokens:
        return PlagiarismScore(overlap_percentage=0.0)

    # Build scoring matrix (sm) for Smith-Waterman
    unique_words = set(t1_tokens) | set(t2_tokens)
    scoring_matrix = {
        w1: {w2: (config.match_score if w1 == w2 else config.mismatch_score) for w2 in unique_words}
        for w1 in unique_words
    }

    # Build k-gram index on the (presumably longer) text for efficiency, let's say text2
    index_t2 = build_ngram_index(t2_tokens, config.k)
    max_overall_sw_score = 0.0

    for i in range(len(t1_tokens) - config.k + 1):
        current_kgram = tuple(t1_tokens[i : i + config.k])
        if current_kgram in index_t2:
            for j_start_pos_t2 in index_t2[current_kgram]:
                # Define windows for Smith-Waterman application
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
                )
                window_score = smith_waterman_window(smith_waterman_params)
                max_overall_sw_score = max(max_overall_sw_score, window_score)

    # Normalize the highest Smith-Waterman score found
    # Denominator could be length of shorter text, or length of text1, or max possible score
    # Using length of shorter text in tokens
    denominator = min(len(t1_tokens), len(t2_tokens))
    normalized_score = (max_overall_sw_score / denominator) if denominator > 0 else 0.0
    # Ensure score is between 0 and 1 (it might exceed 1 if match_score > 1 and many matches)
    normalized_score = min(max(normalized_score, 0.0), 1.0)

    return PlagiarismScore(overlap_percentage=normalized_score)


def calculate_overlap_coefficient(text1: str, text2: str) -> OverlapCoefficient:
    """Calculate the overlap coefficient between two texts."""
    set1 = set(simple_tokenize(text1))  # Using simple_tokenize for consistency
    set2 = set(simple_tokenize(text2))
    intersection_len = len(set1.intersection(set2))
    min_len = min(len(set1), len(set2))
    coefficient = (intersection_len / min_len) if min_len > 0 else 0.0
    return OverlapCoefficient(coefficient=coefficient)


def calculate_sorensen_dice_coefficient(text1: str, text2: str) -> SorensenDiceCoefficient:
    """Calculate the Sørensen-Dice coefficient between two texts."""
    set1 = set(simple_tokenize(text1))
    set2 = set(simple_tokenize(text2))
    intersection_len = len(set1.intersection(set2))
    total_tokens_sum = len(set1) + len(set2)
    coefficient = (2 * intersection_len / total_tokens_sum) if total_tokens_sum > 0 else 0.0
    return SorensenDiceCoefficient(coefficient=coefficient)


def get_char_by_char_equality_optimized(s1_in: Optional[str], s2_in: Optional[str]) -> CharEqualityScore:
    """Compare two strings character by character with geometrically decaying weights."""
    if s1_in is None or s2_in is None:
        return CharEqualityScore(score=0.0)

    s1, s2 = str(s1_in), str(s2_in)  # Ensure strings
    min_len = min(len(s1), len(s2))
    total_score = 0.0
    current_weight = 1.0

    for i in range(min_len):
        if s1[i] == s2[i]:
            total_score += current_weight
        current_weight *= 0.5  # Geometric decay
    return CharEqualityScore(score=total_score)


def create_semantic_graph_spacy(text: str, spacy_nlp_model: Any) -> Optional[nx.Graph]:  # noqa: ANN401
    """Create a semantic graph from text using spaCy dependencies."""
    if spacy_nlp_model is None:
        logger.warning("spaCy model not loaded, cannot create semantic graph.")
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
    """Calculate similarity between two semantic graphs based on Jaccard index of nodes and edges."""
    if graph1 is None or graph2 is None or graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
        return SemanticGraphSimilarity(similarity=0.0, nodes_jaccard=0.0, edges_jaccard=0.0)

    nodes1, nodes2 = set(graph1.nodes), set(graph2.nodes)
    edges1, edges2 = set(graph1.edges), set(graph2.edges)

    union_nodes_len = len(nodes1.union(nodes2))
    nodes_jaccard = len(nodes1.intersection(nodes2)) / union_nodes_len if union_nodes_len > 0 else 0.0

    union_edges_len = len(edges1.union(edges2))
    edges_jaccard = len(edges1.intersection(edges2)) / union_edges_len if union_edges_len > 0 else 0.0

    # Combine node and edge similarities (simple average here)
    overall_similarity = (nodes_jaccard + edges_jaccard) / 2.0
    return SemanticGraphSimilarity(
        similarity=overall_similarity,
        nodes_jaccard=nodes_jaccard,
        edges_jaccard=edges_jaccard,
    )


# --- Lexical/Clustering Features ---


def preprocess_tfidf(text: str, *, lowercase: bool = True, remove_punct: bool = True) -> str:
    """Prepare text for TF-IDF: lowercase, remove punctuation."""
    if not isinstance(text, str):
        return ""
    processed_text = text
    if lowercase:
        processed_text = processed_text.lower()
    if remove_punct:
        processed_text = re.sub(r"[^\w\s]", " ", processed_text)  # Replace punct with space
    return processed_text.strip()  # Remove leading/trailing spaces


def extract_lexical_features(
    model_answers: list[str],
    student_answers: list[str],
    linkage_method: str = "average",
    distance_metric: str = "sqeuclidean",
    cluster_dist_thresh: float = 0.5,
) -> LexicalFeaturesAnalysis:
    """Extract lexical and clustering-based features for student answers relative to model answers.

    Args:
        model_answers: List of model/reference answer strings.
        student_answers: List of student answer strings.
        linkage_method: Linkage method for hierarchical clustering.
        distance_metric: Distance metric for pdist and silhouette_samples.
        cluster_dist_thresh: Distance threshold for forming flat clusters.

    Returns:
        LexicalFeaturesAnalysis model containing a list of features for each student.

    """
    if not model_answers or not student_answers:
        logger.warning("extract_lexical_features: model_answers or student_answers is empty.")
        return LexicalFeaturesAnalysis(student_features=[])

    all_texts_processed = [preprocess_tfidf(t) for t in model_answers + student_answers]
    num_models = len(model_answers)
    num_total_texts = len(all_texts_processed)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix: np.ndarray
    try:
        # Ensure there's content to vectorize
        if not any(all_texts_processed):
            logger.warning("All texts are empty after preprocessing in extract_lexical_features.")
            return LexicalFeaturesAnalysis(student_features=[])

        tfidf_matrix = vectorizer.fit_transform(all_texts_processed).toarray()
    except ValueError:  # Catch "empty vocabulary" errors
        logger.exception(
            "TF-IDF Vectorization error in extract_lexical_features: %s. "
            "This can happen if all texts are empty or contain only stopwords after preprocessing.",
        )
        # Return empty features if TF-IDF fails critically
        return LexicalFeaturesAnalysis(student_features=[])

    # Pairwise distances for clustering and cophenetic distance
    # pdist requires at least 2 samples if X is 1D, or 2 features if X is NxD (N>1)
    min_matrix_size = 2
    if tfidf_matrix.shape[0] < min_matrix_size or tfidf_matrix.shape[1] == 0:
        logger.warning(f"Not enough samples or features for pdist. Matrix shape: {tfidf_matrix.shape}")
        # Create default features for each student indicating an issue
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
        metric=distance_metric,  # type: ignore[arg-type] # Pylance stub issue for scipy pdist metric
    )

    # Hierarchical Clustering
    linkage_matrix: np.ndarray
    try:
        linkage_matrix = linkage(pairwise_dist_matrix_condensed, method=linkage_method)
    except ValueError:  # linkage needs more than 1 observation
        logger.exception(
            "Linkage error in extract_lexical_features. Matrix shape: %s.",
            tfidf_matrix.shape,
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

    cophenetic_coeffs: np.ndarray
    _, cophenetic_coeffs = cophenet(linkage_matrix, pairwise_dist_matrix_condensed)
    cophenetic_dist_matrix_square: np.ndarray = squareform(cophenetic_coeffs)

    # Flat Clustering and Silhouette Scores
    cluster_labels: np.ndarray = fcluster(linkage_matrix, t=cluster_dist_thresh, criterion="distance")

    silhouette_vals: np.ndarray
    num_unique_labels = len(np.unique(cluster_labels))
    if (
        num_unique_labels > 1 and num_unique_labels < num_total_texts
    ):  # silhouette_samples needs 1 < n_labels < n_samples
        silhouette_vals = silhouette_samples(tfidf_matrix, cluster_labels, metric=distance_metric)
    else:  # Not enough distinct clusters or all samples in one cluster
        silhouette_vals = np.zeros(num_total_texts)

    student_feature_list: list[LexicalClusterFeature] = []
    for idx, _ in enumerate(student_answers):
        student_global_idx = num_models + idx  # Index in the combined tfidf_matrix

        # Cophenetic distances from this student to all model answers
        student_coph_to_models: np.ndarray = cophenetic_dist_matrix_square[student_global_idx, :num_models]

        student_cluster_label_val: int = cluster_labels[student_global_idx].item()  # Use .item() for scalar
        student_cluster_size_val: int = int((cluster_labels == student_cluster_label_val).sum())

        features_for_student = LexicalClusterFeature(
            coph_min=float(student_coph_to_models.min()) if student_coph_to_models.size > 0 else 0.0,
            coph_mean=float(student_coph_to_models.mean()) if student_coph_to_models.size > 0 else 0.0,
            coph_max=float(student_coph_to_models.max()) if student_coph_to_models.size > 0 else 0.0,
            cluster_label=student_cluster_label_val,
            cluster_size=student_cluster_size_val,
            is_outlier=int(student_cluster_size_val == 1),
            silhouette=float(silhouette_vals[student_global_idx].item()),  # Use .item() for scalar
        )
        student_feature_list.append(features_for_student)

    return LexicalFeaturesAnalysis(student_features=student_feature_list)


# --- Main Analysis Orchestration Functions ---

import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.translate.meteor_score import meteor_score

reference = [["This", "is", "a", "reference", "summary"]]
generated = ["This", "is", "a", "generated", "summary"]
score = meteor_score(reference, generated)
print(score)


def run_single_pair_text_analysis(
    inputs: SinglePairAnalysisInput,
    existing_graph: Optional[nx.Graph] = None,
) -> SinglePairAnalysisResult:
    """Analyze a single model answer against a single student answer.

    Args:
        inputs: SinglePairAnalysisInput containing the model answer, student text, and parameters.
        existing_graph: An optional pre-built graph. If None, a local graph for the pair will be built.

    Returns:
        SinglePairAnalysisResult with computed similarity metrics for the pair.

    """
    logger.info(
        f"Starting single pair analysis for student text (first 30 chars): '{inputs.student_text[:30]}...' "
        f"and model answer (first 30 chars): '{inputs.model_answer[:30]}...'",
    )
    results = SinglePairAnalysisResult()

    # Graph Similarity
    graph_to_use: Optional[nx.Graph] = existing_graph
    if graph_to_use is None:
        logger.debug("No existing graph provided for single pair, building a local one.")
        pair_word_vecs = create_word_vectors([inputs.model_answer, inputs.student_text])
        if pair_word_vecs.word_matrix_csr_scipy is not None and pair_word_vecs.words_vocabulary:
            graph_to_use = build_graph_efficiently(pair_word_vecs)
        else:
            logger.warning("Could not build local graph for single pair analysis.")

    if graph_to_use and graph_to_use.number_of_nodes() > 0:
        results.graph_similarity = calculate_graph_similarity(
            graph_to_use,
            inputs.student_text,
            inputs.model_answer,
        )
    else:
        logger.info("Graph for single pair is empty or not built; skipping graph similarity.")
        results.graph_similarity = GraphSimilarityOutput(
            similarity_score=0.0,
            subgraph_nodes=0,
            subgraph_edges=0,
            message="Graph not available or empty.",
        )

    # Plagiarism Score
    sw_config = SmithWatermanConfig(
        k=inputs.plagiarism_k,
        window_radius=inputs.plagiarism_window_radius,
    )
    results.plagiarism_score = compute_plagiarism_score_fast(
        inputs.student_text,
        inputs.model_answer,
        config=sw_config,
    )

    # Other direct similarity metrics
    results.overlap_coefficient = calculate_overlap_coefficient(inputs.student_text, inputs.model_answer)
    results.dice_coefficient = calculate_sorensen_dice_coefficient(inputs.student_text, inputs.model_answer)
    results.char_equality_score = get_char_by_char_equality_optimized(inputs.student_text, inputs.model_answer)

    # Semantic Graph Similarity (Optional - requires spaCy setup)
    # try:
    #     if nlp: # Assuming nlp is loaded globally if this section is uncommented
    #         s_graph1 = create_semantic_graph_spacy(inputs.student_text, nlp)
    #         s_graph2 = create_semantic_graph_spacy(inputs.model_answer, nlp)
    #         results.semantic_graph_similarity = calculate_semantic_graph_similarity_spacy(s_graph1, s_graph2)
    #     else:
    #         logger.info("spaCy model (nlp) not available for semantic graph similarity in single pair analysis.")
    # except NameError: # If nlp is not defined at all
    #     logger.info("spaCy (nlp variable) not defined; skipping semantic graph similarity for single pair.")

    logger.info("Single pair analysis finished.")
    return results


def run_full_text_analysis(
    inputs: FullTextAnalysisInput,
) -> tuple[FullTextAnalysisResult, Optional[nx.Graph]]:
    """Orchestrate a full text analysis pipeline for multiple student texts against multiple model answers.

    Args:
        inputs: A FullTextAnalysisInput Pydantic model containing model answers,
                student texts, and any necessary parameters.

    Returns:
        A tuple containing:
            - FullTextAnalysisResult: Pydantic model with all computed metrics and features.
            - Optional[nx.Graph]: The corpus graph if built, otherwise None.

    """
    logger.info("Starting full text analysis pipeline...")
    results = FullTextAnalysisResult()  # Initialize empty result object
    corpus_graph: Optional[nx.Graph] = None  # Initialize corpus_graph

    # 1. Build Corpus Graph (optional, can be time-consuming)
    all_corpus_texts_for_graph = inputs.model_answers + inputs.student_texts
    word_vec_result = create_word_vectors(all_corpus_texts_for_graph)
    if word_vec_result.word_matrix_csr_scipy is not None and word_vec_result.words_vocabulary:
        corpus_graph = build_graph_efficiently(word_vec_result)
        if corpus_graph and corpus_graph.number_of_nodes() > 0:  # Check if graph was actually built
            results.corpus_graph_metrics = GraphMetrics(
                nodes=corpus_graph.number_of_nodes(),
                edges=corpus_graph.number_of_edges(),
                density=nx.density(corpus_graph) if corpus_graph.number_of_nodes() > 1 else 0.0,
            )
        else:
            logger.warning("Corpus graph was attempted but resulted in an empty graph.")
            corpus_graph = None  # Ensure it's None if empty
    else:
        logger.warning("Corpus graph could not be built due to issues with word vector creation.")

    # 2. Lexical/Clustering Features for all students
    try:
        lexical_features_result = extract_lexical_features(
            model_answers=inputs.model_answers,
            student_answers=inputs.student_texts,
            linkage_method=inputs.lexical_linkage_method,
            distance_metric=inputs.lexical_distance_metric,
            cluster_dist_thresh=inputs.lexical_cluster_dist_thresh,
        )
        results.student_lexical_features = lexical_features_result
    except Exception:
        logger.exception("Error extracting lexical features:")

    # 3. Per-student analysis (similarity to model answers)
    per_student_results_list: list[dict[str, Any]] = []

    for idx, student_text_item in enumerate(inputs.student_texts):
        student_specific_analysis: dict[str, Any] = {"student_text_index": idx}

        graph_sims_to_models_list: list[float] = []
        plagiarism_scores_to_models_list: list[float] = []
        overlap_coeffs_to_models_list: list[float] = []
        dice_coeffs_to_models_list: list[float] = []
        char_eq_scores_to_models_list: list[float] = []
        # semantic_graph_sims_to_models_list: list[float] = []

        for model_text_item in inputs.model_answers:
            # Use the single_pair_analysis function for individual metrics
            single_pair_input = SinglePairAnalysisInput(
                model_answer=model_text_item,
                student_text=student_text_item,
                plagiarism_k=inputs.plagiarism_k,
                plagiarism_window_radius=inputs.plagiarism_window_radius,
            )
            # Pass the pre-built corpus_graph to avoid rebuilding it for each pair for graph similarity
            pair_analysis_result = run_single_pair_text_analysis(single_pair_input, existing_graph=corpus_graph)

            if pair_analysis_result.graph_similarity:
                graph_sims_to_models_list.append(pair_analysis_result.graph_similarity.similarity_score)
            if pair_analysis_result.plagiarism_score:
                plagiarism_scores_to_models_list.append(pair_analysis_result.plagiarism_score.overlap_percentage)
            if pair_analysis_result.overlap_coefficient:
                overlap_coeffs_to_models_list.append(pair_analysis_result.overlap_coefficient.coefficient)
            if pair_analysis_result.dice_coefficient:
                dice_coeffs_to_models_list.append(pair_analysis_result.dice_coefficient.coefficient)
            if pair_analysis_result.char_equality_score:
                char_eq_scores_to_models_list.append(pair_analysis_result.char_equality_score.score)
            # if pair_analysis_result.semantic_graph_similarity:
            #     semantic_graph_sims_to_models_list.append(pair_analysis_result.semantic_graph_similarity.similarity)

        # Aggregate scores (e.g., average or max)
        student_specific_analysis["graph_similarity_to_model_avg"] = (
            np.mean(graph_sims_to_models_list).item() if graph_sims_to_models_list else None
        )
        student_specific_analysis["plagiarism_score_to_model_max"] = (
            np.max(plagiarism_scores_to_models_list).item() if plagiarism_scores_to_models_list else None
        )
        student_specific_analysis["overlap_coefficient_to_model_avg"] = (
            np.mean(overlap_coeffs_to_models_list).item() if overlap_coeffs_to_models_list else None
        )
        student_specific_analysis["dice_coefficient_to_model_avg"] = (
            np.mean(dice_coeffs_to_models_list).item() if dice_coeffs_to_models_list else None
        )
        student_specific_analysis["char_equality_to_model_avg"] = (
            np.mean(char_eq_scores_to_models_list).item() if char_eq_scores_to_models_list else None
        )
        # student_specific_analysis["semantic_graph_similarity_to_model_avg"] = (
        #    np.mean(semantic_graph_sims_to_models_list).item() if semantic_graph_sims_to_models_list else None
        # )

        per_student_results_list.append(student_specific_analysis)

    results.per_student_analysis = per_student_results_list
    logger.info("Full text analysis pipeline finished.")
    return results, corpus_graph


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging level for the example
    # logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    logger.info("Starting example script for text analysis...")

    # Example data
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
        "",  # Test with an empty student answer
        "   ",  # Test with a student answer with only spaces
    ]
    # human_scores_main = np.array([85, 90, 40, 100, 75]) # Example human scores

    # Create input model for full analysis
    analysis_input_data = FullTextAnalysisInput(
        model_answers=model_answers_main,
        student_texts=student_texts_main,
        plagiarism_k=4,  # Example override
        lexical_cluster_dist_thresh=0.6,  # Example override
    )

    # Run the full analysis
    analysis_output: FullTextAnalysisResult
    main_corpus_graph: Optional[nx.Graph] = None  # Initialize here
    try:
        analysis_output, main_corpus_graph = run_full_text_analysis(analysis_input_data)
    except Exception as e_main_analysis:
        logger.critical("Main analysis pipeline failed: %s", e_main_analysis, exc_info=True)
        sys.exit(1)

    # --- Print selected results from full analysis ---
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

    # --- Example: Run single pair analysis ---
    print("\n--- Single Pair Analysis Example ---")
    single_model = "The quick brown fox jumps over the lazy dog."
    single_student = "A fast, dark-colored fox leaps above a sleepy canine."

    single_pair_input_data = SinglePairAnalysisInput(
        model_answer=single_model,
        student_text=single_student,
        plagiarism_k=3,
        plagiarism_window_radius=20,
    )

    # We can pass the `main_corpus_graph` if we want to use it for graph similarity,
    # or let `run_single_pair_text_analysis` build a local one if `main_corpus_graph` is None or not suitable.
    # For this example, let's assume we might want to use the main graph if available.
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

    # --- Example: Using one of the full analysis pairs for individual metric check (as before) ---
    s1 = model_answers_main[0]  # "Education is the passport to the future..."
    s2 = student_texts_main[3]  # "The future belongs to those who prepare for it today; education is their passport..."

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

    # Semantic Graph Similarity (if spaCy was active and model loaded)
    # if 'nlp' in globals() and nlp: # Check if nlp was loaded (it's commented out above)
    #     spacy_graph1 = create_semantic_graph_spacy(s1, nlp)
    #     spacy_graph2 = create_semantic_graph_spacy(s2, nlp)
    #     sem_graph_sim_s1s2 = calculate_semantic_graph_similarity_spacy(spacy_graph1, spacy_graph2)
    #     print(f"Semantic Graph Similarity (spaCy, s1 vs s2): {sem_graph_sim_s1s2.similarity:.4f}")
    # else:
    #     print("spaCy model not loaded (or section commented out), skipping semantic graph similarity for s1 vs s2.")

    logger.info("Example script finished.")
