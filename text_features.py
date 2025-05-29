"""Text Feature Extraction and Analysis Module.

This module provides a comprehensive suite of tools for extracting various
features from textual data, primarily aimed at text comparison, similarity
assessment, and plagiarism detection in educational contexts (e.g., comparing
student answers to model answers).

Key Capabilities:
- Tokenization: Includes basic tokenization and utilities for preparing text for
  further analysis (e.g., `simple_tokenize`, `preprocess_sw`, `preprocess_tfidf`).
- Word Vectorization: Uses `CountVectorizer` to create word-document matrices
  (`create_word_vectors`).
- Graph-based Features:
    - Builds word co-occurrence graphs based on cosine similarity of word vectors
      derived from the corpus (`build_graph_efficiently`).
    - Calculates similarity between texts based on the density of their common
      words' subgraph within the corpus graph (`calculate_graph_similarity`).
- Plagiarism Detection:
    - Implements a fast variant of the Smith-Waterman algorithm using k-gram
      indexing and windowed application to identify local sequence similarities,
      normalized into a plagiarism score (`compute_plagiarism_score_fast`,
      `smith_waterman_window`).
- Lexical and Syntactic Features:
    - Calculates overlap coefficients (`calculate_overlap_coefficient`) and
      Sørensen-Dice coefficients (`calculate_sorensen_dice_coefficient`).
    - Provides a character-by-character equality score with geometric decay
      (`get_char_by_char_equality_optimized`).
    - (Commented out) Includes stubs for Semantic Role Labeling (SRL) similarity
      using spaCy, which could be activated for deeper semantic analysis.
- Clustering-based Lexical Features:
    - Extracts features based on hierarchical clustering of TF-IDF vectors of texts.
      This includes cophenetic distances to model answers, cluster labels, sizes,
      outlier status, and silhouette scores (`extract_lexical_features`).
      These features can help identify groups of similar texts.
- Orchestration:
    - `run_single_pair_text_analysis`: Analyzes a single pair of texts (e.g.,
      one student answer vs. one model answer) for various similarity metrics.
    - `run_full_text_analysis`: Orchestrates the analysis of multiple student texts
      against multiple model answers, generating a corpus graph, lexical features,
      and aggregated per-student similarity scores.

Pydantic Models:
The module extensively uses Pydantic models for structured input and output,
ensuring data validation and clear contracts for function arguments and return values.
Examples include `WordVectorCreationResult`, `GraphMetrics`, `SmithWatermanParams`,
`LexicalClusterFeature`, `FullTextAnalysisInput`, `FullTextAnalysisResult`, etc.

Dependencies:
- `networkx` for graph operations.
- `numpy` for numerical operations, especially array manipulations.
- `pydantic` for data modeling and validation.
- `scipy` for sparse matrices and clustering algorithms.
- `scikit-learn` for TF-IDF vectorization and cosine similarity.
- `nltk` (specifically for `meteor_score` in a commented-out example, but generally useful).
- `spacy` (optional, for commented-out semantic graph and SRL features).

The module is designed to be relatively modular, allowing individual feature
extraction functions to be used independently or as part of the larger analysis pipelines.
"""

# Use annotations for cleaner type hinting (requires Python 3.7+)
from __future__ import annotations

import logging
import re
import sys
import time
import warnings
from collections import defaultdict
from typing import Any, Optional  # Added Union earlier, now just using specific types

import nltk # Moved from bottom for E402
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
    """Holds the result of word vector creation using CountVectorizer.

    This model stores the sparse word-document matrix and the vocabulary list
    generated during the vectorization process.
    """

    word_matrix_csr_scipy: Optional[ScipyCSRMatrix] = Field(
        default=None,
        description=(
            "Word-document matrix in SciPy CSR (Compressed Sparse Row) format. "
            "This matrix typically has documents as rows and words (vocabulary) as columns. "
            "Note: SciPy sparse matrices are not directly serializable by Pydantic "
            "without `arbitrary_types_allowed=True`."
        ),
    )
    words_vocabulary: list[str] = Field(
        default_factory=list,
        description="List of unique words (features) identified by the vectorizer, forming the vocabulary.",
    )

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True  # Allows complex types like scipy.sparse.csr_matrix


class GraphMetrics(BaseModel):
    """Represents basic metrics of a graph.

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
    """Parameters for the Smith-Waterman local alignment algorithm variant.

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
        ..., description="Penalty for introducing a gap in the alignment (should be negative or zero).",
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
    """Stores lexical and clustering-based features for a single text (e.g., a student answer).

    These features are derived from comparing the text against a set of model/reference texts
    using TF-IDF and hierarchical clustering.
    """

    # Cosine similarity features were commented out in the original, retaining that.
    # cosine_min: float = Field(..., description="Minimum cosine similarity to model answers.")
    # cosine_mean: float = Field(..., description="Mean cosine similarity to model answers.")
    # cosine_max: float = Field(..., description="Maximum cosine similarity to model answers.")

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
    # student_idx: int = Field(..., description="Original index of the student text.") # Commented out
    # text: str = Field(..., description="The original student text.") # Commented out


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
    """Input parameters for the comprehensive text analysis pipeline (`run_full_text_analysis`).

    This model gathers all necessary data and configuration to analyze a collection
    of student texts against a collection of model answers.
    """

    model_answers: list[str] = Field(..., description="A list of model/reference answer strings.")
    student_texts: list[str] = Field(..., description="A list of student-provided text strings to be analyzed.")

    # Parameters for plagiarism detection (Smith-Waterman variant)
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
    # Note: Smith-Waterman match_score, mismatch_score, gap_penalty will use defaults
    # from SmithWatermanConfig unless explicitly added here and passed through to a
    # custom SmithWatermanConfig instance if that function is called directly.
    # For run_full_text_analysis, these are implicitly passed via the default SmithWatermanConfig.

    # Parameters for lexical feature extraction (hierarchical clustering)
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
    """Stores the comprehensive results from analyzing a set of student texts against model answers.

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
            "for one student text compared against all model answers. Keys in the dict "
            "indicate the type of score (e.g., 'graph_similarity_to_model_avg', "
            "'plagiarism_score_to_model_max')."
        ),
    )
    # Example structure for an item in `per_student_analysis`:
    # {
    #     "student_text_index": int, # Index of the student text in the input list
    #     "graph_similarity_to_model_avg": Optional[float], # Average graph similarity to model answers
    #     "plagiarism_score_to_model_max": Optional[float], # Maximum plagiarism score against any model answer
    #     "overlap_coefficient_to_model_avg": Optional[float], # Average overlap coefficient
    #     "dice_coefficient_to_model_avg": Optional[float], # Average Dice coefficient
    #     "char_equality_to_model_avg": Optional[float], # Average character equality score
    #     # "semantic_graph_similarity_to_model_avg": Optional[float] # If spaCy features were active
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
#                 if child.dep_ in ("nsubj", "dobj", "nsubjpass", "agent", "attr", "acomp", "xcomp"):
#                      # Added agent for passive, attr/acomp/xcomp for linking verbs/complements
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
    """Tokenizes a text string by converting to lowercase, removing punctuation, and splitting by whitespace.

    Args:
        text (str): The input text string.

    Returns:
        list[str]: A list of processed tokens. Returns an empty list if input is not a string.

    """
    if not isinstance(text, str):
        logger.warning("simple_tokenize received non-string input: %s. Returning empty list.", type(text))
        return []
    text_lower = text.lower()
    # Remove punctuation by replacing non-alphanumeric characters (excluding whitespace) with an empty string.
    text_no_punct = re.sub(r"[^\w\s]", "", text_lower)
    # Split by whitespace and filter out any empty strings that might result from multiple spaces.
    return [token for token in text_no_punct.split() if token]


def create_word_vectors(texts: list[str]) -> WordVectorCreationResult:
    """Creates a word-document matrix from a list of texts using CountVectorizer.

    The function uses `simple_tokenize` for tokenization. The resulting matrix
    is a sparse representation (CSR format) of word counts per document.

    Args:
        texts (list[str]): A list of text strings (documents).

    Returns:
        WordVectorCreationResult: A Pydantic model containing the sparse word-document
                                  matrix (`word_matrix_csr_scipy`) and the list of
                                  vocabulary words (`words_vocabulary`). Returns an empty
                                  result if no valid texts are provided or if an error occurs.

    """
    # Initialize CountVectorizer with the custom tokenizer.
    # `token_pattern=None` is important when a custom tokenizer is provided.
    vectorizer = CountVectorizer(
        tokenizer=simple_tokenize,
        token_pattern=None,
    )
    # Filter out any non-string or empty/whitespace-only texts to prevent errors in vectorizer.
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not valid_texts:
        logger.warning("create_word_vectors: No valid (non-empty, string) texts provided. Returning empty result.")
        return WordVectorCreationResult()

    try:
        # Fit the vectorizer to the texts and transform them into a word-document matrix.
        word_matrix: ScipyCSRMatrix = vectorizer.fit_transform(valid_texts)
        # Get the list of unique words (vocabulary) learned by the vectorizer.
        words: list[str] = vectorizer.get_feature_names_out().tolist()
        return WordVectorCreationResult(word_matrix_csr_scipy=word_matrix, words_vocabulary=words)
    except Exception as e:  # Catching generic Exception, consider more specific ones if known.
        logger.exception(f"Error in create_word_vectors during vectorization: {e}")
        return WordVectorCreationResult()  # Return empty result on error.


def build_graph_efficiently(word_matrix_result: WordVectorCreationResult) -> nx.Graph:
    """Builds a word co-occurrence graph based on cosine similarity of word vectors.

    Nodes in the graph are words from the vocabulary. An edge exists between two
    words if their cosine similarity (based on their occurrences across documents)
    exceeds a defined threshold. Edge weights represent this similarity.

    Args:
        word_matrix_result (WordVectorCreationResult): The output from `create_word_vectors`,
                                                     containing the word-document matrix
                                                     and vocabulary.

    Returns:
        nx.Graph: A NetworkX graph where nodes are words and edges are weighted by
                  their cosine similarity. Returns an empty graph if input is invalid
                  or if a MemoryError occurs.

    """
    if word_matrix_result.word_matrix_csr_scipy is None or not word_matrix_result.words_vocabulary:
        logger.warning("build_graph_efficiently: word_matrix or words vocabulary is empty. Returning empty graph.")
        return nx.Graph()

    word_matrix = word_matrix_result.word_matrix_csr_scipy
    words = word_matrix_result.words_vocabulary

    start_time = time.time()
    logger.info(f"Building graph for {len(words)} words...")

    # The word_matrix has documents as rows and words as columns.
    # To get word vectors (how words are distributed across documents), we need the transpose.
    # .T converts CSR to CSC (Compressed Sparse Column), which is efficient for column operations.
    try:
        # Convert word vectors to a dense array for `cosine_similarity`.
        # Note: Using .toarray() can be memory-intensive for very large vocabularies
        # as it creates a dense matrix. For extremely large datasets, alternative approaches
        # like sparse cosine similarity calculations or further optimizations
        # (e.g., feature selection, dimensionality reduction before this step)
        # might be necessary if memory becomes a constraint.
        word_vectors: np.ndarray = word_matrix.T.toarray()
    except MemoryError:
        logger.exception(
            "MemoryError converting sparse matrix to dense for graph building. "
            "Vocabulary size: %s. Consider using methods suitable for sparse "
            "matrices if memory is a constraint.",
            len(words),
        )
        # Fallback: return an empty graph or raise the error. Here, returning empty graph.
        return nx.Graph()  # Or `raise` if this should be a fatal error.

    if word_vectors.shape[0] == 0:  # Should correspond to len(words) == 0, already checked.
        logger.warning("Word vectors are empty (no words in vocabulary). Returning empty graph.")
        return nx.Graph()

    # Calculate pairwise cosine similarity between all word vectors.
    # `similarity_matrix[i, j]` will be the similarity between `words[i]` and `words[j]`.
    similarity_matrix = cosine_similarity(word_vectors)
    logger.debug(f"  Similarity matrix calculation took {time.time() - start_time:.2f}s")

    graph_build_start = time.time()
    num_words = len(words)
    graph = nx.Graph()
    graph.add_nodes_from(words)  # Add all words from the vocabulary as nodes.

    similarity_threshold = 0.01  # Minimum similarity for an edge to be created.
    edges_added_count = 0

    # Iterate through the upper triangle of the similarity matrix to avoid duplicate edges
    # and self-loops (i != j).
    for i in range(num_words):
        for j in range(i + 1, num_words):  # j starts from i + 1
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                # Add an edge between words[i] and words[j] with weight as their similarity.
                graph.add_edge(words[i], words[j], weight=similarity)
                edges_added_count += 1

    logger.debug(f"  Graph construction (added {edges_added_count} edges) took {time.time() - graph_build_start:.2f}s")
    logger.info(
        f"Graph building finished. Total time: {time.time() - start_time:.2f}s. "
        f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}",
    )
    return graph


def calculate_graph_similarity(graph: nx.Graph, text1: str, text2: str) -> GraphSimilarityOutput:
    """Calculates similarity between two texts based on the density of their common words' subgraph.

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
                               Returns a score of 0.0 if no common words are found in the graph.

    """
    # Tokenize the input texts to get sets of unique words.
    words1 = set(simple_tokenize(text1))
    words2 = set(simple_tokenize(text2))

    # Find common words between the two texts that are also present as nodes in the main graph.
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

    # Create a subgraph consisting only of these common words and their edges from the main graph.
    subgraph = graph.subgraph(common_words_in_graph)
    sub_nodes = subgraph.number_of_nodes()
    sub_edges = subgraph.number_of_edges()

    graph_sim_score = 0.0
    sub_density_val: Optional[float] = None  # Explicitly Optional float
    message: Optional[str] = None  # Explicitly Optional str

    # Density is typically meaningful for graphs with at least 2 nodes.
    # nx.density() handles graphs with 0 or 1 node (returns 0 or NaN).
    min_sub_nodes_for_density = 2
    if sub_nodes < min_sub_nodes_for_density:
        message = f"Subgraph has < {min_sub_nodes_for_density} nodes ({sub_nodes}), so density is considered 0."
        # graph_sim_score remains 0.0, sub_density_val remains None (or can be set to 0.0)
        sub_density_val = 0.0
    else:
        try:
            current_density = nx.density(subgraph)
            # nx.density can return NaN for graphs with nodes but no possible edges (e.g., isolated nodes if N < 2, caught above)
            # or if the graph is degenerate in some way for the formula.
            if np.isnan(current_density):
                logger.debug(
                    "Subgraph density calculated as NaN (likely isolated nodes or specific graph structure), treating as 0.",
                )
                sub_density_val = 0.0
            else:
                sub_density_val = float(current_density)  # Ensure it's a float
            graph_sim_score = sub_density_val  # Use density as the similarity score.
            message = f"Subgraph successfully created: {sub_nodes} nodes, {sub_edges} edges."
        except ZeroDivisionError:  # Should be caught by sub_nodes < 2, but as a safeguard.
            message = "Density calculation failed due to ZeroDivisionError (unexpected for N >= 2)."
            sub_density_val = 0.0  # graph_sim_score remains 0.0

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


# --- Smith-Waterman and Other Similarity Functions ---


def preprocess_sw(text: str, *, lowercase: bool = True, remove_punct: bool = True) -> list[str]:
    """Preprocesses text for Smith-Waterman algorithm by lowercasing, removing punctuation, and splitting into tokens.

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
        # Replace punctuation with space to handle word boundaries correctly, then split.
        processed_text = re.sub(r"[^\w\s]", " ", processed_text)
    # Split by whitespace and filter out any empty strings resulting from multiple spaces or leading/trailing spaces.
    return [token for token in processed_text.split() if token]


def build_ngram_index(tokens: list[str], k: int) -> dict[tuple[str, ...], list[int]]:
    """Builds an index mapping each k-gram in a list of tokens to all its starting positions.

    This index is used in the fast Smith-Waterman variant to quickly find
    potential matching regions between two texts.

    Args:
        tokens (list[str]): A list of tokens representing a text.
        k (int): The size of the k-grams to index.

    Returns:
        dict[tuple[str, ...], list[int]]: A dictionary where keys are k-gram tuples
                                          and values are lists of starting indices
                                          where each k-gram appears in the `tokens` list.
                                          Returns an empty dictionary if k is invalid,
                                          tokens are empty, or tokens length is less than k.

    """
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    # Validate inputs: k must be positive, tokens list must not be empty and must be at least k tokens long.
    if k <= 0 or not tokens or len(tokens) < k:
        logger.debug(
            "Invalid input for build_ngram_index (k=%s, len(tokens)=%s). Returning empty index.", k, len(tokens),
        )
        return dict(index)  # Return as a plain dict

    # Slide a window of size k across the tokens.
    for i in range(len(tokens) - k + 1):
        kgram = tuple(tokens[i : i + k])  # Create a tuple for the k-gram (tuples are hashable for dict keys).
        index[kgram].append(i)  # Store the starting position of this k-gram.
    return dict(index)  # Convert defaultdict to dict for the final output.


def smith_waterman_window(params: SmithWatermanParams) -> float:
    """Performs the Smith-Waterman local alignment algorithm on specified windows of two token lists.

    This function calculates the alignment scores within the given windows and returns
    the maximum score found in the H (scoring) matrix. This score represents the
    highest degree of local similarity between the two token subsequences.

    Args:
        params (SmithWatermanParams): A Pydantic model containing the token lists (t1, t2),
                                      scoring matrix (sm), gap penalty, and window
                                      boundaries (win1, win2) for both token lists.

    Returns:
        float: The maximum alignment score found within the specified windows.
               Returns 0.0 if either window results in an empty token list.

    """
    # Destructure parameters for convenience
    t1, t2, sm, gap_penalty, win1, win2 = params.t1, params.t2, params.sm, params.gap_penalty, params.win1, params.win2

    # Extract the subsequences (windows) from the original token lists
    a1 = t1[win1[0] : win1[1]]
    a2 = t2[win2[0] : win2[1]]
    n, m = len(a1), len(a2)

    # If either subsequence is empty, no alignment is possible.
    if n == 0 or m == 0:
        return 0.0

    # Initialize the H matrix (scoring matrix) with zeros.
    # Dimensions are (n+1) x (m+1) to accommodate initial gaps.
    h_matrix = np.zeros((n + 1, m + 1), dtype=float)
    max_score_in_window = 0.0  # To track the highest score in this window.

    # Fill the H matrix based on Smith-Waterman recurrence relation.
    for i in range(1, n + 1):  # Iterate through tokens of the first window (a1)
        for j in range(1, m + 1):  # Iterate through tokens of the second window (a2)
            # Get match/mismatch score from the provided scoring matrix `sm`.
            # `a1[i-1]` and `a2[j-1]` are the current tokens being compared.
            # Default to a mismatch_score (e.g., -1.0 or as defined in `sm`) if a word pair is not explicitly in `sm`.
            # This shouldn't happen if `sm` is built correctly from all unique words in t1 and t2.
            match_val = sm.get(a1[i - 1], {}).get(a2[j - 1], params.mismatch_score)

            # Calculate scores from three possible previous cells:
            # 1. Diagonal: Alignment of a1[i-1] and a2[j-1]
            score_diag = h_matrix[i - 1, j - 1] + match_val
            # 2. Up: Gap in a2 (aligning a1[i-1] with a gap)
            score_up = h_matrix[i - 1, j] + gap_penalty
            # 3. Left: Gap in a1 (aligning a2[j-1] with a gap)
            score_left = h_matrix[i, j - 1] + gap_penalty

            # Current cell value is the maximum of these, but not less than 0 (characteristic of local alignment).
            h_matrix[i, j] = max(0, score_diag, score_up, score_left)
            # Update the maximum score found so far in this window.
            max_score_in_window = max(max_score_in_window, h_matrix[i, j])

    return max_score_in_window


def compute_plagiarism_score_fast(
    text1: str,
    text2: str,
    config: SmithWatermanConfig,
) -> PlagiarismScore:
    """Computes a plagiarism score using a k-gram indexed, windowed Smith-Waterman algorithm.

    This method speeds up alignment by first finding exact k-gram matches between
    the two texts. Smith-Waterman is then applied only to windows around these
    k-gram matches, rather than the entire texts. The highest score obtained from
    these windowed alignments is normalized by the length of the shorter text to
    produce a plagiarism score between 0 and 1.

    Args:
        text1 (str): The first text string (e.g., student text).
        text2 (str): The second text string (e.g., potential source text or model answer).
        config (SmithWatermanConfig): Configuration for the Smith-Waterman algorithm,
                                      including k-gram size, window radius, match/mismatch
                                      scores, and gap penalty.

    Returns:
        PlagiarismScore: A Pydantic model containing the normalized `overlap_percentage`
                         (plagiarism score). Returns a score of 0.0 if either text is empty
                         after preprocessing.

    """
    # Preprocess texts into lists of tokens.
    t1_tokens = preprocess_sw(text1)
    t2_tokens = preprocess_sw(text2)

    if not t1_tokens or not t2_tokens:
        logger.debug("One or both texts are empty after preprocessing. Plagiarism score is 0.")
        return PlagiarismScore(overlap_percentage=0.0)

    # Build a scoring matrix (sm) for Smith-Waterman based on unique words in both texts.
    # This defines scores for matching or mismatching any pair of words.
    unique_words = set(t1_tokens) | set(t2_tokens)
    scoring_matrix = {
        w1: {w2: (config.match_score if w1 == w2 else config.mismatch_score) for w2 in unique_words}
        for w1 in unique_words
    }

    # Build a k-gram index on one of the texts (e.g., text2, assuming it might be longer or the source).
    # This index maps k-grams to their starting positions in t2_tokens.
    index_t2 = build_ngram_index(t2_tokens, config.k)
    if not index_t2:  # If index is empty (e.g. t2_tokens shorter than k)
        logger.debug("K-gram index for text2 is empty. Plagiarism score is 0.")
        return PlagiarismScore(overlap_percentage=0.0)

    max_overall_sw_score = 0.0

    # Iterate through k-grams in the first text (t1_tokens).
    for i in range(len(t1_tokens) - config.k + 1):
        current_kgram = tuple(t1_tokens[i : i + config.k])
        # If this k-gram from t1 exists in the index of t2:
        if current_kgram in index_t2:
            # For each occurrence of this k-gram in t2:
            for j_start_pos_t2 in index_t2[current_kgram]:
                # Define windows around the k-gram match in both texts.
                # Window start: k-gram start - window_radius (but not less than 0).
                # Window end: k-gram start + k + window_radius (but not beyond text length).
                w1_start = max(0, i - config.window_radius)
                w1_end = min(len(t1_tokens), i + config.k + config.window_radius)
                w2_start = max(0, j_start_pos_t2 - config.window_radius)
                w2_end = min(len(t2_tokens), j_start_pos_t2 + config.k + config.window_radius)

                # Prepare parameters for Smith-Waterman on these windows.
                smith_waterman_params = SmithWatermanParams(
                    t1=t1_tokens,
                    t2=t2_tokens,
                    sm=scoring_matrix,
                    gap_penalty=config.gap_penalty,
                    win1=(w1_start, w1_end),
                    win2=(w2_start, w2_end),
                    mismatch_score=config.mismatch_score,
                )
                # Calculate Smith-Waterman score for this specific pair of windows.
                window_score = smith_waterman_window(smith_waterman_params)
                # Keep track of the maximum score found across all windows.
                max_overall_sw_score = max(max_overall_sw_score, window_score)

    # Normalize the highest Smith-Waterman score.
    # A common normalization is by the length of the shorter text (in tokens),
    # assuming match_score is 1. This gives a sense of overlap percentage.
    denominator = (
        min(len(t1_tokens), len(t2_tokens)) * config.match_score
    )  # Max possible score for shorter text if perfect match
    if denominator == 0:  # Avoid division by zero if match_score is 0 or texts were empty
        normalized_score = 0.0
    else:
        normalized_score = max_overall_sw_score / denominator

    # Ensure the score is clamped between 0.0 and 1.0.
    # It might exceed 1.0 if match_score > 1 or if normalization logic changes.
    normalized_score = min(max(normalized_score, 0.0), 1.0)

    return PlagiarismScore(overlap_percentage=normalized_score)


def calculate_overlap_coefficient(text1: str, text2: str) -> OverlapCoefficient:
    """Calculates the overlap coefficient between two texts based on their token sets.

    The overlap coefficient is defined as: |set1 intersect set2| / min(|set1|, |set2|).
    It measures the degree of overlap relative to the smaller set.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.

    Returns:
        OverlapCoefficient: Pydantic model containing the calculated coefficient.
                            Returns 0.0 if the smaller set is empty.

    """
    set1 = set(simple_tokenize(text1))
    set2 = set(simple_tokenize(text2))
    intersection_len = len(set1.intersection(set2))
    min_set_len = min(len(set1), len(set2))
    coefficient = (intersection_len / min_set_len) if min_set_len > 0 else 0.0
    return OverlapCoefficient(coefficient=coefficient)


def calculate_sorensen_dice_coefficient(text1: str, text2: str) -> SorensenDiceCoefficient:
    """Calculates the Sørensen-Dice coefficient (or Dice score) between two texts based on their token sets.

    The Sørensen-Dice coefficient is defined as: 2 * |set1 intersect set2| / (|set1| + |set2|).
    It measures the similarity between two sets, ranging from 0 (no overlap) to 1 (identical sets).

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.

    Returns:
        SorensenDiceCoefficient: Pydantic model containing the calculated coefficient.
                                 Returns 0.0 if both sets are empty.

    """
    set1 = set(simple_tokenize(text1))
    set2 = set(simple_tokenize(text2))
    intersection_len = len(set1.intersection(set2))
    sum_of_set_lengths = len(set1) + len(set2)
    coefficient = (2 * intersection_len / sum_of_set_lengths) if sum_of_set_lengths > 0 else 0.0
    # If both sets are empty, sum_of_set_lengths is 0. If intersection_len is also 0,
    # this correctly yields 0.0. If both sets were identical and non-empty, it's 1.0.
    # If both sets are empty and we want to consider them perfectly similar, this should be 1.0.
    # However, standard Dice for empty sets is 0. Let's stick to that unless a different definition is required.
    # If both sets are empty, len(set1)=0, len(set2)=0, intersection_len=0, sum_of_set_lengths=0 -> coefficient=0.0. This is fine.
    return SorensenDiceCoefficient(coefficient=coefficient)


def get_char_by_char_equality_optimized(s1_in: Optional[str], s2_in: Optional[str]) -> CharEqualityScore:
    """Compares two strings character by character, applying a geometrically decaying weight for matches.

    The score starts with a weight of 1.0 for the first character match. Each subsequent
    match contributes its current weight to the total score, and the weight for the
    next position is halved. This gives higher importance to initial sequence matches.

    Args:
        s1_in (Optional[str]): The first input string.
        s2_in (Optional[str]): The second input string.

    Returns:
        CharEqualityScore: Pydantic model containing the calculated score.
                           Returns a score of 0.0 if either input string is None.

    """
    if s1_in is None or s2_in is None:
        logger.debug("One or both input strings are None for char_by_char_equality. Score is 0.")
        return CharEqualityScore(score=0.0)

    s1, s2 = str(s1_in), str(s2_in)  # Ensure inputs are strings.
    min_len = min(len(s1), len(s2))  # Compare up to the length of the shorter string.
    total_score = 0.0
    current_weight = 1.0  # Initial weight for a match at the first position.

    for i in range(min_len):
        if s1[i] == s2[i]:  # If characters at the current position match.
            total_score += current_weight
        current_weight *= 0.5  # Geometric decay for the weight at the next position.

    return CharEqualityScore(score=total_score)


def create_semantic_graph_spacy(text: str, spacy_nlp_model: Any) -> Optional[nx.Graph]:  # noqa: ANN401
    """Creates a semantic graph from text using spaCy's dependency parse.

    Nodes in the graph represent tokens, identified by their index in the document.
    Node attributes include the token's text, lemma, and part-of-speech tag.
    Edges represent syntactic dependencies between tokens, labeled with the dependency type.
    This function is part of the commented-out spaCy integration.

    Args:
        text (str): The input text string.
        spacy_nlp_model (Any): An initialized spaCy language model (e.g., `nlp = spacy.load("en_core_web_sm")`).

    Returns:
        Optional[nx.Graph]: A NetworkX graph representing the semantic structure,
                            or None if the spaCy model is not available/loaded.

    """
    if spacy_nlp_model is None:  # Guard against uninitialized spaCy model
        logger.warning("spaCy model (spacy_nlp_model) not loaded. Cannot create semantic graph.")
        return None
    # Process the text with the spaCy model to get a Doc object.
    doc = spacy_nlp_model(text)
    graph = nx.Graph()
    # Iterate through tokens in the processed document.
    for token in doc:
        # Add each token as a node. Node ID is token.i (index of token in Doc).
        # Store text, lemma, and POS tag as node attributes.
        graph.add_node(token.i, text=token.text, lemma=token.lemma_, pos=token.pos_)
        # Add edges for syntactic dependencies.
        # An edge connects the current token (head) to its children in the dependency tree.
        for child in token.children:
            graph.add_edge(token.i, child.i, label=child.dep_)  # Edge label is the dependency type.
    return graph


def calculate_semantic_graph_similarity_spacy(
    graph1: Optional[nx.Graph],
    graph2: Optional[nx.Graph],
) -> SemanticGraphSimilarity:
    """Calculates similarity between two semantic graphs (from spaCy) based on Jaccard index of nodes and edges.

    This function assumes graphs are generated by `create_semantic_graph_spacy` or have a similar structure.
    The overall similarity is a simple average of the Jaccard similarity of their node sets
    and the Jaccard similarity of their edge sets.
    This function is part of the commented-out spaCy integration.


    Args:
        graph1 (Optional[nx.Graph]): The first semantic graph.
        graph2 (Optional[nx.Graph]): The second semantic graph.

    Returns:
        SemanticGraphSimilarity: Pydantic model containing the overall similarity,
                                 node Jaccard index, and edge Jaccard index.
                                 Returns all zeros if either graph is None or empty.

    """
    # Handle cases where graphs are None or empty to prevent errors.
    if graph1 is None or graph2 is None or graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
        logger.debug("One or both semantic graphs are None or empty. Similarity is 0.")
        return SemanticGraphSimilarity(similarity=0.0, nodes_jaccard=0.0, edges_jaccard=0.0)

    # Get sets of nodes and edges for Jaccard calculation.
    # Node comparison is based on node IDs (token indices from spaCy).
    nodes1, nodes2 = set(graph1.nodes), set(graph2.nodes)
    # Edge comparison is based on (node1_id, node2_id) tuples.
    # Note: For undirected graphs, NetworkX might store edges as (u,v) where u<v.
    # If graphs could have different ordering for the same edge, normalization might be needed
    # (e.g., `set(tuple(sorted(edge)) for edge in graph1.edges)`), but typically not an issue here.
    edges1, edges2 = set(graph1.edges), set(graph2.edges)

    # Calculate Jaccard similarity for nodes.
    intersection_nodes_len = len(nodes1.intersection(nodes2))
    union_nodes_len = len(nodes1.union(nodes2))
    nodes_jaccard = intersection_nodes_len / union_nodes_len if union_nodes_len > 0 else 0.0

    # Calculate Jaccard similarity for edges.
    intersection_edges_len = len(edges1.intersection(edges2))
    union_edges_len = len(edges1.union(edges2))
    edges_jaccard = intersection_edges_len / union_edges_len if union_edges_len > 0 else 0.0

    # Combine node and edge similarities. A simple average is used here.
    # Other weighting schemes could be applied if desired.
    overall_similarity = (nodes_jaccard + edges_jaccard) / 2.0

    return SemanticGraphSimilarity(
        similarity=overall_similarity,
        nodes_jaccard=nodes_jaccard,  # Jaccard index of graph nodes
        edges_jaccard=edges_jaccard,  # Jaccard index of graph edges
    )


# --- Lexical/Clustering Features ---


def preprocess_tfidf(text: str, *, lowercase: bool = True, remove_punct: bool = True) -> str:
    """Prepares text for TF-IDF vectorization by lowercasing and removing punctuation.

    Args:
        text (str): The input text string.
        lowercase (bool): Whether to convert the text to lowercase. Defaults to True.
        remove_punct (bool): Whether to remove punctuation. Defaults to True.

    Returns:
        str: The processed text string. Returns an empty string if input is not a string.

    """
    if not isinstance(text, str):  # Basic type check
        logger.warning("preprocess_tfidf received non-string input: %s. Returning empty string.", type(text))
        return ""
    processed_text = text
    if lowercase:
        processed_text = processed_text.lower()
    if remove_punct:
        # Replace punctuation with a space to ensure words separated by punctuation are treated as distinct.
        processed_text = re.sub(r"[^\w\s]", " ", processed_text)
    return processed_text.strip()  # Remove any leading/trailing whitespace that might have been introduced.


def extract_lexical_features(
    model_answers: list[str],
    student_answers: list[str],
    linkage_method: str = "average",  # Default linkage method for hierarchical clustering
    distance_metric: str = "sqeuclidean",  # Default distance metric for pdist
    cluster_dist_thresh: float = 0.5,  # Default distance threshold for forming flat clusters
) -> LexicalFeaturesAnalysis:
    """Extracts lexical and clustering-based features for student answers relative to model answers.

    This function performs several steps:
    1. Preprocesses all model and student answers.
    2. Computes TF-IDF vectors for all texts.
    3. Calculates pairwise distances between all TF-IDF vectors.
    4. Performs hierarchical clustering on these distances.
    5. Computes cophenetic distances, which measure the distances preserved by the clustering.
    6. Forms flat clusters based on a distance threshold.
    7. Calculates silhouette scores for each text to measure cluster cohesion and separation.
    8. For each student answer, it compiles features:
        - Min, mean, and max cophenetic distance to the model answers.
        - Assigned cluster label and size.
        - Outlier status (if cluster size is 1).
        - Silhouette score.

    Args:
        model_answers (list[str]): A list of model/reference answer strings.
        student_answers (list[str]): A list of student answer strings.
        linkage_method (str): The linkage method to use for hierarchical clustering
                              (e.g., 'average', 'ward', 'complete'). Passed to `scipy.cluster.hierarchy.linkage`.
        distance_metric (str): The distance metric to use for `scipy.spatial.distance.pdist`
                               and `sklearn.metrics.silhouette_samples` (e.g., 'cosine', 'euclidean', 'sqeuclidean').
        cluster_dist_thresh (float): The distance threshold used by
                                     `scipy.cluster.hierarchy.fcluster` to form flat clusters.

    Returns:
        LexicalFeaturesAnalysis: A Pydantic model containing a list of `LexicalClusterFeature`
                                 objects, one for each student answer. Returns an empty list
                                 of features if inputs are empty or if critical errors occur
                                 (e.g., TF-IDF vectorization fails).

    """
    if not model_answers or not student_answers:
        logger.warning(
            "extract_lexical_features: model_answers or student_answers list is empty. Returning empty features.",
        )
        return LexicalFeaturesAnalysis(student_features=[])

    # Preprocess all texts (model answers + student answers) for TF-IDF
    all_texts_processed = [preprocess_tfidf(t) for t in model_answers + student_answers]
    num_models = len(model_answers)
    num_total_texts = len(all_texts_processed)

    # TF-IDF Vectorization: Convert texts into numerical feature vectors.
    vectorizer = TfidfVectorizer()
    tfidf_matrix: np.ndarray  # Will hold the dense TF-IDF matrix.
    try:
        # Check if all texts are empty after preprocessing, which would cause vectorizer to fail.
        if not any(all_texts_processed):  # `any` checks if at least one string is non-empty.
            logger.warning(
                "All texts are empty after preprocessing in extract_lexical_features. Cannot compute TF-IDF.",
            )
            return LexicalFeaturesAnalysis(student_features=[])  # Return empty features.

        tfidf_matrix = vectorizer.fit_transform(all_texts_processed).toarray()
    except ValueError as e:  # Catch errors like "empty vocabulary" if all texts are stopwords or too short.
        logger.exception(
            f"TF-IDF Vectorization error in extract_lexical_features: {e}. "
            "This can happen if texts are empty or contain only stopwords after preprocessing.",
        )
        # Return empty features if TF-IDF fails critically.
        return LexicalFeaturesAnalysis(student_features=[])

    # Calculate pairwise distances between all TF-IDF vectors.
    # `pdist` requires at least 2 samples if input is 1D, or at least 2 features if NxD (N>1).
    min_samples_for_pdist = 2
    if tfidf_matrix.shape[0] < min_samples_for_pdist or tfidf_matrix.shape[1] == 0:  # Not enough texts or no features.
        logger.warning(
            f"Not enough samples ({tfidf_matrix.shape[0]}) or features ({tfidf_matrix.shape[1]}) "
            "for pdist. Cannot proceed with clustering-based lexical features.",
        )
        # Create default 'error' features for each student answer.
        error_feature = LexicalClusterFeature(
            coph_min=0.0,
            coph_mean=0.0,
            coph_max=0.0,
            cluster_label=-1,
            cluster_size=0,
            is_outlier=1,
            silhouette=0.0,  # Using -1 for undefined cluster
        )
        return LexicalFeaturesAnalysis(student_features=[error_feature for _ in student_answers])

    # `pdist` returns a condensed distance matrix (1D array).
    pairwise_dist_matrix_condensed: np.ndarray = pdist(
        tfidf_matrix,
        metric=distance_metric,  # type: ignore[arg-type] # Pylance can have issues with scipy/sklearn type stubs.
    )

    # Hierarchical Clustering: Group texts based on their pairwise distances.
    linkage_matrix: np.ndarray
    try:
        # `linkage` performs hierarchical/agglomerative clustering.
        linkage_matrix = linkage(pairwise_dist_matrix_condensed, method=linkage_method)
    except ValueError as e:  # `linkage` also needs more than 1 observation.
        logger.exception(
            f"Linkage error in extract_lexical_features (TF-IDF matrix shape: {tfidf_matrix.shape}): {e}.",
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

    # Cophenetic Correlation Coefficient: Measures how well the clustering preserves original pairwise distances.
    # `cophenet` returns the cophenetic correlation coefficient itself and the cophenetic distance matrix (condensed).
    try:
        _cophenetic_corr_coeff, cophenetic_distances_condensed = cophenet(
            linkage_matrix, pairwise_dist_matrix_condensed,
        )
    except Exception as e:  # Catch any error during cophenet calculation
        logger.exception(f"Cophenet calculation error: {e}. Using zero matrix for cophenetic distances.")
        cophenetic_distances_condensed = np.zeros_like(pairwise_dist_matrix_condensed)  # Fallback

    # Convert the condensed cophenetic distance matrix to its square form for easier indexing.
    cophenetic_dist_matrix_square: np.ndarray = squareform(cophenetic_distances_condensed)

    # Flat Clustering: Form flat clusters from the hierarchical clustering using a distance threshold.
    cluster_labels: np.ndarray = fcluster(linkage_matrix, t=cluster_dist_thresh, criterion="distance")

    # Silhouette Scores: Measure how similar an object is to its own cluster compared to other clusters.
    silhouette_vals: np.ndarray
    num_unique_labels = len(np.unique(cluster_labels))
    # `silhouette_samples` requires 1 < n_labels < n_samples (number of unique cluster labels).
    if num_unique_labels > 1 and num_unique_labels < num_total_texts:
        silhouette_vals = silhouette_samples(tfidf_matrix, cluster_labels, metric=distance_metric)
    else:  # If all samples are in one cluster or each sample is its own cluster.
        logger.debug(
            f"Cannot compute meaningful silhouette scores (num_unique_labels={num_unique_labels}, "
            f"num_total_texts={num_total_texts}). Setting silhouette scores to 0.",
        )
        silhouette_vals = np.zeros(num_total_texts)  # Assign 0 if silhouette cannot be computed.

    student_feature_list: list[LexicalClusterFeature] = []
    for idx, _ in enumerate(student_answers):
        student_global_idx = num_models + idx  # Student's index in the combined `all_texts_processed` list.

        # Get cophenetic distances from the current student answer to all model answers.
        # These are rows/columns 0 to num_models-1 in `cophenetic_dist_matrix_square`.
        student_coph_to_models: np.ndarray = cophenetic_dist_matrix_square[student_global_idx, :num_models]

        student_cluster_label_val: int = cluster_labels[student_global_idx].item()  # Get scalar value.
        # Count how many texts belong to the same cluster as the current student.
        student_cluster_size_val: int = int((cluster_labels == student_cluster_label_val).sum())

        features_for_student = LexicalClusterFeature(
            coph_min=float(student_coph_to_models.min()) if student_coph_to_models.size > 0 else 0.0,
            coph_mean=float(student_coph_to_models.mean()) if student_coph_to_models.size > 0 else 0.0,
            coph_max=float(student_coph_to_models.max()) if student_coph_to_models.size > 0 else 0.0,
            cluster_label=student_cluster_label_val,
            cluster_size=student_cluster_size_val,
            is_outlier=int(student_cluster_size_val == 1),  # An outlier if its cluster size is 1.
            silhouette=float(silhouette_vals[student_global_idx].item()),  # Get scalar value.
        )
        student_feature_list.append(features_for_student)

    return LexicalFeaturesAnalysis(student_features=student_feature_list)


# --- Main Analysis Orchestration Functions ---

# Example of NLTK METEOR score - this section seems out of place with the rest of the module's focus.
# It's also missing context for `reference` and `generated` if not run as __main__.
# Consider moving to an example script or integrating it if METEOR is a desired feature.
# For now, keeping it as is from original, but noting its odd placement.
# import nltk # Already moved to top

# The nltk.download calls originally here (around line 1338) are problematic for module-level execution.
# They should ideally be handled by an explicit setup script or within the application's main entry point.
# nltk.download("wordnet", quiet=True)
# nltk.download("omw-1.4", quiet=True)

# These lines will execute when the module is imported, which might not be intended.
# reference_meteor_example = [["This", "is", "a", "reference", "summary"]]
# generated_meteor_example = ["This", "is", "a", "generated", "summary"]
# meteor_example_score = meteor_score(reference_meteor_example, generated_meteor_example) # meteor_score not defined
# print(f"METEOR Score (Example at import time): {meteor_example_score}") # This print will also occur at import.


def run_single_pair_text_analysis(
    inputs: SinglePairAnalysisInput,
    existing_graph: Optional[nx.Graph] = None,  # Allow passing a pre-built corpus graph
) -> SinglePairAnalysisResult:
    """Analyzes a single model answer against a single student answer for various similarity metrics.

    This function computes:
    - Graph-based similarity (if `existing_graph` is provided or a local one can be built).
    - Plagiarism score using a fast Smith-Waterman variant.
    - Overlap coefficient.
    - Sørensen-Dice coefficient.
    - Character-by-character equality score.
    - (Optionally, if spaCy is enabled) Semantic graph similarity.

    Args:
        inputs (SinglePairAnalysisInput): A Pydantic model containing the model answer string,
                                          student text string, and parameters for plagiarism detection
                                          (k-gram size, window radius).
        existing_graph (Optional[nx.Graph]): An optional pre-built word co-occurrence graph.
                                             If None, a local graph for the specific pair will be
                                             attempted for graph similarity calculation.

    Returns:
        SinglePairAnalysisResult: A Pydantic model containing all computed similarity metrics
                                  for the input pair of texts.

    """
    logger.info(
        f"Starting single pair analysis for student text (first 30 chars): '{inputs.student_text[:30]}...' "
        f"and model answer (first 30 chars): '{inputs.model_answer[:30]}...'",
    )
    results = SinglePairAnalysisResult()  # Initialize an empty result object.

    # --- Graph Similarity ---
    graph_to_use: Optional[nx.Graph] = existing_graph
    if graph_to_use is None:  # If no corpus graph is provided, try to build one for this pair.
        logger.debug("No existing graph provided for single pair analysis; attempting to build a local one.")
        # Create word vectors and graph just for this pair.
        pair_word_vecs = create_word_vectors([inputs.model_answer, inputs.student_text])
        if pair_word_vecs.word_matrix_csr_scipy is not None and pair_word_vecs.words_vocabulary:
            graph_to_use = build_graph_efficiently(pair_word_vecs)
        else:
            logger.warning("Could not build local graph for single pair analysis (word vector creation failed).")

    if graph_to_use and graph_to_use.number_of_nodes() > 0:
        results.graph_similarity = calculate_graph_similarity(
            graph_to_use,
            inputs.student_text,  # Student text for comparison
            inputs.model_answer,  # Model answer for comparison
        )
    else:
        logger.info("Graph for single pair analysis is empty or could not be built; skipping graph similarity.")
        results.graph_similarity = GraphSimilarityOutput(  # Populate with default "not applicable" values.
            similarity_score=0.0,
            subgraph_nodes=0,
            subgraph_edges=0,
            message="Graph not available or empty for this pair.",
        )

    # --- Plagiarism Score (Smith-Waterman variant) ---
    # Configure Smith-Waterman parameters from the input model.
    sw_config = SmithWatermanConfig(
        k=inputs.plagiarism_k,
        window_radius=inputs.plagiarism_window_radius,
        # match_score, mismatch_score, gap_penalty will use defaults from SmithWatermanConfig definition
        # unless inputs.SinglePairAnalysisInput is extended to include them.
    )
    results.plagiarism_score = compute_plagiarism_score_fast(
        inputs.student_text,  # Text 1 for plagiarism check
        inputs.model_answer,  # Text 2 for plagiarism check
        config=sw_config,
    )

    # --- Other Direct Similarity Metrics ---
    results.overlap_coefficient = calculate_overlap_coefficient(inputs.student_text, inputs.model_answer)
    results.dice_coefficient = calculate_sorensen_dice_coefficient(inputs.student_text, inputs.model_answer)
    results.char_equality_score = get_char_by_char_equality_optimized(inputs.student_text, inputs.model_answer)

    # --- Semantic Graph Similarity (Optional - requires spaCy setup and model) ---
    # This section remains commented as per the original structure and current focus.
    # If spaCy is to be used, `nlp` model needs to be loaded and passed appropriately.
    # try:
    #     if 'nlp' in globals() and nlp: # Check if global 'nlp' (spaCy model) is loaded.
    #         s_graph_student = create_semantic_graph_spacy(inputs.student_text, nlp)
    #         s_graph_model = create_semantic_graph_spacy(inputs.model_answer, nlp)
    #         results.semantic_graph_similarity = calculate_semantic_graph_similarity_spacy(s_graph_student, s_graph_model)
    #     else:
    #         logger.info("spaCy model (nlp) not available. Skipping semantic graph similarity in single pair analysis.")
    # except NameError: # If 'nlp' is not defined at all.
    #     logger.info("spaCy (nlp variable) not defined. Skipping semantic graph similarity for single pair.")

    logger.info(f"Single pair analysis finished for student '{inputs.student_text[:30]}...'.")
    return results


def run_full_text_analysis(
    inputs: FullTextAnalysisInput,
) -> tuple[FullTextAnalysisResult, Optional[nx.Graph]]:
    """Orchestrates a comprehensive text analysis pipeline.

    This function processes multiple student texts against multiple model answers.
    It performs the following main steps:
    1. Builds a word co-occurrence graph from the entire corpus (all model and student texts).
    2. Extracts lexical and clustering-based features for each student answer relative to model answers.
    3. For each student answer, calculates various similarity scores against each model answer,
       then aggregates these scores (e.g., average similarity to models, max plagiarism score).

    Args:
        inputs (FullTextAnalysisInput): A Pydantic model containing lists of model answers
                                        and student texts, along with configuration parameters
                                        for plagiarism detection and lexical feature extraction.

    Returns:
        tuple[FullTextAnalysisResult, Optional[nx.Graph]]:
            - FullTextAnalysisResult: A Pydantic model containing all computed metrics and features,
                                      including corpus graph metrics, student lexical features,
                                      and per-student aggregated similarity scores.
            - Optional[nx.Graph]: The generated corpus graph if successfully built, otherwise None.

    """
    logger.info(
        "Starting full text analysis pipeline for %d student texts and %d model answers.",
        len(inputs.student_texts),
        len(inputs.model_answers),
    )
    results = FullTextAnalysisResult()  # Initialize an empty result object.
    corpus_graph: Optional[nx.Graph] = None  # To store the graph built from all texts.

    # --- Step 1: Build Corpus Graph (from all model and student texts) ---
    # This graph represents word co-occurrences across the entire dataset.
    # It can be time-consuming for large corpora.
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
                # Density is defined for N > 1. nx.density handles N <= 1 returning 0 or NaN.
                density=nx.density(corpus_graph) if corpus_graph.number_of_nodes() > 1 else 0.0,
            )
            logger.info(
                "Corpus graph built successfully: %d nodes, %d edges.",
                results.corpus_graph_metrics.nodes,
                results.corpus_graph_metrics.edges,
            )
        else:
            logger.warning("Corpus graph construction resulted in an empty or invalid graph.")
            corpus_graph = None  # Ensure it's None if not properly built.
    else:
        logger.warning("Corpus graph could not be built: word vector creation failed or yielded empty results.")

    # --- Step 2: Lexical/Clustering Features for all student answers ---
    # These features are derived from TF-IDF and hierarchical clustering of students relative to models.
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
    except Exception as e:  # Catch broad exception to ensure pipeline continues if this step fails.
        logger.exception(f"Error extracting lexical features: {e}")
        results.student_lexical_features = None  # Or LexicalFeaturesAnalysis(student_features=[])

    # --- Step 3: Per-student analysis (similarity of each student to all model answers) ---
    per_student_results_list: list[dict[str, Any]] = []  # To store results for each student.
    logger.info("Starting per-student analysis against model answers...")

    for student_idx, student_text_item in enumerate(inputs.student_texts):
        student_specific_analysis_dict: dict[str, Any] = {"student_text_index": student_idx}

        # Lists to store scores of this student against EACH model answer
        graph_sims_to_models: list[float] = []
        plagiarism_scores_to_models: list[float] = []
        overlap_coeffs_to_models: list[float] = []
        dice_coeffs_to_models: list[float] = []
        char_eq_scores_to_models: list[float] = []
        # semantic_graph_sims_to_models: list[float] = [] # If spaCy features were active

        # Compare the current student's text against every model answer.
        for _, model_text_item in enumerate(inputs.model_answers):
            # Prepare input for single pair analysis.
            single_pair_input_params = SinglePairAnalysisInput(
                model_answer=model_text_item,
                student_text=student_text_item,
                plagiarism_k=inputs.plagiarism_k,  # Use k from overall config
                plagiarism_window_radius=inputs.plagiarism_window_radius,  # Use radius from overall config
            )
            # Perform analysis for this student-model pair.
            # Crucially, pass the `corpus_graph` if available to avoid rebuilding it repeatedly.
            pair_analysis_result = run_single_pair_text_analysis(
                single_pair_input_params,
                existing_graph=corpus_graph,
            )

            # Collect scores from the pair analysis.
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
            # if pair_analysis_result.semantic_graph_similarity:
            #     semantic_graph_sims_to_models.append(pair_analysis_result.semantic_graph_similarity.similarity)

        # Aggregate the collected scores for the current student (e.g., average, max).
        # .item() converts numpy float to Python float if numpy array results from mean/max.
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
        # student_specific_analysis_dict["semantic_graph_similarity_to_model_avg"] = (
        #    np.mean(semantic_graph_sims_to_models).item() if semantic_graph_sims_to_models else None
        # )
        per_student_results_list.append(student_specific_analysis_dict)
        logger.debug(f"Finished analysis for student index {student_idx}.")

    results.per_student_analysis = per_student_results_list
    logger.info("Full text analysis pipeline finished successfully.")
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
