"""Orchestrates the essay scoring process by integrating various feature extractors and similarity calculators.

This module defines the main `score_essay` function, which takes an essay and a
reference text, and returns a comprehensive set of scores encapsulated in the
`EssayScores` Pydantic model. It utilizes semantic similarity models, keyword
matching, POS-based scoring, and other text feature analyses.
"""

from __future__ import annotations

from app_types import EssayScores, KeywordMatcherConfig, SimilarityMetrics, SinglePairAnalysisResult
from key_word_match import SimilarityCalculator  # For general similarity metrics
from keyword_matcher import KeywordMatcher  # For keyword-based matching
from pos_score import score_pos  # For POS-based scoring
from semantic_match import SemanticCosineSimilarity  # For semantic similarity

# Import global models and settings loaded in settings.py
from settings import semantic_model, settings, similarity_config
from text_features import SinglePairAnalysisInput, run_single_pair_text_analysis  # For detailed text feature analysis


def score_essay(essay: str, reference: str) -> EssayScores:
    """Scores an essay against a reference text using a variety of metrics.

    This function computes:
    - Semantic similarity (cosine similarity using Sentence Transformers).
    - A broad range of classical similarity metrics (e.g., Levenshtein, Jaccard, TF-IDF cosine).
    - Detailed text feature analysis (e.g., plagiarism score, graph similarity).
    - Keyword matching scores (coverage and vocabulary cosine).
    - Part-of-Speech (POS) based similarity score.

    Args:
        essay (str): The essay text to be scored.
        reference (str): The reference text to compare against.

    Returns:
        EssayScores: A Pydantic model instance containing all the calculated scores.
                     If semantic similarity calculation fails, its specific score might be None.

    """
    # Step 1: Initialize Semantic Similarity Model
    # This uses a pre-loaded SentenceTransformer model (`semantic_model` from settings.py)
    # and configuration from the global `settings` object.
    sentence_semantic_model = SemanticCosineSimilarity(
        model=semantic_model,  # Pre-loaded SentenceTransformer model
        chunk_size=settings.semantic.chunk_size,
        overlap=settings.semantic.overlap,  # Added overlap based on SemanticCosineSimilarity's __init__
        batch_size=settings.semantic.batch_size,
        # device is typically handled by SentenceTransformer itself based on model loading,
        # but if SemanticCosineSimilarity takes it, it would be: device=settings.semantic.device
    )

    # Step 2: Initialize General Similarity Calculator
    # This calculator computes a wide range of classical similarity metrics.
    # It uses a pre-defined `similarity_config` from settings.py.
    general_similarity_calculator = SimilarityCalculator(
        config=similarity_config,  # Configuration for various preprocessing options
    )

    # Step 3: Calculate Semantic Similarity Score (e.g., Cosine)
    # This focuses on the meaning/context captured by embeddings.
    semantic_similarity_result = sentence_semantic_model.calculate_similarity(
        essay,
        reference,
        metrics_to_calculate=["cosine"],  # Specify only cosine for the main semantic_score
    )
    # Extract the cosine score if the calculation was successful.
    main_semantic_score: float | None = None
    if semantic_similarity_result and semantic_similarity_result.cosine is not None:
        main_semantic_score = semantic_similarity_result.cosine
    else:
        # Log if primary semantic score calculation failed.
        # In a real app, you might want more robust error handling or default values.
        print(f"Warning: Semantic cosine similarity calculation failed for essay: {essay[:50]}...")

    # Step 4: Perform Detailed Text Feature Analysis (from text_features.py)
    # This includes plagiarism checks, graph-based similarity, etc.
    text_features_input = SinglePairAnalysisInput(
        model_answer=reference,
        student_text=essay,
        # Uses default plagiarism_k and plagiarism_window_radius from SinglePairAnalysisInput
    )
    detailed_text_analysis_results: SinglePairAnalysisResult = run_single_pair_text_analysis(text_features_input)

    # Step 5: Calculate Keyword Matching Scores
    # This assesses keyword coverage and vocabulary similarity.
    # Configure KeywordMatcher (e.g., enable POS tagging for keyword extraction).
    keyword_matcher_config = KeywordMatcherConfig(use_pos_tagging=True)
    keyword_matcher_instance = KeywordMatcher(config=keyword_matcher_config)
    keyword_matching_scores = keyword_matcher_instance.find_matches_and_score(
        paragraph_a=reference,  # Keywords extracted from reference
        paragraph_b=essay,  # Found in essay
    )

    # Step 6: Calculate Part-of-Speech (POS) Based Score
    # This compares structural POS patterns between the texts.
    essay_pos_similarity_score = score_pos(model_text=reference, candidate_text=essay)

    # Step 7: Calculate General Similarity Metrics (classical methods)
    # This provides a broad spectrum of similarity measures.
    all_similarity_metrics: SimilarityMetrics = general_similarity_calculator.calculate_single_pair(reference, essay)

    # Step 8: Assemble all scores into the EssayScores Pydantic model.
    # This ensures a structured output.
    return EssayScores(
        semantic_score=main_semantic_score,  # The primary semantic similarity score
        similarity_metrics=all_similarity_metrics,  # Collection of various other similarity scores
        text_score=detailed_text_analysis_results,  # Results from text_features analysis
        keyword_matcher=keyword_matching_scores.keywords_matcher_result,  # Scores from KeywordMatcher
        pos_score=essay_pos_similarity_score,  # Score from POS pattern matching
    )

    # Placeholder for final aggregation or further processing if needed.
    # For now, just returning the comprehensive scores as is.
