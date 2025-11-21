"""
Orchestrates the essay scoring process by integrating various feature extractors and similarity calculators.

This module defines the main `score_essay` function, which takes an essay and a
reference text, and returns a comprehensive set of scores encapsulated in the
`EssayScores` Pydantic model. It utilizes semantic similarity models, keyword
matching, POS-based scoring, and other text feature analyses.
"""

from __future__ import annotations

from app_types import EssayScores, KeywordMatcherConfig, SimilarityMetrics, SinglePairAnalysisResult
from key_word_match import SimilarityCalculator
from keyword_matcher import KeywordMatcher
from pos_score import score_pos
from semantic_match import SemanticCosineSimilarity
from settings import semantic_model, settings, similarity_config
from text_features import SinglePairAnalysisInput, run_single_pair_text_analysis


def score_essay(essay: str, reference: str) -> EssayScores:
    """
    Scores an essay against a reference text using a variety of metrics.

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
    sentence_semantic_model = SemanticCosineSimilarity(
        model=semantic_model,
        chunk_size=settings.semantic.chunk_size,
        overlap=settings.semantic.overlap,
        batch_size=settings.semantic.batch_size,
    )

    general_similarity_calculator = SimilarityCalculator(
        config=similarity_config,
    )

    semantic_similarity_result = sentence_semantic_model.calculate_similarity(
        essay,
        reference,
        metrics_to_calculate=["cosine"],
    )
    main_semantic_score: float | None = None
    if semantic_similarity_result and semantic_similarity_result.cosine is not None:
        main_semantic_score = semantic_similarity_result.cosine
    else:
        print(f"Warning: Semantic cosine similarity calculation failed for essay: {essay[:50]}...")

    text_features_input = SinglePairAnalysisInput(
        model_answer=reference,
        student_text=essay,
    )
    detailed_text_analysis_results: SinglePairAnalysisResult = run_single_pair_text_analysis(text_features_input)

    keyword_matcher_config = KeywordMatcherConfig(use_pos_tagging=True)
    keyword_matcher_instance = KeywordMatcher(config=keyword_matcher_config)
    keyword_matching_scores = keyword_matcher_instance.find_matches_and_score(
        paragraph_a=reference,
        paragraph_b=essay,
    )

    essay_pos_similarity_score = score_pos(model_text=reference, candidate_text=essay)

    all_similarity_metrics: SimilarityMetrics = general_similarity_calculator.calculate_single_pair(reference, essay)

    return EssayScores(
        semantic_score=main_semantic_score,
        similarity_metrics=all_similarity_metrics,
        text_score=detailed_text_analysis_results,
        keyword_matcher=keyword_matching_scores.keywords_matcher_result,
        pos_score=essay_pos_similarity_score,
    )
