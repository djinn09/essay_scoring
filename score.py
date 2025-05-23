"""Module defines data models for essay grading.

It includes:
- EssayScores: A model for storing essay scoring details.
- score_essay: A function that scores essays against reference texts.
"""

from __future__ import annotations

from app_types import EssayScores, KeywordMatcherConfig
from key_word_match import SimilarityCalculator
from keyword_matcher import KeywordMatcher
from pos_score import score_pos
from semantic_match import SemanticCosineSimilarity
from settings import semantic_model, settings, similarity_config
from text_features import SinglePairAnalysisInput, run_single_pair_text_analysis


def score_essay(essay: str, reference: str) -> EssayScores:
    """Score an essay based on its semantic similarity to a reference text.

    Args:
        essay (str): The essay to be scored.
        reference (str): The reference text to compare against.

    Returns:
        float: A score between 0 and 1 representing the semantic similarity.

    """
    # Initialize the semantic cosine similarity model
    sentence_semantic_model = SemanticCosineSimilarity(
        model=semantic_model,
        chunk_size=settings.semantic.chunk_size,
        batch_size=settings.semantic.batch_size,
    )
    # Initialize the keyword similarity calculator
    keyword_similarity_calculator = SimilarityCalculator(
        config=similarity_config,
    )
    # Calculate the semantic similarity score
    semantic_score = sentence_semantic_model.calculate_similarity(
        essay,
        reference,
        metrics_to_calculate=["cosine"],
    )
    text_features_input = SinglePairAnalysisInput(
        model_answer=reference,
        student_text=essay,
    )
    individual_pair_results = run_single_pair_text_analysis(text_features_input)
    # Calculate the keyword similarity score
    config_pos = KeywordMatcherConfig(use_pos_tagging=True)
    keyword_matcher = KeywordMatcher(config=config_pos)
    keyword_matcher_results = keyword_matcher.find_matches_and_score(reference, essay)
    essay_pos_score = score_pos(reference, essay)
    similarity_metrics = keyword_similarity_calculator.calculate_single_pair(reference, essay)
    if semantic_score is not None:
        return EssayScores(
            semantic_score=semantic_score.cosine,
            similarity_metrics=similarity_metrics,
            text_score=individual_pair_results,
            keyword_matcher=keyword_matcher_results.keywords_matcher_result,
            pos_score=essay_pos_score,
        )
    return EssayScores(
        similarity_metrics=similarity_metrics,
        text_score=individual_pair_results,
        keyword_matcher=keyword_matcher_results.keywords_matcher_result,
        pos_score=essay_pos_score,
    )
