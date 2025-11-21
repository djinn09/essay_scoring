"""
Main execution module for the essay grading application.

This module serves as the entry point for running the essay grading process.
It demonstrates a simple use case of scoring an essay against a reference text.
"""

import logging

from app_types import EssayScores  # Defines the structure for essay scores.
from score import score_essay  # The primary function for scoring an essay.


def main() -> EssayScores:
    """
    Execute the primary essay grading process with sample inputs.

    This function calls the `score_essay` function with predefined essay and
    reference texts and logs the resulting scores.

    Returns:
        EssayScores: An object containing the calculated scores for the sample essay.
    """
    scores = score_essay(
        essay="This is a sample essay text.",
        reference="This is a reference text for comparison.",
    )
    logging.info(f"Essay scores: {scores}")
    return scores


if __name__ == "__main__":
    main()
