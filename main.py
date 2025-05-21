"""Main module for essay grading."""
import logging

from app_types import EssayScores
from score import score_essay


def main()-> EssayScores:
    """Run the essay grading process."""
    scores = score_essay(essay="This is a sample essay text.", reference="This is a reference text for comparison.")
    logging.info(f"Essay scores: {scores}")
    return scores


# Example cleanup for testing if run directly
if __name__ == "__main__":
    main()
