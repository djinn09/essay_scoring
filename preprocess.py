"""
Text Preprocessing and Evaluation Utilities.

This module provides a collection of functions for cleaning, evaluating, and
manipulating text strings, often used for assessing the quality or validity
of textual answers. Functions include checks for minimum length, word repetition,
punctuation removal, and analysis of POS tag patterns.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

try:
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    nltk_stopwords = None
    word_tokenize = None
    print("NLTK not found. Some preprocessing functions may not work as expected.")


if "string" in globals():
    PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
else:
    PUNCTUATION_TABLE = {ord(c): None for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"}


def eval_based_on_length(answer: str, min_length: int = 8) -> bool:
    """
    Check if the number of words in a given string exceeds a minimum length.

    Args:
        answer (str): The candidate answer string.
        min_length (int): The minimum number of words the answer should have. Defaults to 8.

    Returns:
        bool: True if the word count of the answer is greater than `min_length`,
              False otherwise (including if `answer` is not a string).
    """
    should_evaluate = False
    if not isinstance(answer, str):
        return should_evaluate

    words = answer.split(" ")
    if len(words) > min_length:
        should_evaluate = True

    return should_evaluate


def remove_unbalanced_repetition(
    answer: str,
    top_words_percentage: float = 0.4,
    frequency_boundary: float = 0.3,
) -> bool:
    """
    Check if the cumulative frequency of most common words is below a boundary.

    This suggests the answer is not overly repetitive.

    Args:
        answer (str): The text string to evaluate.
        top_words_percentage (float): The percentage to determine the 'top' words.
                                      The current calculation `math.ceil(doc_len % top_words_percentage)`
                                      is non-standard for this purpose.
        frequency_boundary (float): The maximum allowable cumulative frequency for these
                                    'top' words. If their combined frequency is less than
                                    this, the function returns True.

    Returns:
        bool: True if the cumulative frequency of 'top' words is below `frequency_boundary`,
              False otherwise. Returns False if answer is empty or only spaces.
    """
    if not isinstance(answer, str) or not answer.strip():
        return False

    split_words = answer.split(" ")
    doc_len = len(split_words)
    if doc_len == 0:
        return False

    word_counts = Counter(split_words)
    num_top_words_to_check = math.ceil(doc_len % top_words_percentage)

    cumulative_frequency_count = 0
    sorted_frequencies = sorted(word_counts.values(), reverse=True)

    for i in range(int(num_top_words_to_check) + 1):
        if i < len(sorted_frequencies):
            cumulative_frequency_count += sorted_frequencies[i]
        else:
            break

    actual_frequency = cumulative_frequency_count / doc_len if doc_len > 0 else 0

    return actual_frequency < frequency_boundary


def eval_based_on_repetition(answer: str, percentage_threshold: float = 0.6) -> bool:
    """
    Check if any single keyword's repetition exceeds a given percentage of total words.

    If a word's frequency relative to the total number of words is higher than
    `percentage_threshold`, the function considers the answer improper and returns False.

    Args:
        answer (str): The text string to evaluate.
        percentage_threshold (float): The maximum allowable percentage for any single word's
                                      repetition. Defaults to 0.6 (60%).

    Returns:
        bool: True if no single word repetition exceeds the threshold, False otherwise
              (or if answer is empty/invalid).
    """
    should_evaluate = False
    if not answer or not isinstance(answer, str):
        return should_evaluate

    answer_cleaned = answer.strip().lower()
    keywords = [word for word in answer_cleaned.split(" ") if word]
    if not keywords:
        return True

    total_words = len(keywords)
    keyword_counts = Counter(keywords)

    for count in keyword_counts.values():
        keyword_percentage = count / total_words
        if keyword_percentage > percentage_threshold:
            return False

    return True


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation from a text string using a pre-compiled translation table.

    Args:
        text (str): The input text string.

    Returns:
        str: The text string with punctuation removed. Returns an empty string if input is not a string.
    """
    if not isinstance(text, str):
        return ""
    return text.translate(PUNCTUATION_TABLE)


def is_answer_exact_match_to_question(answer: str, question: str) -> bool:
    """
    Check if the candidate answer is an exact match to the question string.

    Args:
        answer (str): The candidate answer string.
        question (str): The question string.

    Returns:
        bool: False if the answer is identical to the question, True otherwise.
    """
    return answer != question


def get_first_unique_word_sequence(text: str) -> str:
    """
    Find the first sequence of unique words in a string until a word repeats.

    Args:
        text (str): The input string.

    Returns:
        str: A string composed of the first sequence of unique words found.
             Returns an empty string if input is not a string.
    """
    if not isinstance(text, str):
        return ""
    tokens = text.split()
    found_words: list[str] = []
    for token in tokens:
        if token not in found_words:
            found_words.append(token)
        else:
            break
    return " ".join(found_words)


def check_for_repeating_sequences(answer: str) -> bool:
    """
    Check if the initial unique word sequence (from `get_first_unique_word_sequence`).

    repeats more than once in the answer.

    Args:
        answer (str): The text string to evaluate.

    Returns:
        bool: True if the initial unique word sequence is found more than once,
              False otherwise or if input is not a string.
    """
    if not isinstance(answer, str):
        return False

    processed_answer = remove_punctuation(answer.lower())
    initial_unique_sequence = get_first_unique_word_sequence(processed_answer)

    if not initial_unique_sequence:
        return False

    occurrences = re.findall(re.escape(initial_unique_sequence), processed_answer)
    return len(occurrences) > 1


def remove_question_from_answer_ends(answer: str, question: str) -> str:
    """
    Remove the `question` string from the beginning or end of the `answer` string, if present.

    Note: This uses `strip()`, which only removes leading/trailing occurrences.
    It does not remove the question if it's embedded within the answer.

    Args:
        answer (str): The answer string.
        question (str): The question string to remove.

    Returns:
        str: The answer string with leading/trailing question text removed.
    """
    if not isinstance(answer, str) or not isinstance(question, str):
        return answer

    if question in answer:
        answer = answer.strip(question)
    return answer


def get_unique_words_from_string(sentence: str) -> set[str]:
    """
    Extract a set of unique, non-stopword words from a sentence.

    The sentence is lowercased, punctuation is removed, tokenized, and stopwords
    are filtered out.

    Args:
        sentence (str): The input sentence string.

    Returns:
        set[str]: A set of unique words from the sentence after processing.

    Raises:
        TypeError: If `sentence` is not a string.
        LookupError: If NLTK data (stopwords, punkt) is not found (propagated from NLTK).
    """
    if not isinstance(sentence, str):
        msg = f"Expected string input for sentence, but got: {type(sentence)}"
        raise TypeError(msg)

    if nltk_stopwords is None or word_tokenize is None:
        print("NLTK stopwords or tokenizer not available. Cannot extract unique words.")
        return set()

    processed_sentence = sentence.translate(PUNCTUATION_TABLE).lower()

    tokens = word_tokenize(processed_sentence)

    try:
        current_stopwords = set(nltk_stopwords.words("english"))
    except AttributeError:
        current_stopwords = set()
    except LookupError:
        print("NLTK 'stopwords' corpus not found. Proceeding without stopword filtering.")
        current_stopwords = set()

    return {word for word in tokens if word not in current_stopwords and word.isalnum()}


def contains_sufficient_grammatical_structure(pos_tags_dict: dict[str, str]) -> bool:
    """
    Check if a sentence, represented by its Part-of-Speech (POS) tags, contains.

    essential grammatical components (like auxiliaries or determiners).

    This can be used to heuristically identify sentences that might be composed
    predominantly of keywords rather than forming full grammatical structures.

    Args:
        pos_tags_dict (dict[str, str]): A dictionary where keys are words and values
                                     are their POS tags. (Note: a list of tag strings
                                     might be more direct if word mapping isn't needed).

    Returns:
        bool: True if at least one of the `required_tags` (e.g., 'AUX', 'DET') is found
              among the POS tags of the sentence. False otherwise, suggesting the sentence
              might lack these common grammatical elements.
    """
    required_grammatical_tags = {"AUX", "DET"}

    unique_tags_in_sentence = set(pos_tags_dict.values())
    print("DEBUG: Unique POS tags in sentence:", unique_tags_in_sentence)

    return any(tag in unique_tags_in_sentence for tag in required_grammatical_tags)
