"""Text Preprocessing and Evaluation Utilities.

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
from typing import Dict  # For type hinting

# Attempt to import NLTK components; these are essential for some functions.
# If NLTK or its data is not available, relevant functions might degrade gracefully or error.
try:
    from nltk.corpus import stopwords as nltk_stopwords  # Renamed to avoid conflict
    from nltk.tokenize import word_tokenize
except ImportError:
    # This will cause functions relying on these to fail if not handled within them.
    # Consider adding fallback mechanisms or clearer error reporting if NLTK is critical.
    nltk_stopwords = None
    word_tokenize = None
    print("NLTK not found. Some preprocessing functions may not work as expected.")


# Global punctuation removal table for efficiency
# This should be defined after importing `string`.
if "string" in globals():
    PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
else:
    # Fallback if string module wasn't imported (e.g. due to test environment)
    # This is less robust than str.maketrans
    PUNCTUATION_TABLE = {ord(c): None for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"}


def eval_based_on_length(answer: str, min_length: int = 8) -> bool:
    """Check if the number of words in a given string exceeds a minimum length.

    Args:
        answer (str): The candidate answer string.
        min_length (int): The minimum number of words the answer should have. Defaults to 8.

    Returns:
        bool: True if the word count of the answer is greater than `min_length`,
              False otherwise (including if `answer` is not a string).

    """
    should_evaluate = False  # Default to False, meaning length criteria not met.
    if not isinstance(answer, str):
        return should_evaluate  # Return False if input is not a string.

    # Split the string by spaces to count words.
    words = answer.split(" ")
    if len(words) > min_length:
        should_evaluate = True  # Length criteria met.

    return should_evaluate


def remove_unbalanced_repetition(
    answer: str,
    top_words_percentage: float = 0.4,
    frequency_boundary: float = 0.3,
) -> bool:
    """Check if the cumulative frequency of most common words is below a boundary.

    This suggests the answer is not overly repetitive.

    Note: The logic for determining 'top' words based on `doc_fre % top_words_percentage`
    seems unusual and might not accurately capture the intended "top percentage of words".
    It's documented here as implemented. A more standard approach would be to sort
    word frequencies and sum up frequencies of words constituting the top X percent.

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
        return False  # Cannot process empty or non-string answers.

    split_words = answer.split(" ")
    doc_len = len(split_words)
    if doc_len == 0:
        return False  # Avoid division by zero later.

    word_counts = Counter(split_words)

    # Current logic for 'top': This calculates `doc_len % top_words_percentage`, which is a modulo operation.
    # This is likely not the intended way to get a "top percentage" of words.
    # For example, if doc_len=100, top_words_percentage=0.4, then 100 % 0.4 = 0.0. math.ceil(0.0) = 0.
    # If doc_len=10, top_words_percentage=0.4, then 10 % 0.4 = 0.0. math.ceil(0.0) = 0.
    # This will result in `top` being 0 or 1 in many practical cases, meaning it only checks
    # the 0th or 1st most frequent word.
    # A more standard interpretation would be to sort words by frequency and take
    # a number of words that account for `top_words_percentage` of unique words or total word count.
    # However, documenting current implementation:
    # The intent seems to be to define how many of the most frequent words to sum up.
    # The current calculation of `top` is: math.ceil(doc_len % top_words_percentage)
    # This should be reviewed for correctness based on desired behavior.
    # Assuming it means to check the `top` most frequent words:
    num_top_words_to_check = math.ceil(doc_len % top_words_percentage)  # Potentially problematic logic for "top".
    # Add a log to show the calculated `num_top_words_to_check` for debugging.
    # print(
    #     f"Debug: doc_len={doc_len}, top_words_percentage={top_words_percentage}, "
    #     f"calculated num_top_words_to_check={num_top_words_to_check}"
    # )

    cumulative_frequency_count = 0
    # Get word frequencies sorted in descending order.
    sorted_frequencies = sorted(word_counts.values(), reverse=True)

    # Sum frequencies of the 'top' most frequent words.
    # The loop goes from 0 up to `num_top_words_to_check` (inclusive of `num_top_words_to_check`).
    for i in range(int(num_top_words_to_check) + 1):
        if i < len(sorted_frequencies):  # Ensure we don't go out of bounds.
            cumulative_frequency_count += sorted_frequencies[i]
        else:
            break  # No more frequencies to sum.

    # Calculate the actual frequency of these top words.
    actual_frequency = cumulative_frequency_count / doc_len if doc_len > 0 else 0

    # Return True if this actual frequency is less than the specified boundary.
    return actual_frequency < frequency_boundary


def eval_based_on_repetition(answer: str, percentage_threshold: float = 0.6) -> bool:
    """Check if any single keyword's repetition exceeds a given percentage of total words.

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
    should_evaluate = False  # Default to False, meaning repetition criteria not met or input invalid.
    if not answer or not isinstance(answer, str):  # Check for empty or non-string input.
        return should_evaluate

    answer_cleaned = answer.strip().lower()
    # Split string by space and remove empty words that might result from multiple spaces.
    keywords = [word for word in answer_cleaned.split(" ") if word]
    if not keywords:  # If no keywords after cleaning (e.g. answer was only spaces).
        return True  # Or False, depending on how empty answers should be treated. Current logic implies True.

    total_words = len(keywords)
    keyword_counts = Counter(keywords)

    # Check frequency of each word.
    for count in keyword_counts.values():
        keyword_percentage = count / total_words
        if keyword_percentage > percentage_threshold:
            return False  # Found a word exceeding the repetition threshold.

    return True  # No single word repetition exceeded the threshold.


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from a text string using a pre-compiled translation table.

    Args:
        text (str): The input text string.

    Returns:
        str: The text string with punctuation removed. Returns an empty string if input is not a string.

    """
    if not isinstance(text, str):
        return ""
    return text.translate(PUNCTUATION_TABLE)


def is_answer_exact_match_to_question(answer: str, question: str) -> bool:
    """Check if the candidate answer is an exact match to the question string.

    Args:
        answer (str): The candidate answer string.
        question (str): The question string.

    Returns:
        bool: False if the answer is identical to the question, True otherwise.

    """
    # This function's original name `que_check` was ambiguous. Renamed for clarity.
    # The original logic returns False if they are the same, which is counter-intuitive
    # for a check named "is_answer_ok" or similar. It's more like "is_answer_not_question".
    # Assuming the goal is to flag answers that are just a copy of the question.
    return answer != question


def get_first_unique_word_sequence(text: str) -> str:
    """Find the first sequence of unique words in a string until a word repeats.

    Note: The original function name `find_first_non_repeating_sentence` was misleading
    as it operates on words, not sentences, and finds a sequence of *unique* words,
    not necessarily a "non-repeating" sentence in the typical sense.

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
        if token not in found_words:  # Check if the token has been seen before in this sequence.
            found_words.append(token)
        else:
            break  # Stop when a word repeats.
    return " ".join(found_words)


def check_for_repeating_sequences(answer: str) -> bool:
    """Check if the initial unique word sequence (from `get_first_unique_word_sequence`).

    repeats more than once in the answer.

    Args:
        answer (str): The text string to evaluate.

    Returns:
        bool: True if the initial unique word sequence is found more than once,
              False otherwise or if input is not a string.

    """
    # Original name `if_repeating_sentence` was potentially misleading.
    if not isinstance(answer, str):
        return False

    processed_answer = remove_punctuation(answer.lower())
    # Get the initial sequence of unique words.
    initial_unique_sequence = get_first_unique_word_sequence(processed_answer)

    if not initial_unique_sequence:  # If no sequence found (e.g., empty answer).
        return False

    # Find all occurrences of this exact sequence in the processed answer.
    # `re.escape` is important if the sequence might contain regex special characters, though less likely here.
    occurrences = re.findall(re.escape(initial_unique_sequence), processed_answer)
    # Return True if the sequence occurs more than once.
    return len(occurrences) > 1


def remove_question_from_answer_ends(answer: str, question: str) -> str:
    """Remove the `question` string from the beginning or end of the `answer` string, if present.

    Note: This uses `strip()`, which only removes leading/trailing occurrences.
    It does not remove the question if it's embedded within the answer.

    Args:
        answer (str): The answer string.
        question (str): The question string to remove.

    Returns:
        str: The answer string with leading/trailing question text removed.

    """
    # Original name `remove_ques_text` was short.
    if not isinstance(answer, str) or not isinstance(question, str):
        return answer  # Or raise error, depending on desired strictness.

    # `strip()` removes all leading/trailing characters found in the `question` string,
    # not necessarily the whole `question` string as a block unless it's an exact prefix/suffix.
    # If the intent is to remove the exact question string as prefix/suffix:
    # if answer.startswith(question): answer = answer[len(question):]
    # if answer.endswith(question): answer = answer[:-len(question)]
    # Current implementation uses `strip`, which has different behavior. Documenting as is.
    if question in answer:  # This check is a bit misleading with `strip`.
        # `strip` works on a set of characters, not a substring.
        # `answer.strip(question)` will remove any leading/trailing characters
        # that are part of the `question` string.
        # For example, if question is "abc", answer "cbaHelloabc" -> "Hello".
        # If question is "abc", answer "abHelloacb" -> "Hello".
        # If question is " What is X?", answer " What is X? It is Y." -> " It is Y." (if question is prefix)
        # A more robust way to remove a prefix/suffix string is answer.removeprefix / .removesuffix (Python 3.9+)
        # or slicing as shown above.
        # For now, keeping original `strip` logic.
        answer = answer.strip(question)
    return answer


def get_unique_words_from_string(sentence: str) -> set[str]:
    """Extract a set of unique, non-stopword words from a sentence.

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

    # Ensure NLTK components are available
    if nltk_stopwords is None or word_tokenize is None:
        # This indicates NLTK wasn't imported correctly or data is missing.
        # Depending on strictness, could raise an error or return empty set with warning.
        print("NLTK stopwords or tokenizer not available. Cannot extract unique words.")
        return set()

    # Remove punctuation and convert to lowercase.
    processed_sentence = sentence.translate(PUNCTUATION_TABLE).lower()

    # Tokenize the sentence.
    tokens = word_tokenize(processed_sentence)

    # Filter out stopwords.
    # Assumes `nltk_stopwords.words('english')` is the source for `stopwords_set`.
    # For robustness, ensure `stopwords_set` is initialized.
    try:
        # This was `stopwords` in original, changed to avoid conflict with imported module name
        current_stopwords = set(nltk_stopwords.words("english"))
    except AttributeError:  # If nltk_stopwords itself is None
        current_stopwords = set()
    except LookupError:  # If 'stopwords' corpus not downloaded
        print("NLTK 'stopwords' corpus not found. Proceeding without stopword filtering.")
        current_stopwords = set()

    return {word for word in tokens if word not in current_stopwords and word.isalnum()}


def contains_sufficient_grammatical_structure(pos_tags_dict: Dict[str, str]) -> bool:
    """Check if a sentence, represented by its Part-of-Speech (POS) tags, contains.

    essential grammatical components (like auxiliaries or determiners).

    This can be used to heuristically identify sentences that might be composed
    predominantly of keywords rather than forming full grammatical structures.

    Args:
        pos_tags_dict (Dict[str, str]): A dictionary where keys are words and values
                                     are their POS tags. (Note: a list of tag strings
                                     might be more direct if word mapping isn't needed).

    Returns:
        bool: True if at least one of the `required_tags` (e.g., 'AUX', 'DET') is found
              among the POS tags of the sentence. False otherwise, suggesting the sentence
              might lack these common grammatical elements.

    """
    # Original name `if_only_keywords` was inverted; this name reflects the logic better.
    # The function checks for the *presence* of essential grammatical tags.
    # If these tags are present, it's *less* likely to be "only keywords".
    required_grammatical_tags = {"AUX", "DET"}  # Tags indicating grammatical structure.

    # Get unique POS tags present in the input.
    unique_tags_in_sentence = set(pos_tags_dict.values())
    print("DEBUG: Unique POS tags in sentence:", unique_tags_in_sentence)  # For debugging, can be removed.

    # Check if any of the required grammatical tags are present in the sentence's tags.
    # If none are found, it implies the sentence might lack common grammatical structure.
    return any(tag in unique_tags_in_sentence for tag in required_grammatical_tags)
