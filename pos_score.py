"""
Calculates a score based on Part-of-Speech (POS) pattern similarity between texts.

This module uses spaCy for POS tagging and word vector similarity. It identifies
specific POS combinations (triplets like ProperNoun-Verb-Noun, and duplets like
ProperNoun-Verb or ProperNoun-Noun) in sentences. The similarity score is
derived from the overlap of these combinations between a model text and a
candidate text.
"""

from __future__ import annotations

from timeit import default_timer as timer

import nltk
import spacy

from config import spacy_model

# Use the globally loaded spaCy model from the config module.
nlp = spacy_model
SIMILARITY_THRESHOLD = 0.3
TRIPLET_LENGTH = 3
DUPLET_LENGTH = 2


def extract_pos_combinations(text: str) -> list[list[str]]:
    """
    Tokenize input text into sentences and extract POS combinations.

    - Triplets: (ProperNoun, Verb, Noun)
    - Duplets: (ProperNoun, Verb) or (ProperNoun, Noun).

    Extracts specific Part-of-Speech (POS) combinations from the input text.

    The function first tokenizes the text into sentences. Then, for each sentence,
    it identifies proper nouns (PROPN), verbs (VERB), and nouns (NOUN).
    It forms:
    - Triplets: (ProperNoun, Verb, Noun) if all three are present.
    - Duplets: (ProperNoun, Verb) or (ProperNoun, Noun) if proper nouns are
               present with either verbs or nouns, but not both.

    Args:
        text (str): The input text string.

    Returns:
        list[list[str]]: A list where each inner list is a POS combination
                         (either a duplet or a triplet of token texts).
                         Returns an empty list if the spaCy model (`nlp`) is not available
                         or if no relevant POS combinations are found.
    """
    if not nlp:
        print("SpaCy model ('nlp') not available. POS combination extraction skipped.")
        return []

    lower_text = text.lower()
    try:
        sentences = nltk.sent_tokenize(lower_text)
    except Exception as e:
        print(f"NLTK sentence tokenization error: {e}. Falling back to splitting by periods.")
        sentences = [s.strip() for s in lower_text.split(".") if s.strip()]

    return pos_combinations(sentences)


def pos_combinations(sentences: list[str]) -> list[list[str]]:
    """
    Extract POS combinations from a list of sentences.

    Given a list of sentences, process each sentence with spaCy and extract
    specific POS combinations. The extracted combinations are either triplets
    (ProperNoun, Verb, Noun) or duplets (ProperNoun, Verb) or (ProperNoun, Noun)
    based on the presence of proper nouns, verbs, and nouns in the sentence.

    Args:
        sentences (list[str]): A list of sentences to process.

    Returns:
        list[list[str]]: A list of extracted POS combinations (either triplets or duplets).
    """
    pos_combinations: list[list[str]] = []
    for _, sentence_text in enumerate(sentences):
        doc = nlp(sentence_text)  # type: ignore[misc]
        proper_nouns: set[str] = set()
        verbs: set[str] = set()
        nouns: set[str] = set()

        for token in doc:
            if token.pos_ == "PROPN":
                proper_nouns.add(token.text)
            elif token.pos_ == "VERB":
                verbs.add(token.text)
            elif token.pos_ == "NOUN":
                nouns.add(token.text)

        if proper_nouns and verbs and not nouns:
            pos_combinations.extend([[on, v] for on in proper_nouns for v in verbs])
        elif proper_nouns and nouns and not verbs:
            pos_combinations.extend([[on, n] for on in proper_nouns for n in nouns])
        elif proper_nouns and verbs and nouns:
            pos_combinations.extend([[on, v, n] for on in proper_nouns for v in verbs for n in nouns])

    return pos_combinations


def match_triplet(triplet_a: tuple[str, str, str], triplet_b: tuple[str, str, str]) -> bool:
    """
    Compare two POS triplets for similarity.

    - Exact match across all three elements
    - If exact mismatch, check spaCy vector similarity for verb and noun.

    Args:
        triplet_a (tuple[str, str, str]): (pronoun, verb, noun) for the first triplet.
        triplet_b (tuple[str, str, str]): (pronoun, verb, noun) for the second triplet.

    Returns:
        bool: True if the triplets match based on exact values or similarity threshold.
    """
    pronoun_a, verb_a, noun_a = triplet_a
    pronoun_b, verb_b, noun_b = triplet_b
    if pronoun_a == pronoun_b and verb_a == verb_b and noun_a == noun_b:
        return True
    try:
        sim_verb = nlp(verb_a).similarity(nlp(verb_b))
        sim_noun = nlp(noun_a).similarity(nlp(noun_b))
    except Exception:
        return False

    return pronoun_a == pronoun_b and sim_verb > SIMILARITY_THRESHOLD and sim_noun > SIMILARITY_THRESHOLD


def match_duplet(elem1_a: str, elem2_a: str, elem1_b: str, elem2_b: str) -> bool:
    """
    Compare two POS duplets for similarity.

    - Exact match across both elements
    - If exact mismatch, check spaCy vector similarity for second element.

    Args:
        elem1_a (str): First element of first duplet.
        elem2_a (str): Second element of first duplet.
        elem1_b (str): First element of second duplet.
        elem2_b (str): Second element of second duplet.

    Returns:
        bool: True if the duplets match based on exact values or similarity threshold.
    """
    if elem1_a == elem1_b and elem2_a == elem2_b:
        return True
    try:
        sim_second = nlp(elem2_a).similarity(nlp(elem2_b))
    except Exception:
        return False

    return elem1_a == elem1_b and sim_second > SIMILARITY_THRESHOLD


def _match_triplet_with_logging(model_combo: list[str], cand_combo: list[str], i: int, j: int) -> bool:
    """
    Helper function to match triplets with logging of similarity scores.

    Args:
        model_combo (list[str]): The triplet from the model text.
        cand_combo (list[str]): The triplet from the candidate text.
        i (int): Index of the model combination.
        j (int): Index of the candidate combination.

    Returns:
        bool: True if match, False otherwise.
    """
    on_m, v_m, n_m = model_combo
    on_c, v_c, n_c = cand_combo
    try:
        sim_verb = nlp(v_m).similarity(nlp(v_c)) if nlp else 0.0
        sim_noun = nlp(n_m).similarity(nlp(n_c)) if nlp else 0.0
    except Exception:
        sim_verb = sim_noun = 0.0
    print(f"[Triplet] Model combo {i} vs Cand combo {j} -> VerbSim={sim_verb:.3f}, NounSim={sim_noun:.3f}")
    return match_triplet((on_m, v_m, n_m), (on_c, v_c, n_c))


def _match_duplet_with_logging(
    model_combo: list[str],
    cand_combo: list[str],
    i: int,
    j: int,
) -> bool:
    """
    Helper function to match duplets with logging of similarity scores.

    Args:
        model_combo (list[str]): The duplet from the model text.
        cand_combo (list[str]): The duplet from the candidate text.
        i (int): Index of the model combination.
        j (int): Index of the candidate combination.

    Returns:
        bool: True if match, False otherwise.
    """
    el1_m, el2_m = model_combo
    el1_c, el2_c = cand_combo
    try:
        sim_second_el = nlp(el2_m).similarity(nlp(el2_c)) if nlp else 0.0
    except Exception:
        sim_second_el = 0.0
    print(f"[Duplet]  Model combo {i} vs Cand combo {j} -> SecondElSim={sim_second_el:.3f}")
    return match_duplet(el1_m, el2_m, el1_c, el2_c)


def calculate_pos_overlap(model_list: list[list[str]], cand_list: list[list[str]]) -> float:
    """
    Compute an overlap score based on matching POS combinations.

    The score is the fraction of POS combinations from the `model_list` that
    find a match in the `cand_list`. A model combination is considered matched
    if it finds at least one similar candidate combination.

    Args:
        model_list (list[list[str]]): A list of POS combinations (triplets or duplets)
                                      extracted from the model text.
        cand_list (list[list[str]]): A list of POS combinations extracted from the
                                     candidate text.

    Returns:
        float: The fraction of model combinations that are matched in the candidate list.
               Returns 0.0 if either list is empty.
    """
    if not model_list or not cand_list:
        return 0.0

    matched_model_combos = 0
    for i, model_combo in enumerate(model_list):
        for j, cand_combo in enumerate(cand_list):
            if (
                len(model_combo) == TRIPLET_LENGTH
                and len(cand_combo) == TRIPLET_LENGTH
                and _match_triplet_with_logging(model_combo, cand_combo, i, j)
            ) or (
                len(model_combo) == DUPLET_LENGTH
                and len(cand_combo) == DUPLET_LENGTH
                and _match_duplet_with_logging(model_combo, cand_combo, i, j)
            ):
                matched_model_combos += 1
                break
    return matched_model_combos / len(model_list) if model_list else 0.0


def score_pos(model_text: str, candidate_text: str) -> float:
    """
    Score texts based on Part-of-Speech (POS) pattern similarity.

    - Extract POS combinations from both texts
    - Calculate and log overlap score details
    - Return normalized score

    Args:
        model_text (str): The reference or model text.
        candidate_text (str): The candidate or student text.

    Returns:
        float: The calculated POS similarity score.
    """
    start = timer()
    print("Extracting POS combinations...")
    model_combos = extract_pos_combinations(model_text)
    cand_combos = extract_pos_combinations(candidate_text)

    print("Calculating POS overlap and logging each comparison...")
    score = calculate_pos_overlap(model_combos, cand_combos)

    elapsed = timer() - start
    print(f"POS score computed in {elapsed:.4f}s -> final score={score:.3f}")
    return score


if __name__ == "__main__":
    if not nlp:
        print("Error: SpaCy model not loaded. Terminating.")
    else:
        examples = [
            (
                "Dr. John Smith presented his new findings from the research. "
                "He said the project was a success because the team worked hard.",
                "John Smith talked about the findings. Smith mentioned the project succeeded. The group was diligent.",
            ),
            (
                "The company launched its innovative product. The CEO, Mrs. Davis, "
                "explained that it would revolutionize the market.",
                "A new device was revealed by the firm. Davis spoke about the plan. It is a game changer.",
            ),
            ("The cat sat on the mat.", "The dog barked at the moon."),
            ("John saw Mary.", "Hello there."),
        ]

        for idx, (m, c) in enumerate(examples, 1):
            print(f"\n--- Example {idx} ---")
            print(f"Model: {m}\nCandidate: {c}")
            print(f"Score: {score_pos(m, c):.4f}")
