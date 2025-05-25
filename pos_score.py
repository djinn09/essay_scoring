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

import nltk  # For sentence tokenization in POS scoring

from config import spacy_model # Imports the globally loaded spaCy model

# Use the globally loaded spaCy model from the config module.
# This assumes config.py has successfully loaded a spaCy model.
nlp = spacy_model
SIMILARITY_THRESHOLD = 0.3 # Threshold for spaCy's vector similarity for matching verbs/nouns.


def extract_pos_combinations(text: str) -> list[list[str]]:
    """Tokenize input text into sentences and extract POS combinations:
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
    if not nlp: # Check if spaCy model is loaded.
        print("SpaCy model ('nlp') not available. POS combination extraction skipped.")
        return []

    lower_text = text.lower() # Convert text to lowercase for consistent processing.
    try:
        # Attempt sentence tokenization using NLTK.
        sentences = nltk.sent_tokenize(lower_text)
    except Exception as e: # Fallback sentence tokenization if NLTK fails.
        print(f"NLTK sentence tokenization error: {e}. Falling back to splitting by periods.")
        sentences = [s.strip() for s in lower_text.split(".") if s.strip()]

    pos_combinations: list[list[str]] = [] # List to store all extracted combinations.
    for _, sentence_text in enumerate(sentences):
        # Process each sentence with spaCy.
        doc = nlp(sentence_text)
        # Sets to store unique proper nouns, verbs, and nouns found in the sentence.
        proper_nouns, verbs, nouns = set(), set(), set()

        # Identify and collect PROPN, VERB, NOUN tokens from the sentence.
        for token in doc:
            if token.pos_ == "PROPN":
                proper_nouns.add(token.text)
            elif token.pos_ == "VERB":
                verbs.add(token.text)
            elif token.pos_ == "NOUN":
                nouns.add(token.text)

        # Logic for forming POS combinations based on presence:
        # Form (ProperNoun, Verb) duplets if nouns are absent but proper nouns and verbs are present.
        if proper_nouns and verbs and not nouns:
            pos_combinations.extend([[pn, v] for pn in proper_nouns for v in verbs])
        # Form (ProperNoun, Noun) duplets if verbs are absent but proper nouns and nouns are present.
        elif proper_nouns and nouns and not verbs:
            pos_combinations.extend([[pn, n] for pn in proper_nouns for n in nouns])
        # Form (ProperNoun, Verb, Noun) triplets if all three POS types are present.
        elif proper_nouns and verbs and nouns:
            pos_combinations.extend([[pn, v, n] for pn in proper_nouns for v in verbs for n in nouns])

    return pos_combinations


def match_triplet(pronoun_a: str, verb_a: str, noun_a: str, pronoun_b: str, verb_b: str, noun_b: str) -> bool:
    """
    Compare two POS triplets for similarity:
    - Exact match across all three elements
    - If exact mismatch, check spaCy vector similarity for verb and noun
    """
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
    Compare two POS duplets for similarity:
    - Exact match across both elements
    - If exact mismatch, check spaCy vector similarity for second element
    """
    if elem1_a == elem1_b and elem2_a == elem2_b:
        return True
    try:
        sim_second = nlp(elem2_a).similarity(nlp(elem2_b))
    except Exception:
        return False

    return elem1_a == elem1_b and sim_second > SIMILARITY_THRESHOLD


def calculate_pos_overlap(model_list: list[list[str]], cand_list: list[list[str]]) -> float:
    """Compute POS overlap score:
    Computes an overlap score based on matching POS combinations.

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
    if not model_list or not cand_list: # If either list is empty, no overlap is possible.
        return 0.0

    matched_model_combos = 0
    # Iterate through each POS combination from the model text.
    for i, model_combo in enumerate(model_list):
        # For each model combination, search for a matching combination in the candidate list.
        for j, cand_combo in enumerate(cand_list):
            match_found = False
            # Check if both are triplets and try to match them.
            if len(model_combo) == 3 and len(cand_combo) == 3:
                # Unpack triplet components.
                pn_m, v_m, n_m = model_combo
                pn_c, v_c, n_c = cand_combo
                # Log similarity details for debugging/analysis (optional).
                try:
                    sim_verb = nlp(v_m).similarity(nlp(v_c)) if nlp else 0.0
                    sim_noun = nlp(n_m).similarity(nlp(n_c)) if nlp else 0.0
                except Exception: # Catch errors if tokens are out-of-vocabulary for spaCy similarity.
                    sim_verb = sim_noun = 0.0
                print(f"[Triplet] Model combo {i} vs Cand combo {j} -> VerbSim={sim_verb:.3f}, NounSim={sim_noun:.3f}")
                if match_triplet(pn_m, v_m, n_m, pn_c, v_c, n_c):
                    match_found = True
            # Check if both are duplets and try to match them.
            elif len(model_combo) == 2 and len(cand_combo) == 2:
                # Unpack duplet components.
                el1_m, el2_m = model_combo
                el1_c, el2_c = cand_combo
                try:
                    sim_second_el = nlp(el2_m).similarity(nlp(el2_c)) if nlp else 0.0
                except Exception:
                    sim_second_el = 0.0
                print(f"[Duplet]  Model combo {i} vs Cand combo {j} -> SecondElSim={sim_second_el:.3f}")
                if match_duplet(el1_m, el2_m, el1_c, el2_c):
                    match_found = True

            if match_found:
                matched_model_combos += 1
                break  # Once a model combination is matched, move to the next model combination.
    # Score is the fraction of model combinations that were matched.
    return matched_model_combos / len(model_list) if model_list else 0.0


def score_pos(model_text: str, candidate_text: str) -> float:
    """Main entry point for POS-based scoring with optional coreference:
    - Extract POS combinations from both texts
    - Calculate and log overlap score details
    - Return normalized score
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
                "Dr. John Smith presented his new findings from the research. He said the project was a success because the team worked hard.",
                "John Smith talked about the findings. Smith mentioned the project succeeded. The group was diligent.",
            ),
            (
                "The company launched its innovative product. The CEO, Mrs. Davis, explained that it would revolutionize the market.",
                "A new device was revealed by the firm. Davis spoke about the plan. It is a game changer.",
            ),
            ("The cat sat on the mat.", "The dog barked at the moon."),
            ("John saw Mary.", "Hello there."),
        ]

        for idx, (m, c) in enumerate(examples, 1):
            print(f"\n--- Example {idx} ---")
            print(f"Model: {m}\nCandidate: {c}")
            print(f"Score: {score_pos(m, c):.4f}")
