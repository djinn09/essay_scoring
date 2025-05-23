from __future__ import annotations

from timeit import default_timer as timer

import nltk  # For sentence tokenization in POS scoring

from config import spacy_model

nlp = spacy_model
SIMILARITY_THRESHOLD = 0.3


def extract_pos_combinations(text: str) -> list[list[str]]:
    """Tokenize input text into sentences and extract POS combinations:
    - Triplets: (ProperNoun, Verb, Noun)
    - Duplets: (ProperNoun, Verb) or (ProperNoun, Noun).

    Returns:
        List of combinations per sentence (order preserved by sentence processing).

    """
    if not nlp:
        print("SpaCy model not available. POS extraction skipped.")
        return []

    lower = text.lower()
    try:
        sentences = nltk.sent_tokenize(lower)
    except Exception as e:
        print(f"NLTK tokenization error: {e}")
        sentences = [s.strip() for s in lower.split(".") if s.strip()]

    combos: list[list[str]] = []
    for _, sent in enumerate(sentences):
        doc = nlp(sent)
        props, verbs, nouns = set(), set(), set()

        # Identify proper nouns, verbs, and nouns in each sentence
        for token in doc:
            if token.pos_ == "PROPN":
                props.add(token.text)
            elif token.pos_ == "VERB":
                verbs.add(token.text)
            elif token.pos_ == "NOUN":
                nouns.add(token.text)

        # Build duplets if verbs or nouns exist without the other
        if props and verbs and not nouns:
            combos.extend([[p, v] for p in props for v in verbs])
        elif props and nouns and not verbs:
            combos.extend([[p, n] for p in props for n in nouns])
        # Build triplets if both verbs and nouns present
        elif props and verbs and nouns:
            combos.extend([[p, v, n] for p in props for v in verbs for n in nouns])

    return combos


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
    - Iterates through each model combination and candidate combination
    - Logs similarity scores and indices for each comparison
    - Returns fraction of model combos matched.
    """
    if not model_list or not cand_list:
        return 0.0

    matched = 0
    # Compare each combination and log details
    for i, m in enumerate(model_list):
        for j, c in enumerate(cand_list):
            # Determine type: triplet vs duplet
            if len(m) == 3 and len(c) == 3:
                # Compute similarity components
                pronoun_a, verb_a, noun_a = m
                pronoun_b, verb_b, noun_b = c
                try:
                    sim_verb = nlp(verb_a).similarity(nlp(verb_b))
                    sim_noun = nlp(noun_a).similarity(nlp(noun_b))
                except Exception:
                    sim_verb = sim_noun = 0.0
                print(f"[Triplet] Model idx {i}, Cand idx {j} -> verb_sim={sim_verb:.3f}, noun_sim={sim_noun:.3f}")
                if match_triplet(pronoun_a, verb_a, noun_a, pronoun_b, verb_b, noun_b):
                    matched += 1
                    break
            elif len(m) == 2 and len(c) == 2:
                elem1_a, elem2_a = m
                elem1_b, elem2_b = c
                try:
                    sim_second = nlp(elem2_a).similarity(nlp(elem2_b))
                except Exception:
                    sim_second = 0.0
                print(f"[Duplet]  Model idx {i}, Cand idx {j} -> sim={sim_second:.3f}")
                if match_duplet(elem1_a, elem2_a, elem1_b, elem2_b):
                    matched += 1
                    break
    return matched / len(model_list)


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
