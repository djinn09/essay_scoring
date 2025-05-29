"""Coreference resolution utilities."""
# ------------------------------------------------------------------------------
# Rule-Based Coreference Resolution using SpaCy
# ------------------------------------------------------------------------------
# This script implements a rule-based approach to coreference resolution,
# a critical task in Natural Language Processing (NLP) for identifying
# expressions in text that refer to the same entity (persons, places, things, etc.).
#
# Approach Inspired By:
# - The concepts outlined in general NLP coreference resolution literature.
# - Utilizes SpaCy (en_core_web_md) for linguistic features:
#   - Part-of-Speech (POS) tagging
#   - Dependency Parsing (Syntactic Structure)
#   - Named Entity Recognition (NER)
#   - Morphological Analysis (Number, Gender)
#   - Word Vectors (for semantic similarity fallback)
#
# Methodology (Rule-Based):
# As described in the article "The Key to Unlocking True Language Understanding:
# Coreference Resolution", this system relies on predefined linguistic rules
# and heuristics based on syntactic and semantic patterns. It processes
# potential referring expressions (mentions), primarily pronouns and proper nouns,
# and attempts to link them to preceding potential antecedents.
#
# Key Rules Implemented:
# 1. Pleonastic 'It' detection (non-referential 'it')
# 2. Reflexive pronoun resolution (e.g., 'himself' -> subject)
# 3. Relative pronoun resolution (e.g., 'who'/'which' -> head noun)
# 4. Quoted speech pronoun resolution ('I'/'we' -> speaker)
# 5. Possessive pronoun resolution ('his'/'her' -> possessor)
# 6. Standard pronoun resolution (backward search with agreement checks)
# 7. Proper noun matching (exact and partial matches)
#
# Scoring & Ambiguity Handling:
# To address the challenge of ambiguity mentioned in the article, rules are
# prioritized, and a confidence score is assigned based on the rule's reliability
# and contextual factors like NER matches, subject salience, and proximity.
# More reliable rules (e.g., Reflexive, Relative) get higher scores.
#
# Context Management:
# Resolving coreferences often depends on context. This system uses a
# configurable sentence window (`search_sentences`) to limit the search space
# for antecedents, balancing recall with the challenge of limited context.
#
# Limitations (Compared to ML/Neural Approaches):
# - Interpretability: Rule-based systems are generally more interpretable.
# - Complexity & Variability: May struggle with the vast complexity and variability
#   of natural language compared to models trained on large corpora.
# - Generalization: May not generalize as well to unseen patterns.
# - World Knowledge: Lacks deep common sense or world knowledge.
# - Definite Noun Phrase & Clausal Coreference: Primarily focuses on pronouns
#   and proper nouns, with limited handling of other referring expression types.
# - Cluster Building: Outputs pairs, not fully resolved entity clusters (though
#   pairs could be used for downstream clustering).
#
# Evaluation:
# Standard metrics like MUC, B-Cubed, and CoNLL F1 are typically used to evaluate
# coreference resolution systems, assessing performance on mentions, links, and chains.
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional  # Import necessary types for annotation

import gender_guesser.detector as gender
import spacy

if TYPE_CHECKING:
    from spacy.tokens import Doc, Token

# --- Constants ---
PROXIMITY_THRESHOLD = 0.05  # Used for candidate scoring
SINGULAR_THEY_THRESHOLD = 0.70  # Used for singular 'they' scoring boost
COLLECTIVE_NOUNS = {"team", "committee", "government", "group", "company", "staff", "jury", "class", "party"}
REPORTING_VERBS = {
    "say",
    "tell",
    "ask",
    "reply",
    "shout",
    "whisper",
    "claim",
    "state",
    "add",
    "explain",
    "note",
    "report",
    "argue",
}
PERSONAL_PRONOUN_LEMMAS = {"he", "she", "it", "they", "we", "i", "you"}
# list of common inanimate nouns likely to be Neut
NEUTER_NOUNS = {
    "car",
    "book",
    "table",
    "house",
    "report",
    "software",
    "cup",
    "water",
    "painting",
    "findings",
    "approach",
    "spirit",
    "puzzle",
    "job",
    "speech",
    "market",
    "party",
    "coffee",
}

# --- Load Model ---
# Note: The spaCy model is loaded globally when this module is imported.
# This can affect startup time and makes the module less suitable for environments
# where spaCy or the specific model is not available or needed immediately.
# Consider deferring model loading to an initialization function or class constructor if needed.
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading en_core_web_md model for coreference resolution...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# --- Helper Functions (with Unspecified Gender Handling) ---

# 1. Pronoun → gender mapping
PRONOUN_GENDER = {
    **dict.fromkeys(["he", "his", "him", "himself"], "Masc"),
    **dict.fromkeys(["she", "her", "hers", "herself"], "Fem"),
    **dict.fromkeys(["it"], "Neut"),
}

# 2. Static name-based hints (small list of common names)
MASC_NAMES = {"john", "paul", "mike", "peter", "bob", "james", "william", "david", "george"}
FEM_NAMES = {"mary", "lisa", "sarah", "alice", "susan", "evelyn", "jane", "elizabeth", "ann", "kate"}
NAME_GENDER = dict.fromkeys(MASC_NAMES, "Masc")
NAME_GENDER.update(dict.fromkeys(FEM_NAMES, "Fem"))

# 3. Common neuter nouns (expandable list)
NEUTER_NOUNS = {"object", "device", "tool", "manager", "car", "book", "company"}

# 4. Initialize gender-guesser detector
DETECTOR = gender.Detector(case_sensitive=False)


def get_gender(token: Token) -> Optional[str]:
    """Determine the 'gender' feature for a spacy Token.

    Order of checks:
      1. Pronoun lookup
      2. SpaCy morphological 'Gender'
      3. Static name lookup
      4. gender-guesser fallback for PERSON entities
      5. Common neuter nouns
      6. Default to 'Unspecified'.
    """
    gender_val = None
    # Normalize lemma and text
    lemma = token.lemma_.lower()
    text = token.text.strip().lower()

    # 1. Pronoun-based gender
    if lemma in PRONOUN_GENDER:
        gender_val = PRONOUN_GENDER[lemma]
    # 2. SpaCy morphological gender
    elif token.morph.get("Gender", []):
        gender_feats = token.morph.get("Gender", [])
        gender_val = gender_feats[0] if gender_feats else None
    # 3. Named entity static lookup (PERSON)
    elif token.ent_type_ == "PERSON":
        if text in NAME_GENDER: # Static name hints
            gender_val = NAME_GENDER[text]
        else: # gender-guesser fallback
            guess = DETECTOR.get_gender(text)
            if guess in ("male", "mostly_male"):
                gender_val = "Masc"
            elif guess in ("female", "mostly_female"):
                gender_val = "Fem"
            else: # PERSON but unknown gender
                gender_val = "Unspecified"
    # 5. Common neuter nouns
    elif token.pos_ == "NOUN" and lemma in NEUTER_NOUNS:
        gender_val = "Neut"
    # 6. Default
    else:
        gender_val = "Unspecified"

    return gender_val


# Pronoun → number mapping
PRONOUN_NUMBER = {
    **dict.fromkeys(["he", "she", "it", "i", "me", "myself", "himself", "herself", "itself"], "Sing"),
    **dict.fromkeys(["we", "they", "us", "them", "ourselves", "themselves"], "Plur"),
}

# POS tag-based number hints
SING_TAGS = {"NN", "NNP"}
PLUR_TAGS = {"NNS", "NNPS"}


def get_number(token: Token) -> list:
    """Determine the 'number' feature for a spacy Token.

    Order of checks:
      1. Pronoun lookup
      2. SpaCy morphological 'Number'
      3. POS tag fallback
      4. Default to 'Sing'.
    """
    lemma = token.lemma_.lower()
    tag = token.tag_
    # 1. Pronoun-based number
    if lemma in PRONOUN_NUMBER:
        return [PRONOUN_NUMBER[lemma]]

    # 2. SpaCy morphological number
    num_feats = token.morph.get("Number", None)
    if num_feats:
        return num_feats
    # 3. POS tag-based fallback
    if tag in SING_TAGS:
        return ["Sing"]
    if tag in PLUR_TAGS:
        return ["Plur"]

    # 4. Default
    return ["Sing"]


def check_agreement(pronoun: Token, candidate: Token) -> tuple[bool, bool]:
    """Check whether a pronoun and a candidate token agree in terms of number and gender.

    Returns a tuple of two booleans. The first boolean indicates whether the agreement check
    succeeded. The second boolean is True if the agreement check passed due to the singular
    'they' case, and False otherwise. This can be used to filter out cases where the agreement
    check is not very informative.

    The agreement check is done in two parts: number and gender.

    For number agreement, the check is done by looking at the morphological features of the
    pronoun and the candidate. If the pronoun is a singular pronoun (e.g. 'he', 'she', 'it'),
    then the candidate must also be singular. If the pronoun is a plural pronoun (e.g. 'they'),
    then the candidate can be either singular or plural. There are two special cases:

    - If the candidate is a person and the pronoun is 'they', then the agreement check passes.
    - If the candidate is a collective noun (e.g. 'team', 'family') and the pronoun is 'they',
      then the agreement check passes.

    For gender agreement, the check is done by looking at the morphological features of the
    pronoun and the candidate. If the pronoun is a gendered pronoun (e.g. 'he', 'she'), then
    the candidate must have the same gender. If the pronoun is a neuter pronoun (e.g. 'it'),
    then the candidate must also be neuter. If the candidate is unspecified, then the
    agreement check passes.

    :param pronoun: The pronoun token.
    :param candidate: The candidate token.
    :return: A tuple of two booleans. The first boolean indicates whether the agreement check
             succeeded. The second boolean is True if the agreement check passed due to the
             singular 'they' case, and False otherwise.
    """
    # --- Number Agreement ---
    pronoun_number = set(get_number(pronoun))
    candidate_number = set(get_number(candidate))
    is_singular_they_case = False
    is_collective_noun_case = False
    # Direct overlap means no mismatch
    if not pronoun_number & candidate_number:
        # Singular 'they' for PERSON
        is_singular_they_case = (
            "Sing" in candidate_number
            and pronoun.lemma_ == "they"
            and (candidate.ent_type_ == "PERSON" or candidate.lemma_ == "friend")
        )
        is_collective_noun_case = (
            "Plur" in pronoun_number
            and "Sing" in candidate_number
            and (candidate.ent_type_ == "ORG" or candidate.lemma_ in COLLECTIVE_NOUNS)
        )
        if not (is_singular_they_case or is_collective_noun_case):
            return False, False  # Failed number agreement

    # --- Gender Agreement ---
    pron_gender = get_gender(pronoun)
    cand_gender = get_gender(candidate)

    # Case 1: Pronoun is Neuter ('it')
    if pron_gender == "Neut":
        # 'it' should only match explicit Neut.
        if cand_gender != "Neut":
            return False, False
        return True, False  # Agreement OK, not singular they

    # Case 2: Pronoun is Gendered ('he', 'she') or Plural ('they')
    # Allow 'Unspecified' candidates to match with specific genders,
    # but disallow direct clashes (Masc vs Fem).
    if cand_gender != "Unspecified" and pron_gender != "Unspecified":  # Both have specified genders
        if (pron_gender == "Masc" and cand_gender == "Fem") or (pron_gender == "Fem" and cand_gender == "Masc"):
            return False, False  # Direct gender clash

    # If no explicit clash, or if one is Unspecified, allow match.
    # - 'he'/'she' can match Masc/Fem respectively, or Unspecified candidate.
    # - 'they' can match Masc, Fem, Neut, or Unspecified candidate (plural or singular).
    return True, is_singular_they_case


# --- Other Helpers (Reflexive, Subject, Sentence, Speaker, Pleonastic - unchanged from v3) ---


def is_reflexive(token: Token) -> bool:
    """Determine whether a token is a reflexive pronoun (e.g., 'himself', 'herself').

    A reflexive pronoun in English ends with 'self' and has the 'PRP' tag.

    :param token: The spaCy token to check.
    :return: True if reflexive pronoun, False otherwise.
    """
    return token.tag_ == "PRP" and token.lemma_.endswith("self")


def find_subject(token: Token) -> Token | None:
    """Attempt to locate the syntactic subject associated with a given token.

    Traverses upward in the dependency tree until a governing verb is found,
    then returns the subject of that verb, if any.

    :param token: The token for which to find the associated subject.
    :return: The subject token, or None if not found.
    """
    head = token.head
    # Traverse up from the token until a verb or the root of the sentence is found.
    while head.pos_ not in ("VERB", "AUX") and head.dep_ != "ROOT" and head.head != head:
        head = head.head  # Move to the head of the current token.

    # If a verb or the root is found, look for its subject(s).
    if head.pos_ in ("VERB", "AUX") or head.dep_ == "ROOT":
        # Prioritize nominal subjects (nsubj, nsubjpass).
        subjects = [c for c in head.children if c.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            # If no nominal subject, look for clausal subjects (csubj, csubjpass).
            subjects = [c for c in head.children if c.dep_ in ("csubj", "csubjpass")]

        if subjects:
            core_subj = subjects[0]  # Take the first subject found.
            # Refinement: If the subject is part of a compound noun, try to get the true head of the compound.
            # This handles cases like "The big black cat" -> "cat" not "The".
            # Ensure the head of the compound still precedes the original token to avoid jumping too far.
            while core_subj.dep_ == "compound" and core_subj.head.i < token.i:
                core_subj = core_subj.head
            # Further refinement for non-PERSON/PROPN subjects
            if core_subj.ent_type_ != "PERSON" and core_subj.pos_ != "PROPN":
                # Search for a PERSON entity in the subject's subtree preceding the subject itself
                person_in_subj = [
                    t for t in core_subj.subtree if t.ent_type_ == "PERSON" and t.i < core_subj.i
                ]
                if person_in_subj:
                    return person_in_subj[-1] # Return the last such PERSON found
            return core_subj
    return None  # No subject found.


def find_speaker(pronoun: Token) -> Token | None:
    """Find the likely speaker associated with a pronoun within its sentence.

    It traverses up the dependency tree looking for a governing reporting verb
    and then tries to extract the subject of that verb.

    :param pronoun: The pronoun token to resolve.
    :return: The token likely referring to the speaker, or None.
    """
    current = pronoun
    # Traverse up the dependency tree within the same sentence.
    while current.head != current and current.sent == pronoun.sent:
        governing_verb = current.head

        # Scenario 1: Pronoun is part of a clause governed by a reporting verb.
        # e.g., Mary said, "I am tired." -> "I" is in ccomp of "said".
        # e.g., Mary asked him if he was okay -> "he" is in advcl/ccomp of "asked"
        # e.g., Mary told him "You are right" -> "You" can be dobj of "told" if "You are right" is direct object.
        if governing_verb.lemma_ in REPORTING_VERBS and (
            current.dep_ in ("ccomp", "advcl", "xcomp") or (current.dep_ == "dobj" and current.pos_ == "PRON")
        ):  # "dobj" more likely if pronoun is object of reporting verb
            speaker = find_subject(governing_verb)  # The subject of the reporting verb is the speaker.
            return speaker if speaker else None

        # Scenario 2: Pronoun is the subject of a clause that itself is governed by a reporting verb.
        # e.g., The report stated that "He was seen..." -> "He" is nsubj of "was seen", "was seen" is ccomp of "stated".
        if current.dep_ == "nsubj" and governing_verb.dep_ == "ccomp" and governing_verb.head.lemma_ in REPORTING_VERBS:
            reporting_verb = governing_verb.head  # This is the actual reporting verb.
            speaker = find_subject(reporting_verb)  # The subject of this reporting verb.
            return speaker if speaker else None

        current = current.head  # Move up the tree.

    return None  # No speaker found through these patterns.


def is_pleonastic_it(token: Token) -> bool:
    """Return True if the token 'it' is used pleonastically (e.g., weather, time, cleft)."""
    if token.lemma_ != "it":  # Must be the pronoun 'it'.
        return False

    # Rule 1: Expletive 'it' (e.g., "It is raining.")
    # SpaCy often tags expletive 'it' with dep_ == "expl".
    if token.dep_ == "expl":
        return True

    # Further checks if 'it' is a nominal subject (nsubj).
    if token.dep_ != "nsubj":
        # If 'it' is not a subject here, less likely to be pleonastic by these rules.
        return False

    verb = token.head  # The verb governed by 'it'.
    pleonastic = False
    if verb.pos_ == "VERB":
        if _check_weather_verbs(verb) or \
           _check_time_attributes(verb) or \
           _check_clausal_complements(verb):
            pleonastic = True
    # Rule 5: Auxiliary 'be' + time/numeric attribute or cleft constructions
    elif verb.lemma_ == "be" and verb.pos_ == "AUX":  # If 'it' is subject of an auxiliary 'be'.
        if _check_time_attributes(verb) or \
           _check_cleft_construction(verb): # Re-use time attribute check
            pleonastic = True

    return pleonastic


def _check_weather_verbs(verb: Token) -> bool:
    """Check if the verb is a weather verb."""
    return verb.lemma_ in {"rain", "snow", "hail", "thunder", "lighten"}


def _check_time_attributes(verb: Token) -> bool:
    """Check for time-related attributes with 'be' verb."""
    if verb.lemma_ == "be":
        attr = next((c for c in verb.children if c.dep_ == "attr"), None)
        if attr:
            subtree = list(attr.subtree)
            has_time_ent = attr.ent_type_ == "TIME" or any(t.ent_type_ == "TIME" for t in subtree)
            has_time_keyword = any(
                t.text.lower() in {"o'clock", "pm", "am", "noon", "midnight", "raining", "snowing", "sunny", "cloudy"}
                for t in subtree
            )
            # Check for "almost" + number (e.g., "It is almost 5.")
            has_almost = any(c.lemma_ == "almost" and c.dep_ == "advmod" for c in attr.children)
            is_num_like = attr.like_num or (attr.text and attr.text[0].isdigit())
            if has_time_ent or has_time_keyword or (has_almost and is_num_like):
                return True
    return False


def _check_clausal_complements(verb: Token) -> bool:
    """Check for verbs of seeming/appearing with clausal complements."""
    return verb.lemma_ in {"seem", "appear", "happen", "matter", "turn out", "look", "sound"} and any(
        c.dep_ in {"ccomp", "csubj", "xcomp", "acomp", "advcl"} for c in verb.children
    )


def _check_cleft_construction(verb: Token) -> bool:
    """Check for cleft constructions."""
    attr = next((c for c in verb.children if c.dep_ == "attr"), None)
    if attr:
        relcl = next((c for c in verb.children if c.dep_ == "relcl" and c.head == attr), None)
        if relcl and relcl.sent == verb.sent and relcl[0].tag_ in ["WP", "WDT"]:
            return True
    return False


# --- End Helper Functions ---


# --- Main Resolution Function (using the updated helpers) ---
# The main function rule_based_coref_resolution_v3 remains unchanged in its
# structure and rule logic, as the changes were within the helper functions.
# Make sure to call the function with the updated helpers integrated.


# --- Main Resolution Function ---
def rule_based_coref_resolution_v4(
    text: str,
    similarity_threshold: float = 0.5,
    *,
    use_similarity_fallback: bool = False,
    search_sentences: int = 2,
) -> list[tuple[dict[str, Any], dict[str, Any], float, str]]:
    """Apply v4 rule-based coreference resolution with Unspecified Gender handling.

    Identifies coreferent mentions (pronouns, proper nouns) and links them to
    their likely antecedents within a specified sentence window, using rules
    based on syntax, NER, morphology, and heuristics.

    Args:
        text (str): The input text document.
        similarity_threshold (float): Minimum word vector similarity for fallback rule.
                                      Only used if `use_similarity_fallback` is True.
        use_similarity_fallback (bool): Enable/disable similarity fallback rule.
        search_sentences (int): Number of sentences (current + preceding N-1)
                                to search backwards for antecedents.

    Returns:
        list[tuple[dict[str, Any], dict[str, Any], float, str]]:
            A list of resolved coreference pairs. Each tuple contains:
            (
                {'text': str, 'start': int, 'end': int}, # Mention span info
                {'text': str, 'start': int, 'end': int}, # Antecedent span info
                float,                                  # Confidence score (0.0-1.0)
                str                                     # Rule/heuristic name triggering the match
            )

    """
    # Process the text with SpaCy NLP pipeline
    doc = nlp(text)
    # list to store results internally (using Token objects)
    coref_results_internal = []
    # Set to keep track of mention token indices that have already been resolved
    processed_mentions = set()

    # --- Processing Loop: Iterate through sentences and tokens ---
    for sent_idx, sentence in enumerate(doc.sents):
        # Define the start token index for the backward search window
        start_search_token_idx = 0
        if search_sentences > 1 and sent_idx > 0:
            sents_list = list(doc.sents)  # Need list access for indexing
            first_sent_idx_in_window = max(0, sent_idx - search_sentences + 1)
            start_search_token_idx = sents_list[first_sent_idx_in_window].start

        # Iterate through tokens within the current sentence
        for i in range(sentence.start, sentence.end):
            token = doc[i]  # The current token being checked as a potential mention

            # Skip if this token has already been resolved as a mention
            if token.i in processed_mentions:
                continue

            # Variables to store the resolution result for this token
            antecedent = None
            confidence = 0.0
            rule = "N/A"

            if token.pos_ == "PRON":
                antecedent, confidence, rule = _resolve_pronoun(
                    token,
                    doc,
                    start_search_token_idx,
                    processed_mentions,
                    similarity_threshold,
                    use_similarity_fallback,
                )
            elif token.pos_ == "PROPN":
                antecedent, confidence, rule = _resolve_proper_noun(
                    token, doc, start_search_token_idx, processed_mentions,
                )

            if antecedent and antecedent.i != token.i:
                coref_results_internal.append((token, antecedent, round(confidence, 2), rule))
                processed_mentions.add(token.i)

    # --- Convert results to final output format with indices ---
    coref_pairs_with_indices = []
    # Iterate through the resolved pairs (stored with Token objects)
    for mention_tok, ant_tok, conf, rule_name in coref_results_internal:
        # Create dictionary for mention span with text and char indices
        mention_span = {
            "text": mention_tok.text,
            "start": mention_tok.idx,  # Character offset of the token's start
            "end": mention_tok.idx + len(mention_tok.text),  # Character offset of the token's end
        }
        # Create dictionary for antecedent span with text and char indices
        antecedent_span = {
            "text": ant_tok.text,
            "start": ant_tok.idx,
            "end": ant_tok.idx + len(ant_tok.text),
        }
        # Append the formatted tuple to the final list
        coref_pairs_with_indices.append((mention_span, antecedent_span, conf, rule_name))

    return coref_pairs_with_indices


def _resolve_pronoun(
    token: Token,
    doc: Doc,
    *,
    start_search_token_idx: int,
    processed_mentions: set,
    similarity_threshold: float,
    use_similarity_fallback: bool,
) -> tuple[Optional[Token], float, str]:
    """Resolve a pronoun token to its antecedent if possible."""
    antecedent = None
    confidence = 0.0
    rule = "N/A"
    best_candidate_info = None

    is_personal_pronoun = token.tag_ == "PRP"
    is_possessive_pronoun = token.tag_ == "PRP$"
    is_relative_possessive = token.tag_ == "WP$"
    is_relative_nonpossessive = token.tag_ in ["WP", "WDT"]
    is_reflexive_pronoun = is_reflexive(token)

    if not (is_personal_pronoun or is_possessive_pronoun or is_relative_possessive or is_relative_nonpossessive):
        return None, 0.0, "N/A"

    if token.lemma_ == "it" and is_pleonastic_it(token):
        processed_mentions.add(token.i)
        return None, 0.0, "N/A"

    if is_reflexive_pronoun:
        subj = find_subject(token)
        if subj:
            antecedent = subj
            confidence = 0.95
            rule = "Reflexive Pronoun -> Subject"
    elif is_relative_nonpossessive:
        potential_antecedent = token.head
        if potential_antecedent.pos_ in ("ADP", "AUX", "VERB"):
            potential_antecedent = potential_antecedent.head
        if potential_antecedent.pos_ in {"NOUN", "PROPN", "PRON"}:
            agrees, _ = check_agreement(token, potential_antecedent)
            if agrees:
                antecedent = potential_antecedent
                confidence = 0.92
                rule = "Relative Pronoun -> Syntactic Head"
    elif not antecedent and token.lemma_ in {"i", "me", "my", "we", "us", "our"}:
        speaker = find_speaker(token)
        if speaker:
            agrees, _ = check_agreement(token, speaker)
            if agrees:
                antecedent = speaker
                confidence = 0.90
                rule = "Quoted Pronoun -> Speaker"
    elif not antecedent and (is_possessive_pronoun or is_relative_possessive):
        potential_candidates = []
        search_rule_name = (
            "Possessive Antecedent" if is_possessive_pronoun else "Relative Possessive (whose) Antecedent"
        )
        for j in range(token.i - 1, start_search_token_idx - 1, -1):
            if j < 0:
                break
            candidate = doc[j]
            if candidate.pos_ not in {"NOUN", "PROPN", "PRON"} or is_reflexive(candidate):
                continue
            agrees, _ = check_agreement(token, candidate)
            if not agrees:
                continue
            cand_score = PROXIMITY_THRESHOLD # Start with base proximity score
            cand_rule_detail = ""
            if candidate.ent_type_ == "PERSON":
                cand_score = max(cand_score, 0.75)
            elif candidate.ent_type_:
                cand_score = max(cand_score, 0.65)
            if candidate.dep_ in ("nsubj", "nsubjpass"):
                subject_bonus = 0.15
                cand_score += subject_bonus
                cand_rule_detail += " (Subject)"
            if candidate.lemma_ in PERSONAL_PRONOUN_LEMMAS:
                cand_score *= 0.70
            distance = token.i - candidate.i
            proximity_factor = max(0.1, 1.0 - (distance / 50.0))
            cand_score *= proximity_factor
            cand_score = min(cand_score, 1.0)
            if cand_score > PROXIMITY_THRESHOLD:
                potential_candidates.append(
                    {
                        "token": candidate,
                        "score": cand_score,
                        "reason": f"{search_rule_name}{cand_rule_detail}",
                        "distance": distance,
                    },
                )
        if potential_candidates:
            potential_candidates.sort(key=lambda x: (-x["score"], x["distance"]))
            best_candidate_info = potential_candidates[0]
    elif not antecedent and is_personal_pronoun:
        potential_candidates = []
        for j in range(token.i - 1, start_search_token_idx - 1, -1):
            if j < 0:
                break
            candidate = doc[j]
            if candidate.pos_ not in {"NOUN", "PROPN", "PRON"} or is_reflexive(candidate):
                continue
            agrees, is_singular_they = check_agreement(token, candidate)
            if not agrees:
                continue
            cand_score = 0.15
            cand_rule_detail = "Agreement"
            if candidate.ent_type_:
                if candidate.ent_type_ == "PERSON" and (token.lemma_ in ["he", "she", "they"]):
                    cand_score = max(cand_score, 0.70)
                    cand_rule_detail = "NER PERSON"
                elif token.lemma_ == "it" and candidate.ent_type_ and candidate.ent_type_ != "PERSON":
                    cand_score = max(cand_score, 0.65)
                    cand_rule_detail = "NER Non-PERSON ('it')"
            if is_singular_they and cand_score < SINGULAR_THEY_THRESHOLD:
                cand_score = max(cand_score, 0.75) # Boost for singular they if not already high
                cand_rule_detail = "Singular They Match"
            if candidate.dep_ in ("nsubj", "nsubjpass"):
                subject_bonus = 0.15
                cand_score += subject_bonus
                cand_rule_detail += " (Subject)" if cand_rule_detail != "Agreement" else "Subject Salience"
            if candidate.lemma_ in PERSONAL_PRONOUN_LEMMAS:
                cand_score *= 0.70
            distance = token.i - candidate.i
            proximity_factor = max(0.1, 1.0 - (distance / 75.0))
            cand_score *= proximity_factor
            cand_score = min(cand_score, 1.0)
            if cand_rule_detail == "Agreement":
                cand_rule_detail += " + Proximity"
            if use_similarity_fallback and cand_score < similarity_threshold:
                try:
                    similarity = token.similarity(candidate)
                    if similarity >= similarity_threshold:
                        sim_score_contribution = 0.1 + (
                            (similarity - similarity_threshold) / (1.0 - similarity_threshold) * 0.3
                        )
                        if sim_score_contribution > (cand_score * 0.1):
                            cand_score += sim_score_contribution
                            cand_score = min(cand_score, 0.9)
                            cand_rule_detail = (
                                f"{cand_rule_detail.replace('Agreement + Proximity', '').strip()} "
                                f"+ Similarity ({similarity:.2f})"
                            )
                            cand_rule_detail = cand_rule_detail.strip().lstrip("+ ")
                except UserWarning:
                    pass
            if cand_score > PROXIMITY_THRESHOLD:
                potential_candidates.append(
                    {
                        "token": candidate,
                        "score": cand_score,
                        "reason": f"Std Pronoun: {cand_rule_detail.strip()}",
                        "distance": distance,
                    },
                )
        if potential_candidates:
            potential_candidates.sort(key=lambda x: (-x["score"], x["distance"]))
            best_candidate_info = potential_candidates[0]

    if best_candidate_info and not antecedent:
        antecedent = best_candidate_info["token"]
        confidence = best_candidate_info["score"]
        rule = best_candidate_info["reason"]

    return antecedent, confidence, rule


def _resolve_proper_noun(
    token: Token, doc: Doc, start_search_token_idx: int, processed_mentions: set,
) -> tuple[Optional[Token], float, str]:
    """Resolve a proper noun token to its antecedent if possible."""
    antecedent = None
    confidence = 0.0
    rule = "N/A"
    potential_pn_antecedents = []
    for j in range(token.i - 1, start_search_token_idx - 1, -1):
        if j < 0:
            break
        candidate = doc[j]
        if candidate.pos_ == "PROPN" and candidate.ent_type_ == "PERSON":
            if candidate.text == token.text:
                potential_pn_antecedents.append(
                    {
                        "token": candidate,
                        "score": 0.98,
                        "type": "Exact",
                        "rule": "PN Exact Match",
                        "distance": token.i - j,
                    },
                )
            candidate_is_longer = len(candidate.text.split()) > 1
            token_is_shorter = len(token.text.split()) == 1
            if candidate_is_longer and token_is_shorter and candidate.text.endswith(token.text):
                prev_token = doc[token.i - 1] if token.i > 0 else None
                is_part_of_prev_pn = prev_token and prev_token.pos_ == "PROPN" and prev_token.ent_iob_ != "O"
                if not is_part_of_prev_pn:
                    potential_pn_antecedents.append(
                        {
                            "token": candidate,
                            "score": 0.95,
                            "type": "Partial",
                            "rule": "PN Partial Match (Last Name)",
                            "distance": token.i - j,
                        },
                    )
    if potential_pn_antecedents:
        potential_pn_antecedents.sort(key=lambda x: (-x["score"], x["distance"]))
        best_pn_match_info = potential_pn_antecedents[0]
        if best_pn_match_info["type"] == "Exact":
            for cand_info in potential_pn_antecedents:
                if (
                    cand_info["type"] == "Partial"
                    and best_pn_match_info["token"].text in cand_info["token"].text.split()
                ):
                    best_pn_match_info = cand_info
                    break
        best_pn_token = best_pn_match_info["token"]
        if best_pn_token.i != token.i and best_pn_token.i not in processed_mentions:
            antecedent = best_pn_token
            confidence = best_pn_match_info["score"]
            rule = best_pn_match_info["rule"]
    return antecedent, confidence, rule


if __name__ == "__main__":
    print("This module is not intended to be run directly.")

    # --- Testing ---
    # [Include the same testing samples and loop as before]
    samples_expert = {
        "Pleonastic It": "It is raining heavily today. It seems that the game will be cancelled.",
        # who -> man
        "Relative Who": "The man who arrived late missed the announcement.",
        # which -> report
        "Relative Which": "The report, which detailed the findings, was released.",
        # whose -> artist
        "Relative Whose": "The artist whose painting won the prize was ecstatic.",
        # It -> cat (subject likely preferred over mouse)
        "Subject Salience": "The cat chased the mouse. It was fast.",
        "Possessive His": "John loves his dog.",  # his -> John
        "Possessive Its": "The company announced its profits.",  # its -> company
        "Quote Possessive": 'Mary said, "My car is blue."',  # My -> Mary
        # Smith -> John Smith (not Jane Smith)
        "PN Partial Refined": ("Prof. John Smith presented. Later, Smith answered questions. Jane Smith watched."),
        # they->team, which->spirit?, their->team
        "Complex Sentence": "Although the team lost, they showed great spirit, which pleased their coach.",
        "Weather/Time It": "It is snowing and it is almost noon.",
        # It -> Pleonastic, who -> Susan
        "Cleft It": "It was Susan who solved the puzzle.",
    }
    samples_advanced = {
        "Appositive": "The CEO of the company, John, gave a speech. He emphasized the importance of innovation.",
        "Possessive": "John's car is red. It is fast.",  # It -> car
        "Definite Description": "The president gave a speech. He emphasized unity.",
        "Simple Morphology": "Sarah went to the market. She bought fruits.",
        "Mixed Case": "My friend Lisa arrived. She said that the party was fun. Lisa loves dancing.",
        "Plural": "The developers released the software. They were proud of it.",  # They -> developers, it -> software
        "Reflexive": "The manager told himself to stay calm.",
        "It ambiguity": "We poured water into the cup until it was full.",  # it -> cup
        "Singular They": "My friend mentioned their new job. They seem happy.",  # They -> friend
        # she->Alice, his->Bob(possessive), he->Bob, it->car
        "Complex": "Alice told Bob that she liked his new car, but he thought it was too flashy.",
        "Quote Simple": 'Mary said, "I need coffee."',  # I -> Mary
        # we -> team? John+team?, They -> team
        "Quote Complex": ('John asked his team, "Can we finish this today?" They replied affirmatively.'),
        # He (inner) -> witness? suspect?, He (outer) -> witness?
        "Quote Nested": ("The report stated, \"The witness claimed, 'He saw the suspect.'\" He later recanted."),
        # He->Peter, She->Susan (test windowing)
        "Sentence Window": "Peter called Mike. He was happy. Later, Susan arrived. She brought cake.",
        # Reed -> Dr. Evelyn Reed
        "Proper Noun Repeat": "Dr. Evelyn Reed published her findings. Reed argued for a new approach.",
        # Smith -> John Smith
        "Proper Noun Partial": "Chairman John Smith entered. Smith looked tired.",
    }
    all_samples = {**samples_advanced, **samples_expert}

    print("--- Running Coreference Resolution v4 (Comments & Annotations) ---")
    for description, text in all_samples.items():
        print(f"\n--- [{description}] ---")
        print(f"Text: {text}")
        try:
            # Call the main resolution function
            pairs_with_indices = rule_based_coref_resolution_v4(text, search_sentences=2, use_similarity_fallback=False)
            print("Coreference pairs (Mention Span, Antecedent Span, Confidence, Rule):")
            if pairs_with_indices:
                # Print results in the desired format
                for pair in pairs_with_indices:
                    mention_span, antecedent_span, conf, rule = pair
                    print(
                        f"  - Mention: '{mention_span['text']}' ({mention_span['start']}:{mention_span['end']}) -> "
                        f"Antecedent: '{antecedent_span['text']}' ({antecedent_span['start']}:{antecedent_span['end']}) "
                        f"(Conf: {conf:.2f}, Rule: {rule})",
                    )
            else:
                print("  No pairs found.")

        except Exception as e:
            # Basic error handling for the loop
            print(f"\n!!! An error occurred processing '{description}': {e} !!!")
            import traceback

            traceback.print_exc()
