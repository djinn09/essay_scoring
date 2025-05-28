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

from typing import TYPE_CHECKING, Any  # Import necessary types for annotation

import gender_guesser.detector as gender
import spacy

if TYPE_CHECKING:
    from spacy.tokens import Token

# --- Constants ---
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


def get_gender(token: Token):
    """Determine the 'gender' feature for a spacy Token.

    Order of checks:
      1. Pronoun lookup
      2. SpaCy morphological 'Gender'
      3. Static name lookup
      4. gender-guesser fallback for PERSON entities
      5. Common neuter nouns
      6. Default to 'Unspecified'.
    """
    # Normalize lemma and text
    lemma = token.lemma_.lower()
    text = token.text.strip().lower()
    # 1. Pronoun-based gender
    if lemma in PRONOUN_GENDER:
        return [PRONOUN_GENDER[lemma]]

    # 2. SpaCy morphological gender
    gender_feats = token.morph.get("Gender", None)
    if gender_feats:
        return gender_feats

    # 3. Named entity static lookup
    if token.ent_type_ == "PERSON":
        # Static name hints
        if text in NAME_GENDER:
            return [NAME_GENDER[text]]
        # 4. gender-guesser fallback
        guess = DETECTOR.get_gender(text)
        if guess in ("male", "mostly_male"):
            return ["Masc"]
        if guess in ("female", "mostly_female"):
            return ["Fem"]
        # PERSON but unknown -> unspecified
        return ["Unspecified"]

    # 5. Common neuter nouns
    if token.pos_ == "NOUN" and lemma in NEUTER_NOUNS:
        return ["Neut"]

    # 6. Default
    return ["Unspecified"]


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
    pron_gender_set = set(get_gender(pronoun))
    cand_gender_set = set(get_gender(candidate))

    # Case 1: Pronoun is Neuter ('it')
    if "Neut" in pron_gender_set:
        # 'it' should only match explicit Neut. Disallow matching Unspecified, Masc, Fem.
        if "Neut" not in cand_gender_set:
            # print(f"Debug Agreemnt [Gender Fail]: 'it' vs non-Neut {candidate} ({cand_gender_set})")
            return False, False
        # Pass if candidate is Neut
        return True, False  # Agreement OK, not singular they

    # Case 2: Pronoun is Gendered ('he', 'she') or Plural ('they')
    # Check for explicit clashes (Masc vs Fem) only if candidate is NOT Unspecified
    if "Unspecified" not in cand_gender_set and (
        ("Masc" in pron_gender_set and "Fem" in cand_gender_set)
        or ("Fem" in pron_gender_set and "Masc" in cand_gender_set)
    ):
        return False, False

    # If no explicit clash, allow match.
    # - 'he'/'she' can match Masc/Fem respectively, or Unspecified.
    # - 'they' can match Masc, Fem, Neut, or Unspecified (plural or singular).
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
        head = head.head # Move to the head of the current token.

    # If a verb or the root is found, look for its subject(s).
    if head.pos_ in ("VERB", "AUX") or head.dep_ == "ROOT":
        # Prioritize nominal subjects (nsubj, nsubjpass).
        subjects = [c for c in head.children if c.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            # If no nominal subject, look for clausal subjects (csubj, csubjpass).
            subjects = [c for c in head.children if c.dep_ in ("csubj", "csubjpass")]

        if subjects:
            core_subj = subjects[0] # Take the first subject found.
            # Refinement: If the subject is part of a compound noun, try to get the true head of the compound.
            # This handles cases like "The big black cat" -> "cat" not "The".
            # Ensure the head of the compound still precedes the original token to avoid jumping too far.
            while core_subj.dep_ == "compound" and core_subj.head.i < token.i:
                core_subj = core_subj.head
            # Further refinement: If the core subject isn't a PERSON or PROPN,
            # check its subtree for a PERSON entity that precedes the core_subj itself.
            # This aims to find a person if the direct subject is more general (e.g., "His team" -> "His").
            if core_subj.ent_type_ != "PERSON" and core_subj.pos_ != "PROPN":
                # Search within the subject's subtree for a PERSON entity appearing before the subject token.
                person_in_subj = [
                    t for t in core_subj.subtree
                    if t.ent_type_ == "PERSON" and t.i < core_subj.i
                ]
                if person_in_subj:
                    return person_in_subj[-1] # Return the last such PERSON found (closest).
            return core_subj
    return None # No subject found.


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
        if governing_verb.lemma_ in REPORTING_VERBS and \
           (current.dep_ in ("ccomp", "advcl", "xcomp") or \
            (current.dep_ == "dobj" and current.pos_ == "PRON")): # "dobj" more likely if pronoun is object of reporting verb
            speaker = find_subject(governing_verb) # The subject of the reporting verb is the speaker.
            return speaker if speaker else None

        # Scenario 2: Pronoun is the subject of a clause that itself is governed by a reporting verb.
        # e.g., The report stated that "He was seen..." -> "He" is nsubj of "was seen", "was seen" is ccomp of "stated".
        if current.dep_ == "nsubj" and \
           governing_verb.dep_ == "ccomp" and \
           governing_verb.head.lemma_ in REPORTING_VERBS:
            reporting_verb = governing_verb.head # This is the actual reporting verb.
            speaker = find_subject(reporting_verb) # The subject of this reporting verb.
            return speaker if speaker else None

        current = current.head # Move up the tree.

    return None # No speaker found through these patterns.


def is_pleonastic_it(token: Token) -> bool:
    """Return True if the token 'it' is used pleonastically (e.g., weather, time, cleft)."""
    if token.lemma_ != "it": # Must be the pronoun 'it'.
        return False

    # Rule 1: Expletive 'it' (e.g., "It is raining.")
    # SpaCy often tags expletive 'it' with dep_ == "expl".
    if token.dep_ == "expl":
        return True

    # Further checks if 'it' is a nominal subject (nsubj).
    if token.dep_ != "nsubj":
        return False # If 'it' is not a subject here, less likely to be pleonastic by these rules.

    verb = token.head # The verb governed by 'it'.

    if verb.pos_ == "VERB":
        # Rule 2: Weather verbs (e.g., "It rains.")
        is_weather_verb = verb.lemma_ in {"rain", "snow", "hail", "thunder", "lighten"}
        if is_weather_verb:
            return True

        # Rule 3: 'It is' + time/weather attribute (e.g., "It is sunny.", "It is 3 o'clock.")
        if verb.lemma_ == "be": # If the verb is 'to be'.
            attr = next((c for c in verb.children if c.dep_ == "attr"), None) # Find attribute complement.
            if attr:
                subtree = list(attr.subtree)
                # Check for TIME entity or common time/weather keywords in the attribute.
                has_time_ent = attr.ent_type_ == "TIME" or any(t.ent_type_ == "TIME" for t in subtree)
                has_time_keyword = any(
                    t.text.lower() in {"o'clock", "pm", "am", "noon", "midnight", "raining", "snowing", "sunny", "cloudy"} for t in subtree
                )
                if has_time_ent or has_time_keyword:
                    return True

        # Rule 4: Verbs of seeming/appearing with clausal complements
        # (e.g., "It seems that...", "It appears to be...")
        if verb.lemma_ in {"seem", "appear", "happen", "matter", "turn out", "look", "sound"} and \
           any(c.dep_ in {"ccomp", "csubj", "xcomp", "acomp", "advcl"} for c in verb.children):
            return True

    # Rule 5: Auxiliary 'be' + time/numeric attribute or cleft constructions
    elif verb.lemma_ == "be" and verb.pos_ == "AUX": # If 'it' is subject of an auxiliary 'be'.
        attr = next((c for c in verb.children if c.dep_ == "attr"), None)
        if attr:
            subtree = list(attr.subtree)
            # Check for time-related attributes.
            is_time_attr = (
                attr.ent_type_ == "TIME" or
                any(t.ent_type_ == "TIME" for t in subtree) or
                any(t.text.lower() in {"o'clock", "pm", "am", "noon", "midnight"} for t in subtree)
            )
            # Check for "almost" + number (e.g., "It is almost 5.")
            has_almost = any(c.lemma_ == "almost" and c.dep_ == "advmod" for c in attr.children)
            is_num_like = attr.like_num or (attr.text and attr.text[0].isdigit())

            if is_time_attr or (has_almost and is_num_like):
                return True

            # Rule 6: Cleft sentences (e.g., "It was John who left.")
            # Check if 'attr' is the head of a relative clause starting with WP/WDT ('who', 'which', 'that').
            # A more robust cleft check might involve looking at the structure more deeply.
            relcl = next((c for c in verb.children if c.dep_ == "relcl" and c.head == attr), None)
            if relcl and relcl.sent == verb.sent and relcl[0].tag_ in ["WP", "WDT"]: # Ensure relcl starts with relative pronoun.
                return True  # Likely a cleft construction.

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
            antecedent = None  # The resolved antecedent Token object
            confidence = 0.0  # Confidence score of the resolution
            rule = "N/A"  # Name of the rule/heuristic that triggered the match
            best_candidate_info = None  # Stores best candidate from backward searches

            # --- A. Pronoun Resolution ---
            # Identify pronoun type based on fine-grained POS tag
            is_personal_pronoun = token.tag_ == "PRP"
            is_possessive_pronoun = token.tag_ == "PRP$"
            is_relative_possessive = token.tag_ == "WP$"  # 'whose'
            is_relative_nonpossessive = token.tag_ in ["WP", "WDT"]  # 'who', 'which', 'that'
            is_reflexive_pronoun = is_reflexive(token)  # Check lemma ends with 'self'

            # Only proceed if the token is some kind of pronoun
            if is_personal_pronoun or is_possessive_pronoun or is_relative_possessive or is_relative_nonpossessive:
                # Rule 0: Skip Pleonastic 'It' - Check before trying to resolve 'it'
                if token.lemma_ == "it" and is_pleonastic_it(token):
                    processed_mentions.add(token.i)  # Mark as processed (but not resolved)
                    continue  # Move to the next token

                # --- High-Confidence Rules (Applied First) ---

                # Rule 1: Reflexive Pronoun Resolution
                if is_reflexive_pronoun:
                    subj = find_subject(token)  # Find subject of the reflexive's clause
                    if subj:
                        antecedent = subj
                        confidence = 0.95
                        rule = "Reflexive Pronoun -> Subject"

                # Rule 2a: Relative Non-Possessive Pronoun Resolution ('who', 'which', 'that')
                # Link to the syntactic head noun phrase
                elif is_relative_nonpossessive:
                    potential_antecedent = token.head  # Initial head from parser
                    # Adjust head if it's a preposition, auxiliary, or intermediate verb
                    if potential_antecedent.pos_ in ("ADP", "AUX", "VERB"):
                        potential_antecedent = potential_antecedent.head
                    # Check if adjusted head is a valid antecedent type
                    if potential_antecedent.pos_ in {"NOUN", "PROPN", "PRON"}:
                        # Check agreement (handles 'who' vs PERSON, 'which' vs non-PERSON)
                        agrees, _ = check_agreement(token, potential_antecedent)
                        if agrees:
                            antecedent = potential_antecedent
                            confidence = 0.92
                            rule = "Relative Pronoun -> Syntactic Head"

                # Rule 3: Quoted Speech Pronoun Resolution ('I', 'me', 'my', 'we', 'us', 'our')
                # Check before general possessive/personal rules for these lemmas
                if not antecedent and token.lemma_ in {"i", "me", "my", "we", "us", "our"}:
                    speaker = find_speaker(token)  # Attempt to find the speaker
                    if speaker:
                        # Check agreement between pronoun and speaker
                        agrees, _ = check_agreement(token, speaker)
                        if agrees:
                            antecedent = speaker
                            confidence = 0.90
                            rule = "Quoted Pronoun -> Speaker"

                # --- Lower-Confidence Rules (Backward Search) ---
                # Run only if no high-confidence rule found an antecedent yet

                # Rule 2b / 4: Possessive Pronouns ('his', 'her', 'its', 'their') + Relative 'whose'
                # Search backwards for the *possessor* entity.
                elif not antecedent and (is_possessive_pronoun or is_relative_possessive):
                    # list to hold potential candidates found during backward search
                    potential_candidates = []
                    # Determine rule name based on pronoun type
                    search_rule_name = (
                        "Possessive Antecedent" if is_possessive_pronoun else "Relative Possessive (whose) Antecedent"
                    )

                    # Iterate backwards from the token before the mention within the search window
                    for j in range(token.i - 1, start_search_token_idx - 1, -1):
                        if j < 0:
                            break  # Safety break
                        candidate = doc[j]  # The potential antecedent token

                        # Basic filtering: Candidate must be Noun, Pronoun, or Proper Noun, and not reflexive
                        if candidate.pos_ not in {"NOUN", "PROPN", "PRON"} or is_reflexive(candidate):
                            continue

                        # Check grammatical agreement (number, gender) for the possessor
                        agrees, _ = check_agreement(token, candidate)
                        if not agrees:
                            continue

                        # --- Candidate Scoring ---
                        cand_score = 0.05  # Base score for passing agreement
                        cand_rule_detail = ""  # Details to append to rule name

                        # Named Entity Bonus (PERSON preferred)
                        if candidate.ent_type_ == "PERSON":
                            cand_score = max(cand_score, 0.75)
                        elif candidate.ent_type_:
                            cand_score = max(cand_score, 0.65)

                        # Subject Salience Bonus
                        if candidate.dep_ in ("nsubj", "nsubjpass"):
                            subject_bonus = 0.15
                            cand_score += subject_bonus
                            cand_rule_detail += " (Subject)"

                        # Pronoun Candidate Penalty (discourage linking pronouns to other pronouns)
                        if candidate.lemma_ in PERSONAL_PRONOUN_LEMMAS:
                            cand_score *= 0.70  # Reduce score

                        # Proximity Decay (closer candidates get higher scores)
                        distance = token.i - candidate.i
                        proximity_factor = max(0.1, 1.0 - (distance / 50.0))  # Decay faster
                        cand_score *= proximity_factor
                        cand_score = min(cand_score, 1.0)  # Cap score

                        # Add candidate to list if score is above minimum threshold
                        if cand_score > 0.05:  # noqa: PLR2004
                            potential_candidates.append(
                                {
                                    "token": candidate,
                                    "score": cand_score,
                                    "reason": f"{search_rule_name}{cand_rule_detail}",
                                    "distance": distance,
                                },
                            )
                    # After searching, select the best candidate (highest score, then closest)
                    if potential_candidates:
                        potential_candidates.sort(key=lambda x: (-x["score"], x["distance"]))
                        best_candidate_info = potential_candidates[0]  # Store best candidate dict

                # Rule 5: Standard Personal Pronouns ('he', 'she', 'it', 'they')
                # General backward search applying multiple heuristics.
                elif not antecedent and is_personal_pronoun:
                    potential_candidates = []
                    # Iterate backwards
                    for j in range(token.i - 1, start_search_token_idx - 1, -1):
                        if j < 0:
                            break
                        candidate = doc[j]

                        # Basic filtering
                        if candidate.pos_ not in {"NOUN", "PROPN", "PRON"} or is_reflexive(candidate):
                            continue

                        # Check agreement (number, gender, singular they)
                        agrees, is_singular_they = check_agreement(token, candidate)
                        if not agrees:
                            continue

                        # --- Candidate Scoring ---
                        cand_score = 0.15  # Base score for agreement (higher for std pronouns)
                        cand_rule_detail = "Agreement"  # Base rule name

                        # Named Entity Bonus
                        if candidate.ent_type_:
                            if candidate.ent_type_ == "PERSON" and (token.lemma_ in ["he", "she", "they"]):
                                cand_score = max(cand_score, 0.70)
                                cand_rule_detail = "NER PERSON"
                            elif token.lemma_ == "it" and candidate.ent_type_ and candidate.ent_type_ != "PERSON":
                                cand_score = max(cand_score, 0.65)
                                cand_rule_detail = "NER Non-PERSON ('it')"

                        # Singular They Bonus (if agreement check flagged it)
                        if is_singular_they and cand_score < 0.70:  # Apply if NER didn't already give high score
                            cand_score = max(cand_score, 0.75)  # Strong boost
                            cand_rule_detail = "Singular They Match"

                        # Subject Salience Bonus
                        if candidate.dep_ in ("nsubj", "nsubjpass"):
                            subject_bonus = 0.15
                            cand_score += subject_bonus
                            # Adjust rule detail name
                            cand_rule_detail += " (Subject)" if cand_rule_detail != "Agreement" else "Subject Salience"

                        # Pronoun Candidate Penalty
                        if candidate.lemma_ in PERSONAL_PRONOUN_LEMMAS:
                            cand_score *= 0.70

                        # Proximity Decay
                        distance = token.i - candidate.i
                        proximity_factor = max(0.1, 1.0 - (distance / 75.0))
                        cand_score *= proximity_factor
                        cand_score = min(cand_score, 1.0)

                        # Adjust rule name if only agreement + proximity contributed significantly
                        if cand_rule_detail == "Agreement":
                            cand_rule_detail += " + Proximity"

                        # Semantic Similarity Fallback (Optional)
                        # Semantic Similarity Fallback (Optional & Experimental)
                        # If enabled and other heuristics provide a low score,
                        # check word vector similarity as a potential tie-breaker or confidence booster.
                        if use_similarity_fallback and cand_score < similarity_threshold:
                            try:
                                # Calculate similarity using pre-trained word vectors from the spaCy model.
                                similarity = token.similarity(candidate)
                                if similarity >= similarity_threshold:
                                    # Scale similarity to a score contribution.
                                    # This scaling is arbitrary and can be tuned.
                                    sim_score_contribution = (
                                        0.1 + (similarity - similarity_threshold) / (1.0 - similarity_threshold) * 0.3 # Max 0.4 if similarity is 1.0
                                    )
                                    # Only apply similarity if it meaningfully improves the score
                                    # and doesn't override a strong syntactic/NER match.
                                    if sim_score_contribution > (cand_score * 0.1): # Heuristic: adds at least 10%
                                        cand_score += sim_score_contribution # Add to existing score
                                        cand_score = min(cand_score, 0.9) # Cap to avoid over-reliance
                                        cand_rule_detail = f"{cand_rule_detail.replace('Agreement + Proximity','').strip()} + Similarity ({similarity:.2f})"
                                        cand_rule_detail = cand_rule_detail.strip().lstrip("+ ")
                            except UserWarning: # spaCy raises UserWarning if vectors are not available or words are OOV.
                                pass  # Ignore warnings if vectors are missing for some tokens.

                        # Add candidate to the list if its score is above a minimum threshold.
                        if cand_score > 0.05:  # noqa: PLR2004 - magic number, threshold for consideration
                            potential_candidates.append(
                                {
                                    "token": candidate,
                                    "score": cand_score,
                                    "reason": f"Std Pronoun: {cand_rule_detail.strip()}", # Construct rule name
                                    "distance": distance,
                                },
                            )
                    # After checking all candidates in the window, select the best one.
                    if potential_candidates:
                        # Sort by score (descending) then distance (ascending).
                        potential_candidates.sort(key=lambda x: (-x["score"], x["distance"]))
                        best_candidate_info = potential_candidates[0] # The best candidate.

                # --- Update main antecedent variables if a backward search found something ---
                # This applies if `best_candidate_info` was populated by either Rule 2b/4 or Rule 5,
                # and no high-priority rule (Rule 1, 2a, 3) had already found an `antecedent`.
                if best_candidate_info and not antecedent:
                    antecedent = best_candidate_info["token"]
                    confidence = best_candidate_info["score"]
                    rule = best_candidate_info["reason"]

            # --- B. Proper Noun (PN) Coreference Logic ---
            # This section handles cases where the current `token` is a proper noun.
            # It attempts to link it to a preceding proper noun.
            elif token.pos_ == "PROPN": # Check if the current token is a proper noun
                potential_pn_antecedents = [] # List to store potential PN antecedent candidates
                # Search backwards from the token before the current PN.
                for j in range(token.i - 1, start_search_token_idx - 1, -1):
                    if j < 0: break # Boundary condition
                    candidate = doc[j] # Potential antecedent token

                    # Candidate must be a Proper Noun and a PERSON entity for this simplified rule.
                    # This can be expanded to other entity types (ORG, GPE) with appropriate logic.
                    if candidate.pos_ == "PROPN" and candidate.ent_type_ == "PERSON":
                        # Rule PN-1: Exact Match
                        # e.g., "Smith" ... "Smith"
                        if candidate.text == token.text:
                            potential_pn_antecedents.append({
                                "token": candidate, "score": 0.98, "type": "Exact",
                                "rule": "PN Exact Match", "distance": token.i - j,
                            })
                        # Rule PN-2: Partial Match (e.g., Full Name -> Last Name)
                        # e.g., "John Smith" ... "Smith"
                        candidate_is_longer = len(candidate.text.split()) > 1
                        token_is_shorter = len(token.text.split()) == 1
                        # Check if the candidate's text ends with the current token's text (e.g., "John Smith" ends with "Smith").
                        if candidate_is_longer and token_is_shorter and candidate.text.endswith(token.text):
                            # Heuristic: check if `token` (e.g. "Smith") is likely standalone,
                            # not part of another multi-word proper noun immediately preceding it.
                            # E.g., avoid matching "Smith" in "James Smith" to "Smith" in "Mr. Smith".
                            prev_token = doc[token.i - 1] if token.i > 0 else None
                            is_part_of_prev_pn = (prev_token and prev_token.pos_ == "PROPN" and prev_token.ent_iob_ != "O")
                            if not is_part_of_prev_pn: # If token is likely standalone or starts a new PN.
                                potential_pn_antecedents.append({
                                    "token": candidate, "score": 0.95, "type": "Partial", # Slightly lower score than exact
                                    "rule": "PN Partial Match (Last Name)", "distance": token.i - j,
                                })
                # Select the best PN match if any candidates were found.
                if potential_pn_antecedents:
                    # Sort by score (descending), then distance (ascending).
                    potential_pn_antecedents.sort(key=lambda x: (-x["score"], x["distance"]))
                    best_pn_match_info = potential_pn_antecedents[0] # Initial best match.

                    # PN Selection Override Logic:
                    # Prefer a partial match (full name as antecedent) if an exact match (e.g., last name)
                    # is part of that full name. E.g., if "Smith" (current token) matches "Smith" (exact antecedent)
                    # but also "John Smith" (partial antecedent), prefer "John Smith".
                    if best_pn_match_info["type"] == "Exact":
                        for cand_info in potential_pn_antecedents:
                            if cand_info["type"] == "Partial" and \
                               best_pn_match_info["token"].text in cand_info["token"].text.split():
                                # If a partial match's antecedent token (e.g. "John Smith") contains
                                # the exact match's antecedent token text (e.g. "Smith").
                                best_pn_match_info = cand_info # Override with the more complete partial match.
                                break # Found preferred partial, no need to check further.

                    # Final check and assignment for PN coreference.
                    best_pn_token = best_pn_match_info["token"]
                    # Avoid self-reference and linking to mentions that have already been resolved as mentions themselves.
                    if best_pn_token.i != token.i and best_pn_token.i not in processed_mentions:
                        antecedent = best_pn_token
                        confidence = best_pn_match_info["score"]
                        rule = best_pn_match_info["rule"]

            # --- Store Result (if an antecedent was found) ---
            # This section is reached after all rules for the current `token` have been evaluated.
            if antecedent and antecedent.i != token.i:
                coref_results_internal.append((token, antecedent, round(confidence, 2), rule))
                # Mark this token as processed so it isn't considered as an antecedent later
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
        antecedent_span = {"text": ant_tok.text, "start": ant_tok.idx, "end": ant_tok.idx + len(ant_tok.text)}
        # Append the formatted tuple to the final list
        coref_pairs_with_indices.append((mention_span, antecedent_span, conf, rule_name))

    return coref_pairs_with_indices


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
        "PN Partial Refined": "Professor John Smith presented. Later, Smith answered questions. Jane Smith watched.",
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
        "Quote Complex": 'John asked his team, "Can we finish this today?" They replied affirmatively.',
        # He (inner) -> witness? suspect?, He (outer) -> witness?
        "Quote Nested": "The report stated, \"The witness claimed, 'He saw the suspect.'\" He later recanted.",
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
                        f"Antecedent: '{antecedent_span['text']}' ({antecedent_span['start']}:{antecedent_span['end']})"
                        f"(Conf: {conf:.2f}, Rule: {rule})",
                    )
            else:
                print("  No pairs found.")

        except Exception as e:
            # Basic error handling for the loop
            print(f"\n!!! An error occurred processing '{description}': {e} !!!")
            import traceback

            traceback.print_exc()
