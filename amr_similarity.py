"""Module for calculating text similarity scores using Abstract Meaning Representation (AMR).

AMR is a semantic representation language that encodes the meaning of a sentence
as a rooted, directed, acyclic graph. This module provides the `AMRSimilarityCalculator`
class to compute various similarity metrics based on these AMR graphs.

Key Features:
- Parses text into AMR graphs using `amrlib`.
- Calculates Smatch (F-score, Precision, Recall) between AMR graphs.
- Computes similarity based on concept overlap (Jaccard Index).
- Computes similarity based on named entity overlap (Jaccard Index).
- Detects and compares negations (currently a placeholder, Jaccard Index on presence).
- Compares root concepts of the AMR graphs.

Dependencies:
- `amrlib`: For AMR parsing. Requires models to be downloaded separately.
  See: https://github.com/bjascob/amrlib
  Installation: `pip install amrlib`
  Model Download: `python -m amrlib.models.download_models`
- `penman`: For working with PENMAN-formatted AMR graphs.
- `rich` (optional): For enhanced logging output.

**Note on AMR Models:** The quality of AMR-based similarity heavily depends on the
accuracy of the AMR parser and the models used. This module expects a pre-loaded
StoG (Stack-Transformer) model from `amrlib`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import amrlib
import penman  # Required for graph manipulation
from amrlib.evaluate.smatch_enhanced import compute_smatch  # Enhanced for more detailed Smatch scores

# --- Rich Logging (Optional) ---
try:
    from rich.logging import RichHandler

    _rich_available = True
except ImportError:
    _rich_available = False
    # Let's make rich optional for the core library, but required for the example's nice output
    # raise ImportError("Missing rich...") # Rich is optional for the library's core functionality


logger = logging.getLogger(__name__)

# Attempt to load the model once when the module is imported.
# This makes it available globally within this module.
# Users of the AMRSimilarityCalculator class will pass this pre-loaded model.
# TODO: Consider making the model path configurable via environment variable or a config file
# for more flexibility, instead of hardcoding.
# For now, this path is a placeholder and would need to be valid in the execution environment.
try:
    # IMPORTANT: The model_dir path below is currently hardcoded.
    # Users should modify this path to point to their downloaded amrlib model directory.
    # For more robust applications, this path should be made configurable,
    # for example, via an environment variable, a configuration file, or by
    # modifying AMRSimilarityCalculator to accept a model path if it were to handle loading.
    STOG_MODEL = amrlib.load_stog_model(
        model_dir="/mnt/e/Machine_learning/NLP-Revise/essay_grading/model_parse_xfm_bart_base-v0_1_0", # Example path
        device="cpu", # Default to CPU, can be changed if GPU is available
    )
    logger.info("Default AMR StoG model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load default AMR StoG model: {e}. AMRSimilarityCalculator may not work unless a model is provided at instantiation.")
    STOG_MODEL = None


# Placeholder concept used when a negation is detected but not associated with a specific concept.
# A more advanced implementation would identify the actual negated concept.
NEGATION_PLACEHOLDER = "HAS_NEGATION"


class AMRSimilarityCalculator:
    """Calculates similarity features based on Abstract Meaning Representation (AMR).

    AMR provides a way to represent the meaning of a sentence as a graph.
    This class uses `amrlib` for parsing text to AMR and `penman` for graph manipulation.

    Requires a pre-loaded `amrlib` parsing model (StoG - Stack-Transformer).
    The model should be passed during instantiation.

    Calculated Features:
    - Smatch (F-score, Precision, Recall): Measures structural similarity between AMR graphs.
    - Concept Overlap (Jaccard Index): Similarity based on common concepts (nodes and literals).
    - Named Entity Overlap (Jaccard Index): Similarity based on common named entities (identified by :name or :wiki links).
    - Negation Overlap (Jaccard Index): Checks for presence of negations (currently a simplified check).
    - Root Concept Similarity: Checks if the main concepts (roots) of the AMR graphs are the same.

    Future features could include similarity based on frames, semantic roles, reentrancies, etc.
    """

    def __init__(self, stog_model: Any) -> None: # TODO: Replace Any with specific amrlib model type
        """Initialize AMRSimilarityCalculator with a pre-loaded `amrlib` StoG model.

        Args:
            stog_model: A pre-loaded `amrlib` Stack-Transformer (StoG) parsing model object.
                        This model is used to parse text into AMR graphs.

        Raises:
            ValueError: If `stog_model` is not provided (is None).

        """
        if not stog_model:
            msg = "A pre-loaded AMR StoG model (stog_model) is required."
            raise ValueError(msg)
        self.stog_model = stog_model
        logger.info("AMRSimilarityCalculator initialized with provided StoG model.")

    # _load_models method was removed as the model is now expected to be pre-loaded and passed to __init__.
    # This simplifies the class and makes model management more explicit for the user.

    def _parse_amr(self, text: str) -> Optional[str]:
        """Parses input text into an AMR graph string in PENMAN format.

        This method handles basic sentence splitting (splitting by periods).
        For more complex texts, a dedicated sentence segmenter might be preferable
        before passing text to this method. Currently, if multiple sentences are
        produced by the split, only the AMR graph for the first sentence is returned.

        Args:
            text (str): The input text (sentence or short paragraph) to parse.

        Returns:
            Optional[str]: The AMR graph in PENMAN string format for the first successfully
                           parsed sentence, or None if parsing fails, the text is empty,
                           or the model is not available.

        """
        if not self.stog_model:
            logger.error("StoG model not available for AMR parsing.")
            return None
        if not text.strip():
            logger.warning("Input text for AMR parsing is empty or whitespace.")
            return None

        try:
            # Naive sentence splitting. amrlib's parse_sents expects a list of sentences.
            # For robust parsing, use a proper sentence tokenizer (e.g., from NLTK or spaCy)
            # before calling this method, or integrate it here.
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            if not sentences:
                logger.warning("No sentences found after splitting input text.")
                return None

            # Parse sentences into AMR graphs (PENMAN format).
            # `parse_sents` returns a list of PENMAN strings, one for each sentence.
            graphs_penman = self.stog_model.parse_sents(sentences)

            # For simplicity, this implementation currently uses only the AMR graph
            # of the first sentence. A more comprehensive approach might involve merging
            # graphs or processing them individually, depending on the use case.
            if graphs_penman and graphs_penman[0]:
                return graphs_penman[0]
            logger.warning("AMR parsing did not yield a graph for the first sentence.")
            return None
        except Exception as e: # Catching a broad exception as amrlib can raise various errors.
            logger.exception(f"AMR parsing failed for text '{text[:50]}...'. Error: {e}")
            return None

    def _get_graph_concepts(self, penman_graph_str: str) -> set[str]:
        """Extracts concepts (instance labels and constants) from an AMR graph string.

        Concepts include:
        1.  Instance labels (e.g., 'dog' from `(d / dog)`).
        2.  Constants that are targets of edges (e.g., "blue" from `:color "blue"`).
        3.  Constants that are targets of attributes (e.g., 5 from `:value 5`).

        Args:
            penman_graph_str (str): The AMR graph in PENMAN string format.

        Returns:
            set[str]: A set of unique concept strings extracted from the graph.
                      Returns an empty set if the input string is empty or parsing fails.

        Raises:
            penman.DecodeError: If the penman library fails to decode the graph string.

        """
        concepts = set()
        if not penman_graph_str:
            return concepts
        try:
            graph = penman.decode(penman_graph_str)
            # `graph.variables()` returns all variables like 'd', 'g', 't' in `(d / dog)`
            variables = graph.variables()

            # 1. Extract concepts from instance declarations (e.g., 'dog' in `(d / dog)`).
            # An instance is a triple (variable, concept, role), e.g., ('d', 'dog', None).
            for _var, concept_label, _role in graph.instances():
                if concept_label:  # Ensure the concept label exists.
                    concepts.add(str(concept_label))

            # 2. Extract constants that appear as targets in graph edges.
            # An edge is a triple (source_variable, role, target_variable_or_constant).
            # Example: `(c / city :name (n / name :op1 "London"))` -> "London" is a constant.
            # Example: `(d / dog :color "blue")` -> "blue" is a constant target in an attribute-like edge.
            # Penman's `edges()` and `attributes()` can sometimes overlap in what they represent based on AMR structure.
            # We iterate through both to be comprehensive.
            for _source_var, _role, target in graph.edges():
                # If the target is not a variable within this graph, it's considered a constant.
                if target not in variables:
                    # Constants are often string literals (e.g., "\"blue\"") or numbers.
                    # `penman` preserves quotes for string literals; strip them for the concept set.
                    constant_val = str(target).strip('"')
                    concepts.add(constant_val)

            # 3. Extract constants that appear as targets in graph attributes.
            # An attribute is similar to an edge but typically points to a literal.
            # Example: `(q / quantity :value 5)` -> 5 is a constant.
            for _source_var, _role, target in graph.attributes():
                # If the target is not a variable, it's a constant.
                if target not in variables:
                    constant_val = str(target).strip('"')
                    concepts.add(constant_val)

        except penman.DecodeError as e:
            logger.exception(f"Penman library failed to decode graph for concept extraction:\n{penman_graph_str}\nError: {e}")
            # Re-raise or handle as per desired error strategy; here, we log and return current concepts.
        except Exception as e:  # Catch other potential errors during processing.
            logger.exception(f"Error processing graph concepts with penman library: {e}\nGraph:\n{penman_graph_str}")
        return concepts

    def _get_named_entities(self, penman_graph_str: str) -> set[str]:
        """Extracts named entities from an AMR graph string.

        Named entities are typically identified by `:wiki` links (pointing to Wikipedia
        titles, often represented as string constants like `"-"` if not linked) or
        by `:name` relations which might point to a `name` instance with operations
        (e.g., `(n / name :op1 "John" :op2 "Doe")`).

        This implementation primarily extracts string literals connected via `:wiki`
        and treats them as named entities. It also extracts string literals from `:name`
        edges if they are direct constants. A more sophisticated NE extraction might
        involve resolving `name` subgraphs.

        Args:
            penman_graph_str (str): The AMR graph in PENMAN string format.

        Returns:
            set[str]: A set of unique named entity strings. Returns an empty set if the
                      input is empty or parsing/processing fails.

        Raises:
            penman.DecodeError: If the penman library fails to decode the graph string.

        """
        nes = set()
        if not penman_graph_str:
            return nes
        try:
            graph = penman.decode(penman_graph_str)
            variables = graph.variables()

            # Look for :wiki or :name relations in attributes.
            # Attributes usually link a variable to a literal (constant).
            for _source_var, role, target in graph.attributes():
                # `:wiki -` means no specific Wikipedia page is linked.
                if role == ":wiki" and target != "-":
                    # Target of :wiki should be a constant (the entity name or link).
                    nes.add(str(target).strip('"'))
                # Handling :name when it directly points to a constant string.
                # Example: (c / country :name "Wonderland")
                elif role == ":name" and target not in variables:
                     nes.add(str(target).strip('"'))


            # Look for :wiki or :name relations in edges.
            # Edges can link variables or a variable to a constant.
            # This is important if a :wiki or :name relation points to a string constant directly.
            for _source_var, role, target in graph.edges():
                if role == ":wiki" and target != "-" and target not in variables:
                    # Ensure target is a constant (not another variable in the graph).
                    nes.add(str(target).strip('"'))
                # If :name points to a constant target in an edge.
                # Example: (p / person :name "Alice")
                elif role == ":name" and target not in variables:
                    nes.add(str(target).strip('"'))
                # More complex :name handling (e.g., `(p / person :name (n / name :op1 "Alice"))`)
                # would require traversing to the `name` instance `n` and collecting its :op parts.
                # This is currently not implemented for simplicity.

        except penman.DecodeError as e:
            logger.exception(f"Penman library failed to decode graph for NE extraction:\n{penman_graph_str}\nError: {e}")
        except Exception as e:
            logger.exception(f"Error processing named entities with penman library: {e}\nGraph:\n{penman_graph_str}")
        return nes

    def _get_negations(self, penman_graph_str: str) -> set[str]:
        """Detects negations in an AMR graph string.

        This method currently uses a simplified approach, checking for the literal
        string ':polarity -' in the PENMAN graph, and returns a placeholder concept if found.
        It does not perform full graph parsing to accurately identify negated concepts.

        A more robust implementation would involve:
        1.  Properly parsing the graph using the `penman` library.
        2.  Traversing the graph to find instances of `:polarity -`.
        3.  Identifying the specific concept or variable that is being negated.

        Args:
            penman_graph_str (str): The AMR graph in PENMAN string format.

        Returns:
            set[str]: A set containing `NEGATION_PLACEHOLDER` if `:polarity -` is found,
                      otherwise an empty set.

        """
        negated_concepts = set()
        if not penman_graph_str:
            return negated_concepts

        try:
            # Current simplified check: looks for the literal string ":polarity -" in the PENMAN graph.
            # This is a basic proxy for negation detection and is less robust than full graph parsing.
            # A more accurate method would use the penman library to parse the graph and
            # then identify nodes/edges explicitly indicating negation (e.g., as shown in the commented-out example below).
            if ":polarity -" in penman_graph_str:
                negated_concepts.add(NEGATION_PLACEHOLDER)
                # Example of how to use penman to find negated concepts (more robust):
                # graph = penman.decode(penman_graph_str)
                # for var, role, value in graph.attributes(): # Or graph.edges() depending on structure
                #     if role == ':polarity' and value == '-':
                #         # `var` is the variable associated with the negation.
                #         # Find the concept of `var`
                #         for inst_var, concept, _ in graph.instances():
                #             if inst_var == var:
                #                 negated_concepts.add(str(concept)) # Or var itself
                #                 break
        except penman.DecodeError as e:
            logger.exception(f"Penman library failed to decode graph for negation detection:\n{penman_graph_str}\nError: {e}")
        except Exception as e: # Catch other potential errors.
            logger.exception(f"Error parsing negations from PENMAN string: {e}\nGraph:\n{penman_graph_str}")
        return negated_concepts

    def _get_root_concept(self, penman_graph_str: str) -> Optional[str]:
        """Extracts the concept of the root node from an AMR graph string.

        The root of the graph is indicated by `graph.top` in the `penman` library.
        This method finds the instance declaration corresponding to this top variable.

        Args:
            penman_graph_str (str): The AMR graph in PENMAN string format.

        Returns:
            Optional[str]: The concept string of the root node (e.g., "want-01"),
                           or None if the root cannot be determined, the input is empty,
                           or parsing fails.

        Raises:
            penman.DecodeError: If the penman library fails to decode the graph string.

        """
        if not penman_graph_str:
            return None
        try:
            graph = penman.decode(penman_graph_str)
            # `graph.top` gives the variable name of the root node (e.g., 'w').
            top_variable = graph.top
            if top_variable is None:
                logger.warning(f"Graph has no designated top variable: {penman_graph_str}")
                return None

            # Find the instance declaration for the top variable to get its concept.
            # An instance is a triple (variable, concept, role).
            for var, concept_label, _role in graph.instances():
                if var == top_variable:
                    return str(concept_label) # Return the concept label as string.
            logger.warning(f"Root concept not found for top variable '{top_variable}' in graph:\n{penman_graph_str}")
            return None # Should ideally not happen if top_variable is valid and in instances.
        except penman.DecodeError as e:
            logger.exception(f"Penman library failed to decode graph for root concept extraction:\n{penman_graph_str}\nError: {e}")
            return None
        except Exception as e:
            logger.exception(f"Error processing root concept with penman library: {e}\nGraph:\n{penman_graph_str}")
            return None

    def calculate_amr_features(self, text1: str, text2: str) -> dict[str, Optional[float]]:
        """Calculates a set of similarity features based on AMR analysis of two texts.

        This method first parses both input texts into AMR graphs. Then, it computes
        various similarity scores based on these graphs, including Smatch, concept overlap,
        named entity overlap, negation presence, and root concept match.

        Args:
            text1 (str): The first text string for comparison.
            text2 (str): The second text string for comparison.

        Returns:
            dict[str, Optional[float]]: A dictionary where keys are feature names
            (e.g., "smatch_fscore", "concept_jaccard") and values are the calculated
            similarity scores as floats. If a specific feature calculation fails or
            if AMR parsing fails for either text, the corresponding value will be None.
            The dictionary also includes placeholders for features not yet implemented.

        """
        results: dict[str, Optional[float]] = {
            "smatch_fscore": None,
            "smatch_precision": None,
            "smatch_recall": None,
            "concept_jaccard": None,
            "named_entity_jaccard": None,
            "negation_jaccard": None, # Based on placeholder detection
            "root_similarity": None,
            # Placeholders for features that could be implemented in the future
            "frame_similarity": None,
            "srl_similarity": None,
            "reentrancy_similarity": None,
            "degree_similarity": None,
            "quantifier_similarity": None,
            "wlk_similarity": None, # Weisfeiler-Lehman Kernel or similar graph kernel
        }

        if not self.stog_model:
            logger.error("AMR StoG model not loaded. Cannot calculate AMR features.")
            return results # Return dictionary with all Nones

        logger.info("Parsing Text 1 for AMR...")
        amr1_penman = self._parse_amr(text1)
        logger.info("Parsing Text 2 for AMR...")
        amr2_penman = self._parse_amr(text2)

        if not amr1_penman or not amr2_penman:
            logger.error("AMR parsing failed for one or both texts. Some features may be unavailable.")
            # We can still return the results dict; features that couldn't be computed will remain None.
            return results

        # 1. Smatch Calculation (Precision, Recall, F-score)
        # Smatch measures the similarity between two AMR graphs.
        try:
            # `compute_smatch` from `amrlib.evaluate.smatch_enhanced` takes lists of PENMAN strings.
            precision, recall, f_score = compute_smatch([amr1_penman], [amr2_penman])
            results["smatch_fscore"] = f_score
            results["smatch_precision"] = precision
            results["smatch_recall"] = recall
            logger.info(f"Smatch calculated: F={f_score:.4f}, P={precision:.4f}, R={recall:.4f}")
        except Exception as e: # Catching broad exception as smatch calculation can fail.
            logger.exception(f"Smatch calculation failed. Error: {e}")

        # 2. Concept Overlap (Jaccard Index)
        # Measures similarity based on the set of concepts present in both AMR graphs.
        try:
            concepts1 = self._get_graph_concepts(amr1_penman)
            concepts2 = self._get_graph_concepts(amr2_penman)
            intersection = len(concepts1.intersection(concepts2))
            union = len(concepts1.union(concepts2))
            # Jaccard Index: |A intersect B| / |A union B|
            # If both sets are empty, Jaccard is 1 (they are perfectly similar in their emptiness).
            # If union is 0 but not both are empty (shouldn't happen if one is non-empty), result is 0.
            if union == 0:
                results["concept_jaccard"] = 1.0 if not concepts1 and not concepts2 else 0.0
            else:
                results["concept_jaccard"] = intersection / union
            logger.debug(
                f"Concepts: Set1={len(concepts1)}, Set2={len(concepts2)}, Jaccard={results['concept_jaccard']:.4f}",
            )
        except Exception as e:
            logger.exception(f"Concept similarity calculation failed. Error: {e}")

        # 3. Named Entity Overlap (Jaccard Index)
        # Measures similarity based on the set of named entities found in both AMR graphs.
        try:
            ne1 = self._get_named_entities(amr1_penman)
            ne2 = self._get_named_entities(amr2_penman)
            intersection = len(ne1.intersection(ne2))
            union = len(ne1.union(ne2))
            if union == 0:
                results["named_entity_jaccard"] = 1.0 if not ne1 and not ne2 else 0.0
            else:
                results["named_entity_jaccard"] = intersection / union
            logger.debug(
                f"Named Entities: Set1={len(ne1)}, Set2={len(ne2)}, Jaccard={results['named_entity_jaccard']:.4f}",
            )
        except Exception as e:
            logger.exception(f"Named entity similarity calculation failed. Error: {e}")

        # 4. Negation Overlap (Simplified - based on placeholder)
        # Checks if both texts contain a negation. Current implementation is basic.
        # A more meaningful score would compare which concepts are negated.
        try:
            neg1 = self._get_negations(amr1_penman) # Set containing NEGATION_PLACEHOLDER or empty
            neg2 = self._get_negations(amr2_penman)

            # If both have the placeholder, they match (1.0). Otherwise, no match (0.0).
            # This is a binary match on the presence of any negation.
            both_have_negation = NEGATION_PLACEHOLDER in neg1 and NEGATION_PLACEHOLDER in neg2
            neither_has_negation = NEGATION_PLACEHOLDER not in neg1 and NEGATION_PLACEHOLDER not in neg2

            if both_have_negation or neither_has_negation:
                 results["negation_jaccard"] = 1.0 # Both have or both don't have negation
            else:
                 results["negation_jaccard"] = 0.0 # One has, one doesn't

            # An alternative Jaccard on the placeholder itself (less intuitive for binary presence):
            # intersection_neg = len(neg1.intersection(neg2))
            # union_neg = len(neg1.union(neg2))
            # if union_neg == 0:
            #    results["negation_jaccard"] = 1.0
            # else:
            #    results["negation_jaccard"] = intersection_neg / union_neg
            logger.debug(f"Negations: Set1 presence={NEGATION_PLACEHOLDER in neg1}, Set2 presence={NEGATION_PLACEHOLDER in neg2}, Match={results['negation_jaccard']:.4f}")

        except Exception as e:
            logger.exception(f"Negation similarity calculation failed. Error: {e}")

        # 5. Root Concept Similarity
        # Checks if the main concept (root) of the AMR graphs is the same.
        try:
            root1 = self._get_root_concept(amr1_penman)
            root2 = self._get_root_concept(amr2_penman)
            # Exact match: 1.0 if roots are identical and not None, 0.0 otherwise.
            results["root_similarity"] = 1.0 if root1 is not None and root1 == root2 else 0.0
            logger.debug(f"Roots: R1='{root1}', R2='{root2}', Sim={results['root_similarity']:.4f}")
        except Exception as e:
            logger.exception(f"Root similarity calculation failed. Error: {e}")

        logger.info("Finished calculating implemented AMR features.")
        return results

# Example usage within the `if __name__ == "__main__":` block
if __name__ == "__main__":
    # --- Configure Rich Logging (if available) ---
    logging.root.handlers.clear() # Clear any existing handlers
    LOG_LEVEL = logging.INFO  # Default log level, change to DEBUG for more verbosity
    logging.root.setLevel(LOG_LEVEL)
    _use_rich_logging = False # Flag to track if Rich logging is active

    if _rich_available:
        try:
            # Setup RichHandler for pretty console logging
            rich_handler = RichHandler(
                level=LOG_LEVEL,
                show_path=False, # Don't show module path in log output
                rich_tracebacks=True, # Enable rich tracebacks for exceptions
                markup=True, # Allow rich markup in log messages
            )
            logging.root.addHandler(rich_handler)
            from rich.console import Console
            _console = Console() # For printing separators or other rich content
            _separator_line = lambda: _console.print("-" * 70, style="dim")
            _use_rich_logging = True
            logger.info("Keyword/Semantic/AMR Example [bold green](using Rich logging)[/bold green]")
        except Exception as e: # Fallback if Rich setup fails for any reason
            _rich_available = False # Ensure flag is correctly set if setup fails
            logger.warning(f"Rich logging setup failed: {e}. Falling back to standard logging.")

    if not _use_rich_logging:
        # Basic logging configuration if Rich is not available or fails
        logging.basicConfig(
            level=LOG_LEVEL,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        _separator_line = lambda: print("-" * 70) # Simple separator for standard logging
        logger.info("Keyword/Semantic/AMR Example (standard logging - install 'rich' for enhanced output)")

    # --- Example Paragraphs for Demonstration ---
    para_a = "The boy wants to visit the park. He likes green trees."
    para_b = "The child desires to go to the green park. He loves trees."
    para_c = "A girl eats an apple quickly. She finished her snack."
    para_empty = "" # Test case with an empty paragraph

    # --- Initialize AMR Similarity Calculator ---
    amr_calculator_instance = None
    enable_amr_similarity_processing = True # Flag to control AMR processing

    if STOG_MODEL is not None: # Check if the global model loaded successfully
        try:
            logger.info("Initializing AMR Similarity Calculator with pre-loaded model...")
            amr_calculator_instance = AMRSimilarityCalculator(stog_model=STOG_MODEL)
            logger.info("AMR Similarity Calculator initialized successfully.")
        except ValueError as e: # Catch specific error from __init__
            logger.error(f"Failed to initialize AMRSimilarityCalculator: {e}")
            enable_amr_similarity_processing = False
        except Exception: # Catch any other unexpected errors during init
            logger.exception("An unexpected error occurred during AMRSimilarityCalculator initialization.")
            enable_amr_similarity_processing = False
    else:
        logger.error("Default AMR StoG model was not loaded. AMR similarity features will be disabled.")
        enable_amr_similarity_processing = False


    # --- Define Pairs of Texts to Process ---
    text_pairs_to_analyze = [
        ("Similar Paragraphs (A vs B)", para_a, para_b),
        ("Dissimilar Paragraphs (A vs C)", para_a, para_c),
        ("Paragraph A vs Empty", para_a, para_empty), # Test with empty string
    ]

    # --- Main Processing Loop ---
    for description, text_sample_1, text_sample_2 in text_pairs_to_analyze:
        if _use_rich_logging:
            logger.info(f"[bold magenta]Processing Pair: {description}[/bold magenta]")
        else:
            logger.info(f"Processing Pair: {description}")

        all_calculated_features = {}

        # Calculate AMR Features if enabled and calculator is available
        if enable_amr_similarity_processing and amr_calculator_instance:
            logger.info("--- Calculating AMR Features ---")
            try:
                amr_feature_results = amr_calculator_instance.calculate_amr_features(text_sample_1, text_sample_2)
                all_calculated_features.update(amr_feature_results)
            except Exception as e: # Catch errors during the feature calculation call itself
                logger.exception(f"Error during AMR Feature calculation for '{description}'. Details: {e}")
        elif not enable_amr_similarity_processing:
            logger.warning("AMR Similarity processing is disabled due to model loading or initialization issues.")
        else: # Should not happen if logic is correct, but as a fallback
            logger.warning("AMR Similarity calculator not available. Skipping AMR features.")


        # --- Print Combined Results for the Current Pair ---
        log_header = f"Results for {description}"
        if _use_rich_logging:
            logger.info(f"[bold cyan]>>> {log_header}:[/bold cyan]")
        else:
            logger.info(f">>> {log_header}:")

        # Log the input texts (first 50 chars for brevity if long)
        logger.info(f' Text 1: "{text_sample_1[:100].strip()}..."' if len(text_sample_1) > 100 else f' Text 1: "{text_sample_1.strip()}"')
        logger.info(f' Text 2: "{text_sample_2[:100].strip()}..."' if len(text_sample_2) > 100 else f' Text 2: "{text_sample_2.strip()}"')

        if not all_calculated_features:
            logger.info("  No features were calculated for this pair.")
        else:
            for feature_name, feature_value in all_calculated_features.items():
                if isinstance(feature_value, float):
                    # Apply color coding for scores if Rich is used
                    if _use_rich_logging and ("score" in feature_name or "similarity" in feature_name or "jaccard" in feature_name):
                        color = "green" if feature_value > 0.6 else "yellow" if feature_value > 0.2 else "red" # noqa: PLR2004
                        logger.info(f"  {feature_name}: [{color}]{feature_value:.4f}[/{color}]")
                    else:
                        logger.info(f"  {feature_name}: {feature_value:.4f}") # Standard float formatting
                else: # For None or other types
                    logger.info(f"  {feature_name}: {feature_value}")
        _separator_line() # Print a separator line after each pair's results

    logger.info("Similarity Calculation Example Finished.")
