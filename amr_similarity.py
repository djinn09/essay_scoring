"""
Module for calculating text similarity scores using Abstract Meaning Representation (AMR).

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
from typing import TYPE_CHECKING, Optional

import amrlib
import penman  # Required for graph manipulation
from amrlib.evaluate.smatch_enhanced import compute_smatch  # Enhanced for more detailed Smatch scores

if TYPE_CHECKING:
    from amrlib.models.StoG import StoG

# --- Rich Logging (Optional) ---
try:
    from rich.logging import RichHandler

    _rich_available = True
except ImportError:
    _rich_available = False


logger = logging.getLogger(__name__)

MAX_TEXT_SAMPLE_LENGTH = 100
try:
    # IMPORTANT: The model_dir path below is currently hardcoded.
    # Users should modify this path to point to their downloaded amrlib model directory.
    STOG_MODEL = amrlib.load_stog_model(
        model_dir="/mnt/e/Machine_learning/NLP-Revise/essay_grading/model_parse_xfm_bart_base-v0_1_0",  # Example path
        device="cpu",  # Default to CPU, can be changed if GPU is available
    )
    logger.info("Default AMR StoG model loaded successfully.")
except Exception:
    logger.exception(
        "Failed to load default AMR StoG model. "
        "AMRSimilarityCalculator may not work unless a model is provided at instantiation.",
    )
    STOG_MODEL = None


# Placeholder concept used when a negation is detected but not associated with a specific concept.
NEGATION_PLACEHOLDER = "HAS_NEGATION"


class AMRSimilarityCalculator:
    """
    Calculates similarity features based on Abstract Meaning Representation (AMR).

    AMR provides a way to represent the meaning of a sentence as a graph.
    This class uses `amrlib` for parsing text to AMR and `penman` for graph manipulation.

    Requires a pre-loaded `amrlib` parsing model (StoG - Stack-Transformer).
    The model should be passed during instantiation.
    """

    def __init__(self, stog_model: StoG) -> None:
        """
        Initialize AMRSimilarityCalculator with a pre-loaded `amrlib` StoG model.

        Args:
            stog_model (StoG): A pre-loaded `amrlib` Stack-Transformer (StoG) parsing model object.
                This model is used to parse text into AMR graphs.

        Raises:
            ValueError: If `stog_model` is not provided (is None).
        """
        if not stog_model:
            msg = "A pre-loaded AMR StoG model (stog_model) is required."
            raise ValueError(msg)
        self.stog_model = stog_model
        logger.info("AMRSimilarityCalculator initialized with provided StoG model.")

    def _parse_amr(self, text: str) -> Optional[str]:
        """
        Parse input text into an AMR graph string in PENMAN format.

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
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            if not sentences:
                logger.warning("No sentences found after splitting input text.")
                return None

            graphs_penman = self.stog_model.parse_sents(sentences)

            if graphs_penman and graphs_penman[0]:
                return graphs_penman[0]
            logger.warning("AMR parsing did not yield a graph for the first sentence.")
            return None
        except Exception:
            logger.exception(f"AMR parsing failed for text '{text[:50]}...'.")
            return None

    def _get_graph_concepts(self, penman_graph_str: str) -> set[str]:
        """
        Extract concepts (instance labels and constants) from an AMR graph string.

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
            variables = graph.variables()

            for _var, concept_label, _role in graph.instances():
                if concept_label:
                    concepts.add(str(concept_label))

            for _source_var, _role, target in graph.edges():
                if target not in variables:
                    constant_val = str(target).strip('"')
                    concepts.add(constant_val)

            for _source_var, _role, target in graph.attributes():
                if target not in variables:
                    constant_val = str(target).strip('"')
                    concepts.add(constant_val)

        except penman.DecodeError:
            logger.exception(f"Penman library failed to decode graph for concept extraction:\n{penman_graph_str}")
        except Exception:
            logger.exception(f"Error processing graph concepts with penman library.\nGraph:\n{penman_graph_str}")
        return concepts

    def _get_named_entities(self, penman_graph_str: str) -> set[str]:
        """
        Extract named entities from an AMR graph string.

        Named entities are typically identified by `:wiki` links or by `:name` relations.
        This implementation primarily extracts string literals connected via `:wiki`
        and treats them as named entities. It also extracts string literals from `:name`
        edges if they are direct constants.

        Args:
            penman_graph_str (str): The AMR graph in PENMAN string format.

        Returns:
            set[str]: A set of unique named entity strings. Returns an empty set if the
                      input is empty or parsing/processing fails.
        """
        nes = set()
        if not penman_graph_str:
            return nes
        try:
            graph = penman.decode(penman_graph_str)
            variables = graph.variables()

            for _source_var, role, target in graph.attributes():
                if role == ":wiki" and target != "-":
                    nes.add(str(target).strip('"'))
                elif role == ":name" and target not in variables:
                    nes.add(str(target).strip('"'))

            for _source_var, role, target in graph.edges():
                if role == ":wiki" and target != "-" and target not in variables:
                    nes.add(str(target).strip('"'))
                elif role == ":name" and target not in variables:
                    nes.add(str(target).strip('"'))

        except penman.DecodeError:
            logger.exception(f"Penman library failed to decode graph for NE extraction:\n{penman_graph_str}")
        except Exception:
            logger.exception(f"Error processing named entities with penman library.\nGraph:\n{penman_graph_str}")
        return nes

    def _get_negations(self, penman_graph_str: str) -> set[str]:
        """
        Detect negations in an AMR graph string.

        This method currently uses a simplified approach, checking for the literal
        string ':polarity -' in the PENMAN graph, and returns a placeholder concept if found.

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
            if ":polarity -" in penman_graph_str:
                negated_concepts.add(NEGATION_PLACEHOLDER)
        except penman.DecodeError:
            logger.exception(f"Penman library failed to decode graph for negation detection:\n{penman_graph_str}")
        except Exception:
            logger.exception(f"Error parsing negations from PENMAN string.\nGraph:\n{penman_graph_str}")
        return negated_concepts

    def _get_root_concept(self, penman_graph_str: str) -> Optional[str]:
        """
        Extract the concept of the root node from an AMR graph string.

        The root of the graph is indicated by `graph.top` in the `penman` library.

        Args:
            penman_graph_str (str): The AMR graph in PENMAN string format.

        Returns:
            Optional[str]: The concept string of the root node (e.g., "want-01"),
                           or None if the root cannot be determined, the input is empty,
                           or parsing fails.
        """
        if not penman_graph_str:
            return None
        try:
            graph = penman.decode(penman_graph_str)
            top_variable = graph.top
            if top_variable is None:
                logger.warning(f"Graph has no designated top variable: {penman_graph_str}")
                return None

            for var, concept_label, _role in graph.instances():
                if var == top_variable:
                    return str(concept_label)
            logger.warning(f"Root concept not found for top variable '{top_variable}' in graph:\n{penman_graph_str}")
            return None
        except penman.DecodeError:
            logger.exception(f"Penman library failed to decode graph for root concept extraction:\n{penman_graph_str}")
            return None
        except Exception:
            logger.exception(f"Error processing root concept with penman library.\nGraph:\n{penman_graph_str}")
            return None

    def calculate_amr_features(self, text1: str, text2: str) -> dict[str, Optional[float]]:
        """
        Calculate a set of similarity features based on AMR analysis of two texts.

        Parse both input texts into AMR graphs. Then, compute
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
        """
        results: dict[str, Optional[float]] = {
            "smatch_fscore": None,
            "smatch_precision": None,
            "smatch_recall": None,
            "concept_jaccard": None,
            "named_entity_jaccard": None,
            "negation_jaccard": None,
            "root_similarity": None,
            "frame_similarity": None,
            "srl_similarity": None,
            "reentrancy_similarity": None,
            "degree_similarity": None,
            "quantifier_similarity": None,
            "wlk_similarity": None,
        }

        if not self.stog_model:
            logger.error("AMR StoG model not loaded. Cannot calculate AMR features.")
            return results

        logger.info("Parsing Text 1 for AMR...")
        amr1_penman = self._parse_amr(text1)
        logger.info("Parsing Text 2 for AMR...")
        amr2_penman = self._parse_amr(text2)

        if not amr1_penman or not amr2_penman:
            logger.error("AMR parsing failed for one or both texts. Some features may be unavailable.")
            return results

        self._calculate_smatch(amr1_penman, amr2_penman, results)
        self._calculate_concept_overlap(amr1_penman, amr2_penman, results)
        self._calculate_named_entity_overlap(amr1_penman, amr2_penman, results)
        self._calculate_negation_overlap(amr1_penman, amr2_penman, results)
        self._calculate_root_similarity(amr1_penman, amr2_penman, results)

        logger.info("Finished calculating implemented AMR features.")
        return results

    def _calculate_smatch(self, amr1_penman: str, amr2_penman: str, results: dict[str, Optional[float]]) -> None:
        """
        Calculate Smatch scores and update the results dictionary.

        Args:
            amr1_penman (str): The first AMR graph in PENMAN format.
            amr2_penman (str): The second AMR graph in PENMAN format.
            results (dict[str, Optional[float]]): The dictionary to update with results.
        """
        try:
            precision, recall, f_score = compute_smatch([amr1_penman], [amr2_penman])
            results["smatch_fscore"] = f_score
            results["smatch_precision"] = precision
            results["smatch_recall"] = recall
            logger.info(f"Smatch calculated: F={f_score:.4f}, P={precision:.4f}, R={recall:.4f}")
        except Exception:
            logger.exception("Smatch calculation failed.")

    def _calculate_concept_overlap(
        self,
        amr1_penman: str,
        amr2_penman: str,
        results: dict[str, Optional[float]],
    ) -> None:
        """
        Calculate concept overlap (Jaccard) and update the results dictionary.

        Args:
            amr1_penman (str): The first AMR graph in PENMAN format.
            amr2_penman (str): The second AMR graph in PENMAN format.
            results (dict[str, Optional[float]]): The dictionary to update with results.
        """
        try:
            concepts1 = self._get_graph_concepts(amr1_penman)
            concepts2 = self._get_graph_concepts(amr2_penman)
            intersection = len(concepts1.intersection(concepts2))
            union = len(concepts1.union(concepts2))
            if union == 0:
                results["concept_jaccard"] = 1.0 if not concepts1 and not concepts2 else 0.0
            else:
                results["concept_jaccard"] = intersection / union
            logger.debug(
                f"Concepts: Set1={len(concepts1)}, Set2={len(concepts2)}, Jaccard={results['concept_jaccard']:.4f}",
            )
        except Exception:
            logger.exception("Concept similarity calculation failed.")

    def _calculate_named_entity_overlap(
        self,
        amr1_penman: str,
        amr2_penman: str,
        results: dict[str, Optional[float]],
    ) -> None:
        """
        Calculate named entity overlap (Jaccard) and update the results dictionary.

        Args:
            amr1_penman (str): The first AMR graph in PENMAN format.
            amr2_penman (str): The second AMR graph in PENMAN format.
            results (dict[str, Optional[float]]): The dictionary to update with results.
        """
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
        except Exception:
            logger.exception("Named entity similarity calculation failed.")

    def _calculate_negation_overlap(
        self,
        amr1_penman: str,
        amr2_penman: str,
        results: dict[str, Optional[float]],
    ) -> None:
        """
        Calculate negation overlap and update the results dictionary.

        Args:
            amr1_penman (str): The first AMR graph in PENMAN format.
            amr2_penman (str): The second AMR graph in PENMAN format.
            results (dict[str, Optional[float]]): The dictionary to update with results.
        """
        try:
            neg1 = self._get_negations(amr1_penman)
            neg2 = self._get_negations(amr2_penman)
            both_have_negation = NEGATION_PLACEHOLDER in neg1 and NEGATION_PLACEHOLDER in neg2
            neither_has_negation = NEGATION_PLACEHOLDER not in neg1 and NEGATION_PLACEHOLDER not in neg2
            if both_have_negation or neither_has_negation:
                results["negation_jaccard"] = 1.0
            else:
                results["negation_jaccard"] = 0.0
            logger.debug(
                f"Negations: Set1 presence={NEGATION_PLACEHOLDER in neg1}, "
                f"Set2 presence={NEGATION_PLACEHOLDER in neg2}, Match={results['negation_jaccard']:.4f}",
            )
        except Exception:
            logger.exception("Negation similarity calculation failed.")

    def _calculate_root_similarity(
        self,
        amr1_penman: str,
        amr2_penman: str,
        results: dict[str, Optional[float]],
    ) -> None:
        """
        Calculate root similarity and update the results dictionary.

        Args:
            amr1_penman (str): The first AMR graph in PENMAN format.
            amr2_penman (str): The second AMR graph in PENMAN format.
            results (dict[str, Optional[float]]): The dictionary to update with results.
        """
        try:
            root1 = self._get_root_concept(amr1_penman)
            root2 = self._get_root_concept(amr2_penman)
            results["root_similarity"] = 1.0 if root1 is not None and root1 == root2 else 0.0
            logger.debug(f"Roots: R1='{root1}', R2='{root2}', Sim={results['root_similarity']:.4f}")
        except Exception:
            logger.exception("Root similarity calculation failed.")


# Example usage within the `if __name__ == "__main__":` block
if __name__ == "__main__":
    # --- Configure Rich Logging (if available) ---
    logging.root.handlers.clear()  # Clear any existing handlers
    LOG_LEVEL = logging.INFO  # Default log level, change to DEBUG for more verbosity
    logging.root.setLevel(LOG_LEVEL)
    _use_rich_logging = False  # Flag to track if Rich logging is active

    if _rich_available:
        try:
            # Setup RichHandler for pretty console logging
            rich_handler = RichHandler(
                level=LOG_LEVEL,
                show_path=False,  # Don't show module path in log output
                rich_tracebacks=True,  # Enable rich tracebacks for exceptions
                markup=True,  # Allow rich markup in log messages
            )
            logging.root.addHandler(rich_handler)
            from rich.console import Console

            _console = Console()  # For printing separators or other rich content
            _separator_line = lambda: _console.print("-" * 70, style="dim")
            _use_rich_logging = True
            logger.info("Keyword/Semantic/AMR Example [bold green](using Rich logging)[/bold green]")
        except Exception as e:  # Fallback if Rich setup fails for any reason
            _rich_available = False  # Ensure flag is correctly set if setup fails
            logger.warning(f"Rich logging setup failed: {e}. Falling back to standard logging.")

    if not _use_rich_logging:
        # Basic logging configuration if Rich is not available or fails
        logging.basicConfig(
            level=LOG_LEVEL,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        _separator_line = lambda: print("-" * 70)  # Simple separator for standard logging
        logger.info("Keyword/Semantic/AMR Example (standard logging - install 'rich' for enhanced output)")

    # --- Example Paragraphs for Demonstration ---
    para_a = "The boy wants to visit the park. He likes green trees."
    para_b = "The child desires to go to the green park. He loves trees."
    para_c = "A girl eats an apple quickly. She finished her snack."
    para_empty = ""  # Test case with an empty paragraph

    # --- Initialize AMR Similarity Calculator ---
    amr_calculator_instance = None
    enable_amr_similarity_processing = True  # Flag to control AMR processing

    if STOG_MODEL is not None:  # Check if the global model loaded successfully
        try:
            logger.info("Initializing AMR Similarity Calculator with pre-loaded model...")
            amr_calculator_instance = AMRSimilarityCalculator(stog_model=STOG_MODEL)
            logger.info("AMR Similarity Calculator initialized successfully.")
        except ValueError:  # Catch specific error from __init__
            logger.exception("Failed to initialize AMRSimilarityCalculator.")
            enable_amr_similarity_processing = False
        except Exception:  # Catch any other unexpected errors during init
            logger.exception("An unexpected error occurred during AMRSimilarityCalculator initialization.")
            enable_amr_similarity_processing = False
    else:
        logger.error("Default AMR StoG model was not loaded. AMR similarity features will be disabled.")
        enable_amr_similarity_processing = False

    # --- Define Pairs of Texts to Process ---
    text_pairs_to_analyze = [
        ("Similar Paragraphs (A vs B)", para_a, para_b),
        ("Dissimilar Paragraphs (A vs C)", para_a, para_c),
        ("Paragraph A vs Empty", para_a, para_empty),  # Test with empty string
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
            except Exception:  # Catch errors during the feature calculation call itself
                logger.exception(f"Error during AMR Feature calculation for '{description}'.")
        elif not enable_amr_similarity_processing:
            logger.warning("AMR Similarity processing is disabled due to model loading or initialization issues.")
        else:  # Should not happen if logic is correct, but as a fallback
            logger.warning("AMR Similarity calculator not available. Skipping AMR features.")

        # --- Print Combined Results for the Current Pair ---
        log_header = f"Results for {description}"
        if _use_rich_logging:
            logger.info(f"[bold cyan]>>> {log_header}:[/bold cyan]")
        else:
            logger.info(f">>> {log_header}:")

        # Log the input texts (first MAX_TEXT_SAMPLE_LENGTH chars for brevity if long)
        logger.info(
            f' Text 1: "{text_sample_1[:MAX_TEXT_SAMPLE_LENGTH].strip()}..."'
            if len(text_sample_1) > MAX_TEXT_SAMPLE_LENGTH
            else f' Text 1: "{text_sample_1.strip()}"',
        )
        logger.info(
            f' Text 2: "{text_sample_2[:MAX_TEXT_SAMPLE_LENGTH].strip()}..."'
            if len(text_sample_2) > MAX_TEXT_SAMPLE_LENGTH
            else f' Text 2: "{text_sample_2.strip()}"',
        )

        if not all_calculated_features:
            logger.info("  No features were calculated for this pair.")
        else:
            for feature_name, feature_value in all_calculated_features.items():
                if isinstance(feature_value, float):
                    # Apply color coding for scores if Rich is used
                    if _use_rich_logging and (
                        "score" in feature_name or "similarity" in feature_name or "jaccard" in feature_name
                    ):
                        color = "green" if feature_value > 0.6 else "yellow" if feature_value > 0.2 else "red"
                        logger.info(f"  {feature_name}: [{color}]{feature_value:.4f}[/{color}]")
                    else:
                        logger.info(f"  {feature_name}: {feature_value:.4f}")  # Standard float formatting
                else:  # For None or other types
                    logger.info(f"  {feature_name}: {feature_value}")
        _separator_line()  # Print a separator line after each pair's results

    logger.info("Similarity Calculation Example Finished.")
