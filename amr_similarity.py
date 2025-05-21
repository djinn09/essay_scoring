"""Module for calculating various text similarity scores.

Includes:
- AMRSimilarityCalculator: For similarity based on Abstract Meaning Representation (AMR).

**AMR Models:** AMR calculations require `amrlib` and its models. See AMRSimilarityCalculator docs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import amrlib
import penman  # Add this import at the top
from amrlib.evaluate.smatch_enhanced import compute_smatch  # Use enhanced for more details

# --- Rich Logging ---
try:
    from rich.logging import RichHandler

    _rich_available = True
except ImportError:
    _rich_available = False
    # Let's make rich optional for the core library, but required for the example's nice output
    # raise ImportError("Missing rich...")


logger = logging.getLogger(__name__)

STOG_MODEL = amrlib.load_stog_model(
    model_dir="/mnt/e/Machine_learning/NLP-Revise/essay_grading/model_parse_xfm_bart_base-v0_1_0",
    device="cpu",
)
# Placeholder for negation concept key
NEGATION_PLACEHOLDER = "HAS_NEGATION"


class AMRSimilarityCalculator:
    """Calculates similarity features based on Abstract Meaning Representation (AMR).

    Requires `amrlib` library and downloaded parsing models.
    See: https://github.com/bjascob/amrlib
    Install: pip install amrlib
    Download Models: python -m amrlib.models.download_models

    Currently calculates:
    - Smatch (F-score, Precision, Recall)
    - Concept Overlap (Jaccard Index)
    - Named Entity Overlap (Jaccard Index based on :name/:wiki links)
    - Negation Overlap (Jaccard Index of negated concepts)
    - Root Concept Similarity (Exact match)

    Other features mentioned (Frames, SRL, Reentrancies, Degrees, Quantifiers, WLK)
    are not implemented due to complexity but could be added with further effort.

    Args:
        stog_model_dir: Path to the downloaded `amrlib` Stack-Transformer (StoG)
                        parsing model directory. Defaults to None (will use amrlib default).
        device: Device for PyTorch models ('cuda', 'cpu', etc.). Defaults to 'cpu'.

    """

    def __init__(self, stog_model: Optional[str] = None) -> None:
        """Initialize AMRSimilarityCalculator with the path to a downloaded `amrlib` StoG model.

        Args:
            stog_model: Path to the downloaded `amrlib` Stack-Transformer (StoG) parsing model directory.

        """
        if not stog_model:
            msg = "AMR model path is required."
            raise ValueError(msg)
        self._load_models(stog_model)

    def _load_models(self, model: Any) -> None:  # noqa: ANN401
        """Load the necessary amrlib models."""
        try:
            logger.info("Loading AMR parsing model (StoG)... This may take a moment.")
            # Load the Stack-Transformer parser model
            self.stog_model = model
            logger.info("AMR StoG model loaded.")

        except Exception as e:
            logger.exception(f"Failed to load amrlib models. AMR features will not work. Error: {e}")

            # Optionally re-raise or just log and let methods fail later
            msg = "Failed to initialize amrlib models."
            raise RuntimeError(msg) from e

    def _parse_amr(self, text: str) -> Optional[str]:
        """Parses a single sentence into AMR PENMAN string format."""
        if not self.stog_model or not text.strip():
            return None
        try:
            # amrlib expects sentences, split paragraphs naively for now
            # A better approach might involve sentence splitting first
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            if not sentences:
                return None

            graphs_penman = self.stog_model.parse_sents(sentences)
            # For simplicity, combine graphs if multiple sentences? Or just use the first?
            # Let's just use the first sentence's graph for this example
            # A proper implementation would handle multi-sentence paragraphs better.
            if graphs_penman:
                return graphs_penman[0]
            return None
        except Exception:
            logger.exception(f"AMR parsing failed for text '{text[:50]}...'")
            return None

    # Example rewriting _get_graph_concepts (Adapt others similarly)
    def _get_graph_concepts(self, penman_graph_str: str) -> set[str]:
        """Extract concepts (instance labels and constants) using the penman library."""
        concepts = set()
        if not penman_graph_str:
            return concepts
        try:
            graph = penman.decode(penman_graph_str)
            variables = graph.variables()  # Get the set of all variable names (e.g., {'d', 'g', 't'})

            # 1. Add concepts from instance declarations (e.g., 'dog' in '(d / dog)')
            for _var, concept, _ in graph.instances():
                if concept:  # Concept is the label after the '/'
                    concepts.add(str(concept))

            # 2. Add constants found as targets in edges
            #    (e.g., "blue" in ':color "blue"')
            for _source, _, target in graph.edges():
                # If the target is NOT one of the graph's variables, it's a constant
                if target not in variables:
                    # penman stores constants as strings; strip quotes from string literals
                    constant_val = str(target).strip('"')
                    concepts.add(constant_val)

            # 3. Add constants found as targets in attributes (similar to edges)
            #    (e.g., 5 in ':value 5')
            for _, _, target in graph.attributes():
                # If the target is NOT one of the graph's variables, it's a constant
                if target not in variables:
                    constant_val = str(target).strip('"')
                    concepts.add(constant_val)

        except penman.DecodeError:
            logger.exception(f"Penman library failed to decode graph:\n{penman_graph_str}")
        except Exception:  # Catch other potential errors during processing
            logger.exception(f"Error processing graph with penman library: \nGraph:\n{penman_graph_str}")
        return concepts

    def _get_named_entities(self, penman_graph_str: str) -> set[str]:
        nes = set()
        if not penman_graph_str:
            return nes
        try:
            graph = penman.decode(penman_graph_str)
            # Look for :wiki or :name attributes/edges
            for _source, role, target in graph.attributes():
                # Note: :name might point to a variable representing a name subgraph
                # :wiki usually points directly to a constant
                if (
                    role == ":wiki" and target != "-"
                ):  # Check role and ignore placeholder '-'. target should be constant.
                    nes.add(str(target).strip('"'))
                # Add more complex logic here if needed to handle :name subgraphs
            # Also check edges if :wiki/:name can appear there
            for _source, role, target in graph.edges():
                if role == ":wiki" and target != "-" and target not in graph.variables():
                    # Ensure target is a constant here too
                    nes.add(str(target).strip('"'))

        except penman.DecodeError:
            logger.exception(f"Penman library failed to decode NEs from graph:\n{penman_graph_str}")
        except Exception:
            logger.exception(f"Error processing NEs with penman library: {e}\nGraph:\n{penman_graph_str}")
        return nes

    def _get_negations(self, penman_graph: str) -> set[str]:
        """Find concepts associated with a negation (:polarity -)."""
        # Find nodes modified by :polarity -
        negated_concepts = set()
        try:
            lines = penman_graph.strip().split("\n")
            # First pass: find variables directly negated
            for line in lines:
                strip_line = line.strip()
                if strip_line.startswith(":polarity -"):
                    # Find the variable this polarity modifies (usually the preceding line)
                    # This parsing is fragile. Proper PENMAN lib needed.
                    pass  # Complex to implement reliably without proper parser

            # For now, just count occurrence of ':polarity -' string as a proxy
            if ":polarity -" in penman_graph:
                negated_concepts.add(NEGATION_PLACEHOLDER)  # Use a placeholder concept

        except Exception:
            logger.exception(f"Error parsing negations from PENMAN:\n{penman_graph}")
        return negated_concepts

    def _get_root_concept(self, penman_graph_str: str) -> Optional[str]:
        if not penman_graph_str:
            return None
        try:
            graph = penman.decode(penman_graph_str)
            top_variable = graph.top  # Get the top variable (e.g., 'w' in '# ::id N ::snt X (w / want)')
            # Find the instance definition for the top variable
            for var, concept, _role in graph.instances():
                if var == top_variable:
                    return str(concept)
            return None  # Should not happen in a valid graph with a top
        except penman.DecodeError:
            logger.exception(f"Penman library failed to decode root from graph:\n{penman_graph_str}")
            return None
        except Exception:
            logger.exception(f"Error processing root with penman library:\nGraph:\n{penman_graph_str}")
            return None

    def calculate_amr_features(self, text1: str, text2: str) -> dict[str, Optional[float]]:
        """Calculate a set of features based on Abstract Meaning Representation (AMR) analysis.

        Features calculated:
            - Smatch F-score, Precision, and Recall
            - Concept Jaccard Similarity
            - Named Entity Jaccard Similarity
            - Negation Jaccard Similarity (placeholder for future implementation)
            - Root Concept Similarity
            - Placeholders for future features:
                - Frame Similarity
                - SRL Similarity
                - Reentrancy Similarity
                - Degree Similarity
                - Quantifier Similarity
                - Walk Similarity

        Returns a dictionary with the calculated features as float values.
        If feature calculation fails, the corresponding field in the dictionary will be None.
        """
        results: dict[str, Optional[float]] = {
            "smatch_fscore": None,
            "smatch_precision": None,
            "smatch_recall": None,
            "concept_jaccard": None,
            "named_entity_jaccard": None,
            "negation_jaccard": None,
            "root_similarity": None,
            # Placeholders for unimplemented features
            "frame_similarity": None,
            "srl_similarity": None,
            "reentrancy_similarity": None,
            "degree_similarity": None,
            "quantifier_similarity": None,
            "wlk_similarity": None,
        }

        if not self.stog_model:
            logger.error("AMR model not loaded, cannot calculate AMR features.")
            return results

        logger.info("Parsing Paragraph A for AMR...")
        amr1_penman = self._parse_amr(text1)
        logger.info("Parsing Paragraph B for AMR...")
        amr2_penman = self._parse_amr(text2)

        if not amr1_penman or not amr2_penman:
            logger.error("AMR parsing failed for one or both texts.")
            return results

        # 1. Smatch Calculation
        try:
            # compute_smatch expects file paths or lists of strings
            # Write temporary strings or use StringIO if needed, or adapt if amrlib handles strings directly
            # For simplicity here, assume compute_smatch can take strings (may need adjustment)
            # It actually returns P, R, F
            precision, recall, f_score = compute_smatch([amr1_penman], [amr2_penman])
            results["smatch_fscore"] = f_score
            results["smatch_precision"] = precision
            results["smatch_recall"] = recall
            logger.info(f"Smatch calculated: F={f_score:.4f}, P={precision:.4f}, R={recall:.4f}")
        except Exception:
            logger.exception("Smatch calculation failed:")

        # 2. Concept Overlap
        try:
            concepts1 = self._get_graph_concepts(amr1_penman)
            concepts2 = self._get_graph_concepts(amr2_penman)
            intersection = len(concepts1.intersection(concepts2))
            union = len(concepts1.union(concepts2))
            results["concept_jaccard"] = (
                intersection / union if union > 0 else 1.0 if not concepts1 and not concepts2 else 0.0
            )
            logger.debug(
                f"Concepts: Set1={len(concepts1)}, Set2={len(concepts2)}, Jaccard={results['concept_jaccard']:.4f}",
            )
        except Exception:
            logger.exception("Concept similarity calculation failed")

        # 3. Named Entity Overlap
        try:
            ne1 = self._get_named_entities(amr1_penman)
            ne2 = self._get_named_entities(amr2_penman)
            intersection = len(ne1.intersection(ne2))
            union = len(ne1.union(ne2))
            results["named_entity_jaccard"] = intersection / union if union > 0 else 1.0 if not ne1 and not ne2 else 0.0
            logger.debug(
                f"Named Entities: Set1={len(ne1)}, Set2={len(ne2)}, Jaccard={results['named_entity_jaccard']:.4f}",
            )
        except Exception:
            logger.exception("Named entity similarity calculation failed")

        # 4. Negation Overlap
        try:
            neg1 = self._get_negations(amr1_penman)
            neg2 = self._get_negations(amr2_penman)
            intersection = len(neg1.intersection(neg2))
            union = len(neg1.union(neg2))
            # Jaccard on the placeholder 'HAS_NEGATION' isn't very meaningful
            # Better: Binary match (1 if both have negation, 0 otherwise)?
            results["negation_jaccard"] = 1.0 if NEGATION_PLACEHOLDER in neg1 and NEGATION_PLACEHOLDER in neg2 else 0.0
            # Alternative: results["negation_jaccard"] = intersection / union if union > 0 else 1.0 if not neg1 and not neg2 else 0.0
            logger.debug(f"Negations: Set1={len(neg1)}, Set2={len(neg2)}, Match={results['negation_jaccard']:.4f}")

        except Exception:
            logger.exception("Negation similarity calculation failed")

        # 5. Root Similarity
        try:
            root1 = self._get_root_concept(amr1_penman)
            root2 = self._get_root_concept(amr2_penman)
            results["root_similarity"] = 1.0 if root1 is not None and root1 == root2 else 0.0
            logger.debug(f"Roots: R1='{root1}', R2='{root2}', Sim={results['root_similarity']:.4f}")
        except Exception:
            logger.exception("Root similarity calculation failed")

        logger.info("Finished calculating implemented AMR features.")
        return results

if __name__ == "__main__":
    # --- Configure Rich Logging ---
    logging.root.handlers.clear()
    LOG_LEVEL = logging.INFO  # Change to DEBUG for more verbose output
    logging.root.setLevel(LOG_LEVEL)
    _use_rich = False
    if _rich_available:
        try:
            rich_handler = RichHandler(level=LOG_LEVEL, show_path=False, rich_tracebacks=True, markup=True)
            logging.root.addHandler(rich_handler)
            from rich.console import Console

            console = Console()
            separator = lambda: console.print("-" * 70, style="dim")
            _use_rich = True
            logger.info("Keyword/Semantic/AMR Example [bold green](using Rich logging)[/bold green]")
        except Exception:  # Fallback if rich setup fails
            _rich_available = False  # Ensure flag is false

    # --- Example Paragraphs ---
    para_a = """
    The boy wants to visit the park. He likes green trees.
    """
    para_b = """
    The child desires to go to the green park. He loves trees.
    """
    para_c = """
    A girl eats an apple quickly. She finished her snack.
    """
    para_empty = ""

    if not _use_rich:
        logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        separator = lambda: print("-" * 70)
        logger.info("Keyword/Semantic/AMR Example (standard logging - install 'rich' for better output)")

    # --- Flags to Enable/Disable Components ---

    # --- Initialize Components ---

    amr_calculator = None

    try:
        logger.info("Initializing AMR Similarity Calculator...")
        # You might need to specify model path: model_dir='/path/to/downloaded/model'
        amr_calculator = AMRSimilarityCalculator(
            stog_model=STOG_MODEL,  # Use the loaded model
        )  # Reuse device if possible
        # Check if models loaded successfully
        if not amr_calculator.stog_model:
            logger.error("AMR Calculator initialized, but model loading failed. Disabling AMR features.")
            ENABLE_AMR_SIMILARITY = False
        else:
            logger.info("AMR Calculator initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize AMRSimilarityCalculator")
        ENABLE_AMR_SIMILARITY = False

    # --- Processing Loop ---
    pairs_to_process = [
        ("Similar Paragraphs (A vs B)", para_a, para_b),
        ("Dissimilar Paragraphs (A vs C)", para_a, para_c),
        ("Empty Paragraph (A vs Empty)", para_a, para_empty),
    ]

    for description, text1, text2 in pairs_to_process:
        logger.info(f"[bold magenta]Processing Pair: {description}[/bold magenta]")
        combined_results = {}

        # 3. AMR Features
        if amr_calculator:
            logger.info("--- Calculating AMR Features ---")
            try:
                amr_results = amr_calculator.calculate_amr_features(text1, text2)
                combined_results.update(amr_results)  # Add results
            except Exception:
                logger.exception("Error during AMR Feature calculation.")
        else:
            logger.warning("AMR Similarity disabled or failed initialization.")

        # --- Print Combined Results for the Pair ---
        logger.info(f"[bold cyan]>>> Combined Results for {description}:[/bold cyan]")
        logger.info(" Text 1: %s", text1)
        logger.info(" Text 2: %s", text2)
        for key, value in combined_results.items():
            if isinstance(value, float):
                # Format floats, potentially add color based on value/type?
                if "score" in key or "similarity" in key or "jaccard" in key:
                    color = "green" if value > 0.6 else "yellow" if value > 0.2 else "red"  # noqa: PLR2004
                    logger.info(f"  {key}: [{color}]{value:.4f}[/{color}]")
                else:
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")  # Print lists, ints, Nones as is
        separator()

    logger.info("Similarity Calculation Example Finished.")
