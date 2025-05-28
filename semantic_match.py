"""Provides semantic similarity calculations using Sentence Transformers.

This module offers the `SemanticCosineSimilarity` class to compute various
similarity and distance metrics (Cosine Similarity, Euclidean Distance, Manhattan Distance)
between two text inputs. It leverages the `sentence-transformers` library to generate
high-quality sentence/text embeddings.

A key feature is its ability to handle texts longer than typical model input limits
by employing a chunking mechanism with configurable overlap. This approach helps
preserve semantic context across chunk boundaries. Chunk embeddings are then aggregated
(currently by averaging) to produce a single representative embedding for the entire text.

The module includes:
- `SemanticCosineSimilarity`: The main class for performing similarity calculations.
- `SimilarityScores`: A Pydantic model for structured output of the calculated scores.
- Metric constants (e.g., `METRIC_COSINE`) for clear metric identification.
- An example usage section (`if __name__ == "__main__":`) demonstrating instantiation,
  calculation of various metrics for different text pairs, and rich logging (if available).

Core Dependencies:
- `torch`: For tensor operations.
- `sentence-transformers`: For loading pre-trained models and generating embeddings.
- `pydantic`: For data validation and structured output (SimilarityScores).
- `rich` (optional, for example): For enhanced console logging in the example.

The design emphasizes modularity, allowing for potential extensions like different
aggregation strategies or additional similarity metrics in the future.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Attempt to import necessary libraries and provide guidance if they are missing
try:
    import torch
    import torch.nn.functional as F  # noqa: N812 - Keep F as standard PyTorch alias
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    # Determine which library is missing for a more specific error message
    missing_lib = ""
    if "torch" in str(e).lower():
        missing_lib = "torch"
    elif "sentence_transformers" in str(e).lower():
        missing_lib = "sentence-transformers"
    else:
        missing_lib = "a required library"  # Generic fallback
    # Construct and raise a new ImportError with installation instructions
    error_message = f"Missing {missing_lib}. Please install it (e.g., `pip install {missing_lib}`). Original error: {e}"
    raise ImportError(error_message) from e


# Attempt to import 'rich' for enhanced console logging.
try:
    from rich.console import Console
    from rich.logging import RichHandler

    _rich_available = True  # Flag to indicate rich is available
except ImportError:
    _rich_available = False  # Flag to indicate rich is not available
    # For this module, rich is considered highly beneficial for the example's output.
    # If the library part needs to function without rich, this error might be conditional.
    error_message = "Missing rich. Please install it (`pip install rich`) for enhanced logging in the example."
    raise ImportError(error_message) from None  # 'from None' suppresses the original ImportError context


# Configure basic logging. This will be used if the module is imported
# or if rich is unavailable. It gets overridden in __main__ if rich is used.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Module-specific logger

# --- Configuration for Similarity Score Interpretation (used in example) ---
GOOD_SIMILARITY_SCORE = float(os.getenv("GOOD_SIMILARITY_SCORE", "0.7"))
BAD_SIMILARITY_SCORE = float(os.getenv("BAD_SIMILARITY_SCORE", "0.3"))

# --- Metric Definitions ---
METRIC_COSINE = "cosine"
METRIC_EUCLIDEAN = "euclidean"
METRIC_MANHATTAN = "manhattan"

ALLOWED_METRICS_MAP: dict[str, str] = {
    "cosine": METRIC_COSINE,
    "euclidean": METRIC_EUCLIDEAN,
    "manhattan": METRIC_MANHATTAN,
}
ALL_METRIC_KEYS: set[str] = set(ALLOWED_METRICS_MAP.values())


# --- Pydantic Model for Similarity Scores ---
class SimilarityScores(BaseModel):
    """Data model for storing calculated similarity scores.

    Allows initialization using internal metric keys (which serve as aliases)
    and allows access via Pythonic attribute names (e.g., `scores.cosine`).
    The `populate_by_name=True` config enables initialization using field names
    as well as aliases.
    """

    model_config = ConfigDict(
        populate_by_name=True,  # Allows initialization by field name or alias
        extra="ignore",         # Ignores extra fields provided during initialization
    )

    cosine: Optional[float] = Field(default=None, description="Cosine similarity score", alias=METRIC_COSINE)
    euclidean: Optional[float] = Field(default=None, description="Euclidean distance score", alias=METRIC_EUCLIDEAN)
    manhattan: Optional[float] = Field(default=None, description="Manhattan (L1) distance score", alias=METRIC_MANHATTAN)


class SemanticCosineSimilarity:
    """Calculates semantic similarity between texts using Sentence Transformers with chunking.

    This class encapsulates the logic for:
    1.  Initializing with a Sentence Transformer model and parameters for chunking.
    2.  Processing input texts: if a text is longer than `chunk_size`, it's
        split into overlapping chunks.
    3.  Generating embeddings for the text (or its chunks) using the provided model.
    4.  Aggregating chunk embeddings (if chunking occurred) into a single
        representative embedding for the entire text, currently by averaging.
    5.  Computing specified similarity and/or distance metrics between the
        embeddings of two texts.
    6.  Returning the results in a structured `SimilarityScores` object.

    The use of overlapping chunks is a strategy to preserve semantic context that
    might otherwise be lost at the boundaries if simple, non-overlapping splitting
    were used, especially for longer texts.

    Attributes:
        model (SentenceTransformer): The Sentence Transformer model instance used for
                                     generating text embeddings.
        chunk_size (int): The target character length for each text chunk. Texts shorter
                          than this are processed as a single chunk.
        overlap (int): The number of characters that adjacent chunks will overlap. This
                       helps maintain context continuity across chunks.
        batch_size (int): The batch size to use when encoding multiple chunks with the
                          Sentence Transformer model. This can optimize performance for
                          texts that are split into many chunks.

    """

    def __init__(
        self,
        model: SentenceTransformer,
        chunk_size: int = 384,  # Default based on common model input sizes (e.g., BERT variants)
        overlap: int = 64,      # Default overlap, provides some context continuity
        batch_size: int = 32,   # Common default batch size for encoding
    ) -> None:
        """Initializes the SemanticCosineSimilarity calculator.

        Args:
            model (SentenceTransformer): An initialized Sentence Transformer model instance.
                                         This model will be used to generate embeddings.
            chunk_size (int): The target character size for each text chunk.
                              Must be greater than `overlap`. Defaults to 384.
            overlap (int): The number of characters for overlap between adjacent chunks.
                           Cannot be negative. Defaults to 64.
            batch_size (int): The batch size for encoding chunks with the sentence
                              transformer model. Must be positive. Defaults to 32.

        Raises:
            TypeError: If `model` is not an instance of `SentenceTransformer`,
                       or if `chunk_size`, `overlap`, or `batch_size` are not integers.
            ValueError: If `chunk_size` is not strictly greater than `overlap`,
                        if `overlap` is negative, or if `batch_size` is not positive.

        """
        # --- Input Validation ---
        if not isinstance(model, SentenceTransformer):
            msg = "Model must be an instance of sentence_transformers.SentenceTransformer."
            logger.error(msg)
            raise TypeError(msg)
        if not all(isinstance(val, int) for val in [chunk_size, overlap, batch_size]):
            msg = "chunk_size, overlap, and batch_size must all be integers."
            logger.error(msg)
            raise TypeError(msg)

        if overlap < 0:
            msg = "Overlap (overlap) cannot be negative."
            logger.error(msg)
            raise ValueError(msg)
        if chunk_size <= overlap: # Must be strictly greater for chunks to make sense
            msg = f"chunk_size ({chunk_size}) must be strictly greater than overlap ({overlap})."
            logger.error(msg)
            raise ValueError(msg)
        if batch_size <= 0:
            msg = "batch_size must be positive."
            logger.error(msg)
            raise ValueError(msg)

        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size

        # Attempt to get the model's name for more informative logging.
        # This tries various common ways a model name might be stored.
        model_name = getattr(getattr(model, "config", {}), "name", None) # E.g. Hugging Face model config
        if not model_name: # Fallback for SentenceTransformer specific storage
            model_name_from_st_config = getattr(model, "_model_config", {}).get("name")
            if model_name_from_st_config:
                model_name = model_name_from_st_config
            elif hasattr(model, "name_or_path"): # Another common attribute in Hugging Face / ST models
                model_name = model.name_or_path
            else: # Generic fallback to class name if no specific name is found
                model_name = model.__class__.__name__

        logger.info(
            f"SemanticCosineSimilarity initialized with model: [cyan]{model_name}[/cyan], "
            f"chunk_size: [yellow]{self.chunk_size}[/yellow], overlap: [yellow]{self.overlap}[/yellow], "
            f"batch_size: [yellow]{self.batch_size}[/yellow]",
        )

    def _get_aggregated_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Encodes a text string into a single aggregated embedding vector.

        If the input `text` is longer than `self.chunk_size` (in characters),
        it is split into overlapping chunks. Each chunk is then encoded individually
        by the Sentence Transformer model. These chunk embeddings are subsequently
        aggregated (currently by averaging) to produce a single representative
        1D embedding vector for the entire text.

        If the text is shorter than or equal to `self.chunk_size`, it is encoded
        directly without chunking.

        Args:
            text (str): The text string to encode.

        Returns:
            Optional[torch.Tensor]: A 1D PyTorch tensor representing the aggregated
            embedding of the input text. Returns `None` if the input `text` is empty,
            not a string, if no valid chunks are produced, or if the encoding process
            fails for any reason.

        """
        # Input validation: Ensure text is a non-empty string.
        if not isinstance(text, str) or not text.strip():
            logger.warning("Input text is empty, whitespace-only, or not a string. Cannot generate embedding.")
            return None

        text_stripped = text.strip()  # Remove leading/trailing whitespace for consistent processing.
        n = len(text_stripped)
        aggregated_embedding: Optional[torch.Tensor] = None  # Initialize to store the final embedding.

        try:
            # Case 1: Text is short enough to be encoded directly (no chunking needed).
            if n <= self.chunk_size:
                # Encode the text. `convert_to_tensor=True` ensures output is a PyTorch tensor.
                # `show_progress_bar=False` as it's a single item.
                embedding: torch.Tensor = self.model.encode(
                    text_stripped,
                    convert_to_tensor=True,
                    show_progress_bar=False, # Not useful for single text/small number of chunks
                )
                # `squeeze()` removes dimensions of size 1. For a single sentence,
                # model.encode might return (1, D); squeeze makes it (D,).
                aggregated_embedding = embedding.squeeze()
            # Case 2: Text is long and requires chunking.
            else:
                # Calculate the step size for creating chunks.
                # Chunks are `chunk_size` long, and subsequent chunks start `step` characters later.
                step = self.chunk_size - self.overlap

                # Create chunks. List comprehension iterates from start to end of text with `step`.
                # Filter out any potential empty strings that might occur if text length
                # is not perfectly divisible or if issues arise from string slicing.
                chunks = [
                    text_stripped[i : i + self.chunk_size]  # Slice the text to get a chunk
                    for i in range(0, n, step)             # Iterate with the calculated step
                    if text_stripped[i : i + self.chunk_size] # Ensure the slice is not empty
                ]
                # Further filter to remove chunks that consist only of whitespace.
                valid_chunks = [chunk for chunk in chunks if chunk.strip()]

                if not valid_chunks:
                    logger.warning(
                        f"Text (length {n}) resulted in no valid (non-empty, non-whitespace) "
                        f"chunks after processing. Text preview: '{text_stripped[:70]}...'",
                    )
                    # aggregated_embedding remains None
                else:
                    logger.debug(
                        f"Encoding [magenta]{len(valid_chunks)}[/magenta] chunks for text (length {n}). "
                        f"Preview: '{text_stripped[:70]}...'",
                    )
                    # Encode all valid chunks. `batch_size` is used by the model internally.
                    # `show_progress_bar` is enabled if there's more than one batch worth of chunks.
                    chunk_embeddings: torch.Tensor = self.model.encode(
                        valid_chunks,
                        batch_size=self.batch_size,
                        convert_to_tensor=True,
                        show_progress_bar=len(valid_chunks) > self.batch_size,
                    )

                    # Validate that the encoding process produced a non-empty tensor.
                    if chunk_embeddings.nelement() == 0: # `nelement()` is num_elements.
                        logger.error(
                            f"SentenceTransformer model encoding returned an empty tensor for chunks of text: "
                            f"'{text_stripped[:70]}...'. This may indicate an issue with the model or input.",
                        )
                        # aggregated_embedding remains None
                    else:
                        # Aggregate chunk embeddings. Currently, this is done by taking the mean
                        # of all chunk embeddings along dimension 0 (the chunk dimension).
                        # The result is a single 1D tensor of shape (D,).
                        aggregated_embedding = torch.mean(chunk_embeddings, dim=0)
                        logger.debug(
                            f"Aggregated embedding shape: "
                            f"{aggregated_embedding.shape if aggregated_embedding is not None else 'None'}. "
                            f"Number of chunks: {len(valid_chunks)}",
                        )

            # Final check: Ensure the resulting embedding is 1D.
            # A 0-dim tensor (scalar) could theoretically occur if model.encode returns a scalar
            # for very short/empty strings and squeeze() is applied, though highly unlikely
            # with standard sentence transformer models which output fixed-size vectors.
            if aggregated_embedding is not None and aggregated_embedding.dim() == 0:
                # If it's a scalar, unsqueeze it to make it a 1D tensor.
                aggregated_embedding = aggregated_embedding.unsqueeze(0)

        except Exception as e: # Catch any unexpected errors during encoding or aggregation.
            logger.exception(f"Failed to encode or aggregate text: '{text_stripped[:70]}...'. Error: {e}")
            aggregated_embedding = None # Ensure None is returned on failure.
        return aggregated_embedding

    def _resolve_requested_metrics(self, metrics_to_calculate: Optional[list[str]]) -> set[str]:
        """Processes a list of user-requested metric names into a set of valid internal metric keys.

        This method validates the requested metric names against `ALLOWED_METRICS_MAP`.
        If `metrics_to_calculate` is `None`, it defaults to calculating only cosine similarity.
        Unknown metric names are logged and ignored.

        Args:
            metrics_to_calculate (Optional[list[str]]): A list of metric names (strings)
                provided by the user (e.g., ["cosine", "euclidean"]). Case-insensitive.

        Returns:
            set[str]: A set of valid internal metric string constants (e.g.,
            `{METRIC_COSINE, METRIC_EUCLIDEAN}`) corresponding to the requested metrics.

        """
        requested_metrics_internal: set[str] = set()
        if metrics_to_calculate is None:
            # Default to cosine similarity if no specific metrics are requested.
            requested_metrics_internal.add(METRIC_COSINE)
        else:
            for metric_name in metrics_to_calculate:
                # Convert to lowercase for case-insensitive matching.
                metric_key = ALLOWED_METRICS_MAP.get(metric_name.lower())
                if metric_key:
                    requested_metrics_internal.add(metric_key)
                else:
                    # Log a warning if an unknown metric is requested.
                    logger.warning(
                        f"Unknown metric '{metric_name}' requested. It will be ignored. "
                        f"Allowed metrics are: {list(ALLOWED_METRICS_MAP.keys())}",
                    )
        return requested_metrics_internal

    def _handle_early_exit_cases(
        self,
        text1: str, # Original text1 for comparison
        text2: str, # Original text2 for comparison
        *, # Force subsequent arguments to be keyword-only for clarity
        is_text1_empty: bool,
        is_text2_empty: bool,
        requested_metrics: set[str],
    ) -> tuple[bool, Optional[SimilarityScores]]:
        """Checks for conditions that allow for an early exit without full embedding generation.

        These conditions include:
        - Texts being identical.
        - Both texts being effectively empty (whitespace-only or empty strings).
        - One text being effectively empty while the other is not.
        - No valid metrics being requested.

        Args:
            text1 (str): The first input text string.
            text2 (str): The second input text string.
            is_text1_empty (bool): True if text1 is considered empty or whitespace-only.
            is_text2_empty (bool): True if text2 is considered empty or whitespace-only.
            requested_metrics (set[str]): A set of valid internal metric keys to calculate.

        Returns:
            tuple[bool, Optional[SimilarityScores]]:
            - The first element (bool) is `True` if processing should proceed to embedding
              generation, `False` otherwise (an early exit condition was met).
            - The second element (Optional[SimilarityScores]) contains the calculated scores
              if an early exit condition provides them (e.g., for identical texts), or `None`
              if comparison is not possible (e.g., one text empty), or an empty
              `SimilarityScores` object if no metrics were requested.

        """
        scores_data: dict[str, float] = {}  # Data to populate SimilarityScores, uses internal metric keys.

        # Case 1: Texts are identical OR both are effectively empty.
        # In these scenarios, similarity is perfect (1.0 for cosine) and distances are zero.
        if text1 == text2 or (is_text1_empty and is_text2_empty):
            log_msg = "Texts are identical" if text1 == text2 else "Both texts are effectively empty"
            logger.info(f"{log_msg}. Returning perfect scores for requested metrics.")
            # Populate scores based on standard definitions for perfect similarity/zero distance.
            if METRIC_COSINE in requested_metrics:
                scores_data[METRIC_COSINE] = 1.0
            if METRIC_EUCLIDEAN in requested_metrics:
                scores_data[METRIC_EUCLIDEAN] = 0.0
            if METRIC_MANHATTAN in requested_metrics:
                scores_data[METRIC_MANHATTAN] = 0.0
            return False, SimilarityScores(**scores_data) # Early exit, scores provided.

        # Case 2: One text is effectively empty, and the other is not.
        # Meaningful semantic comparison is not possible in this state.
        if is_text1_empty != is_text2_empty: # XOR condition: one is true, the other is false.
            logger.warning(
                "One input text is effectively empty (or whitespace) while the other is not. "
                "Semantic comparison is not meaningful. Returning None.",
            )
            return False, None # Early exit, no scores possible.

        # Case 3: No valid metrics were requested by the user.
        # This could happen if `metrics_to_calculate` was an empty list or contained only invalid names.
        if not requested_metrics:
            logger.info(
                "No valid metrics were requested (or list was empty). "
                "Returning an empty SimilarityScores object (all metrics None).",
            )
            return False, SimilarityScores() # Early exit, returns an empty scores object.

        # If none of the above conditions are met, proceed to embedding generation.
        return True, None

    def _calculate_metrics_for_embeddings(
        self,
        emb1: torch.Tensor, # Assumed to be a 1D tensor
        emb2: torch.Tensor, # Assumed to be a 1D tensor
        requested_metrics: set[str],
    ) -> SimilarityScores:
        """Calculates the specified similarity/distance metrics given two 1D embedding tensors.

        Args:
            emb1 (torch.Tensor): The 1D embedding vector for the first text.
            emb2 (torch.Tensor): The 1D embedding vector for the second text.
            requested_metrics (set[str]): A set of valid internal metric keys indicating
                                          which metrics to compute.

        Returns:
            SimilarityScores: A Pydantic model instance populated with the calculated scores
                              for the requested metrics. Metrics not requested will be `None`.

        """
        scores_data: dict[str, float] = {}  # To store calculated scores with internal keys.

        # Reshape 1D embeddings to 2D (1, D) as PyTorch functions often expect batch input.
        emb1_reshaped = emb1.unsqueeze(0) # Changes shape from (D,) to (1, D)
        emb2_reshaped = emb2.unsqueeze(0) # Changes shape from (D,) to (1, D)

        # Calculate Cosine Similarity if requested.
        if METRIC_COSINE in requested_metrics:
            # `F.cosine_similarity` computes similarity along a dimension.
            # For (1,D) and (1,D) inputs, it returns a tensor with one element.
            cosine_sim_tensor = F.cosine_similarity(emb1_reshaped, emb2_reshaped)
            # Clamp the value to ensure it's strictly within [-1.0, 1.0] due to potential float precision issues.
            cosine_sim = max(-1.0, min(1.0, cosine_sim_tensor.item()))
            scores_data[METRIC_COSINE] = cosine_sim

        # Calculate Euclidean Distance (L2 norm of the difference) if requested.
        if METRIC_EUCLIDEAN in requested_metrics:
            # `torch.cdist` computes pairwise distances. `p=2.0` specifies Euclidean.
            # For (1,D) and (1,D) inputs, it returns a (1,1) tensor.
            euclidean_dist_tensor = torch.cdist(emb1_reshaped, emb2_reshaped, p=2.0)
            scores_data[METRIC_EUCLIDEAN] = euclidean_dist_tensor.item()

        # Calculate Manhattan Distance (L1 norm of the difference) if requested.
        if METRIC_MANHATTAN in requested_metrics:
            # `p=1.0` specifies Manhattan distance.
            manhattan_dist_tensor = torch.cdist(emb1_reshaped, emb2_reshaped, p=1.0)
            scores_data[METRIC_MANHATTAN] = manhattan_dist_tensor.item()

        logger.debug(f"Calculated scores from embeddings (raw data for Pydantic model): {scores_data}")
        # Populate and return the Pydantic model using the calculated scores.
        # Metrics not in scores_data will remain default (None) in SimilarityScores.
        return SimilarityScores(**scores_data)

    def calculate_similarity(
        self,
        text1: str,
        text2: str,
        metrics_to_calculate: Optional[list[str]] = None, # Default is None, handled by _resolve_requested_metrics
    ) -> Optional[SimilarityScores]:
        """Calculates specified similarity and/or distance metrics between two text strings.

        This is the main public method of the class. It orchestrates the process of:
        1. Validating requested metrics.
        2. Checking for early exit conditions (e.g., identical texts, empty inputs).
        3. Generating aggregated embeddings for both texts using `_get_aggregated_embedding`.
        4. Calculating the requested metrics based on these embeddings using `_calculate_metrics_for_embeddings`.

        Args:
            text1 (str): The first text string for comparison.
            text2 (str): The second text string for comparison.
            metrics_to_calculate (Optional[list[str]]): A list of metric names to calculate.
                Valid names are "cosine", "euclidean", "manhattan" (case-insensitive).
                If `None` (default), only "cosine" similarity is calculated.
                If an empty list `[]` is provided or if all requested metric names are
                invalid, an empty `SimilarityScores` object (all fields `None`) is
                returned, assuming no other failures prevent this (like one text being empty).

        Returns:
            Optional[SimilarityScores]: A `SimilarityScores` object containing the calculated
            metrics. Returns `None` if a critical error occurs (e.g., embedding generation
            fails for one or both texts, or if one text is empty while the other is not,
            making comparison impossible).

        """
        # Step 1: Resolve and validate the list of metrics the user wants to calculate.
        # This converts user-friendly names to internal constants and filters invalid ones.
        requested_metrics = self._resolve_requested_metrics(metrics_to_calculate)

        # Step 2: Determine if texts are effectively empty (empty string or whitespace-only).
        # `str.strip()` removes whitespace; if the result is empty, the string is effectively empty.
        is_text1_empty = not isinstance(text1, str) or not text1.strip()
        is_text2_empty = not isinstance(text2, str) or not text2.strip()

        # Step 3: Check for early exit conditions.
        # This avoids unnecessary embedding computation if results can be determined directly.
        proceed_to_embeddings, early_result = self._handle_early_exit_cases(
            text1, # Pass original text1 for direct comparison if needed
            text2, # Pass original text2 for direct comparison if needed
            is_text1_empty=is_text1_empty,
            is_text2_empty=is_text2_empty,
            requested_metrics=requested_metrics,
        )

        if not proceed_to_embeddings:
            # If `proceed_to_embeddings` is False, an early exit condition was met.
            # `early_result` will be the `SimilarityScores` object or `None`.
            return early_result

        # Step 4: If no early exit, proceed to generate embeddings.
        # This is the potentially time-consuming part.
        try:
            logger.debug("Generating embedding for text 1...")
            emb1 = self._get_aggregated_embedding(text1) # Handles chunking internally
            logger.debug("Generating embedding for text 2...")
            emb2 = self._get_aggregated_embedding(text2) # Handles chunking internally

            # Step 5: Validate the generated embeddings.
            if emb1 is None or emb2 is None:
                logger.error(
                    "Could not generate valid embeddings for one or both texts. "
                    "This might be due to empty inputs after processing or model errors.",
                )
                return None # Cannot proceed without both embeddings.

            # Ensure embeddings are 1D tensors as expected for metric calculations.
            if emb1.dim() != 1 or emb2.dim() != 1:
                logger.error(
                    f"Embeddings have unexpected dimensions after aggregation: "
                    f"emb1 shape: {emb1.shape}, emb2 shape: {emb2.shape}. Expected 1D tensors.",
                )
                return None # Metric calculations assume 1D embeddings.

            # Step 6: All checks passed, calculate metrics using the generated embeddings.
            return self._calculate_metrics_for_embeddings(emb1, emb2, requested_metrics)

        except Exception as e: # Catch any other unexpected errors during the process.
            logger.exception(
                f"An unexpected error occurred during embedding generation or metric calculation for texts: "
                f"'{text1[:70]}...' vs '{text2[:70]}...'. Error: {e}",
            )
            return None # Return None on unexpected failure.


# --- Example Usage ---
if __name__ == "__main__":
    # This block executes only when the script is run directly.
    print("Running Semantic Similarity Example...")

    # --- Configure Rich Logging for the Example ---
    logging.root.handlers.clear()  # Remove default handlers
    LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)  # Default to INFO
    logging.root.setLevel(LOG_LEVEL)

    console: Optional[Console] = None
    if _rich_available:
        rich_handler = RichHandler(
            level=LOG_LEVEL,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
            console=Console(stderr=True),  # Logs to stderr
        )
        logging.root.addHandler(rich_handler)
        console = Console()  # For printing separators
        separator = lambda: console.print("-" * 60, style="dim") if console else print("-" * 60)
        logger.info("Starting Semantic Similarity Example [bold green](using Rich logging)[/bold green]")
    else:
        # Fallback to basic logging
        logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        separator = lambda: print("-" * 60)
        logger.info("Starting Semantic Similarity Example (using standard logging)")

    try:
        # --- Configuration for the Example ---
        MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
        CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "384"))
        OVERLAP = int(os.environ.get("OVERLAP", "64"))
        BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))

        # --- Initialize Model ---
        logger.info(f"Loading Sentence Transformer model: [cyan]{MODEL_NAME}[/cyan]...")
        # Automatically select device: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: [yellow]{device}[/yellow]")
        model = SentenceTransformer(MODEL_NAME, device=device)
        logger.info("Model loaded successfully.")

        # --- Initialize Semantic Similarity Calculator ---
        semantic_calculator = SemanticCosineSimilarity(
            model=model, chunk_size=CHUNK_SIZE, overlap=OVERLAP, batch_size=BATCH_SIZE,
        )

        # --- Example Texts ---
        text_a = "The quick brown fox jumps over the lazy dog."
        text_b = "A fast, dark-colored fox leaps above a sleepy canine."
        text_c = "The weather is pleasant today, ideal for outdoor activities."
        text_empty = ""
        text_whitespace = "   "  # Effectively empty

        # --- Define Test Cases ---
        # Each case specifies a description, two texts, and which metrics to calculate.

        test_cases = [
            {"desc": "Similar Texts (Default - Cosine only)", "t1": text_a, "t2": text_b, "metrics": None},
            {
                "desc": "Similar Texts (All Metrics)",
                "t1": text_a,
                "t2": text_b,
                "metrics": ["cosine", "euclidean", "manhattan"],
            },
            {
                "desc": "Similar Texts (Cosine & Manhattan)",
                "t1": text_a,
                "t2": text_b,
                "metrics": ["cosine", "manhattan"],
            },
            {
                "desc": "Dissimilar Texts (All Metrics)",
                "t1": text_a,
                "t2": text_c,
                "metrics": list(ALLOWED_METRICS_MAP.keys()),
            },
            {
                "desc": "Identical Texts (All Metrics)",
                "t1": text_a,
                "t2": text_a,
                "metrics": ["cosine", "euclidean", "manhattan"],
            },
            {"desc": "One Empty (Default - Cosine)", "t1": text_a, "t2": text_empty, "metrics": None},
            {
                "desc": "One Whitespace (All Metrics)",
                "t1": text_a,
                "t2": text_whitespace,
                "metrics": ["cosine", "euclidean"],
            },
            {"desc": "Both Empty (Euclidean only)", "t1": text_empty, "t2": text_empty, "metrics": ["euclidean"]},
            {
                "desc": "Empty vs Whitespace (All)",
                "t1": text_empty,
                "t2": text_whitespace,
                "metrics": ["cosine", "euclidean"],
            },
            {
                "desc": "Invalid Metric Requested",
                "t1": text_a,
                "t2": text_b,
                "metrics": ["cosine", "nonexistent_metric"],
            },
            {"desc": "Empty Metrics List (No calculation)", "t1": text_a, "t2": text_b, "metrics": []},
        ]

        # --- Run Test Cases ---
        for case in test_cases:
            logger.info(f"Calculating for: [bold yellow]{case['desc']}[/bold yellow]")
            # scores_obj is now Optional[SimilarityScores]
            scores_obj: Optional[SimilarityScores] = semantic_calculator.calculate_similarity(
                case["t1"], case["t2"], metrics_to_calculate=case["metrics"],
            )

            if scores_obj is not None:
                # Use model_dump to get a dictionary of calculated scores.
                # by_alias=True gives keys like METRIC_COSINE (e.g., "cosine_similarity").
                # exclude_none=True removes metrics that were not calculated (are None).
                calculated_scores_dict = scores_obj.model_dump(by_alias=True, exclude_none=True)

                if not calculated_scores_dict:  # True if SimilarityScores() was returned and all fields are None
                    logger.info(f"Result - {case['desc']}: No metrics calculated or returned an empty set.")
                elif _rich_available:
                    output_parts = []
                    # Iterate through the dumped dictionary
                    for metric_key_alias, value in calculated_scores_dict.items():
                        metric_name_formatted = metric_key_alias.replace("_", " ").title()
                        color = "default"
                        if metric_key_alias == METRIC_COSINE:
                            color = (
                                "green"
                                if value > GOOD_SIMILARITY_SCORE
                                else "yellow"
                                if value > BAD_SIMILARITY_SCORE
                                else "red"
                            )
                        elif "distance" in metric_key_alias:
                            if value < 0.5:
                                color = "green"
                            elif value < 1.0:
                                color = "yellow"
                            else:
                                color = "red"
                        output_parts.append(f"{metric_name_formatted}=[{color}]{value:.4f}[/{color}]")
                    logger.info("Result - %s: %s", case["desc"], " | ".join(output_parts))
                else:  # Standard logging
                    log_parts = [f"Result - {case['desc']}:"]
                    for metric_key_alias, value in calculated_scores_dict.items():
                        metric_name_formatted = metric_key_alias.replace("_", " ").title()
                        log_parts.append(f"{metric_name_formatted}={value:.4f}")
                    logger.info(" | ".join(log_parts))
            else:  # scores_obj is None
                logger.warning(f"Result - {case['desc']}: Calculation Failed or Not Applicable (returned None)")
            separator()

    except ImportError:
        logger.exception("Example cannot run due to missing libraries:")
    except ValueError:
        logger.exception("Configuration error in example:")
    except Exception:
        # Catch any other unexpected errors during the example execution.
        logger.exception("An unexpected error occurred in the example:")

    logger.info("Semantic Similarity Example Finished")
