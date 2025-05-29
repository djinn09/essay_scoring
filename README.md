# Text Analysis and Similarity Toolkit

## Overview
A Python library providing a comprehensive suite of tools for various text similarity and analysis tasks. It is designed to support applications such as automated essay grading, plagiarism detection, semantic comparison, and general text feature extraction. The toolkit integrates classical lexical and syntactic methods with modern semantic approaches using AMR and Sentence Transformers.

## Key Features
*   **Classical Text Similarity Metrics (`key_word_match.py`):**
    *   Calculates a wide array of traditional similarity and distance metrics, including:
        *   Sequence-based: Levenshtein Distance, Jaro-Winkler, Longest Common Subsequence (LCS).
        *   Token-based: Jaccard Index, Cosine Similarity (on character n-grams).
        *   Fuzzy Matching: RapidFuzz (Ratio, Partial Ratio, Token Set/Sort Ratio) and optionally FuzzyWuzzy.
    *   NLP-inspired: BLEU score and BM25 score.
    *   TF-IDF based: Cosine Similarity, Euclidean/Manhattan/Minkowski/Jaccard/Hamming distances on TF-IDF vectors.
    *   Supports parallel processing for multiple text pair comparisons.
*   **AMR Similarity (`amr_similarity.py`):**
    *   Calculates text similarity based on Abstract Meaning Representation (AMR) graphs.
    *   Metrics: Smatch (F-score, Precision, Recall), Concept Overlap, Named Entity Overlap, Negation Overlap, Root Concept Similarity.
    *   Requires `amrlib` and pre-trained AMR parsing models.
*   **Semantic Cosine Similarity (`semantic_match.py`):**
    *   Computes semantic similarity using Sentence Transformer models.
    *   Handles long texts via automated chunking and embedding aggregation.
    *   Metrics: Cosine Similarity, Euclidean Distance, Manhattan Distance.
*   **Advanced Text Feature Extraction & Analysis (`text_features.py`):**
    *   Word co-occurrence graph construction (from document-term matrix) and graph-based similarity (subgraph density).
    *   Plagiarism detection using a k-gram indexed, windowed Smith-Waterman algorithm.
    *   Clustering-based lexical features: TF-IDF vectorization, hierarchical clustering, cophenetic distance, and silhouette scores for analyzing collections of texts.
*   **Keyword Matching (`keyword_matcher.py`):**
    *   Extracts keywords from a source text (optionally using POS tagging and lemmatization).
    *   Calculates Keyword Coverage: Proportion of source keywords found in a target text.
    *   Calculates Vocabulary Cosine Similarity: Based on shared non-stopword vocabularies.
*   **Coreference Resolution (`coreference.py`):**
    *   Rule-based system using spaCy to identify and link coreferent mentions (pronouns, proper nouns).
    *   Includes rules for pleonastic 'it', reflexive/relative/possessive/personal pronouns, and proper noun matching.
*   **POS-based Similarity (`pos_score.py`):**
    *   Extracts and compares specific Part-of-Speech (POS) patterns (triplets and duplets like Noun-Verb-Noun) between texts.
    *   Uses spaCy for POS tagging and word vector similarity for comparing pattern components.
*   **(Optional/Disabled by default) spaCy-based Semantic Graphs:**
    *   Functionality exists in `text_features.py` for creating semantic graphs from spaCy dependency parses and calculating similarity between them (currently commented out in main analysis pipelines).
*   **Configurable:** Uses Pydantic models for structured input/output. Key parameters for various components (TF-IDF, semantic models, plagiarism detection) can be configured via YAML files and environment variables, managed by `config.py` and `settings.py`.

## Code Flow
*   **Entry Point:** `main.py` demonstrates a basic usage pattern, calling `score_essay` from `score.py`.
*   **Orchestration:** The `score.py` module is the primary orchestrator, integrating various calculators and feature extractors to produce a comprehensive set of scores for an essay compared against a reference text.
*   **Core Modules:**
    *   `key_word_match.py`: Provides the `SimilarityCalculator` for a wide range of classical text similarity metrics.
    *   `amr_similarity.py`: Handles AMR-based similarity calculations.
    *   `semantic_match.py`: Manages Sentence Transformer-based semantic similarity, including text chunking.
    *   `text_features.py`: Contains tools for plagiarism detection (Smith-Waterman), word co-occurrence graph analysis, and clustering-based lexical feature extraction.
    *   `keyword_matcher.py`: Implements keyword extraction, coverage scoring, and vocabulary cosine similarity.
    *   `coreference.py`: Provides rule-based coreference resolution.
    *   `pos_score.py`: Calculates similarity based on POS patterns.
    *   `preprocess.py`: Contains various text preprocessing utilities used by different modules.
    *   `config.py` & `settings.py`: Manage application configuration, including model paths, parameters for algorithms, and global initialization of models like SentenceTransformers and spaCy.
    *   `app_types.py`: Defines Pydantic models for data structures (inputs, outputs, configurations) used throughout the project, ensuring data consistency.
*   **Logging:** Logging is set up in `settings.py`, intended to use a utility from `logger_utill.py` (if available) or falls back to standard Python logging.

## Installation and Setup

It's highly recommended to use a virtual environment (like Python's built-in `venv` or `conda`) to manage project dependencies.

### 1. Create and Activate a Virtual Environment

**Using `venv`:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

**Using `conda`:**
```bash
conda create -n essayenv python=3.12  # Ensure Python 3.12+ as per pyproject.toml
conda activate essayenv
```

### 2. Install Dependencies

This project's dependencies are defined in `pyproject.toml`.

**Using `pip` (with `uv` for PyTorch CPU version):**

If you have `uv` (a fast Python package installer and resolver) installed, it can respect the PyTorch CPU source specified in `pyproject.toml`:
```bash
pip install uv
uv pip install .
```

If you are using `pip` directly without `uv`, you might need to install PyTorch separately or ensure your `pip` version correctly handles the source directives if possible. For a direct `pip` install from the project root (where `pyproject.toml` is located):
```bash
pip install .
```
This will install all dependencies listed in `pyproject.toml`. Key libraries include:
`amrlib`, `penman`, `torch` (CPU version specified), `sentence-transformers`, `scikit-learn`, `networkx`, `numpy`, `pydantic`, `scipy`, `nltk`, `spacy`, `rapidfuzz`, `fuzzywuzzy`, `rank_bm25`, `gender-guesser`, `textstat`.

Refer to `pyproject.toml` for the complete and versioned list of dependencies.

### 3. Download NLTK Resources

Certain features require NLTK data. Download them by running the following in your Python environment:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
```
(The `quiet=True` flag is used in the code for non-interactive downloads; running these commands directly is also fine.)

### 4. Download AMR Models (`amrlib`)

The AMR similarity features require pre-trained parsing models from `amrlib`.
```bash
python -m amrlib.models.download_models
```
After downloading, you need to configure the model path.
*   **Current Implementation:** The path is hardcoded in `amr_similarity.py` near the line `STOG_MODEL = amrlib.load_stog_model(...)`. You **must** update this path to your downloaded model location (e.g., if you download to a `./models/amr` directory, it might be `./models/amr/model_parse_xfm_bart_base-v0_1_0`).
*   **Recommendation:** For robust applications, this path should be made configurable (e.g., via an environment variable loaded in `config.py` or a path in your `dev.yaml`).

### 5. Download SpaCy Models (for Coreference and other spaCy-dependent features)

Features like coreference resolution and some POS tagging rely on spaCy and its models.
1.  **Install spaCy** (if not already installed as part of project dependencies):
    ```bash
    pip install spacy
    ```
2.  **Download a spaCy model** (e.g., `en_core_web_sm` or `en_core_web_md` as configured):
    ```bash
    python -m spacy download en_core_web_sm
    # or for the medium model often used:
    python -m spacy download en_core_web_md
    ```
    The specific model configured can be found in your `envs/*.yaml` file or `config.py` under `spacy_config`. The default in `config.py` is `en_core_web_sm`, but `envs/dev.yaml` specifies `en_core_web_md`. Ensure the correct one is downloaded.
3.  **Note on spaCy features in `text_features.py`:** Some advanced spaCy-based features (like semantic graph similarity using dependency parses) are defined in `text_features.py` but are currently commented out in the main analysis pipelines. To use them, you would need to uncomment the relevant sections in that file.

### 6. Environment Configuration
This project uses YAML files (in the `envs/` directory, e.g., `dev.yaml`) and `.env` files (e.g., `dev.env`) for managing environment-specific settings. The active environment is typically determined by the `APP_ENV` environment variable.
*   Copy or rename `envs/dev.yaml.example` to `envs/dev.yaml` and `dev.env.example` to `dev.env` if they exist and modify them as needed.
*   Ensure model paths and other configurations in these files are correct for your setup.

## How to Run

### Running the Main Example
The `main.py` script provides a simple example of how to use the `score_essay` function (defined in `score.py`). You can run it directly:
```bash
python main.py
```
This will process a sample essay against a reference text and output the scores based on the configurations in `settings.py` and `config.py`.

### Exploring Individual Modules
Many modules contain their own executable blocks (`if __name__ == "__main__":`) that demonstrate their specific functionalities in more detail. These are excellent for understanding the capabilities of each component:

*   `python amr_similarity.py`: Shows AMR similarity calculations for example texts. (Requires AMR model path to be correctly set in the script itself).
*   `python semantic_match.py`: Demonstrates semantic similarity using Sentence Transformers with various options, configurable via environment variables for the example.
*   `python text_features.py`: Provides examples of graph-based analysis, plagiarism detection, and lexical feature extraction.
*   `python key_word_match.py`: Shows examples of classical text similarity metrics using the `SimilarityCalculator`.
*   `python keyword_matcher.py`: Illustrates keyword extraction and matching.
*   `python coreference.py`: Demonstrates the rule-based coreference resolution system (uses a spaCy model loaded within the script).
*   `python pos_score.py`: Shows POS-based similarity scoring (uses a spaCy model loaded via `config.py`).
*   `python config.py`: Contains an example that tests and demonstrates the configuration loading mechanism itself (e.g., from YAML and .env files).

### Using as a Library
To integrate this toolkit into your own project, you can import the necessary classes and functions from the various modules. For example, to use the AMR similarity calculator:

```python
from amr_similarity import AMRSimilarityCalculator, STOG_MODEL 
# Note: STOG_MODEL is loaded within amr_similarity.py using a hardcoded path by default.
# Ensure this path is correct or modify the loading mechanism.

if STOG_MODEL is not None:
    calculator = AMRSimilarityCalculator(stog_model=STOG_MODEL)
    scores = calculator.calculate_amr_features("text one", "text two")
    print(scores)
else:
    print("AMR STOG_MODEL not loaded. Cannot run AMR similarity example.")
```
Always check how models and configurations are loaded in each module if you intend to use them independently. The `settings.py` module attempts to centralize some model loading and configuration.

## Configuration

Effective configuration is crucial for optimal performance and accuracy.

### 1. Primary Configuration Method (`config.py` & `settings.py`)

*   **Centralized Settings:** The project uses `config.py` to define Pydantic models for settings and `get_settings()` function to load them. `settings.py` calls `get_settings()` to create a global `settings` object.
*   **Sources & Precedence:**
    1.  **YAML Files:** Configuration is primarily loaded from YAML files in the `envs/` directory (e.g., `envs/dev.yaml`). Values here generally take precedence.
    2.  **`.env` Files:** Environment-specific `.env` files (e.g., `dev.env`) are loaded into process environment variables. Variables already in the environment usually take precedence over `.env` file values (unless `override=True` is used in `load_dotenv`).
    3.  **Environment Variables:** Actual process environment variables.
    4.  **Model Defaults:** Default values defined in the Pydantic models in `config.py`.
*   **Environment Selection:** The `APP_ENV` environment variable (defaulting to "dev") determines which configuration files are loaded (e.g., `dev.env`, `envs/dev.yaml`).
*   **Key Configurable Parameters (via YAML/Env in `config.py`):**
    *   Application settings: `app.name`, `app.version`, `app.debug`, `app.log_level`.
    *   Semantic model settings: `semantic.model_name`, `semantic.chunk_size`, `semantic.overlap`, `semantic.batch_size`, `semantic.device`.
    *   SpaCy model settings: `spacy_config.model_name`, `spacy_config.batch_size`, `spacy_config.device`.

### 2. Model Path Configurations

*   **AMR Model Path (`amr_similarity.py`):**
    *   The path to the `amrlib` StoG parsing model (`STOG_MODEL`) is **currently hardcoded** within the `amr_similarity.py` module itself.
    *   **Action Required:** You **must** modify the `model_dir` parameter in the `amrlib.load_stog_model()` call within `amr_similarity.py` to point to your downloaded AMR models.
    *   The `config.py` system does *not* currently override this hardcoded path for `STOG_MODEL` used in `amr_similarity.py`. The comment in `amr_similarity.py` about `AMR_MODEL_PATH` in `config.py` seems to be a note for future improvement rather than current functionality for this specific model instance.
*   **Sentence Transformer Models (`semantic_match.py`, loaded via `settings.py`):**
    *   The primary Sentence Transformer model is configured via `settings.semantic.model_name` (from `config.py`). This can be set in `envs/dev.yaml` or via the `SEMANTIC__MODEL_NAME` environment variable.
*   **SpaCy Models (used by `coreference.py`, `pos_score.py`, loaded via `config.py`/`settings.py`):**
    *   The spaCy model is configured via `settings.spacy_config.model_name`. This can be set in `envs/dev.yaml` or via the `SPACY_CONFIG__MODEL_NAME` environment variable.

### 3. Algorithm-Specific Parameters & Other Settings

*   **Similarity Metrics (`key_word_match.py`):**
    *   The `SimilarityCalculatorConfig` (defined in `app_types.py`) is used to configure the `SimilarityCalculator`. An instance of this is created in `settings.py` as `similarity_config`, currently using default values from `SimilarityCalculatorConfig`.
    *   To customize TF-IDF parameters (token pattern, n-gram range, min/max df) or preprocessing (lemmatization, stopwords) for the main `SimilarityCalculator` used in `score.py`, you would ideally modify `settings.py` to populate `similarity_config` from the main `settings` object, or directly adjust the `SimilarityCalculatorConfig` instantiation there.
*   **Plagiarism Detection (`text_features.py`):**
    *   The `SmithWatermanConfig` model (in `text_features.py`) defines parameters like `k`, `window_radius`, `match_score`, etc. These are typically passed as arguments when calling `compute_plagiarism_score_fast` or `run_single_pair_text_analysis`. The `FullTextAnalysisInput` model also has fields for `plagiarism_k` and `plagiarism_window_radius`.
*   **Keyword Matching (`keyword_matcher.py`):**
    *   The `KeywordMatcherConfig` (in `app_types.py`) allows configuration of lemmatization, POS tagging, allowed POS tags, and custom stopwords for keyword extraction. An instance is created in `score.py`.
*   **Coreference Resolution (`coreference.py`):**
    *   Parameters like `similarity_threshold`, `use_similarity_fallback`, and `search_sentences` can be adjusted when calling `rule_based_coref_resolution_v4`.
*   **POS Scoring (`pos_score.py`):**
    *   `SIMILARITY_THRESHOLD` is a global constant in this module affecting matching logic.
*   **Semantic Match (`semantic_match.py`):**
    *   The `if __name__ == "__main__":` block in `semantic_match.py` shows that `SemanticCosineSimilarity` is configured with `chunk_size`, `overlap`, `batch_size` which are read from environment variables like `CHUNK_SIZE` for that specific example. The main instantiation in `score.py` uses values from the global `settings.semantic` object.

For most parameters, refer to the Pydantic models in `app_types.py` and `config.py`, and the signatures/docstrings of specific functions or classes in their respective modules.
