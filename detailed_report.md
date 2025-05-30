# Detailed Technical Report: Text Analysis and Similarity Toolkit

## 1. Introduction

This report provides a comprehensive overview of the architecture, code flow, and module responsibilities of the "Text Analysis and Similarity Toolkit". The toolkit is designed for advanced text similarity, feature extraction, and automated essay scoring, integrating both classical and modern NLP techniques. The following sections break down the system for technical and management audiences, highlighting extensibility, modularity, and practical usage.

---

## 2. High-Level Architecture

The toolkit is organized as a modular Python library, with each module responsible for a specific aspect of text analysis or similarity computation. The orchestration is handled by a central scoring module, with configuration and logging managed globally. The design supports both command-line execution and integration as a library in other projects.

### Key Architectural Features
- **Modular Design:** Each major feature (e.g., classical similarity, AMR, semantic, keyword, POS, plagiarism, coreference) is implemented in a separate module.
- **Central Orchestration:** The `score.py` module acts as the main orchestrator, integrating all features.
- **Configurable:** Uses Pydantic models and YAML/.env files for flexible configuration.
- **Extensible:** New metrics or analysis modules can be added with minimal changes to the core.
- **Logging:** Centralized, with optional rich output for enhanced debugging and reporting.

---

## 3. Code Flow and Execution

### 3.1. Entry Point
- **`main.py`**: The script entry point. Calls `score_essay` with sample data and logs/prints the results. Can be run directly or imported.

### 3.2. Orchestration
- **`score.py`**: Defines the `score_essay` function, which:
  - Loads configuration and initializes all calculators.
  - Calls each module's main scoring or feature extraction method.
  - Aggregates all results into a unified `EssayScores` object.
  - Returns the results for further use or reporting.

### 3.3. Configuration and Logging
- **`settings.py`**: Loads settings from YAML/.env/environment, initializes the global logger, and loads the semantic model.
- **`config.py`**: Defines all configuration models and the `get_settings()` loader.
- **`logger_utils.py`**: Sets up logging, with support for rich output if available.

---

## 4. Module Responsibilities and Features

### 4.1. Classical Text Similarity (`key_word_match.py`)
- **SimilarityCalculator**: Implements traditional metrics:
  - Sequence-based: Levenshtein, Jaro-Winkler, LCS
  - Token-based: Jaccard, Cosine (n-grams)
  - Fuzzy: RapidFuzz, FuzzyWuzzy
  - BLEU, BM25, TF-IDF (Cosine, Euclidean, Manhattan, etc.)
- **TFIDFCalculator**: Handles TF-IDF vectorization and similarity.
- **Usage**: Called by `score_essay` for baseline similarity metrics.

### 4.2. AMR Similarity (`amr_similarity.py`)
- **AMRSimilarityCalculator**: Uses Abstract Meaning Representation (AMR) graphs for deep semantic comparison.
  - Metrics: Smatch, Concept Overlap, Named Entity Overlap, Negation, Root Concept Similarity
  - Requires pre-trained AMR models (configurable path).
- **Usage**: Optional, invoked by `score_essay` if enabled in config.

### 4.3. Semantic Cosine Similarity (`semantic_match.py`)
- **SemanticCosineSimilarity**: Leverages Sentence Transformers for semantic similarity.
  - Handles long texts via chunking and aggregation.
  - Metrics: Cosine, Euclidean, Manhattan distances on embeddings.
- **Usage**: Called by `score_essay` for modern semantic similarity.

### 4.4. Keyword Matching (`keyword_matcher.py`)
- **KeywordMatcher**: Extracts keywords (with optional POS/lemmatization), computes coverage and vocabulary similarity.
- **Usage**: Used by `score_essay` for keyword-based metrics.

### 4.5. POS-based Similarity (`pos_score.py`)
- **score_pos**: Compares part-of-speech patterns (e.g., Noun-Verb-Noun) between texts using spaCy.
- **Usage**: Used by `score_essay` for syntactic similarity.

### 4.6. Advanced Text Features (`text_features.py`)
- **Graph-based Similarity**: Constructs word co-occurrence graphs, computes subgraph density.
- **Plagiarism Detection**: Implements a k-gram indexed, windowed Smith-Waterman algorithm.
- **Clustering Features**: TF-IDF vectorization, hierarchical clustering, cophenetic distance, silhouette scores.
- **Main Methods**: `run_single_pair_text_analysis`, `run_full_text_analysis` (for batch/corpus analysis).
- **Usage**: Optional, for advanced analysis or research.

### 4.7. Coreference Resolution (`coreference.py`)
- **rule_based_coref_resolution_v4**: Rule-based system using spaCy for linking pronouns/proper nouns to antecedents.
  - Implements rules for pleonastic 'it', reflexive/relative/possessive/personal pronouns, proper noun matching.
  - Helper functions: `get_gender`, `check_agreement`, `find_subject`, `find_speaker`.
- **Usage**: Optional, for feature extraction or standalone use.

### 4.8. Preprocessing (`preprocess.py`)
- Utility functions for text normalization, tokenization, lemmatization, etc.
- Used by multiple modules for consistent preprocessing.

### 4.9. Data Models (`app_types.py`)
- Pydantic models for all major data structures (inputs, outputs, configs, feature sets).
- Ensures type safety and validation across modules.

---

## 5. Configuration System

- **Centralized via Pydantic**: All settings (app, semantic, spaCy, etc.) are defined in Pydantic models.
- **Flexible Sources**: Loads from YAML files, .env files, and environment variables, with clear precedence.
- **Model Paths**: AMR, Sentence Transformer, and spaCy model paths are configurable.
- **Algorithm Parameters**: Most algorithms (TF-IDF, plagiarism, keyword extraction, etc.) are configurable via YAML/env.

---

## 6. Extensibility and Modularity

- **Adding New Metrics**: Implement a new class/function in a separate module, update `score_essay` to call it, and add results to the output model.
- **Configuration**: Add new config fields to `config.py` and YAML/env files as needed.
- **Testing/Debugging**: Each module can be run independently for development and testing (see `if __name__ == "__main__"` blocks).

---

## 7. Execution Flow (Step-by-Step)

1. **Startup**: User runs `main.py` or imports the library.
2. **Configuration**: `settings.py` loads all settings and initializes logging and models.
3. **Scoring**: `score_essay` is called with essay/reference text.
4. **Feature Extraction**: Each module computes its respective metrics/features.
5. **Aggregation**: All results are collected into a single output object (`EssayScores`).
6. **Output**: Results are returned, printed, or logged for further use.

---

## 8. Practical Usage and Recommendations

- **For Developers**: Import and use individual modules/classes for custom pipelines or research.
- **For End Users**: Run `main.py` for a full scoring pipeline, or use the toolkit as a library in larger applications.
- **For Management**: The toolkit is robust, extensible, and ready for integration into automated essay grading, plagiarism detection, and advanced text analytics platforms.

---

## 9. Summary Table: Module Responsibilities

| Module                | Main Classes/Functions                | Purpose/Features                                 |
|-----------------------|---------------------------------------|--------------------------------------------------|
| main.py               | main                                  | Entry point, demo usage                          |
| score.py              | score_essay                           | Orchestrates all scoring                         |
| key_word_match.py     | SimilarityCalculator, TFIDFCalculator | Classical similarity metrics                     |
| amr_similarity.py     | AMRSimilarityCalculator               | AMR-based similarity                             |
| semantic_match.py     | SemanticCosineSimilarity              | Sentence Transformer-based similarity            |
| keyword_matcher.py    | KeywordMatcher                        | Keyword extraction and matching                  |
| pos_score.py          | score_pos                             | POS pattern similarity                           |
| text_features.py      | run_single_pair_text_analysis, ...    | Graph, plagiarism, clustering features           |
| coreference.py        | rule_based_coref_resolution_v4        | Rule-based coreference resolution                |
| preprocess.py         | (various)                             | Text preprocessing utilities                     |
| config.py/settings.py | get_settings, settings                | Configuration and global settings                |
| logger_utils.py       | setup_global_logger                   | Logging setup                                    |
| app_types.py          | (Pydantic models)                     | Data models for all modules                      |

---

## 10. Conclusion

The "Text Analysis and Similarity Toolkit" is a powerful, modular, and extensible platform for advanced text analytics. Its architecture supports both research and production use cases, with a focus on configurability, robustness, and ease of integration. The codebase is well-structured for future enhancements and can be adapted to a wide range of NLP-driven applications.

---

*Prepared by: Automated Code Analysis (GitHub Copilot)*
*Date: May 28, 2025*
