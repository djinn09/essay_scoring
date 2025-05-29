# Code Flow and Execution Summary

This document summarizes the code flow, class/method execution, and module responsibilities for the essay scoring toolkit. It is based on the analysis of all Python files in the codebase (excluding environment/config folders).

---

## 1. **Entry Point and Orchestration**

- **`main.py`**
  - Entry point for running the toolkit as a script.
  - Calls `score_essay` from `score.py` with sample essay/reference.
  - Prints/logs the resulting scores.

- **`score.py`**
  - Central orchestrator for essay scoring.
  - Defines `score_essay(essay, reference)`:
    - Instantiates/configures all major calculators (classical, semantic, AMR, keyword, POS, etc.).
    - Calls each module's main scoring/extraction method.
    - Aggregates results into an `EssayScores` model (from `app_types.py`).

---

## 2. **Core Modules and Their Execution**

### a. **Classical Text Similarity** (`key_word_match.py`)
- **`SimilarityCalculator`**: Main class for classical metrics (Levenshtein, Jaccard, BLEU, BM25, TF-IDF, etc.).
  - Methods: `calculate_all`, `levenshtein`, `jaccard`, `tfidf_cosine`, etc.
- **`TFIDFCalculator`**: Handles TF-IDF vectorization and similarity.
- Used by: `score_essay` (via `SimilarityCalculator`).

### b. **AMR Similarity** (`amr_similarity.py`)
- **`AMRSimilarityCalculator`**: Computes AMR-based similarity (Smatch, concept overlap, etc.).
  - Uses a pre-loaded `STOG_MODEL` (AMR parser).
- Used by: `score_essay` (if AMR enabled).

### c. **Semantic Cosine Similarity** (`semantic_match.py`)
- **`SemanticCosineSimilarity`**: Uses Sentence Transformers for semantic similarity.
  - Handles chunking, batching, and device management.
- Used by: `score_essay` (via `SemanticCosineSimilarity`).

### d. **Keyword Matching** (`keyword_matcher.py`)
- **`KeywordMatcher`**: Extracts keywords, computes coverage and vocabulary similarity.
- Used by: `score_essay` (via `KeywordMatcher`).

### e. **POS-based Similarity** (`pos_score.py`)
- **`score_pos`**: Compares POS patterns (e.g., Noun-Verb-Noun) between texts.
- Used by: `score_essay`.

### f. **Advanced Text Features** (`text_features.py`)
- **Graph-based similarity**: Word co-occurrence graphs, subgraph density.
- **Plagiarism detection**: Smith-Waterman algorithm, k-gram windowing.
- **Clustering features**: TF-IDF, hierarchical clustering, silhouette scores.
- **Main methods/classes**:
  - `run_single_pair_text_analysis`, `run_full_text_analysis` (for batch/corpus analysis).
  - `FullTextAnalysisInput`, `FullTextAnalysisResult` (Pydantic models for input/output).
- Used optionally in `score_essay` or for deeper analysis.

### g. **Coreference Resolution** (`coreference.py`)
- **`rule_based_coref_resolution_v4`**: Rule-based system using spaCy for linking pronouns/proper nouns to antecedents.
  - Implements rules for pleonastic 'it', reflexive/relative/possessive/personal pronouns, proper noun matching.
  - Helper functions: `get_gender`, `check_agreement`, `find_subject`, `find_speaker`.
- Used optionally for feature extraction or as a standalone module.

### h. **Preprocessing** (`preprocess.py`)
- Utility functions for text normalization, tokenization, lemmatization, etc.
- Used by multiple modules for consistent preprocessing.

### i. **Configuration and Settings**
- **`config.py`**: Defines Pydantic models for all configuration (app, semantic, spacy, etc.).
  - `get_settings()` loads settings from YAML, .env, and environment variables.
- **`settings.py`**: Loads settings, initializes global logger, and semantic model.
- **`logger_utill.py`**: Sets up logging (optionally with rich output).

### j. **Data Models** (`app_types.py`)
- Pydantic models for all major data structures (inputs, outputs, configs, feature sets).
- Used throughout the codebase for type safety and validation.

---

## 3. **Execution Flow (Typical Run)**

1. **Startup**: `main.py` calls `score_essay`.
2. **Configuration**: `settings.py` loads config, initializes logger and semantic model.
3. **Scoring**: `score_essay` orchestrates all calculators:
   - Classical metrics via `SimilarityCalculator`.
   - Semantic similarity via `SemanticCosineSimilarity`.
   - AMR similarity via `AMRSimilarityCalculator` (if enabled).
   - Keyword and POS-based metrics.
   - Optionally, advanced features (plagiarism, clustering, coreference).
4. **Aggregation**: Results are collected into an `EssayScores` object.
5. **Output**: Scores are returned, printed, or logged.

---

## 4. **Class/Method Execution Summary**

- **Each major class** (e.g., `SimilarityCalculator`, `AMRSimilarityCalculator`, `SemanticCosineSimilarity`, `KeywordMatcher`) exposes a main method (usually `calculate_*` or `score_*`) that is called by the orchestrator (`score_essay`).
- **Helper classes/functions** (e.g., `TFIDFCalculator`, `run_single_pair_text_analysis`, `rule_based_coref_resolution_v4`) are used internally by the main classes or for advanced analysis.
- **Configuration and logging** are initialized at startup and used throughout.

---

## 5. **Summary Table: Module Responsibilities

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
| logger_utill.py       | setup_global_logger                   | Logging setup                                    |
| app_types.py          | (Pydantic models)                     | Data models for all modules                      |

---

## 6. **Notes**
- Each module can be run/tested independently (see `if __name__ == "__main__"` blocks).
- Most modules rely on configuration loaded at startup.
- The codebase is modular and extensible for new metrics/features.

---

*Generated by automated code analysis.*
