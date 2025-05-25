"""
Settings and Global Initialization Module for Essay Grading.

This module is responsible for:
1. Retrieving application-wide configuration settings using the `config.py` module.
2. Setting up global logging using a utility from `logger_utill.py`.
   (Note: `logger_utill.py` must be present in the project).
3. Initializing and making globally available the primary SentenceTransformer model
   used for semantic similarity calculations.
4. Initializing a default configuration for `SimilarityCalculator`.

The objects initialized here (like `settings`, `semantic_model`, `similarity_config`)
are intended to be imported and used by other modules in the application.
"""

import logging
import os

from sentence_transformers import SentenceTransformer

from app_types import SimilarityCalculatorConfig # Pydantic model for SimilarityCalculator configuration.
from config import get_settings # Function to load comprehensive application settings.
# The following import assumes `logger_utill.py` exists and provides `setup_global_logger`.
# If `logger_utill.py` is missing, this line will cause an ImportError.
from logger_utill import setup_global_logger # Utility for global logger configuration.

# Load application settings from config.py (which handles .env, YAML, etc.)
settings = get_settings()

# Determine the log level to use, prioritizing environment variable over config file.
log_level_to_use = os.getenv("LOG_LEVEL", settings.app.log_level).upper()

# --- Global Logger Setup ---
# This sets up the root logger for the application.
# It relies on `setup_global_logger` from `logger_utill.py`.
# If `logger_utill.py` or the function is missing, this will fail.
try:
    setup_global_logger(
        log_level=log_level_to_use,
        app_name=settings.app.name, # Use app name from settings for the logger.
    )
except NameError:
    logging.basicConfig(level=log_level_to_use) # Fallback to basicConfig if setup_global_logger is not found
    logging.warning("`setup_global_logger` not found (likely `logger_utill.py` is missing). Using basic logging configuration.")
except Exception as e:
    logging.basicConfig(level=log_level_to_use)
    logging.error(f"Failed to setup global logger via `setup_global_logger`: {e}. Using basic logging configuration.")


# Get a logger instance for this settings module, using the globally configured logger.
# The name of the logger is derived from the application name in settings.
app_logger = logging.getLogger(settings.app.name)
app_logger.info(f"Application settings loaded successfully for environment: '{settings.env}'")
# Log the full settings object in debug mode for troubleshooting.
app_logger.debug(f"Full application settings object: {settings.model_dump_json(indent=2)}")


# --- Initialize Global Semantic Model ---
# This loads the SentenceTransformer model at import time, making it globally available.
# This is a significant side effect: importing this module will trigger model download
# and loading if the model isn't cached, which can be time-consuming.
# For applications requiring faster startup or conditional model loading,
# consider deferring this to a dedicated initialization function or class.
app_logger.info(
    f"Initializing global semantic model '{settings.semantic.model_name}' on device '{settings.semantic.device}'...",
)
try:
    semantic_model = SentenceTransformer(
        settings.semantic.model_name,
        device=settings.semantic.device,
    )
    app_logger.info(f"Global semantic model '{settings.semantic.model_name}' initialized successfully.")
except Exception as e:
    app_logger.exception(
        f"Failed to initialize global semantic model '{settings.semantic.model_name}'. "
        f"Semantic similarity features will likely fail. Error: {e}"
    )
    semantic_model = None # Ensure variable exists even if loading fails.


# --- Initialize Default SimilarityCalculator Configuration ---
# This creates a default configuration instance for the SimilarityCalculator.
# Note: This specific `similarity_config` instance uses the defaults from
# `SimilarityCalculatorConfig` Pydantic model and is NOT directly populated by values
# from the main `settings` object (e.g., from YAML or .env files for these specific fields).
# If `SimilarityCalculatorConfig` needs to be configurable via the main settings mechanism,
# it should be nested within the `Settings` model in `config.py`.
similarity_config = SimilarityCalculatorConfig()
app_logger.debug(
    f"Default SimilarityCalculatorConfig initialized: {similarity_config.model_dump_json(indent=2)}"
)
