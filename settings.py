"""
Settings and Global Initialization Module for Essay Grading.

This module is responsible for:
1. Retrieving application-wide configuration settings using the `config.py` module.
2. Setting up global logging using a utility from `logger_utils.py`.
   (Note: `logger_utils.py` must be present in the project).
3. Initializing and making globally available the primary SentenceTransformer model
   used for semantic similarity calculations.
4. Initializing a default configuration for `SimilarityCalculator`.

The objects initialized here (like `settings`, `semantic_model`, `similarity_config`)
are intended to be imported and used by other modules in the application.
"""

import logging
import os

from sentence_transformers import SentenceTransformer

from app_types import SimilarityCalculatorConfig
from config import get_settings
from logger_utils import setup_global_logger

settings = get_settings()

log_level_to_use = os.getenv("LOG_LEVEL", settings.app.log_level).upper()

try:
    setup_global_logger(
        log_level=log_level_to_use,
        app_name=settings.app.name,
    )
except NameError:
    logging.basicConfig(level=log_level_to_use)
    logging.warning(
        "`setup_global_logger` not found (likely `logger_utils.py` is missing). Using basic logging configuration.",
    )
except Exception:
    logging.basicConfig(level=log_level_to_use)
    logging.exception(
        "Failed to setup global logger via `setup_global_logger`. Using basic logging configuration.",
    )


app_logger = logging.getLogger(settings.app.name)
app_logger.info(f"Application settings loaded successfully for environment: '{settings.env}'")
app_logger.debug(f"Full application settings object: {settings.model_dump_json(indent=2)}")


app_logger.info(
    f"Initializing global semantic model '{settings.semantic.model_name}' on device '{settings.semantic.device}'...",
)
try:
    semantic_model = SentenceTransformer(
        settings.semantic.model_name,
        device=settings.semantic.device,
    )
    app_logger.info(f"Global semantic model '{settings.semantic.model_name}' initialized successfully.")
except Exception:
    app_logger.exception(
        f"Failed to initialize global semantic model '{settings.semantic.model_name}'. "
        f"Semantic similarity features will likely fail.",
    )
    semantic_model = None

similarity_config = SimilarityCalculatorConfig()
app_logger.debug(
    f"Default SimilarityCalculatorConfig initialized: {similarity_config.model_dump_json(indent=2)}",
)
