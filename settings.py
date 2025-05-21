"""Settings module for essay grading.

This module initializes the semantic model using SentenceTransformer
and retrieves configuration settings.
"""

import logging
import os

from sentence_transformers import SentenceTransformer

from app_types import SimilarityCalculatorConfig
from config import get_settings
from logger_utill import setup_global_logger

settings = get_settings()

log_level_to_use = os.getenv("LOG_LEVEL", settings.app.log_level).upper()

# The 'app_name' for the logger comes from settings
setup_global_logger(
    log_level=log_level_to_use,
    app_name=settings.app.name,  # Use app name from settings for the initial log message
)

# Get a logger instance for this core setup module.
# This logger will use the global configuration.
app_logger = logging.getLogger(settings.app.name)  # Use the app name from settings as the logger name
app_logger.info(f"Application settings loaded for env: {settings.env}")
app_logger.debug(f"Full settings object: {settings.model_dump_json(indent=2)}")


# --- 3. Initialize Semantic Model ---
app_logger.info(
    f"Initializing semantic model '{settings.semantic.model_name}' on device '{settings.semantic.device}'...",
)

similarity_config = SimilarityCalculatorConfig()

semantic_model = SentenceTransformer(
    settings.semantic.model_name,
    device=settings.semantic.device,
)
