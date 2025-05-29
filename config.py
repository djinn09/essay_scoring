"""Configuration module for the Essay Grading application.

This module defines configuration models, settings, and utilities for loading
and validating application settings from environment variables, .env files, and YAML files.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import spacy
import yaml
from dotenv import load_dotenv  # Import python-dotenv

# For Pydantic V2 with pydantic-settings
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Module-level Constants ---
VALID_ENVS: set[str] = {"dev", "prod"}  # Define valid environments here


# --- Configuration Models ---


class AppConfig(BaseModel):
    """Application configuration settings.

    Attributes:
        name (str): The name of the application.
        version (str): The version of the application.
        debug (bool): Flag to enable or disable debug mode.
        log_level (str): The logging level for the application (e.g., INFO, DEBUG).

    """

    name: str = Field("Essay Grader", description="The name of the application.")
    version: str = Field("1.0", description="The version of the application.")
    debug: bool = Field(default=False, description="Whether debug mode is enabled, typically more verbose.")
    log_level: str = Field("INFO", description="The logging level (e.g., DEBUG, INFO, WARNING).")


class SemanticConfig(BaseModel):
    """Configuration for semantic similarity calculations.

    Attributes:
        model_name (str): Name or path of the SentenceTransformer model to be used.
        chunk_size (int): Target character size for splitting texts into chunks before embedding.
        overlap (int): Number of characters to overlap between adjacent chunks.
        batch_size (int): Batch size for encoding texts/chunks with the SentenceTransformer model.
        device (Literal["cpu", "cuda", "mps"]): The hardware device to run the model on.

    """

    model_name: str = Field("all-MiniLM-L6-v2", description="The name or path of the SentenceTransformer model.")
    chunk_size: int = Field(384, description="Size of text chunks for processing by the semantic model.")
    overlap: int = Field(64, description="Overlap size between text chunks for semantic processing.")
    batch_size: int = Field(32, description="Batch size for semantic model inference.")
    device: Literal["cpu", "cuda", "mps"] = Field(
        "cpu", description="Device to run the semantic model on ('cpu', 'cuda', 'mps').",
    )

    class Config:
        """Configuration class for SemanticConfig.

        Attributes:
            env_prefix (str): The prefix for environment variables.

        """

        env_prefix = "SEMANTIC_"


class SpacyConfig(BaseModel):
    """Configuration for spaCy model.

    Attributes:
        model_name (str): The name or path of the spaCy model to load (e.g., "en_core_web_sm").
        batch_size (int): Batch size for spaCy's NLP processing pipeline if applicable (e.g., nlp.pipe).
        device (Literal["cpu", "cuda"]): The preferred hardware device for spaCy ('cpu' or 'cuda' if available).

    """

    model_name: str = Field(
        "en_core_web_sm", description="The name of the spaCy model to use (e.g., 'en_core_web_sm').",
    )
    batch_size: int = Field(32, description="Batch size for spaCy processing pipelines.")
    device: Literal["cpu", "cuda"] = Field("cpu", description="Device preference for spaCy ('cpu' or 'cuda').")

    class Config:
        """Configuration class for SpacyConfig.

        Attributes:
            env_prefix (str): The prefix for environment variables.

        """

        env_prefix = "SPACY_"


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        env (str): The environment to run in (dev, prod, etc.).
        app (AppConfig): Application configuration.
        semantic (SemanticConfig): Semantic similarity configuration.

    """

    env: str = "dev"
    app: AppConfig = AppConfig()
    semantic: SemanticConfig = SemanticConfig()
    spacy_config: SpacyConfig = SpacyConfig()
    model_config = SettingsConfigDict(
        env_file=None,  # Explicitly disable default .env loading by pydantic-settings
        # if we manually load first.
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # For loading nested Pydantic models from env vars like APP__DEBUG=True
        extra="ignore",  # Ignore extra fields from environment or YAML not defined in the model.
    )

    # Note: Placing logic like spacy.prefer_gpu() directly in the class body like this
    # means it executes when the class is defined. This might be too early if `spacy_config`
    # itself is meant to be loaded from env/YAML.
    # A better place might be after settings are fully loaded, or in an __init__ if conditional.
    # However, Pydantic BaseSettings usually don't have complex __init__.
    # For now, this reflects the original structure.
    if spacy_config.device == "cuda":  # This check uses the default SpacyConfig.device value here.
        spacy.prefer_gpu()  # type: ignore # noqa: PGH003
    # --- Load SpaCy Model (used for both coreference resolution and POS similarity scoring) ---
    # This model loading is done at import time of this config module.

    @field_validator("env")  # Validates the 'env' field after it's populated.
    @classmethod  # Needs to be a classmethod for Pydantic v2 validators.
    def check_env_is_valid(cls, v: str) -> str:
        """Validate the env field and ensure it is a valid environment.

        Args:
            v (str): The value of the env field.

        Raises:
            ValueError: If the environment is not valid.

        Returns:
            str: The validated environment.

        """
        env_lower = v.lower()
        if env_lower not in VALID_ENVS:
            msg = f"Invalid environment '{v}'. Must be one of {VALID_ENVS}"
            raise ValueError(msg)
        return env_lower


# --- Settings Loading Function ---


def _load_env_file(effective_env: str, config_dir: Path) -> None:
    """Loads environment-specific .env file."""
    env_filename = f"{effective_env}.env"
    env_file_path = config_dir / env_filename
    if env_file_path.exists() and env_file_path.is_file():
        logging.info(f"Loading environment variables from: {env_file_path}")
        load_dotenv(dotenv_path=env_file_path, override=False)
    else:
        logging.info(f"Info: Environment file not found at {env_file_path}. Skipping manual .env load.")


def _load_yaml_config(effective_env: str, config_dir: Path) -> dict:
    """Loads YAML configuration file."""
    yaml_config_path = config_dir / "envs" / f"{effective_env}.yaml"
    file_config = {}
    if yaml_config_path.exists():
        try:
            with yaml_config_path.open("r") as f:
                loaded_yaml = yaml.safe_load(f)
                if isinstance(loaded_yaml, dict):
                    file_config = loaded_yaml
                    logging.info(f"Successfully loaded YAML config from: {yaml_config_path}")
                else:
                    logging.warning(f"Warning: YAML file {yaml_config_path} did not contain a dictionary. Ignoring.")
        except yaml.YAMLError:
            logging.exception(f"Error parsing YAML file {yaml_config_path}")
        except Exception:
            logging.exception(f"Error reading file {yaml_config_path}")
    else:
        logging.info(f"Info: YAML config file not found at {yaml_config_path}. Skipping.")
    return file_config


def _validate_and_log_env_settings(settings: Settings, effective_env: str, file_config: dict) -> None:
    """Validates and logs environment settings."""
    if settings.env != effective_env and (
        "env" not in file_config or file_config.get("env", "").lower() != settings.env.lower()
    ):
        logging.info(
            f"Warning: Final settings.env ('{settings.env}') differs from the "
            f"loading environment ('{effective_env}'). This might happen if environment "
            f"variables (ENV, not from {effective_env}.env if override=False) set 'ENV' differently.",
        )
    elif (
        settings.env != effective_env
        and "env" in file_config
        and file_config.get("env", "").lower() == settings.env.lower()
    ):
        pass  # YAML explicitly set it, already warned.
    else:
        logging.info(f"Settings.env correctly reflects loading environment: '{settings.env}'")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings."""
    effective_env = os.getenv("APP_ENV", "dev").lower()
    if effective_env not in VALID_ENVS:
        logging.info(f"Warning: APP_ENV='{effective_env}' is not one of {VALID_ENVS}. Falling back to 'dev'.")
        effective_env = "dev"

    logging.info(f"--- Loading settings for environment: '{effective_env}' ---")

    config_dir = Path(__file__).parent

    _load_env_file(effective_env, config_dir)
    file_config = _load_yaml_config(effective_env, config_dir)

    try:
        init_data = file_config.copy()
        if "env" not in init_data:
            init_data["env"] = effective_env
        elif init_data.get("env", "").lower() != effective_env:
            logging.warning(
                f"Warning: YAML file specifies 'env: {init_data['env']}', "
                f"which differs from the loading environment '{effective_env}'. "
                f"The YAML value will be used for settings.env.",
            )

        settings = Settings(**init_data)
        _validate_and_log_env_settings(settings, effective_env, file_config)

        logging.info("Settings loaded successfully.")
        return settings
    except ValidationError:
        logging.exception("Error validating settings")
        msg = "Failed to load or validate application settings."
        raise SystemExit(msg) from e
    except Exception:
        logging.exception("An unexpected error occurred during settings initialization")
        msg = "Critical error during settings initialization."
        raise SystemExit(msg) from e


# Create a global settings instance.
# This line will execute `get_settings()` when the `config.py` module is imported,
# making the settings available globally as `config.settings`.
settings = get_settings()

# Load spaCy model globally upon import, using the loaded settings.
# This is a significant side effect for a configuration module.
# It means importing `config` anywhere will trigger this model loading.
# If the model is large or loading is slow, this can impact startup time
# or test execution if tests import modules that eventually import `config`.
# Consider deferring model loading to where it's explicitly needed,
# e.g., within an application context or service initializer.
try:
    spacy_model = spacy.load(settings.spacy_config.model_name)
    logging.info(f"spaCy model '{settings.spacy_config.model_name}' loaded successfully at import time.")
except OSError:
    logging.exception(
        f"Failed to load spaCy model '{settings.spacy_config.model_name}' at import time. "
        "spaCy-dependent features will not be available. "
        "Ensure the model is downloaded (e.g., python -m spacy download en_core_web_sm).",
    )
    spacy_model = None  # Ensure spacy_model is defined even if loading fails.
except Exception:  # Catch any other unexpected errors
    logging.exception(
        f"An unexpected error occurred while loading spaCy model '{settings.spacy_config.model_name}' at import time.",
    )
    spacy_model = None


# --- Example Usage (only runs if script is executed directly) ---
if __name__ == "__main__":
    # --- Setup for Example ---
    logging.info("--- Running Example ---")
    # Simulate setting the environment variable that controls which .env and .yaml to load
    os.environ["APP_ENV"] = "dev"
    logging.info(f"APP_ENV set to: {os.getenv('APP_ENV')}")

    script_dir = Path(__file__).parent
    envs_yaml_dir = script_dir / "envs"
    envs_yaml_dir.mkdir(exist_ok=True)

    # Create dummy dev.yaml
    dev_yaml_path = envs_yaml_dir / "dev.yaml"
    with dev_yaml_path.open("w") as f:
        yaml.dump(
            {
                "app": {"debug": True, "name": "App Name from dev.yaml"},  # YAML highest prio for these
                "semantic": {"model_name": "model-from-dev-yaml"},
                # "env": "prod" # Test case: YAML overrides APP_ENV for the settings.env field
            },
            f,
        )
    logging.info(f"Created dummy file: {dev_yaml_path}")

    # Create dummy dev.env file (in script directory)
    dev_env_path = script_dir / "dev.env"
    with dev_env_path.open("w") as f:
        f.write('APP__VERSION="1.1-from-dev.env"\n')
        f.write("SEMANTIC__BATCH_SIZE=70\n")
        f.write('APP__NAME="App Name from dev.env"\n')  # Will be overridden by YAML
    logging.info(f"Created dummy file: {dev_env_path}")

    # Simulate setting *actual* process environment variables
    # (these override .env file if load_dotenv(override=False))
    # To test .env override, comment these out or set override=True in load_dotenv
    os.environ["APP__VERSION"] = "1.2-from-PROCESS-ENV"
    os.environ["SEMANTIC__DEVICE"] = "cuda"
    logging.info("Simulated PROCESS environment variables set:")
    logging.info(f"  APP__VERSION={os.getenv('APP__VERSION')}")  # Should win over dev.env's APP__VERSION
    logging.info(f"  SEMANTIC__DEVICE={os.getenv('SEMANTIC__DEVICE')}")

    # --- logging.info Final Settings ---
    logging.info("\n--- Final Settings ---")
    if settings:
        logging.info(f"Settings.env: {settings.env}")
        logging.info(f"App Name: {settings.app.name}")
        logging.info(f"App Version: {settings.app.version}")
        logging.info(f"App Debug: {settings.app.debug}")
        logging.info(f"Semantic Model: {settings.semantic.model_name}")
        logging.info(f"Semantic Batch Size: {settings.semantic.batch_size}")
        logging.info(f"Semantic Device: {settings.semantic.device}")

    # Expected with default load_dotenv(override=False):
    # Settings.env: dev (from APP_ENV, as not overridden by YAML in this test setup)
    # App Name: App Name from dev.yaml (YAML overrides dev.env and PROCESS-ENV for 'name')
    # App Version: 1.2-from-PROCESS-ENV (PROCESS-ENV overrides dev.env)
    # App Debug: True (from dev.yaml)
    # Semantic Model: model-from-dev-yaml (from dev.yaml)
    # Semantic Batch Size: 70 (from dev.env, as not in YAML or PROCESS-ENV)
    # Semantic Device: cuda-from-PROCESS-ENV (from PROCESS-ENV, overrides model default)

    # --- Clean up ---
    logging.info("\n--- Cleaning up dummy files ---")
    dev_yaml_path.unlink(missing_ok=True)
    from contextlib import suppress

    with suppress(OSError):
        envs_yaml_dir.rmdir()
    dev_env_path.unlink(missing_ok=True)
    if "APP_ENV" in os.environ:
        del os.environ["APP_ENV"]  # Clean up this test specific one
    if "APP__VERSION" in os.environ:
        del os.environ["APP__VERSION"]
    if "SEMANTIC__DEVICE" in os.environ:
        del os.environ["SEMANTIC__DEVICE"]
    # If APP__NAME was set in process env for testing, clean it too
    if "APP__NAME" in os.environ:
        del os.environ["APP__NAME"]
    logging.info("Cleanup complete.")
