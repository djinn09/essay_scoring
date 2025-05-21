"""Configuration module for the Essay Grading application.

This module defines configuration models, settings, and utilities for loading
and validating application settings from environment variables, .env files, and YAML files.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv  # Import python-dotenv

# For Pydantic V2 with pydantic-settings
from pydantic import BaseModel, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Module-level Constants ---
VALID_ENVS: set[str] = {"dev", "prod"}  # Define valid environments here


# --- Configuration Models ---


class AppConfig(BaseModel):
    """Application configuration settings.

    Attributes:
        name (str): The name of the application.
        version (str): The version of the application.
        debug (bool): Whether debug mode is enabled.

    """

    name: str = "Essay Grader"
    version: str = "1.0"
    debug: bool = False
    log_level: str = "INFO"


class SemanticConfig(BaseModel):
    """Configuration for semantic similarity calculations.

    Attributes:
        model_name (str): The name of the model to use.
        chunk_size (int): Size of text chunks for processing.
        overlap (int): Overlap between text chunks.
        batch_size (int): Batch size for model inference.
        device (Literal): Device to run the model on ("cpu", "cuda", "mps").

    """

    model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 384
    overlap: int = 64
    batch_size: int = 32
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    class Config:
        """Configuration class for SemanticConfig.

        Attributes:
            env_prefix (str): The prefix for environment variables.

        """

        env_prefix = "SEMANTIC_"


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

    model_config = SettingsConfigDict(
        env_file=None,  # Explicitly disable default .env loading here if we manually load
        env_file_encoding="utf-8",  # Still useful if a default .env was used
        env_nested_delimiter="__",
        extra="ignore",
    )

    @field_validator("env")
    @classmethod
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings.

    1. Determine `effective_env` from APP_ENV.
    2. Manually load `<effective_env>.env` into process environment variables.
       (Variables already in the process environment take precedence over these).
    3. Load YAML config `envs/<effective_env>.yaml`.
    4. Initialize Pydantic Settings, which will:
       a. Use values from `file_config` (YAML) passed as init args (highest precedence).
       b. Then use actual process environment variables (which now include those from <effective_env>.env).
       c. Then use model defaults.

    """
    effective_env = os.getenv("APP_ENV", "dev").lower()
    if effective_env not in VALID_ENVS:
        print(f"Warning: APP_ENV='{effective_env}' is not one of {VALID_ENVS}. Falling back to 'dev'.")
        effective_env = "dev"

    print(f"--- Loading settings for environment: '{effective_env}' ---")

    config_dir = Path(__file__).parent

    # --- Manually load environment-specific .env file into process environment ---
    # Variables already in os.environ will NOT be overwritten by load_dotenv by default.
    env_filename = f"{effective_env}.env"
    env_file_path = config_dir / env_filename
    if env_file_path.exists() and env_file_path.is_file():
        print(f"Loading environment variables from: {env_file_path}")
        load_dotenv(dotenv_path=env_file_path, override=False)  # override=False is default
    # set to True if you want .env to win over existing env vars
    else:
        print(f"Info: Environment file not found at {env_file_path}. Skipping manual .env load.")

    # --- Load YAML config based on environment ---
    yaml_config_path = config_dir / "envs" / f"{effective_env}.yaml"
    file_config = {}
    if yaml_config_path.exists():
        try:
            with yaml_config_path.open("r") as f:
                loaded_yaml = yaml.safe_load(f)
                if isinstance(loaded_yaml, dict):
                    file_config = loaded_yaml
                    print(f"Successfully loaded YAML config from: {yaml_config_path}")
                else:
                    print(f"Warning: YAML file {yaml_config_path} did not contain a dictionary. Ignoring.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_config_path}: {e}")
        except Exception as e:
            print(f"Error reading file {yaml_config_path}: {e}")
    else:
        print(f"Info: YAML config file not found at {yaml_config_path}. Skipping.")

    # --- Instantiate Settings ---
    try:
        # `file_config` (from YAML) is spread into init_kwargs here.
        # Pydantic Settings loading order:
        # 1. Arguments passed to the initializer (i.e., `env=effective_env` and `**file_config`).
        # 2. Environment variables (now includes those from the loaded <effective_env>.env).
        # 3. Values from a default `.env` file (if `env_file` in model_config was set and not None).
        # 4. Default field values in the model.

        # Ensure 'env' field in Settings is correctly set by effective_env,
        # unless explicitly overridden by the YAML file itself with a different value.
        init_data = file_config.copy()
        if "env" not in init_data:  # Only set 'env' from effective_env if not in YAML
            init_data["env"] = effective_env
        elif init_data.get("env", "").lower() != effective_env:
            print(
                f"Warning: YAML file specifies 'env: {init_data['env']}', "
                f"which differs from the loading environment '{effective_env}'. "
                f"The YAML value will be used for settings.env.",
            )

        settings = Settings(**init_data)

        # Final check/log for the 'env' field source
        if settings.env != effective_env and (
            "env" not in file_config or file_config.get("env", "").lower() != settings.env.lower()
        ):
            print(
                f"Warning: Final settings.env ('{settings.env}') differs from the "
                f"loading environment ('{effective_env}'). This might happen if environment "
                f"variables (ENV, not from {effective_env}.env if override=False) set 'ENV' differently.",
            )
        elif (
            settings.env != effective_env
            and "env" in file_config
            and file_config.get("env", "").lower() == settings.env.lower()
        ):
            pass  # YAML explicitly set it, already warned above.
        else:
            print(f"Settings.env correctly reflects loading environment: '{settings.env}'")

        print("Settings loaded successfully.")
        return settings
    except ValidationError as e:
        print(f"Error validating settings: {e}")
        msg = "Failed to load or validate application settings."
        raise SystemExit(msg) from e
    except Exception as e:
        print(f"An unexpected error occurred during settings initialization: {e}")
        msg = "Critical error during settings initialization."
        raise SystemExit(msg) from e


# --- Example Usage ---
if __name__ == "__main__":
    # --- Setup for Example ---
    print("--- Running Example ---")
    # Simulate setting the environment variable that controls which .env and .yaml to load
    os.environ["APP_ENV"] = "dev"
    print(f"APP_ENV set to: {os.getenv('APP_ENV')}")

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
    print(f"Created dummy file: {dev_yaml_path}")

    # Create dummy dev.env file (in script directory)
    dev_env_path = script_dir / "dev.env"
    with dev_env_path.open("w") as f:
        f.write('APP__VERSION="1.1-from-dev.env"\n')
        f.write("SEMANTIC__BATCH_SIZE=70\n")
        f.write('APP__NAME="App Name from dev.env"\n')  # Will be overridden by YAML
    print(f"Created dummy file: {dev_env_path}")

    # Simulate setting *actual* process environment variables (these override .env file if load_dotenv(override=False))
    # To test .env override, comment these out or set override=True in load_dotenv
    os.environ["APP__VERSION"] = "1.2-from-PROCESS-ENV"
    os.environ["SEMANTIC__DEVICE"] = "cuda"
    print("Simulated PROCESS environment variables set:")
    print(f"  APP__VERSION={os.getenv('APP__VERSION')}")  # Should win over dev.env's APP__VERSION
    print(f"  SEMANTIC__DEVICE={os.getenv('SEMANTIC__DEVICE')}")

    # --- Load Settings ---
    settings = get_settings()

    # --- Print Final Settings ---
    print("\n--- Final Settings ---")
    if settings:
        print(f"Settings.env: {settings.env}")
        print(f"App Name: {settings.app.name}")
        print(f"App Version: {settings.app.version}")
        print(f"App Debug: {settings.app.debug}")
        print(f"Semantic Model: {settings.semantic.model_name}")
        print(f"Semantic Batch Size: {settings.semantic.batch_size}")
        print(f"Semantic Device: {settings.semantic.device}")

    # Expected with default load_dotenv(override=False):
    # Settings.env: dev (from APP_ENV, as not overridden by YAML in this test setup)
    # App Name: App Name from dev.yaml (YAML overrides dev.env and PROCESS-ENV for 'name')
    # App Version: 1.2-from-PROCESS-ENV (PROCESS-ENV overrides dev.env)
    # App Debug: True (from dev.yaml)
    # Semantic Model: model-from-dev-yaml (from dev.yaml)
    # Semantic Batch Size: 70 (from dev.env, as not in YAML or PROCESS-ENV)
    # Semantic Device: cuda-from-PROCESS-ENV (from PROCESS-ENV, overrides model default)

    # --- Clean up ---
    print("\n--- Cleaning up dummy files ---")
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
    print("Cleanup complete.")
