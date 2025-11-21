"""Utility functions for configuring and managing global application logging."""

# logger_utils.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_rich_available = False
try:
    from rich.console import Console
    from rich.logging import RichHandler

    _rich_available = True
except ImportError:
    RichHandler = None
    Console = None

logger = logging.getLogger("my_app")
_separator_func: Callable[[], None] = lambda: logging.info("-" * 50)


def setup_global_logger(
    log_level: str | int = "INFO",
    app_name: str = "Application",
    *,
    force_basic_logging: bool = False,
) -> None:
    """
    Configure the global root logger with either RichHandler or basic logging.

    This function clears existing handlers on the root logger, sets the specified
    log level, and configures either a RichHandler (if 'rich' is available and
    not forced to basic) or a basic console logger. It also sets a global
    separator function (`_separator_func`) appropriate for the logging type.

    Args:
        log_level (str | int, optional): The logging level to set for the root logger.
            Defaults to "INFO".
        app_name (str, optional): The name of the application, used in the initial
            log message. Defaults to "Application".
        force_basic_logging (bool, optional): If True, forces basic logging even if
            'rich' is available. Defaults to False.
    """
    global _separator_func
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.root.setLevel(log_level)

    if _rich_available and RichHandler and Console and not force_basic_logging:
        rich_handler = RichHandler(
            level=log_level,
            show_path=False,
            show_level=True,
            show_time=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
            console=Console(stderr=True),
        )
        logging.root.addHandler(rich_handler)
        _console_for_print = Console()
        _separator_func = lambda: _console_for_print.print("-" * 50, style="dim")
        initial_logger = logging.getLogger(app_name) if app_name else logging.root
        initial_logger.setLevel(log_level)
        initial_logger.info(f"Starting {app_name} [bold green](using Rich logging)[/bold green]")
    else:
        if not _rich_available and not force_basic_logging:
            print(
                "WARNING: rich library not found. Falling back to basic logging. "
                "Install with 'pip install rich' for enhanced output.",
            )
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=log_level, format=log_format, force=True)
        _separator_func = lambda: logging.info("-" * 50)
        initial_logger = logging.getLogger(app_name) if app_name else logging.root
        initial_logger.setLevel(log_level)
        initial_logger.info(f"Starting {app_name} (using standard logging)")


def get_separator_func() -> Callable[[], None]:
    """
    Return the currently configured separator function.

    The separator function is used to print a visual separator line in the logs,
    and its implementation (rich-based or basic logging-based) is determined
    by the `setup_global_logger` function.

    Returns:
        Callable[[], None]: A function that, when called, prints a separator line.
    """
    return _separator_func
