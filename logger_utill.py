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

logger = logging.getLogger("my_app")  # Default logger name, can be overridden
_separator_func: Callable[[], None] = lambda: logging.info("-" * 50)


def setup_global_logger(
    log_level: str | int = "INFO",
    app_name: str = "Application",
    *,
    force_basic_logging: bool = False,
) -> None:
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
            # If rich was expected but not found, print a warning or raise error
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
    return _separator_func


# Expose logging itself so other modules can just import logging from here if they want
# import logging # This line is already effectively done by `import logging` at the top.
