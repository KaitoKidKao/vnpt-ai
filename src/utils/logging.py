"""Color-coded logging utilities using colorama."""

import re

from colorama import Fore, Style, init

init(autoreset=True)

PREFIX_COLORS: dict[str, str] = {
    "Main": Fore.CYAN,
    "Pipeline": Fore.MAGENTA,
    "Done": Fore.GREEN,
    "Stats": Fore.YELLOW,
    "Error": Fore.RED,
    "Warning": Fore.YELLOW,
    "Info": Fore.BLUE,
    "RAG": Fore.BLUE,
    "Router": Fore.CYAN,
    "Logic": Fore.MAGENTA,
    "Safety": Fore.YELLOW,
}

BRACKET_PATTERN = re.compile(r"^\[([^\]]+)\]")


def get_prefix_color(prefix: str) -> str:
    """Get color for a given prefix, with fallback for question IDs."""
    if prefix in PREFIX_COLORS:
        return PREFIX_COLORS[prefix]
    if prefix.startswith("Q") or prefix[0].isdigit():
        return Fore.LIGHTBLUE_EX
    return Fore.WHITE


def print_log(message: str, end: str = "\n") -> None:
    """Print a log message with colored bracket prefix.

    Args:
        message: Log message, optionally starting with [Prefix].
        end: String appended after the message (default newline).
    """
    match = BRACKET_PATTERN.match(message)
    if match:
        prefix = match.group(1)
        color = get_prefix_color(prefix)
        colored_prefix = f"{color}[{prefix}]{Style.RESET_ALL}"
        rest = message[match.end():]
        print(f"{colored_prefix}{rest}", end=end)
    else:
        print(message, end=end)


def log_main(message: str) -> None:
    """Log with [Main] prefix."""
    print_log(f"[Main] {message}")


def log_pipeline(message: str) -> None:
    """Log with [Pipeline] prefix."""
    print_log(f"[Pipeline] {message}")


def log_done(message: str) -> None:
    """Log with [Done] prefix."""
    print_log(f"[Done] {message}")


def log_error(message: str) -> None:
    """Log with [Error] prefix."""
    print_log(f"[Error] {message}")


def log_stats(message: str) -> None:
    """Log with [Stats] prefix."""
    print_log(f"[Stats] {message}")


def print_separator(char: str = "=", width: int = 50) -> None:
    """Print a separator line."""
    print(Fore.WHITE + char * width + Style.RESET_ALL)


def print_header(title: str, char: str = "=", width: int = 50) -> None:
    """Print a centered header with separators."""
    print_separator(char, width)
    padding = (width - len(title)) // 2
    print(Fore.WHITE + " " * padding + title + Style.RESET_ALL)
    print_separator(char, width)

