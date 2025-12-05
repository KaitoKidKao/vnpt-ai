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
    "Direct": Fore.CYAN,
}

BRACKET_PATTERN = re.compile(r"^\s*\[([^\]]+)\]")


def get_prefix_color(prefix: str) -> str:
    """Get color for a given prefix, with fallback for question IDs."""
    prefix_clean = prefix.strip()
    if prefix_clean in PREFIX_COLORS:
        return PREFIX_COLORS[prefix_clean]
    if prefix_clean.startswith("Q") or (prefix_clean and prefix_clean[0].isdigit()):
        return Fore.LIGHTBLUE_EX
    return Fore.WHITE


def print_log(message: str, end: str = "\n") -> None:
    """Print a log message with colored bracket prefix.
    
    Supports leading whitespace before the bracket prefix.
    
    Args:
        message: Log message, optionally starting with [Prefix] (may have leading whitespace).
        end: String appended after the message (default newline).
    """
    match = BRACKET_PATTERN.match(message)
    if match:
        prefix = match.group(1)
        color = get_prefix_color(prefix)
        
        # Find the start and end positions of the bracket prefix
        prefix_start = match.start()
        prefix_end = match.end()
        
        # Extract leading whitespace, colored prefix, and rest of message
        leading_whitespace = message[:prefix_start]
        colored_prefix = f"{color}[{prefix}]{Style.RESET_ALL}"
        rest = message[prefix_end:]
        
        print(f"{leading_whitespace}{colored_prefix}{rest}", end=end)
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
