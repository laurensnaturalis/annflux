"""File utility functions."""
import os


def append_to_filename(filepath: str, suffix: str) -> str:
    """
    Append a suffix to a filename before its extension.

    Args:
        filepath: Original file path (e.g., "data.csv" or "/path/to/file.txt")
        suffix: Suffix to append before the extension (e.g., "_backup")

    Returns:
        New file path with suffix inserted (e.g., "data_backup.csv")

    Examples:
        >>> append_to_filename("data.csv", "_backup")
        'data_backup.csv'
        >>> append_to_filename("/path/to/file.txt", "_2024")
        '/path/to/file_2024.txt'
        >>> append_to_filename("file", "_v2")
        'file_v2'
    """
    base, ext = os.path.splitext(filepath)
    return f"{base}{suffix}{ext}"
