"""Exact-duplicate detection by content hash."""
from __future__ import annotations

from typing import Optional


class DuplicateTracker:
    """First file seen with a given hash is kept; later ones are duplicates.

    Because files are processed in sorted order the choice is deterministic.
    """

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}

    def check(self, file_hash: str, filename: str) -> Optional[str]:
        """Return the filename this one duplicates, or None if it is new."""
        original = self._seen.get(file_hash)
        if original is None:
            self._seen[file_hash] = filename
        return original
