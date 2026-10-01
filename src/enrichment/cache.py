"""Tiny thread-safe in-memory cache: one GitHub lookup per username per run."""
from __future__ import annotations

import threading
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class MemoryCache(Generic[T]):
    def __init__(self) -> None:
        self._data: dict[str, T] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
            return value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._data[key] = value
