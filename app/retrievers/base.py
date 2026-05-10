from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[dict]:
        """Return list of source dicts (no threshold filtering)."""
        ...
