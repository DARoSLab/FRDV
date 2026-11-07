from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Motion:
    """A motion clip as joint positions in the source's raw units and axes."""

    names: list[str]
    positions: np.ndarray            # (T, J, 3)
    fps: float
    rest: np.ndarray | None = None   # (J, 3) rest/T-pose, same units/axes (optional)
    parents: list[int] | None = None  # (J,) parent index, -1 for roots (optional, for drawing)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"joint '{name}' not in motion; available: {', '.join(self.names)}") from None
