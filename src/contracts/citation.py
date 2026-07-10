"""Citation contract."""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.types import (
    PageNumber,
    SourceURL,
)


@dataclass(
    frozen=True,
    slots=True,
)
class Citation:
    """Presentation-ready citation for a generated answer."""

    title: str

    heading: str | None

    source_url: SourceURL

    page_start: PageNumber

    page_end: PageNumber