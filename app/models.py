from dataclasses import dataclass


@dataclass
class Chunk:
    id: int
    url: str  # full citation URL (including #anchor when available)
    title: str
    breadcrumb: str
    text: str
    source_type: str  # docs | repo | issue | external
    score: float = 0.0
