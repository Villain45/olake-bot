from qdrant_client import QdrantClient

from .config import Settings


def make_client(s: Settings, *, timeout: int = 10) -> QdrantClient:
    """Connect the API and indexer to the configured Qdrant Cloud cluster."""
    url = s.qdrant_url.strip()
    api_key = s.qdrant_api_key.strip()
    if not url or not api_key:
        raise ValueError("Set QDRANT_URL and QDRANT_API_KEY to connect to Qdrant Cloud.")
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)
