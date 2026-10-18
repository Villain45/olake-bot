"""Run the existing indexer, then retain its current and previous published indexes.

Only this wrapper performs retention. Direct ``python -m ingest.build_index`` runs
are unchanged, and untracked staging collections are never cleanup candidates.
"""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import models as m

from app.config import get_settings
from app.qdrant import make_client


@dataclass(frozen=True)
class IndexSnapshot:
    current: str | None
    previous: str | None


def _aliases(client) -> dict[str, str]:
    return {item.alias_name: item.collection_name for item in client.get_aliases().aliases}


def _owned(name: str, base: str) -> bool:
    return name == base or re.fullmatch(re.escape(base) + r"_build_[0-9a-f]{20}", name) is not None


def _require_index(client, name: str, base: str) -> None:
    if not _owned(name, base):
        raise RuntimeError(f"Refusing retention for an unrelated collection: {name}")
    if not client.collection_exists(name) or client.count(name, exact=True).count <= 0:
        raise RuntimeError(f"Published index is missing or empty: {name}")


def capture_index(client, base: str) -> IndexSnapshot:
    aliases = _aliases(client)
    current = aliases.get(f"{base}_current")
    previous = aliases.get(f"{base}_previous")
    if current is None and client.collection_exists(base):
        current = base
    for name in {current, previous} - {None}:
        _require_index(client, name, base)
    if previous is not None and current is None:
        raise RuntimeError("A rollback alias exists without a current index; resolve it before refreshing")
    return IndexSnapshot(current, previous)


class RetentionLedger:
    """Remember only retired, previously published indexes for safe deletion retries."""

    def __init__(self, path, scope: str):
        self.path = Path(path)
        self.scope = scope
        self.pending: set[str] = set()
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("Invalid retention ledger; no indexes were deleted")
            # Cached state from another cluster or collection cannot authorize deletion.
            if data.get("scope") != scope:
                return
            pending = data.get("pending")
            if not isinstance(pending, list) or not all(isinstance(name, str) for name in pending):
                raise ValueError("Invalid retention candidates; no indexes were deleted")
            self.pending = set(pending)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix="retention-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({"version": 1, "scope": self.scope, "pending": sorted(self.pending)}, handle)
                handle.write("\n")
            temporary.replace(self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def add(self, name: str) -> None:
        self.pending.add(name)
        self._save()

    def discard(self, name: str) -> None:
        self.pending.discard(name)
        self._save()


def _check_publication(aliases, base, current, previous) -> None:
    if aliases.get(f"{base}_current") != current or aliases.get(f"{base}_previous") != previous:
        raise RuntimeError("Published aliases changed during retention; cleanup stopped")


def finish_refresh(client, base: str, before: IndexSnapshot, ledger: RetentionLedger) -> list[str]:
    """Called only after a successful build; preserve all data on uncertain ownership."""
    # Validate the entire candidate set before any alias update or deletion.
    for name in ledger.pending | ({before.current, before.previous} - {None}):
        if not _owned(name, base):
            raise RuntimeError(f"Unrelated retention candidate; cleanup stopped: {name}")
    aliases = _aliases(client)
    current = aliases.get(f"{base}_current")
    if current is None:
        raise RuntimeError("The build did not publish a current alias; cleanup skipped")
    _require_index(client, current, base)
    if aliases.get(f"{base}_previous") != before.previous:
        raise RuntimeError("Rollback alias changed during the build; cleanup skipped")

    previous = before.previous
    if current != before.current:
        if before.current is not None:
            _require_index(client, before.current, base)
        # Persist provenance before removing the alias that protects this candidate.
        if before.previous is not None and before.previous not in {current, before.current}:
            ledger.add(before.previous)
        operations = []
        if before.previous is not None:
            operations.append(m.DeleteAliasOperation(delete_alias=m.DeleteAlias(alias_name=f"{base}_previous")))
        if before.current is not None:
            operations.append(m.CreateAliasOperation(create_alias=m.CreateAlias(
                alias_name=f"{base}_previous", collection_name=before.current)))
        if operations:
            _check_publication(_aliases(client), base, current, before.previous)
            client.update_collection_aliases(change_aliases_operations=operations)
        previous = before.current

    deleted = []
    _check_publication(_aliases(client), base, current, previous)
    if previous is not None:
        _require_index(client, previous, base)
    for candidate in sorted(ledger.pending):
        # Freshly protect every alias, including aliases belonging to other consumers.
        aliases = _aliases(client)
        _check_publication(aliases, base, current, previous)
        if candidate in {current, previous} or candidate in aliases.values():
            continue
        if not client.collection_exists(candidate):
            ledger.discard(candidate)
            continue
        # Recheck after the existence request, immediately before the destructive call.
        aliases = _aliases(client)
        _check_publication(aliases, base, current, previous)
        if candidate in aliases.values():
            continue
        if not client.delete_collection(candidate):
            raise RuntimeError(f"Deletion was not confirmed for retired index: {candidate}")
        ledger.discard(candidate)
        deleted.append(candidate)
        print(f"Deleted retired index: {candidate}", flush=True)
    print(f"Retention complete: current={current}; rollback={previous}; "
          f"{len(deleted)} older indexes deleted", flush=True)
    return deleted


def main() -> int:
    settings = get_settings()
    scope = hashlib.sha256(f"{settings.qdrant_url.strip().rstrip('/')}\n{settings.collection}".encode()).hexdigest()
    ledger = RetentionLedger(f"{settings.index_cache_path}.retention.json", scope)
    client = make_client(settings, timeout=60)
    try:
        before = capture_index(client, settings.collection)
        # The child keeps the existing model loading, locking, progress and publication behavior.
        # A nonzero exit exits here without rotating rollback aliases or deleting anything.
        subprocess.run([sys.executable, "-m", "ingest.build_index"], check=True)
        try:
            finish_refresh(client, settings.collection, before, ledger)
        except Exception as exc:
            raise RuntimeError("Index build succeeded, but retention did not finish; "
                               "the published index remains available") from exc
    finally:
        client.close()
    return 0


def cli() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("Refresh interrupted; completed index data was left intact", file=sys.stderr, flush=True)
        return 130
    except subprocess.CalledProcessError as exc:
        print("Index build failed; retention was not run", file=sys.stderr, flush=True)
        return exc.returncode if 0 < exc.returncode < 256 else 1
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
