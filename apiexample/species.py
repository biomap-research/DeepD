"""Resolve a eukaryotic species scientific name to the internal species id.

Client-side helper used by ``cli.py``.  The inference gateway identifies a
species by an internal id in the ``species_raw`` parameter; this module loads
the compact lookup table ``data/species_name_to_id.json`` beside this module and resolves a
user-friendly scientific name (for example ``"Caenorhabditis elegans"``) to
that internal id, so callers only ever need to work with scientific names.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent
DEFAULT_MAP_FILE = _HERE / "data" / "species_name_to_id.json"

# Gateway placeholder when no species is requested (worker uses [spMASK]).
SPECIES_RAW_EMPTY = "''"

_CACHE: Dict[str, Any] = {}


class SpeciesError(ValueError):
    """Unknown scientific name or invalid species id."""


def species_raw_is_unset(raw: Any) -> bool:
    """True when ``species_raw`` means no species ([spMASK] on the worker)."""
    if raw is None:
        return True
    text = str(raw).strip()
    if not text:
        return True
    return text == SPECIES_RAW_EMPTY


def _normalize_name(name: str) -> str:
    """Collapse whitespace and strip surrounding brackets/sort order markers."""
    text = re.sub(r"\s+", " ", str(name).strip())
    text = re.sub(r"^\[", "", text)
    text = re.sub(r"\]$", "", text)
    return text


def load_species_map(path: str | Path | None = None) -> dict[str, Any]:
    """Load (and cache) the species name -> species_id lookup table."""
    map_file = Path(path) if path else DEFAULT_MAP_FILE
    if not map_file.is_file():
        raise FileNotFoundError(
            f"species map not found at {map_file}. "
            "Expected apiexample/data/species_name_to_id.json."
        )
    if "path" not in _CACHE or _CACHE.get("path") != str(map_file.resolve()):
        with open(map_file, encoding="utf-8") as f:
            _CACHE["path"] = str(map_file.resolve())
            _CACHE["doc"] = json.load(f)
    return _CACHE["doc"]


def resolve_species(
    name: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a scientific name to its species record.

    Returns the full record dict from ``data/species_name_to_id.json`` whose
    ``species_id`` field is the internal id accepted by the gateway as
    ``species_raw``.

    Raises ``SpeciesError`` with a helpful message when the name is unknown.
    """
    doc = load_species_map(path)
    species = doc["species"]

    clean = _normalize_name(name)

    record = species.get(clean)
    if record is None:
        aliases = {key.lower(): key for key in species}
        matched = aliases.get(clean.lower())
        if matched is not None:
            record = species[matched]

    if record is None:
        known = list(species)  # lazy, only constructed on failure
        raise SpeciesError(
            f"unknown species: {name!r}. "
            f"Try a scientific name such as {known[:5]!r} (see {DEFAULT_MAP_FILE.name})"
        )
    return dict(record)


def to_species_raw(name: str, path: str | Path | None = None) -> str:
    """Shortcut: scientific name -> internal species id used by the gateway."""
    return str(resolve_species(name, path=path)["species_id"])


def guess_species(record: dict[str, Any], path: str | Path | None = None) -> str:
    """Return a scientific name for a species record (reverse lookup)."""
    doc = load_species_map(path)
    rec_id = str(record["species_id"])
    for name, rec in doc["species"].items():
        if str(rec["species_id"]) == rec_id:
            return name
    raise KeyError(f"species record not found in map: {rec_id}")


if __name__ == "__main__":  # pragma: no cover
    import sys

    demo = ["Caenorhabditis elegans", "Homo sapiens", "Mus musculus"]
    if len(sys.argv) > 1:
        demo = sys.argv[1:]
    for n in demo:
        try:
            rec = resolve_species(n)
            print(f"{n} -> species_id={rec['species_id']} taxid={rec.get('taxid')}")
        except SpeciesError as exc:
            print(f"{n} -> ERROR: {exc}")