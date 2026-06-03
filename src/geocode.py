"""
Open-Meteo geocoding for the "new project" setup wizard (Phase B).

A thin, dependency-light wrapper around the free Open-Meteo geocoding API used
to resolve a city name to coordinates + timezone.  All network and parsing
errors are swallowed and surfaced as an empty list so the wizard can always
fall back to manual latitude/longitude entry.
"""
from __future__ import annotations

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# ASCII digraph → Unicode substitutions tried as fallbacks when an ASCII-only
# query returns no results.  Ordered from most specific (digraphs) to least
# specific (single chars) so "oe"→"ø" is tried before "o"→"ø".
_FALLBACK_SUBS: list[tuple[str, str]] = [
    ("oe", "ø"),   # sonderborg → sønderborg (Danish/Norwegian)
    ("oe", "ö"),   # goeteborg  → göteborg   (German/Swedish)
    ("ae", "æ"),   # taarnby    → –          (Danish)
    ("ae", "ä"),   # muenchen   → münchen    (German)
    ("aa", "å"),   # aalborg    → åalborg    (Danish — "aa" is the ASCII form)
    ("ue", "ü"),   # muenchen fallback
    ("o",  "ø"),   # sonderborg → sønderborg (single-char, tried last)
    ("a",  "å"),   # arhus      → århus
]


def _fetch(query: str, count: int) -> list[dict]:
    """Single API call; returns parsed list or [] on any error."""
    params = {"name": query, "count": count, "language": "en", "format": "json"}
    try:
        resp = requests.get(GEOCODE_URL, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except Exception:
        return []
    return [
        {
            "name": r.get("name"),
            "country": r.get("country"),
            "country_code": r.get("country_code"),
            "admin1": r.get("admin1"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "timezone": r.get("timezone"),
            "population": r.get("population"),
        }
        for r in results
    ]


def geocode_city(name: str, count: int = 5) -> list[dict]:
    """Look up *name* and return up to *count* candidate locations.

    Each candidate is a dict with keys: ``name``, ``country``,
    ``country_code`` (ISO-3166 alpha-2), ``admin1``, ``latitude``,
    ``longitude``, ``timezone``, ``population``.

    When the original query returns no results and the query is pure ASCII,
    common ASCII→diacritic substitutions are tried automatically so that
    e.g. "sonderborg" finds "Sønderborg" and "aalborg" finds "Aalborg".

    Returns ``[]`` for blank input, no results, or any network/parse error so
    the caller can offer manual lat/lon entry instead of crashing.
    """
    name = str(name).strip()
    if not name:
        return []

    results = _fetch(name, count)
    if results:
        return results

    # No results — try diacritic variants for ASCII-only queries
    if not name.isascii():
        return []

    name_l = name.lower()
    tried: set[str] = {name_l}
    for old, new in _FALLBACK_SUBS:
        if old in name_l:
            variant = name_l.replace(old, new, 1)
            if variant in tried:
                continue
            tried.add(variant)
            results = _fetch(variant, count)
            if results:
                return results

    return []
