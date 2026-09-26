"""
Text normalization for business names and addresses.

Optimised for large-scale processing (millions of rows) using vectorised
pandas string operations wherever possible.  Per-row Python loops are
avoided; the heavy lifting (lowercasing, regex substitution, accent
folding) is done through pandas ``.str`` accessors which call C under
the hood.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Set

import pandas as pd

from .config import CFG

logger = logging.getLogger(__name__)

# ── Indic Script Transliteration ─────────────────────────────────────────
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    _HAS_INDIC = True
except ImportError:
    _HAS_INDIC = False

_ANY_INDIC = re.compile(r"[\u0900-\u0D7F]")

_INDIC_RANGES = [
    (re.compile(r"[\u0900-\u097F]"), getattr(sanscript, "DEVANAGARI", None) if _HAS_INDIC else None),
    (re.compile(r"[\u0A80-\u0AFF]"), getattr(sanscript, "GUJARATI", None) if _HAS_INDIC else None),
    (re.compile(r"[\u0980-\u09FF]"), getattr(sanscript, "BENGALI", None) if _HAS_INDIC else None),
    (re.compile(r"[\u0C80-\u0CFF]"), getattr(sanscript, "KANNADA", None) if _HAS_INDIC else None),
    (re.compile(r"[\u0B80-\u0BFF]"), getattr(sanscript, "TAMIL", None) if _HAS_INDIC else None),
    (re.compile(r"[\u0C00-\u0C7F]"), getattr(sanscript, "TELUGU", None) if _HAS_INDIC else None),
    (re.compile(r"[\u0D00-\u0D7F]"), getattr(sanscript, "MALAYALAM", None) if _HAS_INDIC else None),
    (re.compile(r"[\u0A00-\u0A7F]"), getattr(sanscript, "GURMUKHI", None) if _HAS_INDIC else None),
    (re.compile(r"[\u0B00-\u0B7F]"), getattr(sanscript, "ORIYA", None) if _HAS_INDIC else None),
]

_DEVA_SUBS = {
    "\u0949": "o", "\u094a": "o", "\u0945": "e",
    "\u0911": "O", "\u090d": "E", "\u093c": "", "\u0950": "om "
}


def _transliterate_indic_text(text: str) -> str:
    """Convert Indic scripts (Devanagari, Gujarati, Bengali, etc.) to Latin phonetic text."""
    if not text or not _HAS_INDIC or not isinstance(text, str):
        return text
    if _ANY_INDIC.search(text):
        for k, v in _DEVA_SUBS.items():
            if k in text:
                text = text.replace(k, v)
        for pattern, scheme in _INDIC_RANGES:
            if scheme is not None and pattern.search(text):
                try:
                    text = transliterate(text, scheme, sanscript.ITRANS)
                except Exception:
                    pass
    return text


# ── Scalar helpers (used in .apply only where vectorised is impossible) ──

def _fold_accents(text: str) -> str:
    """NFKD decompose then strip combining marks:  é→e  ç→c  ô→o  etc."""
    if not text:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _extract_numbers_scalar(text: str) -> set:
    """Return set of digit-sequences found in *text*."""
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


# ── Pre-compiled regex patterns ──────────────────────────────────────────

_RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s\-]")
_RE_MULTI_SPACE  = re.compile(r"\s+")

# Multi-word legal suffixes checked first, then single-word
_MULTI_SUFFIXES = [
    "private limited", "pvt ltd", "pvt limited", "private ltd",
    "praiveta limiteda", "praiveta ltd", "pvt limiteda",
]
_SINGLE_SUFFIX_SET: Set[str] = set()  # filled at import time


def _init_suffix_set():
    global _SINGLE_SUFFIX_SET
    _SINGLE_SUFFIX_SET = {s.lower() for s in CFG.legal_suffixes}
    # Add common Hindi/Indic transliterated legal suffixes
    _SINGLE_SUFFIX_SET.update({"limiteda", "kampani", "stora", "elaelapi"})

_init_suffix_set()


def _strip_suffix_scalar(name: str) -> str:
    """Remove trailing legal suffixes from an already-normalised name."""
    if not name:
        return name
    # Multi-word first (longest match)
    for ms in _MULTI_SUFFIXES:
        if name.endswith(ms):
            name = name[: -len(ms)].rstrip(" -")
            break
    # Then single-word trailing token
    parts = name.rsplit(maxsplit=1)
    if len(parts) == 2 and parts[1] in _SINGLE_SUFFIX_SET:
        name = parts[0].rstrip(" -")
    return name


def _get_tokens_scalar(text: str, min_len: int = 3) -> list:
    """Split text into unique tokens ≥ min_len, sorted."""
    if not text:
        return []
    return sorted({t for t in text.split() if len(t) >= min_len})


# ── Vectorised preprocessing ────────────────────────────────────────────

def preprocess_dataframe(
    df: pd.DataFrame,
    suffixes: list,
    abbrevs: dict,
) -> pd.DataFrame:
    """Add cleaned columns to *df* using vectorised string ops.

    Adds:  name_clean, addr_clean, name_tokens, addr_tokens, addr_numbers,
           name_script_dropped, addr_script_dropped.

    Works on DataFrames with columns: entity_id, business_name,
    business_address, country.  NaN values are handled gracefully.
    """
    out = df.copy()

    # ── Fill NaN ──────────────────────────────────────────────────
    out["business_name"]    = out["business_name"].fillna("")
    out["business_address"] = out["business_address"].fillna("")

    # ── Transliterate Indic scripts BEFORE ascii normalization ────
    s_raw = out["business_name"]
    if out["business_name"].str.contains(_ANY_INDIC, regex=True).any():
        s = out["business_name"].apply(_transliterate_indic_text)
    else:
        s = out["business_name"]

    a_raw = out["business_address"]
    if out["business_address"].str.contains(_ANY_INDIC, regex=True).any():
        a = out["business_address"].apply(_transliterate_indic_text)
    else:
        a = out["business_address"]

    # ── name_clean (vectorised pipeline) ─────────────────────────
    s = s.str.lower()
    s = s.apply(_fold_accents)
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(_RE_NON_ALPHANUM, " ", regex=True)
    s = s.str.replace(_RE_MULTI_SPACE, " ", regex=True).str.strip()
    out["name_clean"] = s.apply(_strip_suffix_scalar)

    # ── Diagnostic: name_script_dropped ───────────────────────────
    out["name_script_dropped"] = (out["name_clean"] == "") & (s_raw.str.strip() != "")
    dropped_count = int(out["name_script_dropped"].sum())
    if dropped_count > 0:
        logger.info(
            f"Diagnostic: {dropped_count:,} / {len(out):,} records ({dropped_count/len(out)*100:.2f}%) "
            f"had name dropped by normalization (name_script_dropped=True)"
        )

    # ── addr_clean (vectorised pipeline) ─────────────────────────
    a = a.str.lower()
    a = a.apply(_fold_accents)
    a = a.str.replace("&", " and ", regex=False)
    a = a.str.replace(_RE_NON_ALPHANUM, " ", regex=True)
    a = a.str.replace(_RE_MULTI_SPACE, " ", regex=True).str.strip()
    if abbrevs:
        sorted_abbrevs = sorted(abbrevs.items(), key=lambda x: len(x[0]), reverse=True)
        for short, full in sorted_abbrevs:
            a = a.str.replace(
                r"\b" + re.escape(short) + r"\b", full, regex=True
            )
    out["addr_clean"] = a

    out["addr_script_dropped"] = (out["addr_clean"] == "") & (a_raw.str.strip() != "")

    # ── Tokens (scalar .apply — fast since strings are already clean)
    out["name_tokens"]  = out["name_clean"].apply(
        lambda x: _get_tokens_scalar(x, CFG.min_token_len)
    )
    out["addr_tokens"]  = out["addr_clean"].apply(
        lambda x: _get_tokens_scalar(x, CFG.min_token_len)
    )

    # ── Address numbers from original address
    out["addr_numbers"] = out["business_address"].apply(_extract_numbers_scalar)

    return out


# ── Convenience wrappers (kept for compatibility) ────────────────────────

def normalize_text(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = _transliterate_indic_text(text)
    text = text.lower()
    text = _fold_accents(text)
    text = text.replace("&", " and ")
    text = _RE_NON_ALPHANUM.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    return text


def normalize_business_name(name: str, suffixes: list) -> str:
    return _strip_suffix_scalar(normalize_text(name))


def normalize_address(addr: str, abbrevs: dict) -> str:
    cleaned = normalize_text(addr)
    if abbrevs:
        tokens = cleaned.split()
        cleaned = " ".join(abbrevs.get(t, t) for t in tokens)
    return cleaned


def extract_numbers(text: str) -> Set[str]:
    return _extract_numbers_scalar(text)


def get_name_tokens(name: str, min_len: int = 3) -> List[str]:
    return _get_tokens_scalar(normalize_text(name), min_len)


def get_address_tokens(addr: str, min_len: int = 3) -> List[str]:
    return _get_tokens_scalar(normalize_text(addr), min_len)
