"""
Data loader module for the Multi-Modal Evidence Review system.

Uses pandas for robust CSV parsing that handles:
- Quoted fields with commas, pipe characters, and multilingual text
- Whitespace stripping on all string fields
- NaN/empty field normalization

Loads all three CSV files at startup into structured Python dicts/lists
so that lookups during processing are O(1) dict access rather than
repeated CSV reads.

Provides:
- load_claims(csv_path) → list[dict]
- load_sample_claims() → list[dict]
- load_user_history() → dict keyed by user_id
- load_evidence_requirements() → list[dict]
- get_user_history(user_id) → dict or None
- get_evidence_requirements(claim_object, issue_family, image_count) → list[dict]
- load_all_data() → combined dict
"""

import pandas as pd
from pathlib import Path
from config import (
    CLAIMS_CSV,
    SAMPLE_CLAIMS_CSV,
    USER_HISTORY_CSV,
    EVIDENCE_REQUIREMENTS_CSV,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a dataframe after loading:
    - Strip leading/trailing whitespace from all string columns
    - Replace NaN with empty string for string columns
    - Strip stray quotes that pandas may leave on fields
    """
    for col in df.columns:
        if df[col].dtype == object:
            # Fill NaN with empty string, then strip whitespace and stray quotes
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.strip('"')
                .str.strip("'")
            )
    return df


def _read_csv(path: Path) -> list[dict]:
    """
    Read a CSV file using pandas with robust settings for:
    - Quoted fields (handles embedded commas, pipes, multilingual text)
    - UTF-8 encoding
    - Proper quoting (QUOTE_ALL style CSVs)

    Returns a list of dicts, one per row.
    """
    df = pd.read_csv(
        path,
        encoding="utf-8",
        quotechar='"',
        skipinitialspace=True,
        na_values=["", "NA", "N/A"],
        keep_default_na=False,
    )
    df = _clean_dataframe(df)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Claims loader
# ---------------------------------------------------------------------------

def load_claims(csv_path: Path | None = None) -> list[dict]:
    """
    Load claims from the given CSV path (defaults to dataset/claims.csv).
    Returns a list of dicts with keys: user_id, image_paths, user_claim, claim_object.
    """
    path = csv_path or CLAIMS_CSV
    return _read_csv(path)


def load_sample_claims() -> list[dict]:
    """Load sample claims (with ground-truth labels) for evaluation."""
    return _read_csv(SAMPLE_CLAIMS_CSV)


# ---------------------------------------------------------------------------
# User history loader + lookup
# ---------------------------------------------------------------------------

_user_history_cache: dict[str, dict] | None = None


def load_user_history() -> dict[str, dict]:
    """
    Load user_history.csv into a dict keyed by user_id.

    Each value is a dict with keys:
        user_id, past_claim_count, accept_claim, manual_review_claim,
        rejected_claim, last_90_days_claim_count, history_flags, history_summary
    """
    global _user_history_cache
    if _user_history_cache is not None:
        return _user_history_cache

    rows = _read_csv(USER_HISTORY_CSV)
    _user_history_cache = {row["user_id"]: row for row in rows}
    return _user_history_cache


def get_user_history(user_id: str) -> dict | None:
    """
    Look up a single user's history by user_id.
    Returns the history dict or None if user_id is not found.
    """
    history = load_user_history()
    return history.get(user_id.strip())


# ---------------------------------------------------------------------------
# Evidence requirements loader + lookup
# ---------------------------------------------------------------------------

_evidence_requirements_cache: list[dict] | None = None


def load_evidence_requirements() -> list[dict]:
    """
    Load evidence_requirements.csv into a list of dicts.

    Each dict has keys:
        requirement_id, claim_object, applies_to, minimum_image_evidence
    """
    global _evidence_requirements_cache
    if _evidence_requirements_cache is not None:
        return _evidence_requirements_cache

    _evidence_requirements_cache = _read_csv(EVIDENCE_REQUIREMENTS_CSV)
    return _evidence_requirements_cache


def get_evidence_requirements(
    claim_object: str, issue_family: str, image_count: int = 1
) -> list[dict]:
    """
    Look up applicable evidence requirements for a given claim_object and issue family.

    Matching logic:
    1. Requirements where claim_object matches exactly OR claim_object == "all"
    2. Requirements where applies_to is a substring match against the issue_family
       (e.g., issue_family="dent" matches applies_to="dent or scratch")
       OR applies_to is a general/reviewability rule
    3. The "multi-image rows" rule (REQ_GENERAL_MULTI_IMAGE) is ONLY included
       when image_count >= 2, since it describes how to handle multiple images.

    Args:
        claim_object: car | laptop | package
        issue_family: the extracted/observed issue type
        image_count: number of submitted images (default 1)

    Returns a list of matching requirement dicts (may be multiple).
    """
    all_reqs = load_evidence_requirements()
    matched = []

    issue_lower = issue_family.lower().strip()

    for req in all_reqs:
        # Check claim_object match
        req_object = req["claim_object"].lower().strip()
        if req_object != "all" and req_object != claim_object.lower().strip():
            continue

        # Check applies_to match
        applies_to = req["applies_to"].lower().strip()

        # Multi-image rule only applies when there are 2+ images
        if applies_to == "multi-image rows":
            if image_count >= 2:
                matched.append(req)
            continue

        # Other general rules always apply
        if applies_to in ("general claim review", "reviewability"):
            matched.append(req)
            continue

        # Check if the issue family appears in the applies_to field
        # e.g., "dent" appears in "dent or scratch"
        # or "crack" appears in "crack, broken, or missing part"
        if issue_lower in applies_to:
            matched.append(req)
            continue

        # Also check reverse: applies_to keywords appear in issue_family
        # e.g., applies_to="crushed, torn, or seal damage" matches issue_family="crushed_packaging"
        applies_keywords = [
            w.strip() for w in applies_to.replace(",", " ").replace("or", " ").split()
        ]
        for keyword in applies_keywords:
            if keyword and len(keyword) > 2 and keyword in issue_lower:
                matched.append(req)
                break

    return matched


# ---------------------------------------------------------------------------
# Convenience: load everything at once
# ---------------------------------------------------------------------------

def load_all_data(claims_path: Path | None = None) -> dict:
    """
    Load all datasets and return a structured dict:
    {
        "claims": list[dict],
        "user_history": dict[str, dict],
        "evidence_requirements": list[dict],
    }
    """
    return {
        "claims": load_claims(claims_path),
        "user_history": load_user_history(),
        "evidence_requirements": load_evidence_requirements(),
    }
