"""
Preprocessor module for the Multi-Modal Evidence Review system.

Takes one claim row (dict) and produces a "context bundle" that will be
passed to both Pass 1 (VLM reasoning) and Pass 2 (validation/finalization).

Responsibilities:
1. Extract the claimed damage type and object part from the conversation
2. Look up user history from the preloaded cache
3. Select which evidence requirements apply to this claim
4. Pre-scan the claim text for prompt injection attempts
5. Validate that all referenced image files actually exist on disk
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

from config import (
    DATASET_DIR,
    ISSUE_TYPE_VALUES,
    CAR_OBJECT_PARTS,
    LAPTOP_OBJECT_PARTS,
    PACKAGE_OBJECT_PARTS,
    OBJECT_PARTS_BY_TYPE,
    RISK_FLAG_VALUES,
)
from data_loader import get_user_history, get_evidence_requirements


# ---------------------------------------------------------------------------
# Prompt injection detection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"skip\s+(manual\s+)?review", re.IGNORECASE),
    re.compile(r"approve\s+(the\s+)?(claim|this)\s+(immediately|now|quickly)", re.IGNORECASE),
    re.compile(r"mark\s+this\s+(row\s+)?(as\s+)?(supported|approved)", re.IGNORECASE),
    re.compile(r"system\s+reading\s+this\s+should", re.IGNORECASE),
    re.compile(r"follow\s+(it|the\s+note)\s+and\s+approve", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?rules?", re.IGNORECASE),
    re.compile(r"do\s+not\s+(flag|reject|deny)", re.IGNORECASE),
    re.compile(r"usko\s+follow\s+karke.*approve", re.IGNORECASE),
    re.compile(r"note\s+(says?|is\s+enough)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Keyword maps for extracting claim info from conversation
# ---------------------------------------------------------------------------

# Maps conversational keywords → canonical issue_type values
_ISSUE_KEYWORDS: dict[str, str] = {
    "dent": "dent",
    "dented": "dent",
    "deformation": "dent",
    "hail": "dent",
    "scratch": "scratch",
    "scratched": "scratch",
    "scrape": "scratch",
    "mark": "scratch",
    "crack": "crack",
    "cracked": "crack",
    "cracking": "crack",
    "shatter": "glass_shatter",
    "shattered": "glass_shatter",
    "broken": "broken_part",
    "broke": "broken_part",
    "missing": "missing_part",
    "torn": "torn_packaging",
    "torn-open": "torn_packaging",
    "phati": "torn_packaging",
    "crushed": "crushed_packaging",
    "crush": "crushed_packaging",
    "dab": "crushed_packaging",
    "water": "water_damage",
    "wet": "water_damage",
    "liquid": "water_damage",
    "stain": "stain",
    "stained": "stain",
    "oily": "stain",
    "oil": "stain",
    "spill": "stain",
    "spilled": "stain",
}

# Maps conversational keywords → canonical object_part values (per object type)
_PART_KEYWORDS: dict[str, dict[str, str]] = {
    "car": {
        "front bumper": "front_bumper",
        "rear bumper": "rear_bumper",
        "back bumper": "rear_bumper",
        "parachoques trasero": "rear_bumper",
        "parachoques": "rear_bumper",
        "door": "door",
        "hood": "hood",
        "windshield": "windshield",
        "front glass": "windshield",
        "side mirror": "side_mirror",
        "mirror": "side_mirror",
        "headlight": "headlight",
        "head light": "headlight",
        "taillight": "taillight",
        "tail light": "taillight",
        "back light": "taillight",
        "fender": "fender",
        "quarter panel": "quarter_panel",
        "body": "body",
        "body panel": "body",
    },
    "laptop": {
        "screen": "screen",
        "display": "screen",
        "pantalla": "screen",
        "keyboard": "keyboard",
        "keys": "keyboard",
        "teclas": "keyboard",
        "trackpad": "trackpad",
        "palm-rest": "trackpad",
        "hinge": "hinge",
        "lid": "lid",
        "corner": "corner",
        "port": "port",
        "base": "base",
        "body": "body",
    },
    "package": {
        "box": "box",
        "corner": "package_corner",
        "package corner": "package_corner",
        "side": "package_side",
        "seal": "seal",
        "label": "label",
        "shipping label": "label",
        "contents": "contents",
        "item": "item",
        "product inside": "item",
        "inside item": "item",
        "item inside": "item",
    },
}


# ---------------------------------------------------------------------------
# Context bundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClaimContext:
    """Context bundle for a single claim, passed to Pass 1 and Pass 2."""

    # Original claim data
    user_id: str
    image_paths_raw: str
    user_claim: str
    claim_object: str

    # Parsed image info
    image_paths: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    images_exist: list[bool] = field(default_factory=list)
    all_images_valid: bool = True

    # Extracted from conversation
    extracted_issue_type: str = "unknown"
    extracted_object_part: str = "unknown"

    # User history
    user_history: dict | None = None
    has_history_risk: bool = False

    # Evidence requirements
    applicable_requirements: list[dict] = field(default_factory=list)

    # Prompt injection detection
    injection_detected: bool = False
    injection_matches: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core preprocessing functions
# ---------------------------------------------------------------------------

def _extract_claim_info(user_claim: str, claim_object: str) -> tuple[str, str]:
    """
    Extract the most likely issue_type and object_part from the conversation text.

    Strategy:
    - Scan the customer's final statements (last few turns) with higher priority
    - Match against keyword maps
    - Return (issue_type, object_part) or ("unknown", "unknown") if unclear
    """
    text_lower = user_claim.lower()

    # Split into turns and prioritize customer statements
    turns = [t.strip() for t in user_claim.split("|")]
    customer_turns = [t for t in turns if t.lower().startswith("customer:") or t.lower().startswith("cliente:")]

    # Use last customer turns (most specific) for extraction
    # Fall back to full text if no customer turns found
    priority_text = " ".join(customer_turns[-3:]).lower() if customer_turns else text_lower

    # --- Extract issue type ---
    found_issue = "unknown"
    for keyword, issue_type in _ISSUE_KEYWORDS.items():
        if keyword in priority_text:
            found_issue = issue_type
            break  # First match from priority text wins

    # If not found in priority text, try full text
    if found_issue == "unknown":
        for keyword, issue_type in _ISSUE_KEYWORDS.items():
            if keyword in text_lower:
                found_issue = issue_type
                break

    # --- Extract object part ---
    found_part = "unknown"
    part_map = _PART_KEYWORDS.get(claim_object, {})

    # Search longer phrases first (more specific)
    sorted_parts = sorted(part_map.keys(), key=len, reverse=True)

    for keyword in sorted_parts:
        if keyword in priority_text:
            found_part = part_map[keyword]
            break

    # Fallback to full text
    if found_part == "unknown":
        for keyword in sorted_parts:
            if keyword in text_lower:
                found_part = part_map[keyword]
                break

    return found_issue, found_part


def _detect_injection(user_claim: str) -> tuple[bool, list[str]]:
    """
    Scan the claim text for prompt injection patterns.

    Returns (detected: bool, matched_patterns: list[str])
    """
    matches = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(user_claim)
        if match:
            matches.append(match.group(0))

    return len(matches) > 0, matches


def _validate_images(image_paths_raw: str) -> tuple[list[str], list[str], list[bool], bool]:
    """
    Parse the semicolon-separated image paths, extract image IDs,
    and check that each file exists on disk.

    Returns (paths, image_ids, exists_flags, all_valid)
    """
    paths = [p.strip() for p in image_paths_raw.split(";") if p.strip()]
    image_ids = []
    exists_flags = []

    for p in paths:
        # Image ID is filename without extension
        img_id = Path(p).stem
        image_ids.append(img_id)

        # Check if file exists relative to dataset dir
        full_path = DATASET_DIR / p
        exists_flags.append(full_path.exists())

    all_valid = all(exists_flags) if exists_flags else False

    return paths, image_ids, exists_flags, all_valid


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_claim(claim_row: dict) -> ClaimContext:
    """
    Build a full context bundle from a single claim row.

    Steps:
    1. Extract issue type and object part from conversation
    2. Look up user history
    3. Select applicable evidence requirements
    4. Detect prompt injection attempts
    5. Validate image file existence

    Args:
        claim_row: dict with keys user_id, image_paths, user_claim, claim_object

    Returns:
        ClaimContext dataclass with all enriched data
    """
    user_id = claim_row["user_id"]
    image_paths_raw = claim_row["image_paths"]
    user_claim = claim_row["user_claim"]
    claim_object = claim_row["claim_object"]

    # 1. Extract claim info from conversation
    extracted_issue, extracted_part = _extract_claim_info(user_claim, claim_object)

    # 2. Look up user history
    user_history = get_user_history(user_id)
    has_history_risk = False
    if user_history:
        flags = user_history.get("history_flags", "")
        has_history_risk = "user_history_risk" in flags or "manual_review_required" in flags

    # 3. Validate image paths (done early so image_count can inform requirement selection)
    paths, image_ids, exists_flags, all_valid = _validate_images(image_paths_raw)

    # 4. Select applicable evidence requirements (multi-image rule depends on count)
    applicable_reqs = get_evidence_requirements(
        claim_object, extracted_issue, image_count=len(paths)
    )

    # 5. Detect prompt injection
    injection_detected, injection_matches = _detect_injection(user_claim)

    return ClaimContext(
        user_id=user_id,
        image_paths_raw=image_paths_raw,
        user_claim=user_claim,
        claim_object=claim_object,
        image_paths=paths,
        image_ids=image_ids,
        images_exist=exists_flags,
        all_images_valid=all_valid,
        extracted_issue_type=extracted_issue,
        extracted_object_part=extracted_part,
        user_history=user_history,
        has_history_risk=has_history_risk,
        applicable_requirements=applicable_reqs,
        injection_detected=injection_detected,
        injection_matches=injection_matches,
    )
