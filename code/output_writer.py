"""
Output writer module for the Multi-Modal Evidence Review system.

Responsibilities:
- Validate every row against the enum constraints before writing
- Write output.csv with exact 14-column schema matching the expected format
- All fields are double-quoted (matching the sample CSV style)
- Crash with a clear error if any invalid enum value slips through
"""

import csv
import sys
from pathlib import Path
from typing import Any

from config import (
    OUTPUT_CSV,
    OUTPUT_COLUMNS,
    CLAIM_OBJECT_VALUES,
    CLAIM_STATUS_VALUES,
    ISSUE_TYPE_VALUES,
    SEVERITY_VALUES,
    RISK_FLAG_VALUES,
    OBJECT_PARTS_BY_TYPE,
    BOOL_TRUE,
    BOOL_FALSE,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class OutputValidationError(Exception):
    """Raised when a row has invalid enum values that would corrupt output."""
    pass


def validate_row(row: dict[str, str], row_index: int) -> None:
    """
    Validate a single output row against all enum constraints.
    Raises OutputValidationError with a clear message if anything is invalid.

    Checks:
    1. All 14 required columns are present
    2. claim_object is valid
    3. evidence_standard_met is "true" or "false"
    4. risk_flags: each flag in the semicolon-separated list is valid
    5. issue_type is valid
    6. object_part is valid for the given claim_object
    7. claim_status is valid
    8. valid_image is "true" or "false"
    9. severity is valid
    10. supporting_image_ids format is correct
    """
    errors = []

    # 1. All columns present
    for col in OUTPUT_COLUMNS:
        if col not in row:
            errors.append(f"Missing column: '{col}'")

    if errors:
        raise OutputValidationError(
            f"Row {row_index}: {'; '.join(errors)}"
        )

    # 2. claim_object
    claim_object = row.get("claim_object", "")
    if claim_object not in CLAIM_OBJECT_VALUES:
        errors.append(
            f"claim_object='{claim_object}' not in {sorted(CLAIM_OBJECT_VALUES)}"
        )

    # 3. evidence_standard_met
    esm = row.get("evidence_standard_met", "")
    if esm not in (BOOL_TRUE, BOOL_FALSE):
        errors.append(
            f"evidence_standard_met='{esm}' must be 'true' or 'false'"
        )

    # 4. risk_flags
    risk_flags_str = row.get("risk_flags", "")
    if risk_flags_str:
        flags = [f.strip() for f in risk_flags_str.split(";")]
        for flag in flags:
            if flag not in RISK_FLAG_VALUES:
                errors.append(
                    f"risk_flags contains invalid value: '{flag}'"
                )

    # 5. issue_type
    issue_type = row.get("issue_type", "")
    if issue_type not in ISSUE_TYPE_VALUES:
        errors.append(
            f"issue_type='{issue_type}' not in {sorted(ISSUE_TYPE_VALUES)}"
        )

    # 6. object_part (validated per claim_object)
    object_part = row.get("object_part", "")
    allowed_parts = OBJECT_PARTS_BY_TYPE.get(claim_object, frozenset())
    if allowed_parts and object_part not in allowed_parts:
        errors.append(
            f"object_part='{object_part}' not valid for "
            f"claim_object='{claim_object}'. Allowed: {sorted(allowed_parts)}"
        )

    # 7. claim_status
    claim_status = row.get("claim_status", "")
    if claim_status not in CLAIM_STATUS_VALUES:
        errors.append(
            f"claim_status='{claim_status}' not in {sorted(CLAIM_STATUS_VALUES)}"
        )

    # 8. valid_image
    vi = row.get("valid_image", "")
    if vi not in (BOOL_TRUE, BOOL_FALSE):
        errors.append(
            f"valid_image='{vi}' must be 'true' or 'false'"
        )

    # 9. severity
    severity = row.get("severity", "")
    if severity not in SEVERITY_VALUES:
        errors.append(
            f"severity='{severity}' not in {sorted(SEVERITY_VALUES)}"
        )

    # 10. supporting_image_ids format
    sids = row.get("supporting_image_ids", "")
    if sids and sids != "none":
        parts = [p.strip() for p in sids.split(";")]
        for part in parts:
            if not part.startswith("img_"):
                errors.append(
                    f"supporting_image_ids contains invalid ID: '{part}' "
                    f"(must start with 'img_' or be 'none')"
                )

    if errors:
        raise OutputValidationError(
            f"Row {row_index} (user_id={row.get('user_id', '?')}): "
            + "; ".join(errors)
        )


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

def build_output_row(
    claim_row: dict[str, str],
    pass2_output: dict[str, str],
) -> dict[str, str]:
    """
    Combine original claim input columns with Pass 2 generated output columns
    into a single row dict with all 14 required columns.

    Args:
        claim_row: original input dict (user_id, image_paths, user_claim, claim_object)
        pass2_output: dict from validate_and_finalize() with 10 generated columns

    Returns:
        Complete 14-column dict ready for CSV writing.
    """
    row = {
        # 4 pass-through input columns
        "user_id": claim_row.get("user_id", ""),
        "image_paths": claim_row.get("image_paths", ""),
        "user_claim": claim_row.get("user_claim", ""),
        "claim_object": claim_row.get("claim_object", ""),
        # 10 generated columns
        "evidence_standard_met": pass2_output.get("evidence_standard_met", "false"),
        "evidence_standard_met_reason": pass2_output.get("evidence_standard_met_reason", ""),
        "risk_flags": pass2_output.get("risk_flags", "none"),
        "issue_type": pass2_output.get("issue_type", "unknown"),
        "object_part": pass2_output.get("object_part", "unknown"),
        "claim_status": pass2_output.get("claim_status", "not_enough_information"),
        "claim_status_justification": pass2_output.get("claim_status_justification", ""),
        "supporting_image_ids": pass2_output.get("supporting_image_ids", "none"),
        "valid_image": pass2_output.get("valid_image", "false"),
        "severity": pass2_output.get("severity", "unknown"),
    }
    return row


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def write_output_csv(
    rows: list[dict[str, str]],
    output_path: Path | None = None,
    validate: bool = True,
) -> Path:
    """
    Write the final output.csv with exact 14-column schema.

    Args:
        rows: list of complete row dicts (14 columns each)
        output_path: where to write (defaults to repo root output.csv)
        validate: if True, validate every row before writing (recommended)

    Returns:
        Path to the written file.

    Raises:
        OutputValidationError: if validate=True and any row has invalid enums
    """
    path = output_path or OUTPUT_CSV

    # Validate all rows first (fail fast before writing anything)
    if validate:
        for i, row in enumerate(rows):
            validate_row(row, row_index=i + 1)

    # Write CSV with double-quoting (matching the expected format)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            quotechar='"',
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Output written: {path} ({len(rows)} rows)")
    return path
