"""
Configuration module for the Multi-Modal Evidence Review system.

Models:
- Pass 1 (VLM): GPT-4o — multimodal image + text analysis
- Pass 2 (Verdict): GPT-4o — text-only final decision
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & API Key
# ---------------------------------------------------------------------------

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set.")

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.resolve()
DATASET_DIR = REPO_ROOT / "dataset"

CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
IMAGES_SAMPLE_DIR = DATASET_DIR / "images" / "sample"
IMAGES_TEST_DIR = DATASET_DIR / "images" / "test"

OUTPUT_CSV = REPO_ROOT / "output.csv"

# ---------------------------------------------------------------------------
# Model Settings
# ---------------------------------------------------------------------------

# Pass 1: GPT-4o multimodal (image + text → structured JSON)
# Limits: 500 RPM, 30K TPM
PASS1_MODEL = "gpt-4o-2024-11-20"
PASS1_TEMPERATURE = 0.0
PASS1_MAX_OUTPUT_TOKENS = 4096
PASS1_RETRY_ATTEMPTS = 5
PASS1_RETRY_DELAY_SECONDS = 2.0

# Pass 2: GPT-4o (verdict finalizer — text only, no images)
# Limits: 500 RPM, 30K TPM
PASS2_MODEL = "gpt-4o-2024-11-20"
PASS2_TEMPERATURE = 0.0
PASS2_MAX_OUTPUT_TOKENS = 2048
PASS2_RETRY_ATTEMPTS = 4
PASS2_RETRY_DELAY_SECONDS = 2.0

# Rate limiting — both models have 500 RPM, no artificial delays needed
MAX_CONCURRENT_REQUESTS = 5
REQUEST_DELAY_SECONDS = 0.5

# ---------------------------------------------------------------------------
# Allowed Enum Values
# ---------------------------------------------------------------------------

CLAIM_STATUS_VALUES = frozenset([
    "supported", "contradicted", "not_enough_information",
])

ISSUE_TYPE_VALUES = frozenset([
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
])

CAR_OBJECT_PARTS = frozenset([
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender",
    "quarter_panel", "body", "unknown",
])

LAPTOP_OBJECT_PARTS = frozenset([
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown",
])

PACKAGE_OBJECT_PARTS = frozenset([
    "box", "package_corner", "package_side", "seal",
    "label", "contents", "item", "unknown",
])

OBJECT_PARTS_BY_TYPE: dict[str, frozenset[str]] = {
    "car": CAR_OBJECT_PARTS,
    "laptop": LAPTOP_OBJECT_PARTS,
    "package": PACKAGE_OBJECT_PARTS,
}

CLAIM_OBJECT_VALUES = frozenset(["car", "laptop", "package"])

RISK_FLAG_VALUES = frozenset([
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
])

SEVERITY_VALUES = frozenset(["none", "low", "medium", "high", "unknown"])

BOOL_TRUE = "true"
BOOL_FALSE = "false"

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]
