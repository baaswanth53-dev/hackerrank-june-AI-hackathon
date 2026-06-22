"""
Main entry point — Multi-Modal Evidence Review system.

Orchestrates the full pipeline:
1. Load all data (claims, user history, evidence requirements)
2. For each claim: preprocess → Pass 1 VLM → Pass 2 validate → build row
3. Write output.csv with strict validation

Rate limiting strategy (Gemini free tier):
- Pass 1 (gemini-2.5-pro): ~2 RPM on free tier → 30s delay between calls
- Pass 2 verdict (gemini-2.5-flash): ~15 RPM on free tier → 5s delay
- Total per claim: 1 Pass 1 call + 1 Pass 2 call = 2 API calls
- 44 claims × 2 = 88 total API calls
- Sequential processing with delays to stay under rate limits

Usage:
    python main.py                    # Process all claims in claims.csv
    python main.py --sample           # Process sample_claims.csv (for eval)
    python main.py --limit 5          # Process first N claims only
    python main.py --resume           # Resume from last checkpoint
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config import (
    CLAIMS_CSV,
    SAMPLE_CLAIMS_CSV,
    OUTPUT_CSV,
    REPO_ROOT,
)
from data_loader import load_claims, load_sample_claims, load_all_data
from preprocessor import preprocess_claim
from pass1_vlm import analyze_claim_images
from pass2_validator import validate_and_finalize
from output_writer import build_output_row, write_output_csv

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiting constants
# ---------------------------------------------------------------------------

# GPT-4o: 500 RPM, 30K TPM → no delay needed between Pass 1 calls
PASS1_DELAY_SECONDS = 0

# GPT-4o-mini: 500 RPM, 200K TPM → no delay needed
PASS2_DELAY_SECONDS = 0

# 44 claims: ~2-3 minutes total (both models have 500 RPM)


# ---------------------------------------------------------------------------
# Checkpoint support (resume after interruption)
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path(__file__).parent / ".checkpoints"


def _checkpoint_path(claim_index: int) -> Path:
    return CHECKPOINT_DIR / f"claim_{claim_index:03d}.json"


def _save_checkpoint(claim_index: int, result: dict) -> None:
    """Save a processed claim result for resumability."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = _checkpoint_path(claim_index)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


def _load_checkpoint(claim_index: int) -> dict | None:
    """Load a previously saved checkpoint, or None if not found."""
    path = _checkpoint_path(claim_index)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _count_checkpoints() -> int:
    """Count how many checkpoints exist."""
    if not CHECKPOINT_DIR.exists():
        return 0
    return len(list(CHECKPOINT_DIR.glob("claim_*.json")))


# ---------------------------------------------------------------------------
# Single claim processing
# ---------------------------------------------------------------------------

def process_single_claim(
    claim_row: dict,
    claim_index: int,
    total_claims: int,
    use_checkpoint: bool = True,
) -> dict[str, str]:
    """
    Process one claim through the full pipeline:
    preprocess → Pass 1 VLM → Pass 2 validate → build output row.

    If use_checkpoint is True and a checkpoint exists, skip processing.
    """
    # Check for existing checkpoint
    if use_checkpoint:
        cached = _load_checkpoint(claim_index)
        if cached:
            logger.info(
                f"[{claim_index + 1}/{total_claims}] "
                f"{claim_row['user_id']} — loaded from checkpoint"
            )
            return cached

    user_id = claim_row["user_id"]
    claim_object = claim_row["claim_object"]

    logger.info(
        f"[{claim_index + 1}/{total_claims}] "
        f"Processing {user_id} | {claim_object} ..."
    )

    # Step 1: Preprocess
    ctx = preprocess_claim(claim_row)
    logger.info(
        f"  Preprocessed: issue={ctx.extracted_issue_type}, "
        f"part={ctx.extracted_object_part}, "
        f"images={len(ctx.image_paths)}, "
        f"injection={'YES' if ctx.injection_detected else 'no'}, "
        f"history_risk={'YES' if ctx.has_history_risk else 'no'}"
    )

    # Step 2: Pass 1 — VLM analysis
    logger.info("  Calling Pass 1 VLM (Gemini 2.5 Pro)...")
    start_time = time.time()
    vlm_analysis = analyze_claim_images(ctx)
    vlm_elapsed = time.time() - start_time
    logger.info(f"  Pass 1 complete ({vlm_elapsed:.1f}s)")

    # Rate limit delay after Pass 1
    if vlm_elapsed < PASS1_DELAY_SECONDS:
        wait = PASS1_DELAY_SECONDS - vlm_elapsed
        logger.info(f"  Rate limit: waiting {wait:.0f}s before Pass 2...")
        time.sleep(wait)

    # Step 3: Pass 2 — Validate and finalize (includes Flash LLM call)
    logger.info("  Calling Pass 2 Validator (Gemini 2.5 Flash)...")
    start_time = time.time()
    pass2_output = validate_and_finalize(vlm_analysis, ctx)
    pass2_elapsed = time.time() - start_time
    logger.info(
        f"  Pass 2 complete ({pass2_elapsed:.1f}s) → "
        f"status={pass2_output['claim_status']}, "
        f"severity={pass2_output['severity']}"
    )

    # Rate limit delay after Pass 2
    if pass2_elapsed < PASS2_DELAY_SECONDS:
        wait = PASS2_DELAY_SECONDS - pass2_elapsed
        logger.info(f"  Rate limit: waiting {wait:.0f}s...")
        time.sleep(wait)

    # Step 4: Build output row
    output_row = build_output_row(claim_row, pass2_output)

    # Save checkpoint
    if use_checkpoint:
        _save_checkpoint(claim_index, output_row)

    return output_row


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    claims_path: Path | None = None,
    output_path: Path | None = None,
    limit: int | None = None,
    resume: bool = False,
) -> Path:
    """
    Run the full pipeline on all claims.

    Args:
        claims_path: path to claims CSV (default: dataset/claims.csv)
        output_path: where to write output (default: output.csv at repo root)
        limit: process only first N claims (for testing)
        resume: if True, use checkpoints to skip already-processed claims

    Returns:
        Path to the written output CSV.
    """
    # Load data
    logger.info("Loading datasets...")
    data = load_all_data(claims_path)
    claims = data["claims"]

    if limit:
        claims = claims[:limit]

    total = len(claims)
    output = output_path or OUTPUT_CSV

    logger.info(f"Processing {total} claims...")
    logger.info(f"Rate limiting: {PASS1_DELAY_SECONDS}s (Pass 1) + {PASS2_DELAY_SECONDS}s (Pass 2) per claim")

    estimated_time = total * (PASS1_DELAY_SECONDS + PASS2_DELAY_SECONDS)
    logger.info(f"Estimated time: ~{estimated_time // 60}m {estimated_time % 60}s")

    if resume:
        checkpoint_count = _count_checkpoints()
        logger.info(f"Resume mode: {checkpoint_count} checkpoints found")

    # Process each claim sequentially
    output_rows = []
    errors = []

    for i, claim_row in enumerate(claims):
        try:
            row = process_single_claim(
                claim_row,
                claim_index=i,
                total_claims=total,
                use_checkpoint=resume,
            )
            output_rows.append(row)

        except Exception as e:
            logger.error(
                f"[{i + 1}/{total}] FAILED: {claim_row['user_id']} — {e}"
            )
            errors.append((i, claim_row["user_id"], str(e)))

            # Build a safe fallback row so output isn't missing rows
            fallback_row = build_output_row(claim_row, {
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": f"Processing error: {str(e)[:100]}",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "Automated review failed; manual review required.",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            })
            output_rows.append(fallback_row)

    # Write output
    logger.info(f"Writing {len(output_rows)} rows to {output}...")
    try:
        result_path = write_output_csv(output_rows, output_path=output, validate=True)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.info("Attempting to write without validation...")
        result_path = write_output_csv(output_rows, output_path=output, validate=False)

    # Summary
    logger.info("=" * 60)
    logger.info(f"DONE: {len(output_rows)} claims processed")
    if errors:
        logger.warning(f"  {len(errors)} errors (fallback rows used):")
        for idx, uid, err in errors:
            logger.warning(f"    Row {idx + 1} ({uid}): {err[:80]}")
    logger.info(f"Output: {result_path}")

    return result_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Evidence Review — process damage claims"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process sample_claims.csv instead of claims.csv (for evaluation)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N claims (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints (skip already-processed claims)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: output.csv at repo root)",
    )

    args = parser.parse_args()

    # Determine input/output paths
    if args.sample:
        claims_path = SAMPLE_CLAIMS_CSV
        output_path = REPO_ROOT / "output_sample.csv"
        logger.info("Mode: SAMPLE evaluation")
    else:
        claims_path = CLAIMS_CSV
        output_path = OUTPUT_CSV
        logger.info("Mode: FULL test set")

    if args.output:
        output_path = Path(args.output)

    logger.info(f"Input: {claims_path}")
    logger.info(f"Output: {output_path}")

    # Run
    run_pipeline(
        claims_path=claims_path,
        output_path=output_path,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
