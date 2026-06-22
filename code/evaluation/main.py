"""
Evaluation script — Multi-Modal Evidence Review system.

Runs the full pipeline on sample_claims.csv (which has ground-truth labels)
and compares predictions against expected outputs.

Produces:
- Per-field accuracy metrics
- Per-claim breakdown (pass/fail per field)
- Confusion matrices for key categorical fields
- Overall accuracy score
- Results written to evaluation/eval_output.csv

Usage:
    cd code/
    python evaluation/main.py                  # Run all 20 sample claims
    python evaluation/main.py --limit 5        # Run first 5 only
    python evaluation/main.py --skip-api       # Use cached results if available
"""

import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directory to path so we can import sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_sample_claims
from preprocessor import preprocess_claim
from pass1_vlm import analyze_claim_images
from pass2_validator import validate_and_finalize
from output_writer import build_output_row, write_output_csv
from config import PASS1_MODEL, PASS2_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent
EVAL_OUTPUT_CSV = EVAL_DIR / "eval_output.csv"
EVAL_CACHE_DIR = EVAL_DIR / ".cache"
EVAL_REPORT_PATH = EVAL_DIR / "evaluation_report.md"

# Model names for display
ACTIVE_PASS1_MODEL = PASS1_MODEL
ACTIVE_PASS2_MODEL = PASS2_MODEL

# Fields to evaluate (categorical / exact match)
EVAL_FIELDS = [
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
    "evidence_standard_met",
    "valid_image",
]

# Fields for partial/fuzzy evaluation
PARTIAL_FIELDS = [
    "risk_flags",
    "supporting_image_ids",
]

# Rate limit delay between claims (seconds)
# Both GPT-4o and GPT-4o-mini have 500 RPM — no significant delay needed.
RATE_LIMIT_DELAY = 1


# ---------------------------------------------------------------------------
# Caching (avoid re-running API calls during iterative development)
# ---------------------------------------------------------------------------

def _cache_path(index: int) -> Path:
    return EVAL_CACHE_DIR / f"sample_{index:03d}.json"


def _save_cache(index: int, row: dict):
    EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(index), "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)


def _load_cache(index: int) -> dict | None:
    path = _cache_path(index)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_claim(claim_row: dict, index: int, total: int, use_cache: bool) -> dict:
    """Process a single claim through the full pipeline."""
    if use_cache:
        cached = _load_cache(index)
        if cached:
            logger.info(f"[{index+1}/{total}] {claim_row['user_id']} — cached")
            return cached

    user_id = claim_row["user_id"]
    logger.info(f"[{index+1}/{total}] Processing {user_id}...")

    # Preprocess
    ctx = preprocess_claim(claim_row)

    # Pass 1: VLM
    vlm_analysis = analyze_claim_images(ctx)

    # Pass 2: Validate and finalize
    pass2_output = validate_and_finalize(vlm_analysis, ctx)

    # Build output row
    output_row = build_output_row(claim_row, pass2_output)

    # Cache result
    _save_cache(index, output_row)

    return output_row


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def normalize_bool(val: str) -> str:
    """Normalize boolean-like values for comparison."""
    v = str(val).lower().strip()
    if v in ("true", "1", "yes"):
        return "true"
    if v in ("false", "0", "no"):
        return "false"
    return v


def compute_risk_flag_overlap(expected: str, actual: str) -> dict:
    """Compute precision/recall/F1 for risk flags (set-based)."""
    exp_set = set(f.strip() for f in expected.split(";") if f.strip() and f.strip() != "none")
    act_set = set(f.strip() for f in actual.split(";") if f.strip() and f.strip() != "none")

    if not exp_set and not act_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": True}

    if not exp_set or not act_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": False}

    tp = len(exp_set & act_set)
    precision = tp / len(act_set) if act_set else 0
    recall = tp / len(exp_set) if exp_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exp_set == act_set,
    }


def evaluate_results(
    predictions: list[dict], ground_truth: list[dict]
) -> dict:
    """Compare predictions against ground truth and compute metrics."""
    results = {
        "total_claims": len(predictions),
        "per_field": {},
        "per_claim": [],
        "confusion": defaultdict(lambda: defaultdict(int)),
        "risk_flags_avg": {"precision": 0, "recall": 0, "f1": 0, "exact_match": 0},
    }

    field_correct = {f: 0 for f in EVAL_FIELDS}
    field_total = {f: 0 for f in EVAL_FIELDS}

    risk_metrics = []

    for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        claim_results = {"user_id": pred.get("user_id", "?"), "fields": {}}

        # Exact match fields
        for field in EVAL_FIELDS:
            expected = normalize_bool(truth.get(field, ""))
            actual = normalize_bool(pred.get(field, ""))

            if expected:  # Only evaluate if ground truth has a value
                field_total[field] += 1
                match = expected == actual
                if match:
                    field_correct[field] += 1
                claim_results["fields"][field] = {
                    "expected": expected, "actual": actual, "match": match
                }

                # Confusion matrix for claim_status
                if field == "claim_status":
                    results["confusion"][expected][actual] += 1

        # Risk flags (set-based)
        exp_flags = truth.get("risk_flags", "none")
        act_flags = pred.get("risk_flags", "none")
        flag_metrics = compute_risk_flag_overlap(exp_flags, act_flags)
        risk_metrics.append(flag_metrics)
        claim_results["risk_flags"] = {
            "expected": exp_flags, "actual": act_flags, **flag_metrics
        }

        results["per_claim"].append(claim_results)

    # Aggregate per-field accuracy
    for field in EVAL_FIELDS:
        total = field_total[field]
        correct = field_correct[field]
        results["per_field"][field] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0,
        }

    # Aggregate risk flag metrics
    if risk_metrics:
        results["risk_flags_avg"] = {
            "precision": sum(m["precision"] for m in risk_metrics) / len(risk_metrics),
            "recall": sum(m["recall"] for m in risk_metrics) / len(risk_metrics),
            "f1": sum(m["f1"] for m in risk_metrics) / len(risk_metrics),
            "exact_match": sum(m["exact_match"] for m in risk_metrics) / len(risk_metrics),
        }

    # Overall accuracy (across all evaluated fields)
    total_checks = sum(field_total.values())
    total_correct = sum(field_correct.values())
    results["overall_accuracy"] = total_correct / total_checks if total_checks > 0 else 0

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(metrics: dict):
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 70)
    print("  EVALUATION REPORT")
    print("=" * 70)

    print(f"\n  Model: {ACTIVE_PASS1_MODEL} (Pass 1), {ACTIVE_PASS2_MODEL} (Pass 2)")
    print(f"  Total claims evaluated: {metrics['total_claims']}")
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.1%}")

    # Per-field accuracy
    print("\n  --- Per-Field Accuracy ---")
    print(f"  {'Field':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*10}")
    for field, data in metrics["per_field"].items():
        print(f"  {field:<25} {data['correct']:>8} {data['total']:>6} {data['accuracy']:>10.1%}")

    # Risk flags
    rf = metrics["risk_flags_avg"]
    print(f"\n  --- Risk Flags (Set-Based) ---")
    print(f"  Precision:   {rf['precision']:.2f}")
    print(f"  Recall:      {rf['recall']:.2f}")
    print(f"  F1:          {rf['f1']:.2f}")
    print(f"  Exact Match: {rf['exact_match']:.1%}")

    # Confusion matrix for claim_status
    print(f"\n  --- Claim Status Confusion Matrix ---")
    statuses = ["supported", "contradicted", "not_enough_information"]
    print(f"  {'Expected \\ Actual':<25}", end="")
    for s in statuses:
        print(f" {s[:12]:>12}", end="")
    print()
    for exp in statuses:
        print(f"  {exp:<25}", end="")
        for act in statuses:
            count = metrics["confusion"].get(exp, {}).get(act, 0)
            print(f" {count:>12}", end="")
        print()

    # Per-claim details
    print(f"\n  --- Per-Claim Results ---")
    for claim in metrics["per_claim"]:
        uid = claim["user_id"]
        fields = claim["fields"]
        all_match = all(f.get("match", True) for f in fields.values())
        status_icon = "✅" if all_match else "❌"
        mismatches = [f for f, d in fields.items() if not d.get("match", True)]
        if mismatches:
            print(f"  {status_icon} {uid}: mismatched={mismatches}")
        else:
            print(f"  {status_icon} {uid}: all fields correct")

    print("\n" + "=" * 70)


def write_report_md(metrics: dict, output_path: Path):
    """Write evaluation report as markdown."""
    lines = [
        "# Evaluation Report",
        "",
        f"**Model:** {ACTIVE_PASS1_MODEL} (Pass 1), {ACTIVE_PASS2_MODEL} (Pass 2)",
        f"**Claims evaluated:** {metrics['total_claims']}",
        f"**Overall accuracy:** {metrics['overall_accuracy']:.1%}",
        "",
        "## Per-Field Accuracy",
        "",
        "| Field | Correct | Total | Accuracy |",
        "|-------|---------|-------|----------|",
    ]
    for field, data in metrics["per_field"].items():
        lines.append(
            f"| {field} | {data['correct']} | {data['total']} | {data['accuracy']:.1%} |"
        )

    rf = metrics["risk_flags_avg"]
    lines += [
        "",
        "## Risk Flags (Set-Based)",
        "",
        f"- Precision: {rf['precision']:.2f}",
        f"- Recall: {rf['recall']:.2f}",
        f"- F1: {rf['f1']:.2f}",
        f"- Exact Match: {rf['exact_match']:.1%}",
        "",
        "## Claim Status Confusion Matrix",
        "",
        "| Expected \\ Actual | supported | contradicted | not_enough_information |",
        "|---|---|---|---|",
    ]
    statuses = ["supported", "contradicted", "not_enough_information"]
    for exp in statuses:
        row_vals = [str(metrics["confusion"].get(exp, {}).get(act, 0)) for act in statuses]
        lines.append(f"| {exp} | {' | '.join(row_vals)} |")

    lines += ["", "## Per-Claim Details", ""]
    for claim in metrics["per_claim"]:
        uid = claim["user_id"]
        fields = claim["fields"]
        mismatches = [(f, d) for f, d in fields.items() if not d.get("match", True)]
        if mismatches:
            details = "; ".join(f"{f}: exp={d['expected']}, got={d['actual']}" for f, d in mismatches)
            lines.append(f"- ❌ **{uid}**: {details}")
        else:
            lines.append(f"- ✅ **{uid}**: all correct")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  📄 Report written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate system on sample_claims.csv")
    parser.add_argument("--limit", type=int, default=None, help="Process first N claims only")
    parser.add_argument("--skip-api", action="store_true", help="Use cached results only")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache, re-run all")
    args = parser.parse_args()

    # Load sample claims (with ground truth)
    sample_claims = load_sample_claims()
    if args.limit:
        sample_claims = sample_claims[:args.limit]

    total = len(sample_claims)
    logger.info(f"Evaluating on {total} sample claims...")
    logger.info(f"Model: {ACTIVE_PASS1_MODEL} (Pass 1), {ACTIVE_PASS2_MODEL} (Pass 2)")

    # Process each claim
    predictions = []
    start_time = time.time()

    for i, claim_row in enumerate(sample_claims):
        use_cache = not args.no_cache

        if args.skip_api:
            cached = _load_cache(i)
            if cached:
                predictions.append(cached)
                continue
            else:
                logger.warning(f"[{i+1}/{total}] No cache for {claim_row['user_id']}, skipping")
                # Use a fallback row
                predictions.append(build_output_row(claim_row, {
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": "Skipped (no cache)",
                    "risk_flags": "none",
                    "issue_type": "unknown",
                    "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "Not processed",
                    "supporting_image_ids": "none",
                    "valid_image": "false",
                    "severity": "unknown",
                }))
                continue

        try:
            row = process_claim(claim_row, i, total, use_cache=use_cache)
            predictions.append(row)
        except Exception as e:
            logger.error(f"[{i+1}/{total}] FAILED: {claim_row['user_id']} — {e}")
            predictions.append(build_output_row(claim_row, {
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": f"Error: {str(e)[:80]}",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "Processing failed.",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            }))

        # Rate limit between claims
        if i < total - 1 and not (use_cache and _load_cache(i) is not None):
            time.sleep(RATE_LIMIT_DELAY)

    elapsed = time.time() - start_time
    logger.info(f"Processing complete in {elapsed:.0f}s")

    # Write predictions CSV
    try:
        write_output_csv(predictions, output_path=EVAL_OUTPUT_CSV, validate=True)
    except Exception as e:
        logger.warning(f"Validation error: {e}")
        write_output_csv(predictions, output_path=EVAL_OUTPUT_CSV, validate=False)

    # Evaluate against ground truth
    metrics = evaluate_results(predictions, sample_claims)

    # Print and save report
    print_report(metrics)
    write_report_md(metrics, EVAL_REPORT_PATH)

    return metrics


if __name__ == "__main__":
    main()
