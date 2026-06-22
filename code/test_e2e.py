"""
End-to-end test script — runs 2 real claims through the full pipeline
with real Gemini API calls. Prints every intermediate step.

Test cases:
1. user_005 (car, door dent, single image) — basic claim
2. user_030 (package, torn seal, Hindi conversation) — multilingual

Usage:
    cd code/
    python test_e2e.py
"""

import json
import sys
import logging
from pprint import pprint

from data_loader import load_claims
from preprocessor import preprocess_claim, ClaimContext
from pass1_vlm import analyze_claim_images
from pass2_validator import (
    enforce_enums,
    match_evidence_rules,
    compute_risk_flags,
    adversarial_guard,
    finalize_verdict,
)
from output_writer import build_output_row, validate_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_context(ctx: ClaimContext):
    """Print the preprocessor context bundle."""
    print(f"  user_id: {ctx.user_id}")
    print(f"  claim_object: {ctx.claim_object}")
    print(f"  image_paths: {ctx.image_paths}")
    print(f"  image_ids: {ctx.image_ids}")
    print(f"  images_exist: {ctx.images_exist}")
    print(f"  all_images_valid: {ctx.all_images_valid}")
    print(f"  extracted_issue_type: {ctx.extracted_issue_type}")
    print(f"  extracted_object_part: {ctx.extracted_object_part}")
    print(f"  has_history_risk: {ctx.has_history_risk}")
    print(f"  injection_detected: {ctx.injection_detected}")
    if ctx.injection_matches:
        print(f"  injection_matches: {ctx.injection_matches}")
    if ctx.user_history:
        print(f"  user_history_flags: {ctx.user_history.get('history_flags')}")
        print(f"  user_history_summary: {ctx.user_history.get('history_summary')}")
    print(f"  applicable_requirements: {len(ctx.applicable_requirements)} rules")
    for req in ctx.applicable_requirements:
        print(f"    - {req['requirement_id']}: {req['applies_to']}")


def run_claim(claim_row: dict, label: str):
    """Run a single claim through the full pipeline with verbose output."""

    print_section(f"CLAIM: {label}")
    print(f"  user_id: {claim_row['user_id']}")
    print(f"  claim_object: {claim_row['claim_object']}")
    print(f"  image_paths: {claim_row['image_paths']}")
    print(f"  user_claim: {claim_row['user_claim'][:200]}...")

    # --- STEP: Preprocess ---
    print_section("PREPROCESSOR — Context Bundle")
    ctx = preprocess_claim(claim_row)
    print_context(ctx)

    # --- STEP: Pass 1 VLM ---
    print_section("PASS 1 — VLM Raw JSON Response")
    import time
    start = time.time()
    vlm_analysis = analyze_claim_images(ctx)
    elapsed = time.time() - start
    print(f"  [Completed in {elapsed:.1f}s]")
    print()
    print(json.dumps(vlm_analysis, indent=2, ensure_ascii=False))

    # --- STEP: Pass 2 Step 1 — Enum Enforcer ---
    print_section("PASS 2 STEP 1 — Enum Enforcer Output")
    step1 = enforce_enums(vlm_analysis, ctx)
    print(f"  issue_type: {step1['issue_type']}")
    print(f"  object_part: {step1['object_part']}")
    print(f"  severity: {step1['severity']}")
    print(f"  claim_status: {step1['claim_status']}")
    print(f"  supporting_image_ids: {step1['supporting_image_ids']}")
    print(f"  raw_justification: {step1['raw_justification'][:200]}")

    # --- STEP: Pass 2 Step 2 — Evidence Rule Matcher ---
    print_section("PASS 2 STEP 2 — Evidence Rule Matcher Output")
    step2 = match_evidence_rules(step1, ctx)
    print(f"  evidence_standard_met: {step2['evidence_standard_met']}")
    print(f"  evidence_standard_met_reason: {step2['evidence_standard_met_reason']}")

    # --- STEP: Pass 2 Step 3 — Risk Flag Computer ---
    print_section("PASS 2 STEP 3 — Risk Flag Computer Output")
    step3 = compute_risk_flags(step2, ctx)
    print(f"  risk_flags: {step3['risk_flags']}")

    # --- STEP: Pass 2 Step 4 — Adversarial Guard ---
    print_section("PASS 2 STEP 4 — Adversarial Guard Output")
    step4 = adversarial_guard(step3, ctx)
    print(f"  risk_flags: {step4['risk_flags']}")
    print(f"  adversarial_detected: {step4.get('_adversarial_detected', False)}")

    # --- STEP: Pass 2 Step 5 — Verdict Finalizer (LLM call) ---
    print_section("PASS 2 STEP 5 — Verdict Finalizer (Gemini Flash)")
    start = time.time()
    step5 = finalize_verdict(step4, ctx)
    elapsed = time.time() - start
    print(f"  [Completed in {elapsed:.1f}s]")
    print(f"  claim_status: {step5['claim_status']}")
    print(f"  claim_status_justification: {step5['claim_status_justification']}")
    print(f"  supporting_image_ids: {step5['supporting_image_ids_str']}")
    print(f"  severity: {step5['severity']}")
    print(f"  valid_image: {step5['valid_image']}")
    print(f"  evidence_standard_met: {step5['evidence_standard_met']}")
    print(f"  evidence_standard_met_reason: {step5['evidence_standard_met_reason']}")
    print(f"  risk_flags: {step5['risk_flags']}")

    # --- Build final output row ---
    print_section("FINAL OUTPUT ROW")
    pass2_output = {
        "evidence_standard_met": step5["evidence_standard_met"],
        "evidence_standard_met_reason": step5["evidence_standard_met_reason"],
        "risk_flags": step5["risk_flags"],
        "issue_type": step5["issue_type"],
        "object_part": step5["object_part"],
        "claim_status": step5["claim_status"],
        "claim_status_justification": step5["claim_status_justification"],
        "supporting_image_ids": step5["supporting_image_ids_str"],
        "valid_image": step5["valid_image"],
        "severity": step5["severity"],
    }
    output_row = build_output_row(claim_row, pass2_output)

    # Validate
    try:
        validate_row(output_row, row_index=1)
        print("  ✓ VALIDATION PASSED")
    except Exception as e:
        print(f"  ✗ VALIDATION FAILED: {e}")

    print()
    for k, v in output_row.items():
        if k == "user_claim":
            print(f"  {k}: {v[:100]}...")
        else:
            print(f"  {k}: {v}")

    return output_row


def main():
    import time
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end test with live Gemini API calls",
        usage="python test_e2e.py [--sample] user_id1 user_id2 ...",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Load claims from sample_claims.csv instead of claims.csv",
    )
    parser.add_argument(
        "user_ids",
        nargs="*",
        default=["user_005", "user_030"],
        help="User IDs to test (default: user_005 user_030)",
    )

    args = parser.parse_args()
    user_ids = args.user_ids

    # Load from sample or test claims
    if args.sample:
        from data_loader import load_sample_claims
        claims = load_sample_claims()
        source = "sample_claims.csv"
    else:
        claims = load_claims()
        source = "claims.csv"

    # Find the first matching claim for each requested user_id, preserving order
    selected = []
    for uid in user_ids:
        match = next((c for c in claims if c["user_id"] == uid), None)
        if match is None:
            print(f"ERROR: {uid} not found in {source}")
            print(f"  Available user_ids: {sorted(set(c['user_id'] for c in claims))}")
            sys.exit(1)
        selected.append((uid, match))

    print("\n" + "=" * 70)
    print(f"  END-TO-END TEST: {len(selected)} Real Claims with Live Gemini API")
    print(f"  Source: {source}")
    print("=" * 70)
    for uid, claim in selected:
        n_imgs = len([p for p in claim["image_paths"].split(";") if p.strip()])
        print(f"  - {uid}: {claim['claim_object']} ({n_imgs} image(s))")
    print()

    # Process each claim with rate-limit pauses in between
    results = []
    for i, (uid, claim) in enumerate(selected):
        label = f"{uid} — {claim['claim_object']}"
        row = run_claim(claim, label)
        results.append((uid, row))

        # If sample data has expected output, show comparison
        if args.sample and "claim_status" in claim:
            print(f"\n  📋 EXPECTED vs ACTUAL:")
            print(f"    claim_status:        expected={claim.get('claim_status'):<25} actual={row['claim_status']}")
            print(f"    issue_type:          expected={claim.get('issue_type'):<25} actual={row['issue_type']}")
            print(f"    object_part:         expected={claim.get('object_part'):<25} actual={row['object_part']}")
            print(f"    severity:            expected={claim.get('severity'):<25} actual={row['severity']}")
            print(f"    evidence_met:        expected={claim.get('evidence_standard_met'):<25} actual={row['evidence_standard_met']}")
            print(f"    valid_image:         expected={claim.get('valid_image'):<25} actual={row['valid_image']}")
            print(f"    supporting_imgs:     expected={claim.get('supporting_image_ids'):<25} actual={row['supporting_image_ids']}")
            print(f"    risk_flags:          expected={claim.get('risk_flags')}")
            print(f"                         actual  ={row['risk_flags']}")

        if i < len(selected) - 1:
            print("\n  ⏳ Rate limit pause (35s) before next claim...")
            time.sleep(35)

    # Summary
    print_section("SUMMARY")
    for uid, row in results:
        print(
            f"  {uid}: status={row['claim_status']}, "
            f"severity={row['severity']}, "
            f"evidence_met={row['evidence_standard_met']}, "
            f"risk_flags={row['risk_flags']}"
        )

    if args.sample:
        # Score: how many fields match expected
        total_checks = 0
        matches = 0
        key_fields = ["claim_status", "issue_type", "object_part", "severity",
                      "evidence_standard_met", "valid_image"]
        for uid, row in results:
            expected_claim = next(c for c in claims if c["user_id"] == uid)
            for field in key_fields:
                if field in expected_claim:
                    total_checks += 1
                    if row.get(field) == expected_claim.get(field):
                        matches += 1
        print(f"\n  Accuracy: {matches}/{total_checks} fields match expected "
              f"({100*matches/total_checks:.0f}%)" if total_checks else "")

    # Write results to test_sample_output.csv
    from output_writer import write_output_csv
    from pathlib import Path
    output_path = Path(__file__).parent / "test_sample_output.csv"
    try:
        all_rows = [row for _, row in results]
        write_output_csv(all_rows, output_path=output_path, validate=True)
        print(f"\n  📄 Results written to: {output_path}")
    except Exception as e:
        print(f"\n  ⚠️ CSV write failed (validation): {e}")
        write_output_csv(all_rows, output_path=output_path, validate=False)
        print(f"  📄 Written without validation to: {output_path}")

    print("\n  ✓ End-to-end test complete!")


if __name__ == "__main__":
    main()
