"""
Pass 2 Validator Module — Multi-Modal Evidence Review system.

Takes raw VLM analysis from Pass 1 and runs 5 sequential steps:
1. Enum Enforcer       — fuzzy-maps VLM strings to valid enum values
2. Evidence Rule Matcher — checks evidence requirements against VLM findings
3. Risk Flag Computer  — merges image quality + user history + adversarial flags
4. Adversarial Guard   — detects injection in images and text
5. Verdict Finalizer   — LLM call (GPT-4o-mini) for final decision

Steps 1-4: deterministic Python. Step 5: lightweight LLM micro-call.

Models:
- Pass 2 verdict: GPT-4o-mini (via OpenAI API)
"""

import json
import re
import time
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from config import (
    OPENAI_API_KEY,
    ISSUE_TYPE_VALUES,
    CLAIM_STATUS_VALUES,
    SEVERITY_VALUES,
    RISK_FLAG_VALUES,
    OBJECT_PARTS_BY_TYPE,
    BOOL_TRUE,
    BOOL_FALSE,
    PASS2_MODEL,
    PASS2_TEMPERATURE,
    PASS2_MAX_OUTPUT_TOKENS,
    PASS2_RETRY_ATTEMPTS,
    PASS2_RETRY_DELAY_SECONDS,
)
from preprocessor import ClaimContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client for verdict
# ---------------------------------------------------------------------------

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_VERDICT_PROMPT: str | None = None


def _get_verdict_prompt() -> str:
    global _VERDICT_PROMPT
    if _VERDICT_PROMPT is None:
        _VERDICT_PROMPT = (_PROMPTS_DIR / "pass2_verdict.txt").read_text(encoding="utf-8")
    return _VERDICT_PROMPT


# ===========================================================================
# STEP 1: ENUM ENFORCER (fuzzy matching)
# BUG FIX #3: "partially_supported" must NOT match "supported".
# The substring check now requires the candidate to be the FULL cleaned value,
# not just a substring of it. We check allowed-in-cleaned but NOT cleaned-in-allowed.
# ===========================================================================

_SYNONYMS: dict[str, str] = {
    "moderate": "medium", "severe": "high", "minor": "low", "slight": "low",
    "significant": "high", "image_blurry": "blurry_image", "blurry": "blurry_image",
    "obstructed": "cropped_or_obstructed", "cropped": "cropped_or_obstructed",
    "glare": "low_light_or_glare", "low_light": "low_light_or_glare",
    "manipulation": "possible_manipulation", "screenshot": "non_original_image",
    "non_original": "non_original_image",
}


def _fuzzy_match(value: str, allowed: frozenset[str], fallback: str = "unknown") -> str:
    """
    Fuzzy-match a VLM string to the closest valid enum value.
    BUG FIX #3: Only checks if an allowed value equals the cleaned input,
    NOT if allowed is a substring of cleaned (which caused "partially_supported"
    to match "supported").
    """
    if not value or not value.strip():
        return fallback

    cleaned = value.lower().strip().replace(" ", "_").replace("-", "_")

    # 1. Exact match
    if cleaned in allowed:
        return cleaned

    # 2. Check if an allowed value appears as a COMPLETE word/segment in the input
    #    (split by underscores/spaces). NOT simple substring — "supported" must not
    #    match inside "partially_supported".
    #    Instead: check if cleaned starts with an allowed value AND the next char
    #    is end-of-string. This prevents partial prefix matching.
    # REMOVED: old substring check that caused Bug #3

    # 3. Synonyms
    if cleaned in _SYNONYMS and _SYNONYMS[cleaned] in allowed:
        return _SYNONYMS[cleaned]

    # 4. Similarity ratio (threshold 0.7 for stricter matching)
    best_score = 0.0
    best_match = fallback
    for candidate in allowed:
        score = SequenceMatcher(None, cleaned, candidate).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= 0.7:
        return best_match

    return fallback


def enforce_enums(vlm_analysis: dict[str, Any], ctx: ClaimContext) -> dict[str, Any]:
    """Step 1: Map all VLM output fields to valid enum values."""
    overall = vlm_analysis.get("overall_assessment", {})
    evidence = vlm_analysis.get("evidence_sufficiency", {})
    per_image = vlm_analysis.get("per_image_analysis", [])
    allowed_parts = OBJECT_PARTS_BY_TYPE.get(ctx.claim_object, frozenset(["unknown"]))

    # Issue type
    raw_issue = overall.get("primary_issue_type", "unknown")
    issue_type = _fuzzy_match(raw_issue, ISSUE_TYPE_VALUES, fallback="unknown")

    # Object part
    raw_part = overall.get("primary_object_part", "unknown")
    object_part = _fuzzy_match(raw_part, allowed_parts, fallback="unknown")

    # Severity
    raw_severity = overall.get("overall_severity", "unknown")
    severity = _fuzzy_match(raw_severity, SEVERITY_VALUES, fallback="unknown")

    # Claim alignment → claim_status
    # BUG FIX #3: Explicit mapping for "partially_supported" and "partially supported"
    raw_alignment = overall.get("claim_alignment", "unclear")
    alignment_map = {
        "supported": "supported",
        "contradicted": "contradicted",
        "contradicts": "contradicted",
        "unclear": "not_enough_information",
        "insufficient_evidence": "not_enough_information",
        "not_enough_information": "not_enough_information",
        # BUG FIX #3: "partially" anything → not_enough_information
        "partially_supported": "not_enough_information",
        "partially supported": "not_enough_information",
        "partially_contradicted": "contradicted",
    }
    cleaned_alignment = raw_alignment.lower().strip().replace(" ", "_")
    claim_status = alignment_map.get(
        cleaned_alignment,
        _fuzzy_match(raw_alignment, CLAIM_STATUS_VALUES, "not_enough_information")
    )

    # Supporting image IDs
    raw_supporting = overall.get("best_supporting_image_ids", [])
    if isinstance(raw_supporting, list):
        supporting_ids = [str(s).strip() for s in raw_supporting if s]
    elif isinstance(raw_supporting, str):
        supporting_ids = [s.strip() for s in raw_supporting.split(";") if s.strip()]
    else:
        supporting_ids = []
    valid_ids = set(ctx.image_ids)
    supporting_ids = [sid for sid in supporting_ids if sid in valid_ids]

    # Raw justification
    raw_justification = overall.get("alignment_reasoning", "")

    return {
        "issue_type": issue_type,
        "object_part": object_part,
        "severity": severity,
        "claim_status": claim_status,
        "supporting_image_ids": supporting_ids,
        "raw_justification": raw_justification,
        "_per_image_analysis": per_image,
        "_overall_assessment": overall,
        "_evidence": evidence,
    }


# ===========================================================================
# STEP 2: EVIDENCE RULE MATCHER
# ===========================================================================

def match_evidence_rules(enforced: dict[str, Any], ctx: ClaimContext) -> dict[str, Any]:
    """Step 2: Check whether images meet evidence requirements."""
    per_image = enforced.get("_per_image_analysis", [])

    object_visible = any(
        ctx.claim_object.lower() in str(img.get("visible_object", "")).lower()
        and str(img.get("relevance_to_claim", "")).lower() in ("high", "medium", "relevant", "partially_relevant")
        for img in per_image
    )

    claimed_part = enforced.get("object_part", ctx.extracted_object_part)
    part_visible = any(
        (claimed_part.lower().replace("_", " ") in str(img.get("visible_part", "")).lower()
         or claimed_part.lower() in str(img.get("visible_part", "")).lower().replace(" ", "_"))
        for img in per_image
    ) if claimed_part != "unknown" else True

    if not per_image:
        evidence_met = False
        reason = "No images available for analysis."
    elif not object_visible:
        evidence_met = False
        reason = f"The claimed {ctx.claim_object} is not clearly visible in any submitted image."
    elif not part_visible and claimed_part != "unknown":
        evidence_met = False
        reason = f"The claimed {claimed_part.replace('_', ' ')} is not visible in the submitted images."
    else:
        evidence_met = True
        reason = enforced.get("raw_justification", "Evidence requirements met.")

    # VLM override
    evidence_section = enforced.get("_evidence", {})
    if evidence_section:
        vlm_met = evidence_section.get("evidence_standard_met")
        if vlm_met is False or str(vlm_met).lower() == "false":
            evidence_met = False
            vlm_reason = evidence_section.get("reason", "")
            if vlm_reason:
                reason = vlm_reason

    enforced["evidence_standard_met"] = BOOL_TRUE if evidence_met else BOOL_FALSE
    enforced["evidence_standard_met_reason"] = reason
    return enforced


# ===========================================================================
# STEP 3: RISK FLAG COMPUTER
# BUG FIX #1: Now detects wrong_object across images (different objects in
# multi-image submissions), not just vs. the claimed object type.
# ===========================================================================

def compute_risk_flags(enforced: dict[str, Any], ctx: ClaimContext) -> dict[str, Any]:
    """Step 3: Compute risk_flags by merging all sources."""
    flags: set[str] = set()
    per_image = enforced.get("_per_image_analysis", [])
    overall = enforced.get("_overall_assessment", {})

    # Source 1: Image quality issues
    for img in per_image:
        for issue in img.get("quality_issues", []):
            matched = _fuzzy_match(issue, RISK_FLAG_VALUES, fallback="")
            if matched:
                flags.add(matched)

    # Source 1b: Authenticity concerns
    for concern in overall.get("authenticity_concerns", []):
        matched = _fuzzy_match(concern, RISK_FLAG_VALUES, fallback="")
        if matched:
            flags.add(matched)

    # Source 1c: Image originality
    for img in per_image:
        if img.get("is_original_photo") is False:
            flags.add("non_original_image")

    # BUG FIX #1: Cross-image object mismatch detection
    # Check if different images show different objects (not just vs claimed type)
    visible_objects = set()
    for img in per_image:
        obj = str(img.get("visible_object", "")).lower().strip()
        if obj and obj not in ("", "unclear", "other"):
            visible_objects.add(obj)
    # If multiple distinct objects found across images → wrong_object
    if len(visible_objects) > 1:
        flags.add("wrong_object")
    # Also: single image showing wrong object vs claim
    for img in per_image:
        obj = str(img.get("visible_object", "")).lower().strip()
        if obj and obj not in ("", ctx.claim_object.lower(), "unclear", "other"):
            flags.add("wrong_object")

    # Source 2: User history
    if ctx.user_history:
        history_flags_str = ctx.user_history.get("history_flags", "")
        if "user_history_risk" in history_flags_str:
            flags.add("user_history_risk")
        if "manual_review_required" in history_flags_str:
            flags.add("manual_review_required")

    # Source 3: Injection detection
    if ctx.injection_detected:
        flags.add("text_instruction_present")
    for img in per_image:
        if img.get("contains_text_instructions", False):
            flags.add("text_instruction_present")

    # Claim mismatch
    if enforced["claim_status"] == "contradicted":
        flags.add("claim_mismatch")

    # Damage not visible
    if enforced.get("evidence_standard_met") == BOOL_FALSE:
        reason_lower = enforced.get("evidence_standard_met_reason", "").lower()
        if "not visible" in reason_lower or "not show" in reason_lower:
            flags.add("damage_not_visible")

    # Business rule: manual_review_required triggers
    trigger_flags = {"user_history_risk", "claim_mismatch", "text_instruction_present", "non_original_image"}
    if flags & trigger_flags:
        flags.add("manual_review_required")

    flags.discard("none")
    flags.discard("")
    enforced["risk_flags"] = ";".join(sorted(flags)) if flags else "none"
    return enforced


# ===========================================================================
# STEP 4: ADVERSARIAL GUARD
# ===========================================================================

def adversarial_guard(enforced: dict[str, Any], ctx: ClaimContext) -> dict[str, Any]:
    """Step 4: Detect and flag injection attempts."""
    per_image = enforced.get("_per_image_analysis", [])
    text_in_images = any(img.get("contains_text_instructions", False) for img in per_image)

    if not ctx.injection_detected and not text_in_images:
        return enforced

    current_flags = set(enforced.get("risk_flags", "none").split(";"))
    current_flags.discard("none")
    current_flags.add("text_instruction_present")
    current_flags.add("manual_review_required")
    enforced["risk_flags"] = ";".join(sorted(current_flags))
    enforced["_adversarial_detected"] = True
    return enforced


# ===========================================================================
# STEP 5: VERDICT FINALIZER (Gemini 3.1 Flash Lite)
# ===========================================================================

def _build_vlm_summary(enforced: dict[str, Any]) -> str:
    per_image = enforced.get("_per_image_analysis", [])
    lines = []
    for img in per_image:
        lines.append(
            f"  {img.get('image_id', '?')}: object={img.get('visible_object', '?')}, "
            f"part={img.get('visible_part', '?')}, damage={img.get('visible_damage_type', 'none')}, "
            f"severity={img.get('damage_severity', '?')}, relevance={img.get('relevance_to_claim', '?')}"
        )
    overall = enforced.get("_overall_assessment", {})
    if overall:
        lines.append(f"  Overall: issue={overall.get('primary_issue_type', '?')}, "
                     f"part={overall.get('primary_object_part', '?')}, "
                     f"alignment={overall.get('claim_alignment', '?')}")
        reasoning = overall.get("alignment_reasoning", "")
        if reasoning:
            lines.append(f"  Reasoning: {reasoning}")
    return "\n".join(lines) if lines else "No VLM analysis available."


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    recovered: dict[str, Any] = {}
    pair_pattern = re.compile(r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"')
    for match in pair_pattern.finditer(text):
        recovered[match.group(1)] = match.group(2)
    trailing = re.search(r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)$', text)
    if trailing and trailing.group(1) not in recovered:
        recovered[trailing.group(1)] = trailing.group(2).strip()
    return recovered if recovered else None


def _call_verdict_llm(prompt: str) -> dict[str, Any] | None:
    """Call GPT-4o-mini for verdict."""
    client = _get_openai_client()
    last_error = None

    for attempt in range(1, PASS2_RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=PASS2_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=PASS2_TEMPERATURE,
                max_tokens=PASS2_MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content
            if not text:
                raise ValueError("Empty verdict response")
            return json.loads(text)

        except json.JSONDecodeError as e:
            logger.warning(f"Verdict attempt {attempt}: JSON error: {e}")
            last_error = e
            if attempt < PASS2_RETRY_ATTEMPTS:
                time.sleep(PASS2_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)))
                continue
            break

        except Exception as e:
            error_str = str(e).lower()
            last_error = e
            is_retryable = any(code in error_str for code in [
                "429", "rate_limit", "500", "503", "unavailable", "timeout", "connection",
            ])
            if is_retryable and attempt < PASS2_RETRY_ATTEMPTS:
                delay = PASS2_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(f"Verdict attempt {attempt} retryable: {e}. Retry in {delay}s")
                time.sleep(delay)
                continue
            logger.error(f"Verdict LLM failed: {e}")
            break

    logger.error(f"Verdict LLM exhausted retries. Last: {last_error}")
    return None


def finalize_verdict(enforced: dict[str, Any], ctx: ClaimContext) -> dict[str, Any]:
    """
    Step 5: Verdict Finalizer — GPT-4o-mini LLM call.

    BUG FIX #2: If verdict says "no damage" or "contradicted" but issue_type
    still shows a damage type, reset issue_type to "none".
    """
    template = _get_verdict_prompt()

    history_summary = "No history available."
    if ctx.user_history:
        history_summary = (
            f"Past: {ctx.user_history.get('past_claim_count', '0')}, "
            f"Rejected: {ctx.user_history.get('rejected_claim', '0')}, "
            f"Flags: {ctx.user_history.get('history_flags', 'none')}, "
            f"{ctx.user_history.get('history_summary', '')}"
        )

    vlm_summary = _build_vlm_summary(enforced)

    prompt = template.replace("{claim_object}", ctx.claim_object)
    prompt = prompt.replace("{user_claim}", ctx.user_claim[:500])
    prompt = prompt.replace("{evidence_standard_met}", enforced.get("evidence_standard_met", BOOL_FALSE))
    prompt = prompt.replace("{evidence_standard_met_reason}", enforced.get("evidence_standard_met_reason", ""))
    prompt = prompt.replace("{risk_flags}", enforced.get("risk_flags", "none"))
    prompt = prompt.replace("{vlm_analysis_summary}", vlm_summary)
    prompt = prompt.replace("{history_summary}", history_summary)

    # Call verdict LLM
    verdict = _call_verdict_llm(prompt)

    if verdict:
        final_status = _fuzzy_match(
            verdict.get("claim_status", enforced["claim_status"]),
            CLAIM_STATUS_VALUES, "not_enough_information"
        )
        final_justification = verdict.get("claim_status_justification", enforced.get("raw_justification", ""))
        final_supporting = verdict.get("supporting_image_ids", "none")
        final_severity = _fuzzy_match(
            verdict.get("severity", enforced["severity"]),
            SEVERITY_VALUES, "unknown"
        )
        # Use verdict's issue_type and object_part if provided (more calibrated)
        verdict_issue = verdict.get("issue_type", "")
        if verdict_issue:
            allowed_parts = OBJECT_PARTS_BY_TYPE.get(ctx.claim_object, frozenset(["unknown"]))
            matched_issue = _fuzzy_match(verdict_issue, ISSUE_TYPE_VALUES, fallback="")
            if matched_issue:
                enforced["issue_type"] = matched_issue
        verdict_part = verdict.get("object_part", "")
        if verdict_part:
            allowed_parts = OBJECT_PARTS_BY_TYPE.get(ctx.claim_object, frozenset(["unknown"]))
            matched_part = _fuzzy_match(verdict_part, allowed_parts, fallback="")
            if matched_part:
                enforced["object_part"] = matched_part
        verdict_reason = verdict.get("evidence_standard_met_reason", "")
        if verdict_reason:
            enforced["evidence_standard_met_reason"] = verdict_reason
    else:
        # Deterministic fallback
        final_status = enforced["claim_status"]
        final_justification = enforced.get("raw_justification", "")
        final_supporting = ";".join(enforced["supporting_image_ids"]) or "none"
        final_severity = enforced["severity"]

    # === CONSISTENCY RULES ===

    # Rule: evidence not met + supported → downgrade to NEI
    if enforced["evidence_standard_met"] == BOOL_FALSE and final_status == "supported":
        final_status = "not_enough_information"

    # BUG FIX #2: If status is contradicted and the VLM saw no damage on the
    # claimed part, issue_type should be "none" (not whatever the VLM reported
    # about other parts). Also applies when severity is "none".
    if final_status == "contradicted" and final_severity == "none":
        enforced["issue_type"] = "none"

    # BUG FIX #2: If status is not_enough_information, issue_type should reflect
    # that we couldn't determine it.
    if final_status == "not_enough_information":
        if enforced["issue_type"] not in ("none", "unknown"):
            enforced["issue_type"] = "unknown"
        final_severity = "unknown"

    # Validate supporting_image_ids
    if isinstance(final_supporting, list):
        final_supporting = ";".join(final_supporting)
    if final_supporting and final_supporting != "none":
        valid_ids = set(ctx.image_ids)
        parts = [s.strip() for s in final_supporting.split(";") if s.strip()]
        validated = [s for s in parts if s in valid_ids]
        final_supporting = ";".join(validated) if validated else "none"

    # Justification fallback
    if not final_justification:
        if final_status == "supported":
            final_justification = f"Image evidence supports the {enforced['issue_type']} claim."
        elif final_status == "contradicted":
            final_justification = "The visual evidence contradicts the user's claim."
        else:
            final_justification = "Insufficient evidence to verify the claim."
    if len(final_justification) > 300:
        final_justification = final_justification[:297] + "..."

    # valid_image: at least one relevant + original image
    per_image = enforced.get("_per_image_analysis", [])
    has_usable = any(
        str(img.get("relevance_to_claim", "")).lower() in ("high", "medium", "relevant", "partially_relevant")
        and img.get("is_original_photo", True) is not False
        for img in per_image
    ) if per_image else False

    enforced["claim_status"] = final_status
    enforced["claim_status_justification"] = final_justification
    enforced["supporting_image_ids_str"] = final_supporting
    enforced["severity"] = final_severity
    enforced["valid_image"] = BOOL_TRUE if has_usable else BOOL_FALSE
    return enforced


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def validate_and_finalize(vlm_analysis: dict[str, Any], ctx: ClaimContext) -> dict[str, str]:
    """Run the full Pass 2 validation pipeline. Returns output dict."""
    result = enforce_enums(vlm_analysis, ctx)
    result = match_evidence_rules(result, ctx)
    result = compute_risk_flags(result, ctx)
    result = adversarial_guard(result, ctx)
    result = finalize_verdict(result, ctx)

    return {
        "evidence_standard_met": result["evidence_standard_met"],
        "evidence_standard_met_reason": result["evidence_standard_met_reason"],
        "risk_flags": result["risk_flags"],
        "issue_type": result["issue_type"],
        "object_part": result["object_part"],
        "claim_status": result["claim_status"],
        "claim_status_justification": result["claim_status_justification"],
        "supporting_image_ids": result["supporting_image_ids_str"],
        "valid_image": result["valid_image"],
        "severity": result["severity"],
    }
