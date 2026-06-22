# Multi-Modal Evidence Review System

An automated damage claim verification system that analyzes submitted images against user claims using a two-pass VLM pipeline with GPT-4o.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Orchestrator)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │data_loader│──▶│ preprocessor │──▶│     pass1_vlm          │  │
│  │           │   │              │   │  (GPT-4o multimodal)   │  │
│  └──────────┘   └──────────────┘   └───────────┬────────────┘  │
│                                                  │               │
│                                                  ▼               │
│                                     ┌────────────────────────┐  │
│                                     │   pass2_validator       │  │
│                                     │  5-step pipeline:       │  │
│                                     │  1. Enum Enforcer       │  │
│                                     │  2. Evidence Matcher    │  │
│                                     │  3. Risk Flag Computer  │  │
│                                     │  4. Adversarial Guard   │  │
│                                     │  5. Verdict Finalizer   │  │
│                                     │     (GPT-4o LLM call)   │  │
│                                     └───────────┬────────────┘  │
│                                                  │               │
│                                                  ▼               │
│                                     ┌────────────────────────┐  │
│                                     │    output_writer        │  │
│                                     │  (validates + writes    │  │
│                                     │   output.csv)           │  │
│                                     └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Setup
cd code/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run on test claims (produces output.csv)
python main.py

# 4. Run evaluation on sample claims
python evaluation/main.py

# 5. Run end-to-end test on specific claims
python test_e2e.py --sample user_001 user_005
```

## File Structure

```
code/
├── main.py                  # Entry point — processes claims.csv → output.csv
├── config.py                # API key, model settings, enum values, file paths
├── data_loader.py           # Loads CSVs into structured dicts with O(1) lookups
├── preprocessor.py          # Builds context bundle per claim (extraction, injection detection)
├── pass1_vlm.py             # VLM call (GPT-4o) — image + text → structured JSON
├── pass2_validator.py       # 5-step validation pipeline → final output fields
├── output_writer.py         # CSV writer with strict enum validation
├── test_e2e.py              # End-to-end test script with verbose output
├── requirements.txt         # Python dependencies
├── .env.example             # API key template
├── .gitignore               # Prevents .env and cache from being committed
├── prompts/
│   ├── pass1_system.txt     # VLM system prompt (analysis rules, originality check)
│   ├── pass1_user.txt       # VLM user prompt template (claim + decision guidance)
│   └── pass2_verdict.txt    # Verdict finalizer prompt (few-shot calibrated)
└── evaluation/
    ├── main.py              # Evaluation script — compares against ground truth
    ├── eval_output.csv      # Predictions on sample_claims.csv
    └── evaluation_report.md # Accuracy metrics and per-claim breakdown
```

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Pass 1 (VLM) | `gpt-4o-2024-11-20` | Multimodal image + text analysis |
| Pass 2 (Verdict) | `gpt-4o-2024-11-20` | Final decision synthesis with few-shot calibration |

## Pipeline Details

### Pass 1: VLM Image Analysis (GPT-4o)

Sends images + claim context to GPT-4o with structured JSON output. The VLM:
- Analyzes each image independently (object, part, damage, severity, quality)
- Checks image originality (AI-generated, stock, screenshot detection)
- Determines claim alignment (supported / contradicted / unclear)
- Prioritizes the claimed part when multiple damage types are visible
- Reports mismatches precisely (actual object/part seen, not claimed)

### Pass 2: Validation Pipeline (4 deterministic steps + 1 LLM call)

| Step | Name | Type | Purpose |
|------|------|------|---------|
| 1 | Enum Enforcer | Deterministic | Fuzzy-maps VLM strings to valid enum values |
| 2 | Evidence Rule Matcher | Deterministic | Checks image evidence meets requirements |
| 3 | Risk Flag Computer | Deterministic | Merges quality + history + adversarial flags |
| 4 | Adversarial Guard | Deterministic | Detects injection in images/text |
| 5 | Verdict Finalizer | LLM (GPT-4o) | Few-shot calibrated final decision |

### Key Design Decisions

- **Two-pass architecture**: Pass 1 is the expensive multimodal call; Pass 2 is mostly deterministic Python with one text-only LLM call for the final verdict.
- **Few-shot calibration**: The verdict prompt includes 8 calibration examples that teach the model the exact severity scale, issue_type taxonomy, and object_part conventions used in the ground truth.
- **Enum enforcement via fuzzy matching**: VLMs are unreliable with exact string formatting. Uses exact → synonyms → SequenceMatcher (≥0.7 threshold) → fallback.
- **Bug fix: partial support handling**: `"partially_supported"` is explicitly mapped to `"not_enough_information"` (not silently matched as `"supported"`).
- **Cross-image mismatch detection**: Detects when different images in the same submission show different objects (not just individual image vs claim type).
- **Issue type consistency**: When verdict is `contradicted` with no damage visible, `issue_type` is reset to `"none"`. When `not_enough_information`, reset to `"unknown"`.
- **Business rules codified**: `manual_review_required` always accompanies `user_history_risk`, `claim_mismatch`, `text_instruction_present`, or `non_original_image`.
- **Prompt injection resistance**: Dual-layer detection (regex in preprocessor + VLM-detected text in images).
- **Strict output validation**: All enum fields validated before CSV write.

## Evaluation Results

On 20 labeled sample claims (GPT-4o for both passes):

| Metric | Score |
|--------|-------|
| **Overall accuracy** | **75.8%** |
| claim_status | **85.0%** |
| object_part | **85.0%** |
| evidence_standard_met | **85.0%** |
| valid_image | **85.0%** |
| issue_type | 60.0% |
| severity | 55.0% |
| Risk flags F1 | **0.81** |
| Risk flags exact match | 65.0% |

### Evaluation Strategy Comparison

| Configuration | Overall | claim_status | object_part | Runtime |
|---------------|---------|-------------|-------------|---------|
| GPT-4o + GPT-4o (few-shot) | **75.8%** | **85%** | **85%** | 123s |
| GPT-4o + GPT-4o (no few-shot) | 68.3% | 80% | 55% | 97s |
| GPT-4o + GPT-4o-mini | 68.3% | 80% | 60% | 114s |
| GPT-4o + Gemini 2.5 Flash | 70.8% | 85% | 65% | 189s |
| GPT-4o + Gemini 3.1 Flash Lite | 69.2% | 85% | 55% | 194s |

The few-shot calibration approach delivered the biggest single improvement (+30% on object_part accuracy).

### Remaining Gap Analysis

The 24.2% accuracy gap comes from:
- **Severity** (55%): Consistent one-level-off disagreements. Model rates slightly higher than ground truth (e.g., `medium` vs `low` for minor dents).
- **Issue type** (60%): Ambiguous cases where multiple types apply (e.g., `broken_part` vs `crack` for a shattered component).
- **Edge cases** (user_002, user_008, user_033): Complex multi-image claims with identity mismatches that require nuanced reasoning.

These are inherent subjectivity boundaries in VLM-based assessment — not bugs.

## Operational Analysis

### Cost Estimate (44 test claims)

| Item | Count | Cost |
|------|-------|------|
| Pass 1 GPT-4o calls (with images) | 44 | ~$2.20 (avg 2K input tokens + image @ $0.05) |
| Pass 2 GPT-4o calls (text only) | 44 | ~$0.44 (avg 1.5K input tokens) |
| **Total estimated** | 88 calls | **~$2.64** |

### Runtime

- 20 sample claims: ~2 minutes
- 44 test claims: ~4-5 minutes
- No artificial delays needed (500 RPM limit handles 88 calls easily)

### Rate Limits

- GPT-4o: 500 RPM, 30K TPM — well within budget for 88 total calls
- Occasional 429s auto-retry successfully (OpenAI SDK handles backoff)

## CLI Options

```bash
# Main pipeline
python main.py                    # Full run: claims.csv → output.csv
python main.py --sample           # Run on sample_claims.csv
python main.py --limit 5          # Process first 5 claims only
python main.py --resume           # Resume from checkpoints

# Evaluation
python evaluation/main.py         # Full evaluation (20 samples)
python evaluation/main.py --limit 5
python evaluation/main.py --skip-api   # Use cached results only

# End-to-end testing
python test_e2e.py user_001 user_005   # Test specific claims from claims.csv
python test_e2e.py --sample user_001   # Test from sample_claims.csv
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o |

## Output Schema (14 columns)

| Column | Values |
|--------|--------|
| user_id | pass-through |
| image_paths | pass-through |
| user_claim | pass-through |
| claim_object | car, laptop, package |
| evidence_standard_met | true, false |
| evidence_standard_met_reason | free text |
| risk_flags | semicolon-separated flags or "none" |
| issue_type | dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown |
| object_part | per-object enum (see config.py) |
| claim_status | supported, contradicted, not_enough_information |
| claim_status_justification | free text (image-grounded) |
| supporting_image_ids | semicolon-separated img_N or "none" |
| valid_image | true, false |
| severity | none, low, medium, high, unknown |

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
