# Multi-Modal Evidence Review System

An automated damage claim verification system that analyzes submitted images against user claims using a two-pass VLM pipeline.

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
│                                     │  (GPT-4o)               │  │
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
├── config.py                # API keys, model settings, enum values, file paths
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
│   └── pass2_verdict.txt    # Verdict finalizer prompt (decision logic)
└── evaluation/
    ├── main.py              # Evaluation script — compares against ground truth
    ├── eval_output.csv      # Predictions on sample_claims.csv
    └── evaluation_report.md # Accuracy metrics and per-claim breakdown
```

## Pipeline Details

### Pass 1: VLM Image Analysis (GPT-4o)

Sends images + claim context to GPT-4o with structured JSON output. The VLM:
- Analyzes each image independently (object, part, damage, severity, quality)
- Checks image originality (AI-generated, stock, screenshot detection)
- Determines claim alignment (supported / contradicted / unclear)
- Prioritizes the claimed part when multiple damage types are visible

### Pass 2: Validation Pipeline (5 deterministic steps + 1 LLM call)

| Step | Name | Type | Purpose |
|------|------|------|---------|
| 1 | Enum Enforcer | Deterministic | Fuzzy-maps VLM strings to valid enum values |
| 2 | Evidence Rule Matcher | Deterministic | Checks image evidence meets requirements |
| 3 | Risk Flag Computer | Deterministic | Merges quality + history + adversarial flags |
| 4 | Adversarial Guard | Deterministic | Detects injection in images/text |
| 5 | Verdict Finalizer | LLM (GPT-4o) | Produces final claim_status + justification |

### Key Design Decisions

- **Two-pass architecture**: Pass 1 is the expensive multimodal call (GPT-4o); Pass 2 is mostly deterministic Python with one text-only LLM call (GPT-4o) for the final verdict.
- **Enum enforcement via fuzzy matching**: VLMs are unreliable with exact string formatting. The enum enforcer uses exact → substring → synonyms → SequenceMatcher (≥0.6 threshold) → fallback.
- **Business rules codified**: `manual_review_required` always accompanies `user_history_risk`, `claim_mismatch`, `text_instruction_present`, or `non_original_image`.
- **Prompt injection resistance**: Dual-layer detection (regex in preprocessor + VLM-detected text in images). Injection never overrides visual evidence.
- **Rate limiting**: Both GPT-4o and GPT-4o-mini have 500 RPM (no artificial delays needed). Checkpoint/resume system for long runs.
- **Strict output validation**: All enum fields are validated before CSV write. Invalid values crash with a descriptive error rather than producing bad output.
- **Fault tolerance**: Failed claims produce fallback rows with `not_enough_information` status rather than crashing the entire pipeline.

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Pass 1 (VLM) | `gpt-4o-2024-11-20` | Multimodal image + text analysis |
| Pass 2 (Verdict) | `gpt-4o-2024-11-20` | Final decision synthesis (text only) |

## Evaluation Results

On 20 labeled sample claims (last run: fresh pipeline execution):

| Metric | Score |
|--------|-------|
| Overall accuracy | **75.0%** |
| claim_status | **80.0%** |
| object_part | **80.0%** |
| valid_image | **95.0%** |
| evidence_standard_met | **80.0%** |
| issue_type | 55.0% |
| severity | 60.0% |
| Risk flags F1 | **0.80** |

Detailed per-claim breakdown available in `evaluation/evaluation_report.md`.

### Performance Notes

- 20 sample claims processed in ~47 seconds (most loaded from checkpoint)
- Fresh claims take ~3-12s each (Pass 1: 2-10s, Pass 2: 1-2s)
- Full 44-claim run estimated at ~3-4 minutes

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
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o (Pass 1 and Pass 2) |

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
