# Evaluation Report

**Model:** gpt-4o-2024-11-20 (Pass 1), gpt-4o-2024-11-20 (Pass 2)
**Claims evaluated:** 20
**Overall accuracy:** 75.8%

## Per-Field Accuracy

| Field | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| claim_status | 17 | 20 | 85.0% |
| issue_type | 12 | 20 | 60.0% |
| object_part | 17 | 20 | 85.0% |
| severity | 11 | 20 | 55.0% |
| evidence_standard_met | 17 | 20 | 85.0% |
| valid_image | 17 | 20 | 85.0% |

## Risk Flags (Set-Based)

- Precision: 0.83
- Recall: 0.79
- F1: 0.81
- Exact Match: 65.0%

## Claim Status Confusion Matrix

| Expected \ Actual | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 12 | 0 | 0 |
| contradicted | 1 | 3 | 1 |
| not_enough_information | 1 | 0 | 2 |

## Per-Claim Details

- ✅ **user_001**: all correct
- ❌ **user_002**: claim_status: exp=not_enough_information, got=supported; issue_type: exp=broken_part, got=scratch; severity: exp=unknown, got=medium; evidence_standard_met: exp=false, got=true
- ✅ **user_004**: all correct
- ❌ **user_007**: issue_type: exp=broken_part, got=crack
- ❌ **user_005**: issue_type: exp=scratch, got=none; severity: exp=low, got=none
- ❌ **user_006**: valid_image: exp=true, got=false
- ✅ **user_003**: all correct
- ❌ **user_008**: issue_type: exp=broken_part, got=none; object_part: exp=front_bumper, got=hood; severity: exp=high, got=none; evidence_standard_met: exp=true, got=false
- ✅ **user_009**: all correct
- ❌ **user_010**: severity: exp=medium, got=high
- ❌ **user_011**: issue_type: exp=stain, got=water_damage
- ❌ **user_012**: severity: exp=low, got=medium
- ❌ **user_018**: issue_type: exp=crack, got=glass_shatter; severity: exp=medium, got=high
- ❌ **user_020**: issue_type: exp=none, got=scratch; severity: exp=none, got=low
- ✅ **user_015**: all correct
- ❌ **user_030**: object_part: exp=seal, got=package_side
- ✅ **user_031**: all correct
- ❌ **user_032**: valid_image: exp=false, got=true
- ❌ **user_033**: claim_status: exp=contradicted, got=not_enough_information; object_part: exp=unknown, got=box; severity: exp=low, got=unknown; evidence_standard_met: exp=true, got=false; valid_image: exp=true, got=false
- ❌ **user_034**: claim_status: exp=contradicted, got=supported; issue_type: exp=none, got=torn_packaging; severity: exp=none, got=medium