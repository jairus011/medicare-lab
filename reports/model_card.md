# Model card: MediCare Lab v1.0

- Intended use: educational inspection of a weak symptom classifier and source lookups.
- Prohibited use: diagnosis, triage, prescribing, medication substitution, or clinical decisions.
- Selected model: Random forest; seed 42; fixed training-only grouped CV selection.
- Input: four reviewed binary symptoms; unselected means absent, not unknown.
- Output: one of 28 exact original condition labels. No calibrated clinical probabilities.
- Training: 157 source records in 139 raw-profile groups; test: 38 records in 33 groups.
- Test accuracy: 0.0526; macro F1: 0.0298.
- Baseline accuracy: 0.0789; baseline macro F1: 0.0052.
- 88 source labels excluded by a predefined minimum-support rule; unknown-condition detection is absent.
- All four-symptom patterns are ambiguous in the complete source. No patient IDs were supplied.
- No external clinical validation, calibration, demographic fairness conclusion, or personalized treatment evidence.
- Exact evaluated pipeline is saved; no full-data refit. Reference data does not enter model training.
- Runtime, data checksum and model checksum: `models/metadata.json`.
- Missing descriptions and references are explicit. Unverified reference entries are opt-in; doses are omitted.
- Source content and reuse permissions have not been independently established.
