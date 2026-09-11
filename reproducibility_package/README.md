# Reproducibility Package

This directory contains the public specification files used by the revised manuscript:

- `annotation_guideline.md`: HKG construction, source-reference, synonym, scope-split, ambiguity, and adjudication rules.
- `hkg_schema.json`: machine-readable HKG node and edge schema.
- `prompts/`: evaluator prompt templates.
- `schemas/hkg_extraction_schema.json`: structured extraction output schema.
- `verifier/ucdg_thresholds_weights.csv`: fixed gate thresholds, routing thresholds, and matching weights.
- `verifier/gate_scorer_record.json`: local gate-scorer inputs, outputs, and ambiguity-routing record.
- `examples/`: representative schema and curriculum-boundary examples.

The versioned public release is available in the repository. This directory provides the annotation specification, HKG schema, evaluator prompts, extraction schema, fixed verifier parameters, interface record, and representative examples; the released HKG tables and implementation are in the repository-level `hkg/` and `src/open_r1/` directories.
