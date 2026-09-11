# Reject-Before-Aggregate Reward Modeling for Curriculum-Constrained Mathematical Reasoning

This repository contains the implementation and public reproducibility materials for manuscript **NEUCOM-D-26-12485**.

## Empirical scope

The reported policy-optimization evidence concerns Qwen2.5-1.5B-Instruct (seed 42) in one Hangzhou/Zhejiang textbook-curriculum mathematical-reasoning setting. The repository materials below are aligned with that evaluated operating point.

## Minimum reproducibility files

- `src/open_r1/rewards.py`: RBA-CAV curriculum reward, deterministic per-reasoning-unit retrieval, local four-gate scoring, ambiguity routing, and final admission rule.
- `src/open_r1/grpo.py`: GRPO training entry point.
- `hkg/`: released HKG node, synonym, split, ambiguity, changelog, and graph files.
- `dataset/dedup_screening_summary.csv`: aggregate deduplication-screening summary used for the released split.
- `src/open_r1/knowledge_base_graph57.json`: runtime hierarchy used by the reward implementation (224 total nodes: 179 fine-grained curriculum nodes and 45 structural nodes).
- `recipes/Qwen2.5-1.5B-Instruct/grpo/config_paper.yaml`: paper-aligned training configuration.
- `reproducibility_package/prompts/`: evaluator prompt templates.
- `reproducibility_package/schemas/hkg_extraction_schema.json`: structured extraction schema.
- `reproducibility_package/verifier/ucdg_thresholds_weights.csv`: gate and routing parameters.
- `reproducibility_package/verifier/gate_scorer_record.json`: local gate-scorer and ambiguity-routing interface record.
- `reproducibility_package/annotation_guideline.md`: construction and adjudication protocol.
- `tests/test_rewards.py`: reward-interface tests.

The versioned public release described in the manuscript is available in this repository. It includes the HKG tables, split and ambiguity records, annotation specifications, evaluator prompts and schemas, fixed verifier parameters, representative boundary examples, the deduplication summary, paper-aligned configuration, and reward implementation. Raw annotation workspaces, source-unit drafts, and copyrighted source passages are not included.

## Human-evaluation boundary

The manuscript reports adult mathematics-education annotators judging model-generated solutions. It does not report a learner-facing intervention or student-participant data. No personally identifying information is included here.

## Configuration summary

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Seed: `42`
- Epochs: `3`
- Difficulty temperature: `10`
- Reward weights: `[1, 1, 1, 2]`
- Routing thresholds: reject `0.45`, accept `0.80`
- Local gate-scorer endpoint: `http://127.0.0.1:5001/score`
- Fallback evaluator: used only for scores in the ambiguity interval `(0.45, 0.80)`

## Citation

```bibtex
@article{chen2026rbacav,
  title   = {Reject-Before-Aggregate Reward Modeling for Curriculum-Constrained Mathematical Reasoning},
  author  = {Chen, Kaiyan and Qiu, Yimiao and Lin, Wang and Xia, Qi and Chen, Jingyuan},
  journal = {Neurocomputing},
  note    = {Manuscript NEUCOM-D-26-12485},
  year    = {2026}
}
```

## License

See `LICENSE`.
