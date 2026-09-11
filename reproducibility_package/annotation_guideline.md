# HKG Annotation Guideline

## Scope

The HKG records curriculum knowledge points used for textbook-curriculum-aligned mathematical solution generation. It is a curriculum coordinate resource, not a universal mathematical difficulty graph.

## Source Documents

Annotators use curriculum standards, textbook volumes, and representative problem collections. Copyrighted source passages are not redistributed. Each retained node records a source locator instead.

## Candidate Extraction

1. Extract mathematical concepts, formulas, operations, theorem names, and recurring textbook problem-solving operators.
2. Prefer fine-grained knowledge points that correspond to a teachable curriculum unit.
3. Exclude generic words that do not identify a mathematical operator.
4. Record implicit operators only when textbook examples require the operator as a recognizable step.

## Synonym Merging

Merge surface forms into one node when they share the same domain, subfield, curriculum stage, and operator scope. Synonyms may include formula names, notation variants, textbook paraphrases, and Chinese/English surface forms.

## Cross-Stage Splitting

Create separate scope-qualified nodes when the same surface form corresponds to distinct curriculum-stage operations. Split when any of the following holds:

1. The expected operation differs across stages.
2. The proof or derivation role differs across stages.
3. The surface form shifts between elementary recognition and advanced operator use.
4. The later-stage use depends on concepts not introduced in the earlier stage.

## Ambiguous Surface Forms

Record an ambiguity entry when a surface form can map to multiple nodes. The ambiguity record should include admissible domains/subfields and disambiguation cues. During extraction, ambiguous forms must pass context, stage, and evidence gates before entering the verified path.

## Source Locator Format

Use locators such as:

- curriculum-standard item identifier;
- textbook grade-semester volume and chapter/section;
- worked-example or exercise-heading locator;
- representative problem-collection identifier.

Do not copy copyrighted textbook passages into the release.

## Stage Assignment

Assign the first grade-semester stage where the concept is systematically introduced, practiced in worked examples, and expected for problem solving. Informal earlier mentions do not determine the stage unless the concept is expected as an operator.

## Adjudication

Two annotators independently propose or validate node records. Disagreements about stage, scope, or split decisions are resolved through adjudication. All split and ambiguous-form decisions should be frozen before final model evaluation.
