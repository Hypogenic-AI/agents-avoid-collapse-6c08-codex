# Research Plan: How Many Agents to Avoid Collapse?

## Motivation & Novelty Assessment

### Why This Research Matters
Recursive synthetic-data loops are becoming a practical governance problem: models already generate training data, summaries, code, and conversations that later systems may absorb. If diversity decays faster than task performance, an LLM ecosystem can appear healthy while silently losing coverage, robustness, and truthfulness.

### Gap in Existing Work
The local literature shows strong evidence for model collapse under recursive training, and strong counterevidence that accumulation or correction can delay or avoid it. What is missing is a population-level view: prior work studies a single retrained model or a single synthetic-data loop, not an ecosystem with multiple partially distinct producers competing to define the next generation's training distribution.

### Our Novel Contribution
This project operationalizes three missing pieces together:
1. An empirical definition of when two LLM agents count as distinct individuals.
2. A population-size sweep over recursive synthetic-data loops.
3. A test of whether any apparent minimum viable population is intrinsic, or instead depends mainly on data-governance regime such as replacement versus accumulation.

### Experiment Justification
- Experiment 1: quantify individuality. Needed because "population size" is meaningless unless we can distinguish true individuals from prompt variants that differ only by sampling noise.
- Experiment 2: recursive replacement loops. Needed to test whether small populations collapse faster than larger populations when synthetic data fully replaces human data.
- Experiment 3: recursive accumulation loops. Needed to test whether collapse is inevitable, or whether a fixed trickle of human data removes the apparent minimum viable population threshold.

## Research Question
Is there a minimum viable population of sufficiently distinct LLM agents below which a recursive information ecosystem collapses, and does that threshold persist once human data accumulation is allowed?

## Background and Motivation
The literature review identifies two robust findings. First, recursive synthetic training tends to erase low-probability modes before headline accuracy fails. Second, anti-collapse interventions such as accumulation and correction can stabilize training. This suggests that "collapse" should be measured as a joint failure of diversity retention and held-out quality, and that any minimum viable population must be estimated conditional on the data-refresh regime rather than treated as a universal constant.

## Hypothesis Decomposition
- H1: In replacement-only recursive loops, ecosystems with fewer distinct producers show faster diversity collapse and worse held-out quality than ecosystems with more distinct producers.
- H2: Population count alone is not sufficient; effective distinctness matters. Multiple near-identical agents behave like a smaller population.
- H3: Under accumulation with even a modest human-data anchor, collapse is substantially delayed or avoided across the tested population sizes.
- H4: A practical "individual" is an agent configuration whose between-agent behavioral distance exceeds within-agent stochastic variance on the same prompt set.

Independent variables:
- Population size: `P in {1, 2, 4}`
- Distinctness regime: `homogeneous`, `role-diverse`
- Data regime: `replacement`, `accumulation`
- Generation index: `g in {0, 1, 2, 3}`

Dependent variables:
- Agent individuality score
- Corpus diversity metrics
- Descendant-model held-out perplexity
- Descendant-model TruthfulQA performance

## Proposed Methodology

### Approach
Use a frontier API model as the real synthetic-data producer and a small local causal LM as the trainable descendant. The API layer supplies authentic LLM behavior; the local descendant makes recursive retraining feasible within one session. This directly tests the user's scenario: LLMs produce text, then later models are trained on that text.

### Experimental Steps
1. Load bounded slices of `wikitext_103_raw`, `oasst1`, and `truthful_qa_generation`.
   Rationale: combine human-written expository text, multi-turn conversational prompts, and a fixed held-out quality benchmark.
2. Instantiate agent populations with shared base model but different system prompts and private memory exemplars.
   Rationale: isolates effective individuality from raw model-family differences.
3. Run an individuality assay on a shared prompt set and compute within-agent versus between-agent distances.
   Rationale: defines who counts as a population member.
4. Generate generation-0 synthetic corpora from the agents.
   Rationale: seed the ecosystem with real API outputs rather than simulated text.
5. Fine-tune a compact local LM for each condition and generation.
   Rationale: creates literal descendant models trained on prior synthetic output.
6. Use each descendant LM to generate the next-generation corpus under either replacement or accumulation.
   Rationale: reproduces recursive self-consumption dynamics while keeping training tractable.
7. Evaluate each descendant on held-out WikiText text and TruthfulQA prompts, and compute diversity metrics on generated corpora.
   Rationale: collapse should degrade both coverage/diversity and external quality.
8. Compare trajectories across population size, distinctness, and data regime.
   Rationale: reveals whether a viable threshold exists and whether it disappears under accumulation.

### Baselines
- Real-data anchor only: descendant trained on human text slice without synthetic recursion.
- `P=1`, homogeneous, replacement: strongest collapse baseline.
- `P=1`, homogeneous, accumulation: minimal anti-collapse baseline.
- `P=4`, role-diverse, replacement: population-size test.
- `P=4`, role-diverse, accumulation: best practical anti-collapse condition in this study.

### Evaluation Metrics
- Individuality ratio: mean between-agent embedding distance divided by mean within-agent distance.
- Lexical diversity: distinct-2, distinct-3, token entropy.
- Cross-sample similarity: self-BLEU and mean cosine similarity between passage embeddings.
- Held-out quality: perplexity on bounded human-written WikiText validation slice.
- Benchmark quality: TruthfulQA judged by exact-match overlap against reference answers plus LLM-judge scoring where needed.
- Collapse score: normalized composite of diversity loss and held-out-quality degradation.

### Statistical Analysis Plan
- Use paired bootstrap confidence intervals for condition differences over prompts and generated samples.
- Use Spearman correlation between individuality ratio and collapse severity.
- Use two-way ANOVA style linear modeling on `population_size`, `distinctness`, and `data_regime` for final-generation collapse scores.
- Report 95% confidence intervals and standardized effect sizes where applicable.
- Treat findings as exploratory but precommitted to the metrics above to avoid post-hoc selection.

## Expected Outcomes
Support for the hypothesis would look like:
- replacement loops collapsing quickly at `P=1`,
- slower collapse at larger `P`,
- weak benefit from larger `P` when agents are behaviorally near-identical,
- much smaller or absent threshold effects under accumulation.

Evidence against the hypothesis would be:
- no monotonic relationship between effective population and collapse,
- collapse equally severe across all `P`,
- or accumulation failing to materially change outcomes.

## Timeline and Milestones
1. Planning and resource review: complete first.
2. Environment and dependency setup: immediate next step.
3. Data loading and validation: build bounded slices and prompt sets.
4. Implementation: generation harness, individuality assay, fine-tuning pipeline, evaluation scripts.
5. Experiments: run all population and regime conditions.
6. Analysis and documentation: produce figures, tables, `REPORT.md`, and reproduction notes.

## Potential Challenges
- API variance and rate limits: mitigate with caching and bounded prompt counts.
- Small-model underfitting: use relative comparisons across matched conditions rather than absolute quality claims.
- TruthfulQA grading noise: combine exact lexical heuristics with manual spot checks and optional LLM-judge scoring.
- Confounding prompt diversity with population size: measure individuality directly and report it, rather than assuming prompt roles create real individuals.

## Success Criteria
- A reproducible pipeline exists from human seed data to recursive descendant training.
- At least one clear collapse trajectory is observed under replacement.
- At least one anti-collapse trajectory is observed under accumulation or higher effective diversity.
- The report gives a concrete operational answer to:
  - whether a minimum viable population appears in this setup,
  - whether collapse is inevitable,
  - and what counts as "different enough" to be an individual.
