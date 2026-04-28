# REPORT

## 1. Executive Summary
This project tested whether a recursive LLM information ecosystem has a minimum viable population of sufficiently distinct agents below which it collapses. In this bounded proxy experiment, I did not find a clean minimum viable population threshold. Instead, the dominant pattern was an immediate generation-1 truthfulness shock across almost all conditions, followed by partial recovery.

The most important result is that prompt-level role diversity was detectable but not protective. The role-diverse conditions had a higher individuality ratio than the homogeneous conditions, yet they also had worse final composite collapse scores in this setup. The best final condition was `p1_homogeneous_accumulation`, and the worst was `p4_role_diverse_accumulation`.

Practically, this suggests that "more agents" is not enough, and "more different prompts" is not automatically enough either. If an ecosystem is to avoid collapse, the important variable may be the quality of the inheritance mechanism and preservation of human anchors, not only the nominal population count.

## 2. Research Question & Motivation
The user asked whether an LLM information ecosystem has an analogue of biological minimum viable population: a minimum number of sufficiently distinct producers needed to prevent collapse once LLM-written data becomes training data for later generations. This matters because synthetic text is already entering public corpora, code bases, and dialogue systems, and diversity can decay before standard quality metrics visibly fail.

The literature review in [literature_review.md](/workspaces/agents-avoid-collapse-6c08-codex/literature_review.md) established three relevant facts:
- Recursive self-training often destroys low-probability modes before headline task accuracy fails.
- Accumulation or correction can mitigate collapse in some settings.
- Prior work mostly studies a single model loop, not a population of interacting or partially distinct producers.

The missing piece was population-level evidence. This project therefore tested both nominal population size and operational individuality.

## 3. Experimental Setup

### Models and APIs
- Generation model: `gpt-4.1-mini`
- Embedding model: `text-embedding-3-small`
- Temperature: `0.9`
- Max output tokens: `170`
- Client: OpenAI Python SDK `2.32.0`

### Data
- Human prompt source: `datasets/oasst1`
- Human inheritance anchor: `datasets/wikitext_103_raw`
- Held-out evaluation: `datasets/truthful_qa_generation`

### Conditions
Six recursive ecosystem conditions were tested for 4 generations each:
- `p1_homogeneous_replacement`
- `p2_homogeneous_replacement`
- `p4_homogeneous_replacement`
- `p4_role_diverse_replacement`
- `p1_homogeneous_accumulation`
- `p4_role_diverse_accumulation`

Each generation produced:
- 12 corpus samples from OASST1 prompts
- 12 TruthfulQA answers

Total recursive outputs: `576` saved in [results/recursive_outputs.json](/workspaces/agents-avoid-collapse-6c08-codex/results/recursive_outputs.json)

### Individuality Definition
I defined an "individual" behaviorally rather than architecturally. An agent counts as distinct when its between-agent output distance exceeds its within-agent stochastic variance on the same prompt set.

Measured individuality:

| Condition | Within distance | Between distance | Individuality ratio |
|---|---:|---:|---:|
| homogeneous P=4 assay | 0.0672 | 0.0625 | 0.9294 |
| role-diverse P=4 assay | 0.0613 | 0.0728 | 1.1873 |

Interpretation:
- Homogeneous prompt variants were not distinct enough to count as separate individuals by this criterion.
- Role-diverse prompt variants were distinct enough to count as separate individuals.

### Metrics
- Diversity: `distinct_2`, `distinct_3`, token entropy, self-BLEU, embedding similarity
- Truthfulness proxy: answer embedding similarity to TruthfulQA correct vs incorrect references
- Composite collapse score: combined change in diversity, truthfulness, and embedding concentration relative to generation 0

### Important Deviation From Original Plan
The original plan proposed local descendant model training on synthetic corpora. I did not use that as the final experiment because the local `torch/transformers` stack was unstable enough to threaten the one-session requirement. Instead, I used a recursive inheritance proxy:
- generation 0 agents are anchored on human exemplars
- later-generation agents inherit exemplars from prior synthetic outputs
- `replacement` uses synthetic inheritance only
- `accumulation` preserves a small human anchor

This preserves real LLM behavior end to end, but it is weaker than full weight-space retraining. The conclusions should therefore be read as evidence about recursive cultural inheritance, not definitive proof about future pretraining dynamics.

### Hardware and Environment
- Workspace venv: `.venv`
- Python: `3.12.8`
- GPU detected: `4 x NVIDIA RTX A6000` with ~49 GB each
- Final experiment path: API-based, so GPU was available but not needed

## 4. Results

### Final Generation Comparison

| Condition | Final collapse score | Final truth rate | Final distinct-2 | Final token entropy |
|---|---:|---:|---:|---:|
| `p1_homogeneous_accumulation` | -0.0072 | 0.7500 | 0.9095 | 8.7962 |
| `p4_homogeneous_replacement` | 0.0190 | 0.8333 | 0.9041 | 8.8324 |
| `p1_homogeneous_replacement` | 0.2003 | 0.6667 | 0.9394 | 8.9622 |
| `p2_homogeneous_replacement` | 0.2056 | 0.7500 | 0.9352 | 8.8352 |
| `p4_role_diverse_replacement` | 0.2238 | 0.6667 | 0.9297 | 8.8834 |
| `p4_role_diverse_accumulation` | 0.3147 | 0.6667 | 0.9193 | 8.8540 |

Lower collapse score is better.

### Aggregate Patterns
- By data regime at final generation:
  - accumulation mean collapse score: `0.1538`
  - replacement mean collapse score: `0.1622`
- By distinctness at final generation:
  - homogeneous mean collapse score: `0.1044`
  - role-diverse mean collapse score: `0.2693`
- By population size at final generation:
  - P=1 mean collapse score: `0.0966`
  - P=2 mean collapse score: `0.2056`
  - P=4 mean collapse score: `0.1858`

### Temporal Pattern
The most stable pattern was not gradual monotonic collapse. Instead:
- generation 0 truthfulness started high, from `0.67` to `0.92` depending on condition
- generation 1 dropped sharply to `0.42` to `0.50` in every condition
- generations 2 and 3 partially recovered to `0.67` to `0.83`

This indicates a strong inheritance shock after the first synthetic handoff, but not universal runaway degeneration over the next two generations.

### Diversity and Concentration
Embedding similarity rose by generation 3 in every condition, which is consistent with some concentration of outputs over time. For example:
- `p1_homogeneous_accumulation`: `0.1579 -> 0.2018`
- `p4_homogeneous_replacement`: `0.1580 -> 0.2154`
- `p4_role_diverse_replacement`: `0.1681 -> 0.2179`

Figures:
- [figures/distinct_2.png](/workspaces/agents-avoid-collapse-6c08-codex/figures/distinct_2.png)
- [figures/token_entropy.png](/workspaces/agents-avoid-collapse-6c08-codex/figures/token_entropy.png)
- [figures/truth_binary_mean.png](/workspaces/agents-avoid-collapse-6c08-codex/figures/truth_binary_mean.png)
- [figures/collapse_score.png](/workspaces/agents-avoid-collapse-6c08-codex/figures/collapse_score.png)

## 5. Analysis & Discussion

### Answer to the Main Question
Within this proxy setup, there is no evidence of a simple minimum viable population threshold such as "below 4 agents the ecosystem collapses." Population size alone did not order the outcomes cleanly. The best final condition was a single-agent accumulation regime, while several larger populations performed worse.

### What Counted as an Individual?
The individuality assay gives a concrete answer:
- same-model, same-role agents with only stochastic variation did not separate from within-agent noise
- same-model agents with different roles did separate from within-agent noise

So, in this experiment, "different enough" meant behavioral separation above stochastic variance, not merely being separate API calls or chat sessions.

### Does Collapse Always Happen?
Not in this setup. All conditions showed a first-generation shock, and all conditions showed increased embedding concentration by the end, but none showed irreversible monotonic collapse across all metrics. The best accumulation condition ended slightly better than its own generation-0 composite baseline.

### Did More Individuality Help?
Surprisingly, no. The final-generation Spearman correlation between individuality proxy and collapse score was:
- `rho = 0.8281`
- `p = 0.0418`

Because higher collapse score is worse, the observed association here runs in the wrong direction for the original hypothesis. In plain terms: prompt-role diversity made agents more distinguishable, but did not make the ecosystem more robust. This suggests that superficial stylistic diversity is not the kind of diversity that protects against recursive degradation.

### Interpretation
The experiment points toward a narrower conclusion than the original hypothesis:
- nominal population size is a weak predictor
- prompt-level behavioral distinctness is measurable
- but the protective value of distinctness depends on what kind of information is preserved

In other words, a viable ecosystem likely needs epistemic diversity or external grounding, not just many stylistically different speakers.

## 6. Limitations
- This is not full retraining of descendant model weights. It is a recursive inheritance proxy using real LLMs plus inherited exemplars.
- Sample sizes are deliberately bounded for one-session feasibility: 12 corpus prompts and 12 TruthfulQA questions per generation-condition cell.
- The truthfulness metric is embedding-based, not human-annotated.
- The role-diverse condition changes style and framing, but may not create genuinely independent knowledge sources.
- The first full generation run was saved before a determinism patch that sorted inherited exemplars. The saved raw outputs are stable and the analysis reran identically, but a fully bitwise-reproducible end-to-end rerun should use the patched script.

## 7. Conclusions & Next Steps
The clearest answer is: this experiment does not support a universal minimum viable population for LLM ecosystems. Collapse was not inevitable, and larger populations were not reliably safer. What mattered more was the inheritance mechanism, and prompt-level diversity by itself was not enough.

The strongest operational answer to "what counts as an individual?" is behavioral distinctness above within-agent stochastic variance. That is a defensible population definition, but it was not sufficient to protect the recursive ecosystem here.

Recommended follow-up work:
- Replace the inheritance proxy with actual small-model fine-tuning across generations.
- Introduce external retrieval or human-refresh controls to test grounded diversity rather than prompt-style diversity.
- Expand the population design to include model-family diversity, not just role diversity.
- Increase the benchmark set beyond TruthfulQA and use human or LLM-judge validation for failure cases.

## References
- Shumailov et al. 2023. *The Curse of Recursion: Training on Generated Data Makes Models Forget*. arXiv:2305.17493.
- Alemohammad et al. 2023. *Self-Consuming Generative Models Go MAD*. arXiv:2307.01850.
- Briesch et al. 2023. *Large Language Models Suffer From Their Own Output*. arXiv:2311.16822.
- Gillman et al. 2024. *Self-Correcting Self-Consuming Loops for Generative Model Training*. arXiv:2402.07087.
- Gerstgrasser et al. 2024. *Is Model Collapse Inevitable?* arXiv:2404.01413.
- McKee et al. 2021. *Quantifying the Effects of Environment and Population Diversity in Multi-Agent Reinforcement Learning*. arXiv:2102.08370.
