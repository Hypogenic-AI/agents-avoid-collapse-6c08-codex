# Resources Catalog

## Summary

This document catalogs the papers, datasets, and code gathered for the project **How Many Agents to Avoid Collapse?** The emphasis is on recursive synthetic-data loops, agent-population diversity, and practical experimental infrastructure.

## Papers

Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|---|---|---:|---|---|
| The Curse of Recursion | Shumailov et al. | 2023 | `papers/2305.17493_curse_of_recursion.pdf` | Foundational model-collapse theory and LLM evidence |
| Self-Consuming Generative Models Go MAD | Alemohammad et al. | 2023 | `papers/2307.01850_self_consuming_models_go_mad.pdf` | Synthetic loop taxonomy and degradation regimes |
| LLMs Suffer From Their Own Output | Briesch et al. | 2023 | `papers/2311.16822_llms_suffer_from_their_own_output.pdf` | Diversity collapse measured directly in LLM outputs |
| Self-Correcting Self-Consuming Loops | Gillman et al. | 2024 | `papers/2402.07087_self_correcting_self_consuming_loops.pdf` | Correction-based stabilization method |
| Is Model Collapse Inevitable? | Gerstgrasser et al. | 2024 | `papers/2404.01413_is_model_collapse_inevitable.pdf` | Accumulation avoids collapse in language-model experiments |
| Population Diversity in MARL | McKee et al. | 2021 | `papers/2102.08370_population_diversity_in_marl.pdf` | Population size and behavioral diversity analog |
| CAMEL | Li et al. | 2023 | `papers/2303.17760_camel_llm_society.pdf` | LLM society / role-playing framework |
| Generative Agents | Park et al. | 2023 | `papers/2304.03442_generative_agents.pdf` | Persistent identity and social simulation |
| Survey on LLM Autonomous Agents | Wang et al. | 2023 | `papers/2308.11432_survey_llm_autonomous_agents.pdf` | Agent architecture survey |
| RoleLLM | Wang et al. | 2023 | `papers/2310.00746_rolellm_role_playing.pdf` | Role consistency benchmark for individuality |

See `papers/README.md` for more detail.

## Datasets

Total datasets downloaded locally: 4

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| WikiText-103 Raw | Hugging Face | 1.81M train rows, 314 MB local | human text seed corpus | `datasets/wikitext_103_raw/` | best local corpus for pilot recursive pretraining |
| OpenAssistant OASST1 | Hugging Face | 84k train rows, 51 MB local | human conversation corpus | `datasets/oasst1/` | strong interaction-diversity dataset |
| TruthfulQA Generation | Hugging Face | 817 examples | evaluation benchmark | `datasets/truthful_qa_generation/` | held-out quality check across generations |
| TinyStories validation subset | Hugging Face | 5k examples, 2.3 MB local | lightweight reproduction subset | `datasets/tinystories_validation_5k/` | full corpus intentionally not downloaded |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| self-correcting-self-consuming | github.com/nate-gillman/self-correcting-self-consuming | official collapse-mitigation code | `code/self-correcting-self-consuming/` | best direct baseline repo |
| CAMEL | github.com/camel-ai/camel | large-scale agent society framework | `code/camel/` | useful for population-size sweeps |
| generative_agents | github.com/joonspk-research/generative_agents | persistent-memory social simulation | `code/generative_agents/` | useful for individuality experiments |
| lm-evaluation-harness | github.com/EleutherAI/lm-evaluation-harness | standardized LLM evaluation | `code/lm-evaluation-harness/` | best held-out benchmark layer |

See `code/README.md` for more detail.

## Resource Gathering Notes

### Search Strategy

- Started with the local `paper-finder` workflow.
- The service stalled, so the search was completed with targeted arXiv lookups and paper-linked repository discovery.
- Prioritized papers that either:
  - directly study recursive synthetic-data collapse, or
  - provide a reusable definition of diversity / individuality / agent population behavior.

### Selection Criteria

- Direct relevance to collapse or anti-collapse mechanisms.
- Availability of open PDFs and preferably code.
- Ability to support automated experiments in the current workspace.

### Challenges Encountered

- The local paper-finder service did not return usable results in time.
- `uv add` attempted to build the workspace as an editable package; fallback to `uv pip install` was required.
- The current `datasets` version treats some split requests conservatively; full TinyStories was avoided to keep the workspace bounded.
- Some candidate datasets such as `lmsys/lmsys-chat-1m` are gated.

### Gaps and Workarounds

- No single dataset directly measures “ecosystem collapse” in an LLM population.
- Workaround: combine a human-origin corpus (`wikitext_103_raw`), a dialogue corpus (`oasst1`), and a fixed evaluation benchmark (`truthful_qa_generation`).
- Full reproduction of `2402.07087` requires external motion datasets and licensed body assets; the repo is cloned and documented, but not fully executed here.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: use `wikitext_103_raw` for recursive language-modeling loops and `oasst1` for multi-agent conversational loops.
2. **Baseline methods**: compare replacement, accumulation, fresh-data injection, and self-correction; then add population-size sweeps.
3. **Evaluation metrics**: track quality with held-out cross-entropy or lm-eval tasks, and track collapse with diversity metrics and role/identity consistency.
4. **Code to adapt/reuse**: start from `code/self-correcting-self-consuming/` for collapse loop structure, `code/camel/` for agent populations, and `code/lm-evaluation-harness/` for stable evaluation.
