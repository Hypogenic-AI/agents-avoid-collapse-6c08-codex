# Literature Review: How Many Agents to Avoid Collapse?

## Review Scope

### Research Question
Is there a minimum viable population of sufficiently distinct LLM agents below which an information ecosystem collapses because recursive synthetic data loops overwhelm diversity?

### Inclusion Criteria
- Directly studies model collapse, self-consuming loops, or recursive synthetic-data training.
- Operationalizes diversity, population diversity, role diversity, or individuality in multi-agent systems.
- Provides code, datasets, or evaluation recipes useful for automated experiments.

### Exclusion Criteria
- Purely application papers without reusable methodology for collapse or diversity.
- Agent papers focused only on task completion with no relevance to identity, population, or feedback loops.

### Time Frame
- Core focus: 2021-2024, with emphasis on 2023-2024 collapse papers.

### Sources
- Local paper downloads from arXiv.
- Repository READMEs from cloned GitHub codebases.
- Targeted manual search after the local paper-finder service stalled.

## Search Log

| Date | Query | Source | Notes |
|---|---|---|---|
| 2026-04-28 | model collapse, self-consuming training, LLM diversity | arXiv/manual | Used because local paper-finder call did not return |
| 2026-04-28 | multi-agent population diversity | arXiv/manual | Added analog evidence from MARL |
| 2026-04-28 | LLM societies, generative agents, role playing | arXiv/manual | Added individuality and ecosystem papers |

## Key Papers

### The Curse of Recursion: Training on Generated Data Makes Models Forget
- **Authors**: Shumailov et al.
- **Year**: 2023
- **Source**: arXiv `2305.17493`
- **Key Contribution**: Introduces model collapse as a general degenerative process under recursive training on generated data.
- **Methodology**: Theory plus experiments on Gaussians, GMMs, VAEs, and OPT-125m trained recursively.
- **Datasets Used**: WikiText-2 for language-model evaluation; synthetic Gaussian and image-generation setups elsewhere.
- **Baselines**: no-preservation, partial-real-data preservation, accumulating mixtures.
- **Results**: Replacing real data with generated data causes tail loss and distribution drift; preserving some real data slows but does not eliminate degradation in the studied settings.
- **Code Available**: public code exists externally, but not cloned here.
- **Relevance to Our Research**: Establishes why diversity collapses first and motivates population/ecosystem-level safeguards.

### Self-Consuming Generative Models Go MAD
- **Authors**: Alemohammad et al.
- **Year**: 2023
- **Source**: arXiv `2307.01850`
- **Key Contribution**: Defines Model Autophagy Disorder and compares recursive training regimes.
- **Methodology**: Analytical and empirical generate-train loops for image models under different real-data refresh strategies.
- **Datasets Used**: Image-generation benchmarks such as MNIST and FFHQ-style settings.
- **Baselines**: fixed-real-data access, fresh-real-data access, synthetic-only loops.
- **Results**: Without enough fresh real data, future generations lose precision or recall progressively.
- **Code Available**: not gathered locally.
- **Relevance to Our Research**: Supports the claim that collapse depends on refresh rate and diversity injection, not only on recursive training itself.

### Large Language Models Suffer From Their Own Output: An Analysis of the Self-Consuming Training Loop
- **Authors**: Briesch et al.
- **Year**: 2023
- **Source**: arXiv `2311.16822`
- **Key Contribution**: First direct LLM study of self-consuming loops.
- **Methodology**: Trains language models on logic-expression data where correctness can be verified exactly; compares four data cycles.
- **Datasets Used**: Synthetic logic-expression corpus with exact correctness labels.
- **Baselines**: full synthetic, balanced, incremental, expanding data cycles.
- **Evaluation Metrics**: correctness and pairwise normalized Levenshtein diversity.
- **Results**: Correctness remains comparatively stable; diversity collapses much earlier. Full-synthetic loops collapse to a single point by generation 39, and even fresh data only slows the decline.
- **Code Available**: none gathered locally.
- **Relevance to Our Research**: Strongest evidence that ecosystem collapse should be defined in terms of diversity loss before raw task accuracy fails.

### Self-Correcting Self-Consuming Loops for Generative Model Training
- **Authors**: Gillman et al.
- **Year**: 2024
- **Source**: arXiv `2402.07087`
- **Key Contribution**: Shows recursive training can be stabilized if synthetic outputs are corrected toward the true distribution.
- **Methodology**: Theoretical correction operator plus experiments on MNIST and text-conditioned human motion diffusion.
- **Datasets Used**: MNIST; HumanML3D built from AMASS; BMLMoVi for filtering; motion prompts.
- **Baselines**: iterative fine-tuning without correction, iterative fine-tuning with correction, baseline training.
- **Evaluation Metrics**: FID-like measures, motion quality, physical plausibility.
- **Results**: Self-correction consistently outperforms uncorrected iterative fine-tuning and can stay competitive even at high synthetic-data ratios.
- **Code Available**: yes, cloned in `code/self-correcting-self-consuming/`.
- **Relevance to Our Research**: Suggests collapse is not inevitable if “individuality-preserving” or “reality-correcting” mechanisms exist.

### Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data
- **Authors**: Gerstgrasser et al.
- **Year**: 2024
- **Source**: arXiv `2404.01413`
- **Key Contribution**: Provides a direct counterexample to inevitability by changing replacement to accumulation.
- **Methodology**: Pretrains small GPT-2/Llama-2 variants on TinyStories; compares data replacement vs data accumulation; proves bounded error in a linear model.
- **Datasets Used**: TinyStories; additional VAE and molecule-generation settings.
- **Baselines**: replacement vs accumulation across model sizes, temperatures, and hyperparameters.
- **Evaluation Metrics**: validation cross-entropy and test error.
- **Results**: Replacement drives collapse; accumulation keeps cross-entropy stable or improved across iterations, with theoretical bounded error.
- **Code Available**: not cloned, but methods are simple to reproduce.
- **Relevance to Our Research**: Critical for the hypothesis because it means the threshold may depend on data governance, not only on population count.

### Quantifying the Effects of Environment and Population Diversity in Multi-Agent Reinforcement Learning
- **Authors**: McKee et al.
- **Year**: 2021
- **Source**: arXiv `2102.08370`
- **Key Contribution**: Quantifies how co-player and environment diversity affect generalization in populations of agents.
- **Methodology**: MARL experiments with procedural environments, varying population size, and an explicit behavioral-diversity measure.
- **Results**: Larger and more diverse co-player populations can improve held-out generalization, though not uniformly.
- **Relevance to Our Research**: Best operational precedent for defining a minimum viable population and measuring behavioral individuality.

### CAMEL / Generative Agents / RoleLLM / Agent Surveys
- **Common contribution**: These papers do not study collapse directly, but they provide the mechanisms needed to define individuality in LLM ecosystems.
- **Methodology**:
  - CAMEL uses role-playing communicative agents and large-scale societies.
  - Generative Agents use memory, reflection, and planning to create persistent identities.
  - RoleLLM benchmarks role consistency and persona enactment.
  - Agent surveys catalog agent architectures and evaluation strategies.
- **Relevance to Our Research**: They provide practical ways to instantiate “distinct LLMs” beyond model weights alone: different memories, roles, prompts, tools, and training mixtures may all count as population members.

## Common Methodologies

- Recursive generate-train loops: used in `2305.17493`, `2307.01850`, `2311.16822`, `2402.07087`, `2404.01413`.
- Replacement vs accumulation regimes: central comparison in `2305.17493` and `2404.01413`.
- Diversity-first diagnostics: explicit in `2311.16822`; implicit in `2305.17493` and `2307.01850`.
- Controlled synthetic tasks with exact labels: used in `2311.16822`.
- Population-diversity measurement: used in `2102.08370`.
- Agent-society simulation: used in `2303.17760` and `2304.03442`.

## Standard Baselines

- Real-only training.
- Synthetic-only replacement loop.
- Mixed real/synthetic loop with fixed ratio.
- Accumulating real-plus-synthetic loop.
- Fresh-data injection / expanding-data loop.
- Self-corrected synthetic loop.
- Small vs large population of interacting agents.

## Evaluation Metrics

- **Perplexity / cross-entropy**: standard quality metric in `2305.17493` and `2404.01413`.
- **Levenshtein diversity**: direct diversity metric in `2311.16822`.
- **FID-style quality metrics**: used in image and motion settings such as `2307.01850` and `2402.07087`.
- **Held-out generalization against unseen co-players**: used in `2102.08370`.
- **Role consistency / persona fidelity**: natural individuality metric suggested by `2310.00746`.

## Datasets in the Literature

- **WikiText-2 / WikiText**: language modeling under recursive retraining.
- **TinyStories**: efficient language-model pretraining corpus for accumulation-vs-replacement tests.
- **Synthetic logic expressions**: exact-correctness LLM collapse testbed.
- **MNIST / FFHQ-like image datasets**: recursive image-generation loops.
- **HumanML3D / AMASS / BMLMoVi**: motion-generation self-correction experiments.

## Gaps and Opportunities

- No paper in this set directly identifies a population threshold for LLM societies.
- Collapse is usually measured for a single recursively retrained model, not an interacting population of distinct models.
- “Individuality” is underspecified. Existing work suggests at least four measurable components:
  - output diversity
  - role consistency
  - memory/history divergence
  - behavioral diversity under shared tasks
- The strongest anti-collapse evidence changes the data regime, not the number of agents. That means population threshold experiments must control for replacement vs accumulation and fresh-data access.

## Recommendations for Our Experiment

- **Recommended datasets**:
  - `datasets/wikitext_103_raw` as the main human-origin text pool.
  - `datasets/oasst1` for multi-turn interaction experiments.
  - `datasets/truthful_qa_generation` as a fixed held-out quality benchmark.
  - `datasets/tinystories_validation_5k` only as a lightweight reproduction check for `2404.01413`.
- **Recommended baselines**:
  - single-model replacement loop
  - single-model accumulation loop
  - multi-agent loop with shared synthetic pool
  - multi-agent loop with partitioned agent-specific memories
  - self-corrected or fresh-data-injected loop
- **Recommended metrics**:
  - held-out cross-entropy or perplexity
  - lexical diversity and self-BLEU style collapse indicators
  - embedding-based pairwise diversity across agents
  - role consistency / identity persistence across turns
  - performance on TruthfulQA or lm-eval tasks
- **Methodological considerations**:
  - Define population size as the number of distinct synthetic data producers, not just chat sessions.
  - Vary distinctness separately from count: same base model with different roles or memories may still increase effective diversity.
  - Treat replacement vs accumulation as a first-order experimental factor, otherwise threshold claims will be confounded.
