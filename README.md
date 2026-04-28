# How Many Agents to Avoid Collapse?

This project studies whether an LLM information ecosystem has a minimum viable population below which recursive self-consumption causes collapse. The final experiment uses real OpenAI API generations in a recursive inheritance loop, then scores diversity and TruthfulQA behavior across population size, prompt-level individuality, and replacement versus accumulation regimes.

Key findings:
- No clean minimum viable population threshold emerged in this proxy setup.
- Prompt-role diversity was measurable, but it did not protect the ecosystem; the role-diverse conditions had the worst final composite collapse scores on average.
- Collapse was not inevitable here: the best final condition was `p1_homogeneous_accumulation`, whose final composite score was slightly better than its generation-0 baseline.
- The largest effect was a generation-1 truthfulness shock across nearly all conditions, followed by partial recovery rather than monotonic collapse.

Reproduce:
```bash
source .venv/bin/activate
python src/run_research.py
python src/analyze_saved_results.py
```

Notes:
- `src/run_research.py` runs the individuality assay and recursive generation experiment.
- `src/analyze_saved_results.py` recomputes metrics and figures from `results/recursive_outputs.json`.
- Full details are in [REPORT.md](/workspaces/agents-avoid-collapse-6c08-codex/REPORT.md).

File structure:
- `planning.md`: experimental plan and hypothesis decomposition
- `src/`: experiment and analysis scripts
- `results/`: raw outputs, caches, metrics tables, analysis summaries
- `figures/`: generated plots for diversity, truthfulness, and collapse
