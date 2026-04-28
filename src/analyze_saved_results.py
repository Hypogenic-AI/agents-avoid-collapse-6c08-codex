from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from run_research import (
    FIGURES_DIR,
    LLMClient,
    RESULTS_DIR,
    compute_corpus_metrics,
    load_prompt_sets,
    run_analysis,
    save_plots,
    score_truthfulness,
    summarize_condition_metrics,
)


def main():
    start = time.time()
    llm = LLMClient()
    output_df = pd.read_json(RESULTS_DIR / "recursive_outputs.json")
    individuality = pd.read_csv(RESULTS_DIR / "individuality_metrics.csv")
    truthful_rows = load_prompt_sets()[1][:12]

    corpus_df = output_df[output_df["task_type"] == "corpus"].copy()
    eval_df = output_df[output_df["task_type"] == "truthfulqa"].copy()

    corpus_metrics = compute_corpus_metrics(llm, corpus_df)
    truth_metrics = score_truthfulness(llm, eval_df, truthful_rows)
    summary = summarize_condition_metrics(corpus_metrics, truth_metrics, individuality)

    corpus_metrics.to_csv(RESULTS_DIR / "corpus_metrics.csv", index=False)
    truth_metrics.to_csv(RESULTS_DIR / "truth_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    save_plots(summary)
    run_analysis(summary)

    payload = {
        "analysis_runtime_seconds": round(time.time() - start, 2),
        "summary_rows": int(len(summary)),
        "figure_dir": str(FIGURES_DIR),
    }
    (RESULTS_DIR / "analysis_run_info.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
