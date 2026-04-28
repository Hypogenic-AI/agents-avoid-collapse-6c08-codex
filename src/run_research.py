from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_from_disk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GENERATION_MODELS = ["gpt-4.1-mini", "gpt-4.1"]
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_WORKERS = 6
NUM_GENERATIONS = 4
CORPUS_PROMPTS = 12
TRUTHFULQA_PROMPTS = 12
INDIVIDUALITY_PROMPTS = 8
HERITAGE_EXAMPLES = 4
MAX_OUTPUT_TOKENS = 170
TEMPERATURE = 0.9

BASE_SYSTEM_PROMPT = (
    "You are contributing training data for future language models. "
    "Write factual, self-contained, medium-length answers. "
    "Avoid bullets, hedging, and meta commentary. "
    "Stay concrete and readable."
)

ROLE_PROMPTS = [
    "Write in the style of a careful historian who emphasizes chronology, institutions, and documented evidence.",
    "Write in the style of a science explainer who prefers mechanisms, definitions, and causal clarity.",
    "Write in the style of a field journalist who prioritizes vivid but factual reporting and clear attribution.",
    "Write in the style of a patient teacher who uses simple explanations and intuitive examples.",
]


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    role_prompt: str


@dataclass(frozen=True)
class Condition:
    condition_id: str
    population_size: int
    distinctness: str
    data_regime: str


class CacheStore:
    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text())
        else:
            self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value):
        self.data[key] = value

    def set_many(self, items: dict[str, object]):
        self.data.update(items)

    def flush(self):
        self.path.write_text(json.dumps(self.data, indent=2))


class LLMClient:
    def __init__(self):
        self.client = OpenAI()
        self.response_cache = CacheStore(RESULTS_DIR / "api_response_cache.json")
        self.embedding_cache = CacheStore(RESULTS_DIR / "embedding_cache.json")

    def generate(self, system_prompt: str, user_prompt: str, metadata: dict) -> dict:
        key = self._hash_payload(
            {
                "kind": "generate",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "metadata": metadata,
                "temperature": TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
        )
        cached = self.response_cache.get(key)
        if cached is not None:
            return cached

        last_error = None
        for model in GENERATION_MODELS:
            for attempt in range(5):
                try:
                    response = self.client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    )
                    result = {
                        "model": model,
                        "text": response.output_text.strip(),
                        "metadata": metadata,
                    }
                    self.response_cache.set(key, result)
                    self.response_cache.flush()
                    return result
                except Exception as exc:  # pragma: no cover - network path
                    last_error = repr(exc)
                    time.sleep(min(2 ** attempt, 20))

        raise RuntimeError(f"Generation failed: {last_error}")

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        outputs = []
        missing = []
        missing_keys = []
        for text in texts:
            key = self._hash_payload({"kind": "embed", "text": text})
            cached = self.embedding_cache.get(key)
            if cached is None:
                missing.append(text)
                missing_keys.append(key)
            else:
                outputs.append(cached)

        if missing:
            batch_size = 64
            for start in range(0, len(missing), batch_size):
                batch_texts = missing[start : start + batch_size]
                batch_keys = missing_keys[start : start + batch_size]
                response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
                self.embedding_cache.set_many(
                    {key: item.embedding for key, item in zip(batch_keys, response.data)}
                )
                self.embedding_cache.flush()

        final = []
        for text in texts:
            key = self._hash_payload({"kind": "embed", "text": text})
            final.append(self.embedding_cache.get(key))
        return final

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in normalize_text(text).split() if token.strip()]


def distinct_n(texts: list[str], n: int) -> float:
    grams = []
    for text in texts:
        toks = tokenize(text)
        grams.extend(tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1)))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def token_entropy(texts: list[str]) -> float:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([count / total for count in counter.values()])
    return float(-(probs * np.log2(probs)).sum())


def mean_self_bleu(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    smoothie = SmoothingFunction().method1
    scores = []
    tokenized = [tokenize(text) for text in texts]
    for idx, candidate in enumerate(tokenized):
        references = tokenized[:idx] + tokenized[idx + 1 :]
        if not candidate or not references:
            continue
        scores.append(sentence_bleu(references, candidate, smoothing_function=smoothie))
    return float(np.mean(scores)) if scores else 0.0


def pairwise_cosine_stats(vectors: list[list[float]]) -> tuple[float, float]:
    if len(vectors) < 2:
        return 0.0, 0.0
    matrix = np.array(vectors)
    sims = cosine_similarity(matrix)
    upper = sims[np.triu_indices_from(sims, k=1)]
    return float(upper.mean()), float(upper.std())


def build_agents(population_size: int, distinctness: str) -> list[AgentSpec]:
    agents = []
    for idx in range(population_size):
        if distinctness == "role_diverse":
            role_prompt = ROLE_PROMPTS[idx % len(ROLE_PROMPTS)]
        else:
            role_prompt = ROLE_PROMPTS[0]
        agents.append(AgentSpec(agent_id=f"agent_{idx}", role_prompt=role_prompt))
    return agents


def load_prompt_sets() -> tuple[list[str], list[dict], list[str]]:
    oasst = load_from_disk(ROOT / "datasets" / "oasst1")
    truthful = load_from_disk(ROOT / "datasets" / "truthful_qa_generation")
    wikitext = load_from_disk(ROOT / "datasets" / "wikitext_103_raw")

    conversation_prompts = []
    for row in oasst["train"]:
        if row["role"] == "prompter":
            text = normalize_text(row["text"])
            if 40 <= len(text) <= 220 and "http" not in text.lower():
                conversation_prompts.append(text)
        if len(conversation_prompts) >= CORPUS_PROMPTS * 4:
            break

    truthful_rows = []
    for row in truthful["validation"]:
        truthful_rows.append(
            {
                "question": normalize_text(row["question"]),
                "best_answer": row["best_answer"],
                "correct_answers": list(row["correct_answers"]),
                "incorrect_answers": list(row["incorrect_answers"]),
            }
        )
        if len(truthful_rows) >= max(TRUTHFULQA_PROMPTS * 3, INDIVIDUALITY_PROMPTS * 2):
            break

    human_exemplars = []
    for row in wikitext["train"]:
        text = normalize_text(row["text"])
        if 80 <= len(text) <= 420 and not text.startswith("="):
            human_exemplars.append(text)
        if len(human_exemplars) >= HERITAGE_EXAMPLES * 4:
            break

    return conversation_prompts, truthful_rows, human_exemplars


def build_system_prompt(agent: AgentSpec, heritage_texts: list[str]) -> str:
    prompt = [BASE_SYSTEM_PROMPT, agent.role_prompt]
    if heritage_texts:
        prompt.append("Inheritance examples from the previous generation:")
        for idx, text in enumerate(heritage_texts, start=1):
            prompt.append(f"Example {idx}: {normalize_text(text)[:700]}")
        prompt.append(
            "Preserve useful patterns from the examples without copying them verbatim. "
            "Prefer factuality, specificity, and variety."
        )
    return "\n\n".join(prompt)


def select_heritage(
    condition: Condition,
    agent: AgentSpec,
    generation: int,
    prior_outputs: dict[str, list[dict]],
    human_exemplars: list[str],
) -> list[str]:
    if generation == 0:
        return human_exemplars[:HERITAGE_EXAMPLES]

    inherited = []
    previous = prior_outputs.get(condition.condition_id, [])
    if condition.distinctness == "role_diverse":
        agent_specific = [row["text"] for row in previous if row["agent_id"] == agent.agent_id]
        inherited.extend(agent_specific[: HERITAGE_EXAMPLES // 2])
    inherited.extend([row["text"] for row in previous[:HERITAGE_EXAMPLES]])

    if condition.data_regime == "accumulation":
        inherited = human_exemplars[:2] + inherited

    unique = []
    seen = set()
    for text in inherited:
        text = normalize_text(text)
        if text not in seen:
            unique.append(text)
            seen.add(text)
        if len(unique) >= HERITAGE_EXAMPLES:
            break
    return unique


def response_task(
    llm: LLMClient,
    condition: Condition,
    agent: AgentSpec,
    generation: int,
    prompt_text: str,
    heritage_texts: list[str],
    task_type: str,
    item_id: str,
) -> dict:
    system_prompt = build_system_prompt(agent, heritage_texts)
    if task_type == "corpus":
        user_prompt = (
            "Respond to the following prompt with a self-contained answer of about 90-140 words.\n\n"
            f"Prompt: {prompt_text}"
        )
    else:
        user_prompt = (
            "Answer the question directly in 1-3 sentences. Prefer factual corrections over myths.\n\n"
            f"Question: {prompt_text}"
        )

    result = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata={
            "condition_id": condition.condition_id,
            "agent_id": agent.agent_id,
            "generation": generation,
            "task_type": task_type,
            "item_id": item_id,
        },
    )
    return {
        "condition_id": condition.condition_id,
        "population_size": condition.population_size,
        "distinctness": condition.distinctness,
        "data_regime": condition.data_regime,
        "agent_id": agent.agent_id,
        "generation": generation,
        "task_type": task_type,
        "item_id": item_id,
        "prompt_text": prompt_text,
        "text": result["text"],
        "model": result["model"],
    }


def run_individuality_assay(llm: LLMClient, truthful_rows: list[dict]) -> pd.DataFrame:
    assay_conditions = [
        Condition("assay_homogeneous_p4", 4, "homogeneous", "replacement"),
        Condition("assay_role_diverse_p4", 4, "role_diverse", "replacement"),
    ]
    prompts = [row["question"] for row in truthful_rows[:INDIVIDUALITY_PROMPTS]]
    all_rows = []

    for condition in assay_conditions:
        agents = build_agents(condition.population_size, condition.distinctness)
        jobs = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for agent in agents:
                for prompt_idx, prompt in enumerate(prompts):
                    for replicate in range(2):
                        jobs.append(
                            pool.submit(
                                response_task,
                                llm,
                                condition,
                                agent,
                                0,
                                prompt,
                                [],
                                "assay",
                                f"{prompt_idx}_rep{replicate}",
                            )
                        )
            for future in as_completed(jobs):
                all_rows.append(future.result())

    assay_df = pd.DataFrame(all_rows).sort_values(["condition_id", "agent_id", "item_id"])
    vectors = llm.embed_many(assay_df["text"].tolist())
    assay_df["embedding"] = vectors
    records = []
    for condition_id, group in assay_df.groupby("condition_id"):
        within = []
        between = []
        grouped = defaultdict(list)
        for _, row in group.iterrows():
            prompt_idx = row["item_id"].split("_rep")[0]
            grouped[(row["agent_id"], prompt_idx)].append(row["embedding"])
        for vecs in grouped.values():
            if len(vecs) == 2:
                within.append(1 - cosine_similarity([vecs[0]], [vecs[1]])[0][0])

        prompt_agent_vectors = defaultdict(dict)
        for _, row in group.iterrows():
            prompt_idx = row["item_id"].split("_rep")[0]
            if prompt_idx not in prompt_agent_vectors:
                prompt_agent_vectors[prompt_idx] = {}
            if row["agent_id"] not in prompt_agent_vectors[prompt_idx]:
                prompt_agent_vectors[prompt_idx][row["agent_id"]] = row["embedding"]

        for prompt_dict in prompt_agent_vectors.values():
            ids = list(prompt_dict.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    sim = cosine_similarity([prompt_dict[ids[i]]], [prompt_dict[ids[j]]])[0][0]
                    between.append(1 - sim)

        records.append(
            {
                "condition_id": condition_id,
                "within_distance_mean": float(np.mean(within)),
                "between_distance_mean": float(np.mean(between)),
                "individuality_ratio": float(np.mean(between) / max(np.mean(within), 1e-6)),
            }
        )

    result = pd.DataFrame(records)
    result.to_csv(RESULTS_DIR / "individuality_metrics.csv", index=False)
    assay_df.drop(columns=["embedding"]).to_json(
        RESULTS_DIR / "individuality_outputs.json", orient="records", indent=2
    )
    return result


def score_truthfulness(llm: LLMClient, eval_df: pd.DataFrame, truthful_rows: list[dict]) -> pd.DataFrame:
    row_lookup = {str(idx): row for idx, row in enumerate(truthful_rows[:TRUTHFULQA_PROMPTS])}
    texts = []
    for _, row in eval_df.iterrows():
        bundle = row_lookup[str(row["item_id"])]
        texts.append(row["text"])
        texts.extend(bundle["correct_answers"])
        texts.extend(bundle["incorrect_answers"])

    embeddings = llm.embed_many(texts)
    idx = 0
    scores = []
    for _, row in eval_df.iterrows():
        bundle = row_lookup[str(row["item_id"])]
        answer_vec = embeddings[idx]
        idx += 1
        correct_vecs = embeddings[idx : idx + len(bundle["correct_answers"])]
        idx += len(bundle["correct_answers"])
        incorrect_vecs = embeddings[idx : idx + len(bundle["incorrect_answers"])]
        idx += len(bundle["incorrect_answers"])
        correct_score = cosine_similarity([answer_vec], correct_vecs).max()
        incorrect_score = cosine_similarity([answer_vec], incorrect_vecs).max()
        margin = float(correct_score - incorrect_score)
        exact_bonus = float(bundle["best_answer"].lower() in row["text"].lower())
        scores.append(
            {
                "condition_id": row["condition_id"],
                "generation": row["generation"],
                "item_id": row["item_id"],
                "truth_margin": margin,
                "truth_binary": float(margin > 0),
                "best_answer_bonus": exact_bonus,
            }
        )
    return pd.DataFrame(scores)


def compute_corpus_metrics(llm: LLMClient, corpus_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition_id, generation), group in corpus_df.groupby(["condition_id", "generation"]):
        texts = group["text"].tolist()
        vectors = llm.embed_many(texts)
        mean_cos, std_cos = pairwise_cosine_stats(vectors)
        rows.append(
            {
                "condition_id": condition_id,
                "generation": generation,
                "population_size": int(group["population_size"].iloc[0]),
                "distinctness": group["distinctness"].iloc[0],
                "data_regime": group["data_regime"].iloc[0],
                "distinct_2": distinct_n(texts, 2),
                "distinct_3": distinct_n(texts, 3),
                "token_entropy": token_entropy(texts),
                "self_bleu": mean_self_bleu(texts),
                "embedding_similarity_mean": mean_cos,
                "embedding_similarity_std": std_cos,
                "mean_length_tokens": float(np.mean([len(tokenize(text)) for text in texts])),
            }
        )
    return pd.DataFrame(rows)


def summarize_condition_metrics(
    corpus_metrics: pd.DataFrame,
    truth_metrics: pd.DataFrame,
    individuality: pd.DataFrame,
) -> pd.DataFrame:
    summary = corpus_metrics.merge(
        truth_metrics.groupby(["condition_id", "generation"], as_index=False)
        .agg(
            truth_margin_mean=("truth_margin", "mean"),
            truth_binary_mean=("truth_binary", "mean"),
            best_answer_bonus_mean=("best_answer_bonus", "mean"),
        ),
        on=["condition_id", "generation"],
        how="left",
    )

    individuality_lookup = {
        "homogeneous": individuality.loc[
            individuality["condition_id"] == "assay_homogeneous_p4", "individuality_ratio"
        ].iloc[0],
        "role_diverse": individuality.loc[
            individuality["condition_id"] == "assay_role_diverse_p4", "individuality_ratio"
        ].iloc[0],
    }
    summary["individuality_ratio_proxy"] = summary["distinctness"].map(individuality_lookup)

    baseline = summary[summary["generation"] == 0][
        ["condition_id", "distinct_2", "token_entropy", "truth_binary_mean", "embedding_similarity_mean"]
    ].rename(
        columns={
            "distinct_2": "distinct_2_g0",
            "token_entropy": "token_entropy_g0",
            "truth_binary_mean": "truth_binary_mean_g0",
            "embedding_similarity_mean": "embedding_similarity_mean_g0",
        }
    )
    summary = summary.merge(baseline, on="condition_id", how="left")
    summary["collapse_score"] = (
        (summary["distinct_2_g0"] - summary["distinct_2"])
        + (summary["token_entropy_g0"] - summary["token_entropy"]) / max(summary["token_entropy_g0"].mean(), 1e-6)
        + (summary["truth_binary_mean_g0"] - summary["truth_binary_mean"])
        + (summary["embedding_similarity_mean"] - summary["embedding_similarity_mean_g0"])
    )
    return summary


def build_conditions() -> list[Condition]:
    return [
        Condition("p1_homogeneous_replacement", 1, "homogeneous", "replacement"),
        Condition("p2_homogeneous_replacement", 2, "homogeneous", "replacement"),
        Condition("p4_homogeneous_replacement", 4, "homogeneous", "replacement"),
        Condition("p4_role_diverse_replacement", 4, "role_diverse", "replacement"),
        Condition("p1_homogeneous_accumulation", 1, "homogeneous", "accumulation"),
        Condition("p4_role_diverse_accumulation", 4, "role_diverse", "accumulation"),
    ]


def run_recursive_experiment(llm: LLMClient) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus_prompts, truthful_rows, human_exemplars = load_prompt_sets()
    conditions = build_conditions()
    prior_outputs: dict[str, list[dict]] = {}
    all_rows = []

    for generation in range(NUM_GENERATIONS):
        generation_rows = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = []
            for condition in conditions:
                agents = build_agents(condition.population_size, condition.distinctness)
                selected_corpus_prompts = corpus_prompts[generation : generation + CORPUS_PROMPTS]
                selected_truth_rows = truthful_rows[generation : generation + TRUTHFULQA_PROMPTS]

                for prompt_idx, prompt in enumerate(selected_corpus_prompts):
                    agent = agents[prompt_idx % len(agents)]
                    heritage = select_heritage(condition, agent, generation, prior_outputs, human_exemplars)
                    futures.append(
                        pool.submit(
                            response_task,
                            llm,
                            condition,
                            agent,
                            generation,
                            prompt,
                            heritage,
                            "corpus",
                            str(prompt_idx),
                        )
                    )
                for prompt_idx, row in enumerate(selected_truth_rows):
                    agent = agents[prompt_idx % len(agents)]
                    heritage = select_heritage(condition, agent, generation, prior_outputs, human_exemplars)
                    futures.append(
                        pool.submit(
                            response_task,
                            llm,
                            condition,
                            agent,
                            generation,
                            row["question"],
                            heritage,
                            "truthfulqa",
                            str(prompt_idx),
                        )
                    )
            for future in as_completed(futures):
                generation_rows.append(future.result())

        all_rows.extend(generation_rows)
        for condition in conditions:
            condition_rows = [
                row
                for row in generation_rows
                if row["condition_id"] == condition.condition_id and row["task_type"] == "corpus"
            ]
            prior_outputs[condition.condition_id] = sorted(
                condition_rows, key=lambda row: (row["item_id"], row["agent_id"])
            )

    output_df = pd.DataFrame(all_rows).sort_values(["condition_id", "generation", "task_type", "item_id"])
    output_df.to_json(RESULTS_DIR / "recursive_outputs.json", orient="records", indent=2)
    return output_df, pd.DataFrame(truthful_rows[:TRUTHFULQA_PROMPTS])


def save_plots(summary: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    plot_specs = [
        ("distinct_2", "Distinct-2"),
        ("token_entropy", "Token Entropy"),
        ("truth_binary_mean", "Truthfulness Rate"),
        ("collapse_score", "Collapse Score"),
    ]
    for metric, label in plot_specs:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=summary,
            x="generation",
            y=metric,
            hue="condition_id",
            marker="o",
        )
        plt.title(label)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{metric}.png", dpi=200)
        plt.close()


def run_analysis(summary: pd.DataFrame):
    final_gen = summary[summary["generation"] == summary["generation"].max()]
    rho, pval = spearmanr(final_gen["individuality_ratio_proxy"], final_gen["collapse_score"])
    analysis = {
        "spearman_individuality_vs_collapse": float(rho) if not math.isnan(rho) else None,
        "spearman_pvalue": float(pval) if not math.isnan(pval) else None,
        "best_final_condition": final_gen.sort_values("collapse_score").iloc[0]["condition_id"],
        "worst_final_condition": final_gen.sort_values("collapse_score").iloc[-1]["condition_id"],
    }
    (RESULTS_DIR / "analysis_summary.json").write_text(json.dumps(analysis, indent=2))


def main():
    start = time.time()
    llm = LLMClient()
    individuality = run_individuality_assay(llm, load_prompt_sets()[1])
    output_df, truthful_eval_rows = run_recursive_experiment(llm)
    corpus_df = output_df[output_df["task_type"] == "corpus"].copy()
    eval_df = output_df[output_df["task_type"] == "truthfulqa"].copy()

    corpus_metrics = compute_corpus_metrics(llm, corpus_df)
    truth_metrics = score_truthfulness(llm, eval_df, truthful_eval_rows.to_dict(orient="records"))
    summary = summarize_condition_metrics(corpus_metrics, truth_metrics, individuality)

    corpus_metrics.to_csv(RESULTS_DIR / "corpus_metrics.csv", index=False)
    truth_metrics.to_csv(RESULTS_DIR / "truth_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    save_plots(summary)
    run_analysis(summary)

    env = {
        "seed": SEED,
        "generation_models": GENERATION_MODELS,
        "embedding_model": EMBEDDING_MODEL,
        "num_generations": NUM_GENERATIONS,
        "corpus_prompts_per_generation": CORPUS_PROMPTS,
        "truthfulqa_prompts_per_generation": TRUTHFULQA_PROMPTS,
        "max_workers": MAX_WORKERS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - start, 2),
    }
    (RESULTS_DIR / "run_config.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
