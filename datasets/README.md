# Downloaded Datasets

This directory contains datasets for the research project. Data files are not committed to git due to size. Small sample files are in `datasets/samples/`.

## Dataset 1: WikiText-103 Raw

### Overview
- Source: `Salesforce/wikitext` (`wikitext-103-raw-v1`)
- Size: train 1,801,350, validation 3,760, test 4,358
- Format: Hugging Face `DatasetDict`
- Task: human-written text corpus for recursive pretraining / synthetic replacement experiments
- Local path: `datasets/wikitext_103_raw/`
- Local size: about 314 MB

### Download Instructions

Using Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
ds.save_to_disk("datasets/wikitext_103_raw")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/wikitext_103_raw")
```

### Sample Data

See `datasets/samples/wikitext_103_raw.json`.

### Notes
- Best local seed corpus for human-origin text in this workspace.
- Recommended as the main “real data” source for pilot collapse experiments.

## Dataset 2: OpenAssistant OASST1

### Overview
- Source: `OpenAssistant/oasst1`
- Size: train 84,437, validation 4,401
- Format: Hugging Face `DatasetDict`
- Task: human assistant conversations for interaction diversity and multi-turn identity experiments
- Local path: `datasets/oasst1/`
- Local size: about 51 MB

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("OpenAssistant/oasst1")
ds.save_to_disk("datasets/oasst1")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/oasst1")
```

### Sample Data

See `datasets/samples/oasst1.json`.

### Notes
- Includes message trees, role labels, and moderation metadata.
- Good source for human conversational variation before introducing synthetic loops.

## Dataset 3: TruthfulQA (Generation)

### Overview
- Source: `truthful_qa` (`generation`)
- Size: validation 817
- Format: Hugging Face `DatasetDict`
- Task: evaluation benchmark for truthfulness under recursive retraining
- Local path: `datasets/truthful_qa_generation/`
- Local size: about 271 KB

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("truthful_qa", "generation")
ds.save_to_disk("datasets/truthful_qa_generation")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/truthful_qa_generation")
```

### Sample Data

See `datasets/samples/truthful_qa_generation.json`.

### Notes
- Useful held-out benchmark to detect quality degradation while diversity changes.
- Recommended as one of the fixed test sets across generations.

## Dataset 4: TinyStories Validation Subset

### Overview
- Source: `roneneldan/TinyStories`
- Size: 5,000 validation examples
- Format: Hugging Face `Dataset`
- Task: lightweight reproduction subset for the accumulation-vs-replacement paper
- Local path: `datasets/tinystories_validation_5k/`
- Local size: about 2.3 MB

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="validation[:5000]")
ds.save_to_disk("datasets/tinystories_validation_5k")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/tinystories_validation_5k")
```

### Sample Data

See `datasets/samples/tinystories_validation_5k.json`.

### Notes
- The paper `2404.01413` uses the full TinyStories corpus; this workspace keeps only a bounded subset locally.
- The older `datasets` version in this environment tends to pull the full corpus when requesting some split patterns, so use the command above carefully if you expand it.

## Recommended External Datasets From the Literature

- HumanML3D / AMASS / BMLMoVi: required for reproducing the human-motion experiments in `2402.07087`.
- Full TinyStories: required for closer reproduction of `2404.01413`.
- WikiText-2: used in `2305.17493` for perplexity tracking over recursive generations.
