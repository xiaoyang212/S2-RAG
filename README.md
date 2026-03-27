# S2-RAG: Decoupled Semantic and Structural Indexing for Long-Document RAG

<p align="center">
  <b>S2-RAG</b> is a dual-index Retrieval-Augmented Generation framework that decouples <b>Semantic</b> and <b>Structural</b> indexing to improve long-document question answering.
</p>

---

## Overview

Large Language Models (LLMs) continue to expand their context windows, yet accurately answering questions and performing multi-hop reasoning over long documents remain challenging. The core difficulty lies in accurately locating dispersed key information and establishing cross-paragraph factual connections.

**S2-RAG** addresses this challenge with a dual-index framework inspired by the human reading strategies of *skimming* and *scanning*:

- **Hierarchical Summary Tree** – supports global semantic localization (skimming), enabling coarse-to-fine retrieval over multi-granularity document summaries.
- **Entity Co-occurrence Graph** – supports local factual navigation (scanning), connecting named entities and key terms that co-occur across chunks without relying on costly LLM-based relation extraction.

During online retrieval, a **dual-path fusion mechanism** jointly ranks candidates from both the semantic and structural paths, improving evidence coverage and generation quality.

> Paper: *S2-RAG: Decoupled Semantic and Structural Indexing for Long-Document Retrieval-Augmented Generation*  
> Authors: Jianing Yang, Guojie Liu, Yiqi Wang, Songlei Jian, Jianfeng Zhang, Shuang Tan, Bao Li  
> College of Computer Science and Technology, National University of Defense Technology

---

## Framework

```
Offline Indexing
  ├── Chunking → Hierarchical Summary Tree (LLM-based summarization)
  └── Chunking → Entity Co-occurrence Graph (NER + POS tagging)

Online Retrieval
  ├── Semantic Path  : Dense retrieval over summary-tree node embeddings
  ├── Structural Path: Budget-controlled graph traversal from query entities
  └── Fusion         : Rank-normalization + weighted score fusion → top-K evidence → LLM answer
```

---

## Installation

```bash
git clone https://github.com/xiaoyang212/S2-RAG.git
cd S2-RAG
pip install -r requirement.txt
```

> **Note:** A CUDA-capable GPU is required. The code has been tested with PyTorch 2.x and CUDA 12.x.

### NLP models

The default extractor uses the SpaCy `en_core_web_lg` model. It will be downloaded automatically on first run. You can also install it manually:

```bash
python -m spacy download en_core_web_lg
```

For Chinese documents, the same model is currently used as a fallback.

---

## Data Preparation

See [`data/README.md`](data/README.md) for instructions on obtaining the evaluation datasets.

Supported datasets:

| Dataset | Description |
|---|---|
| **NovelQA** | Multiple-choice QA over novels exceeding 200K tokens |
| **InfiniteChoice** | Multiple-choice subset of InfiniteBench (English) |
| **InfiniteQA** | Free-form QA subset of InfiniteBench (English) |

Expected directory layout after downloading:

```
data/
├── NovelQA/
│   └── ...
└── InfiniteBench/
    ├── longbook_choice_eng.jsonl
    └── longbook_qa_eng.jsonl
```

---

## Configuration

Copy and edit the example configuration file:

```bash
cp config/example.yaml config/my_config.yaml
```

Key configuration options:

```yaml
dataset:
  dataset_name: NovelQA           # NovelQA | InfiniteQALoader | InfiniteChoice
  dataset_path: ./data/NovelQA

llm:
  llm_path: path/to/your/llm      # HuggingFace model path or name
  llm_device: cuda                 # e.g., cuda:0

paths:
  log_path: ./log
  answer_path: ./answer/test
  cache_path: ./cache/test

extractor:
  language: "en"                   # en | zh
  method: "Spacy"                  # Spacy | NLTK

cluster:
  length: 1200                     # tokens per chunk
  overlap: 50                      # token overlap between chunks
  merge_num: 5                     # chunks merged per summary node
  force_Reextract: False           # re-extract graph even if cache exists

retriever:
  kwargs:
    device: cuda
    shortest_path_k: 4             # max hop distance for graph retrieval
    merge_num: 5
    overlap: 50
    tokenizer: path/to/your/llm    # same as llm_path
    max_chunk_setting: 25          # max retrieved chunks
    use_fusion: True               # combine semantic + structural paths
```

---

## Usage

### Full pipeline (parallel indexing + QA)

```bash
python main.py --config config/my_config.yaml
```

### Cache-ready pipeline (skip indexing if caches exist)

```bash
python main_cacheready.py --config config/my_config.yaml
```

Both scripts will:
1. Load the dataset.
2. Build the hierarchical summary tree and entity co-occurrence graph in parallel.
3. Run the dual-path retrieval and generate answers using the configured LLM.
4. Save results as JSON files under `answer_path`.

---

## Experimental Results

Results on three long-document QA benchmarks (Accuracy for NovelQA / InfiniteChoice, ROUGE-L for InfiniteQA):

| Method | NovelQA (Qwen) | InfiniteChoice (Qwen) | InfiniteQA (Qwen) | NovelQA (Llama) | InfiniteChoice (Llama) | InfiniteQA (Llama) |
|---|---|---|---|---|---|---|
| VanillaRAG | 43.82 | 39.30 | 14.10 | 32.97 | 33.19 | 8.01 |
| RAPTOR | 38.44 | 43.67 | 11.65 | 42.21 | 40.17 | 9.86 |
| GGraphRAG | 34.06 | 36.68 | 5.22 | 36.01 | 34.06 | 3.62 |
| LGraphRAG | 39.09 | 37.55 | 9.63 | 48.34 | 43.23 | 7.33 |
| LightRAG | 39.87 | 38.43 | 10.32 | 47.74 | 42.79 | 9.01 |
| HippoRAG | 49.85 | 50.66 | 14.23 | 49.63 | 49.78 | 13.67 |
| **S2-RAG (Ours)** | **49.76** | **56.77** | **18.05** | **48.50** | **52.84** | **15.37** |

Backbone models: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Dense retrieval uses BGE-M3. Experiments were run on a single NVIDIA RTX 4090 GPU (48 GB).

---

## Project Structure

```
S2-RAG/
├── main.py               # Full pipeline with parallel indexing
├── main_cacheready.py    # Pipeline that reuses existing index caches
├── build_tree.py         # Hierarchical summary tree construction
├── extract_graph.py      # Entity co-occurrence graph construction
├── query.py              # Dual-path retriever
├── process_utils.py      # Subprocess helpers for parallel processing
├── prompt_dict.py        # Prompt templates
├── dataloader.py         # Dataset loading utilities
├── utils.py              # Shared utilities (logging, text splitting, …)
├── config/
│   └── example.yaml      # Example configuration
├── data/
│   └── README.md         # Data download instructions
└── requirement.txt       # Python dependencies
```

---

## License

This project is released under the [MIT License](LICENSE).
