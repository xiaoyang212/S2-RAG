# How to obtain the data

## NovelQA

NovelQA is introduced in the paper  [NovelQA: Benchmarking Question Answering on Documents Exceeding 200K Tokens](https://arxiv.org/abs/2403.12766).

Since some books included in NovelQA are not publicly distributable, we cannot directly provide the full dataset in this repository. If you need access to the complete dataset, please contact the authors of NovelQA.

For the accessible portion of the dataset, you can follow the instructions below and place the files under `./data/NovelQA`:

## InfiniteBench

InfiniteBench is introduced in the paper [\inftyBench: Extending Long Context Evaluation Beyond 100K Tokens](https://arxiv.org/abs/2402.13718).

In our experiments, we use the English multiple-choice and English QA subsets from InfiniteBench, namely:

* `longbook_choice_eng.jsonl`
* `longbook_qa_eng.jsonl`

You can download these files from the official InfiniteBench release and place them under `./data/InfiniteBench`.

For convenience, you may also use the provided downloading script:

```bash
## Directory structure
After downloading the data, the directory structure should look like this:

data/
├── NovelQA/
│   └── ...
└── InfiniteBench/
    ├── longbook_choice_eng.jsonl
    ├── longbook_qa_eng.jsonl

```
## Notes

* Please make sure you comply with the original dataset licenses and usage restrictions.
* We only provide instructions for obtaining the datasets and do not redistribute restricted data.



