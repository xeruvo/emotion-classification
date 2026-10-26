# Emotion Classification with BERT

This student project fine-tunes `bert-base-uncased` for multi-label emotion
classification on the GoEmotions dataset. A text can receive more than one
emotion label. The project is educational and is not a psychological
assessment or production moderation system.

## Task and labels

The notebook loads the raw GoEmotions training split from Hugging Face and
uses the first 7,000 records for this project. It verifies the 28 expected
emotion columns. Neutral-only examples are removed, while examples containing
one or more non-neutral emotions are preserved.

The resulting model has 27 output labels:

`admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`,
`confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`,
`embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`,
`nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`,
`sadness`, and `surprise`.

The model uses independent sigmoid probabilities with a fixed threshold of
0.5. It is not a single-label classifier.

## Methodology

- Model: `BertForSequenceClassification`
- Base model: `bert-base-uncased`
- Problem type: multi-label classification
- Loss: Transformers' multi-label path using `BCEWithLogitsLoss`
- Maximum token length: 128
- Seed: 42
- Training epochs: 6
- Threshold: 0.5, fixed before final evaluation
- Mixed precision: enabled when CUDA is available

Unique text values are split into approximately:

- 80% training
- 10% validation
- 10% final test

Splitting on unique text groups prevents identical text from appearing in
different partitions. The validation set is used for checkpoint selection. The
final test set is passed to the trainer only after training and is not used for
model selection, threshold tuning, or hyperparameter decisions. The notebook
prints actual partition sizes and label-count summaries at runtime.

Because multi-label stratification is not implemented, label distributions are
reported for inspection. Final evaluation includes micro, macro, and weighted
metrics, a per-label report with support, and the number of labels with no
positive test examples.

## Files

```text
emotion-classification/
├── Emotion Classification (3) - Copy.ipynb  # Training and evaluation
├── app.py                                   # Optional inference-only demo
├── requirements.txt                         # Direct project dependencies
├── requirements_emotion - Copy.txt          # Original environment snapshot
└── README.md
```

## Installation

Python 3.12 was used for the verified run. Create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install a CUDA-enabled PyTorch build using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
The verified environment used the CUDA 12.8 wheel:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The first full run downloads GoEmotions and `bert-base-uncased` from Hugging
Face. Internet access is required unless they are already cached.

## Training and evaluation

Open the notebook in Jupyter or VS Code and run it from top to bottom. Training
and final evaluation are enabled in the current validated notebook:

```python
trainer.train()
evaluate_test_set(trainer)
```

The notebook saves checkpoints under `results/` and the selected model and
tokenizer under `emotion_model/`. These generated artifacts are excluded from
Git. A compatible CUDA GPU is strongly recommended; CPU execution may be slow.

## Verified GPU run and results

The corrected notebook was executed end to end using:

- Windows 11
- Python 3.12.12
- NVIDIA GeForce RTX 4060 Laptop GPU
- 8,188 MiB reported by `nvidia-smi`
- NVIDIA driver 591.66
- PyTorch `2.11.0+cu128`
- PyTorch CUDA runtime `12.8`
- `torch.cuda.is_available() == True`

The six-epoch run completed successfully with mixed precision in approximately
**370 seconds** and reached approximately **1,896 MiB** peak allocated GPU
memory. The final model was selected using validation macro F1 before the final
test evaluation.

The usable filtered data contained **5,043 rows and 4,839 unique texts**:

| Partition | Rows | Unique texts |
|---|---:|---:|
| Training | 4,041 | 3,871 |
| Validation | 500 | 484 |
| Final test | 502 | 484 |

Text overlap between every pair of partitions was zero.

At the fixed sigmoid threshold of 0.5, the final test metrics were:

| Average | Precision | Recall | F1 |
|---|---:|---:|---:|
| Micro | 0.5960 | 0.2826 | 0.3834 |
| Macro | 0.3844 | 0.1904 | 0.2368 |
| Weighted | 0.5152 | 0.2826 | 0.3432 |

All 27 labels had at least one positive test example. Per-label results:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| admiration | 0.60 | 0.46 | 0.52 | 54 |
| amusement | 0.81 | 0.57 | 0.67 | 30 |
| anger | 1.00 | 0.19 | 0.32 | 16 |
| annoyance | 0.31 | 0.14 | 0.19 | 36 |
| approval | 0.47 | 0.33 | 0.39 | 58 |
| caring | 0.33 | 0.12 | 0.18 | 16 |
| confusion | 0.67 | 0.11 | 0.20 | 35 |
| curiosity | 0.56 | 0.36 | 0.43 | 42 |
| desire | 0.00 | 0.00 | 0.00 | 11 |
| disappointment | 0.33 | 0.07 | 0.11 | 15 |
| disapproval | 0.36 | 0.23 | 0.28 | 40 |
| disgust | 0.00 | 0.00 | 0.00 | 15 |
| embarrassment | 0.00 | 0.00 | 0.00 | 10 |
| excitement | 0.50 | 0.10 | 0.16 | 21 |
| fear | 0.00 | 0.00 | 0.00 | 4 |
| gratitude | 0.82 | 0.65 | 0.73 | 43 |
| grief | 0.00 | 0.00 | 0.00 | 1 |
| joy | 0.79 | 0.45 | 0.58 | 33 |
| love | 0.70 | 0.75 | 0.72 | 28 |
| nervousness | 0.00 | 0.00 | 0.00 | 5 |
| optimism | 0.62 | 0.18 | 0.28 | 28 |
| pride | 0.00 | 0.00 | 0.00 | 6 |
| realization | 0.50 | 0.06 | 0.10 | 36 |
| relief | 0.00 | 0.00 | 0.00 | 7 |
| remorse | 0.00 | 0.00 | 0.00 | 9 |
| sadness | 0.75 | 0.33 | 0.46 | 18 |
| surprise | 0.25 | 0.05 | 0.08 | 20 |

These are corrected, verified holdout results from the 27-label methodology.
They do not establish clinical or production performance.

## Inference demo

The optional Gradio app loads an existing `emotion_model/` directory and does
not retrain:

```powershell
python app.py
```

If `emotion_model/` is missing, the app reports that the notebook training
workflow must be run first. The saved model was loaded successfully by
`app.py` and tested with three ordinary example sentences, producing five
ranked emotion probabilities for each.

## Historical results

The original notebook outputs came from 5,043 filtered rows, a 1,009-row
evaluation set, and a 28-output model whose `neutral` label had no positive
examples. They reported:

- Evaluation loss: 0.1360796093940735
- Micro F1: 0.31776765375854216
- Macro F1: 0.17251991152567253

Those values are historical context only and are not results from the corrected
27-label, three-way methodology.

## Dataset, privacy, and licensing

The Hugging Face dataset card identifies GoEmotions as Apache License 2.0 and
describes approximately 58,000 English Reddit comments labeled for 27 emotions
or Neutral. The raw configuration includes Reddit usernames and metadata. The
dataset card warns that usernames can sometimes be linked to real identities
and documents Reddit/user-population and annotation biases. Review the
[GoEmotions dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions)
and license notices before redistribution. This repository does not include a
dataset copy.

The `bert-base-uncased` model card identifies the model as Apache License 2.0.
Review the [model card](https://huggingface.co/google-bert/bert-base-uncased)
and retain required notices if redistributing artifacts. This repository does
not include generated model weights.

## Limitations

- A 7,000-example subset is used rather than the complete dataset.
- Multi-label distributions are inspected but not fully stratified.
- Results come from one fixed split and one threshold.
- Rare labels have very small test support and several have zero predicted
  positives at the fixed threshold.
- Results depend on the pretrained model, software versions, seeds, and
  hardware.
- The task is an educational NLP demonstration, not a psychological assessment
  or a production moderation system.
