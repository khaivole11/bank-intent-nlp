# Banking Intent Classification with Unsloth

This project fine-tunes instruction-tuned LLMs to classify banking intents on the BANKING77 dataset. It includes data sampling and preprocessing, Unsloth + LoRA training, evaluation with validation and test splits, and a standalone inference class that loads a saved checkpoint.

## Project overview

- Sampled a subset of BANKING77 and created train/test CSVs.
- Built a preprocessing script that normalizes labels and formats prompts.
- Fine-tuned a model with Unsloth (LoRA, 4-bit) and documented hyperparameters.
- Added a 90/10 train/validation split (stratified) and saved both validation and test metrics.
- Implemented a standalone inference class with `__init__` and `__call__`.
- Added quick tests to validate the split and the training pipeline.

Dataset: BANKING77
https://huggingface.co/datasets/PolyAI/banking77

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## Data prepration

The preprocessing script downloads BANKING77 metadata, builds a label map, samples data per label, and writes `sample_data/train.csv` and `sample_data/test.csv`.

```bash
python scripts/preprocess_data.py
```

Optional arguments:

- `--num_train_samples` (default: 3850)
- `--num_test_samples` (default: 770)
- `--seed` (default: 42)
- `--use_all` (use the full dataset)

## Training

Default training uses `configs/train.yaml`:

```bash
bash ./train.sh
```

Override model name or max length:

```bash
bash ./train.sh unsloth/gemma-4-e2b-it 512
```

The training script saves the final checkpoint to `outputs/checkpoints/final` and metrics to `outputs/checkpoints/metrics.json`.

### Key hyperparameters default (from configs/train.yaml)

| Item              | Value                                                         |
| ----------------- | ------------------------------------------------------------- |
| Model             | unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit                        |
| Max length        | 256                                                           |
| Load in 4-bit     | true                                                          |
| LoRA r            | 16                                                            |
| LoRA alpha        | 16                                                            |
| LoRA dropout      | 0.0                                                           |
| Target modules    | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Batch size        | 8                                                             |
| Grad accumulation | 2 (effective batch size 16)                                   |
| Learning rate     | 2e-4                                                          |
| Epochs            | 3                                                             |
| Weight decay      | 0.01                                                          |
| Warmup ratio      | 0.1                                                           |
| Save strategy     | epoch                                                         |
| Eval max samples  | 1000                                                          |
| Seed              | 42                                                            |

Optimizer: Hugging Face Trainer default (AdamW).

### Validation strategy

- Split train into 90% train and 10% validation (stratified).
- Evaluate on validation for overfitting signals.
- Evaluate on full test set for final metrics.

## Inference

CLI inference:

```bash
bash ./inference.sh "I want to close my account" outputs/checkpoints/final
```

Python usage (standalone class):

```python
from scripts.inference import IntentClassification

predictor = IntentClassification(model_path="outputs/checkpoints/final")
label = predictor("I want to close my account")
print(label)
```

The inference class loads config from `configs/inference.yaml` and uses the `__init__` and `__call__` interface required by the assignment.

## Benchmark (Kaggle fine-tunes)

Add each Kaggle run as a row. The example below uses the current checkpoint metrics available in `outputs/checkpoints/metrics.json`.

| Run   | Kaggle link                                                   | Model                                  | Quantization | Max length | Valid accuracy | Valid F1 macro | Eval accuracy | Eval F1 macro | Notes                                                                        |
| ----- | ------------------------------------------------------------- | -------------------------------------- | ------------ | ---------- | -------------- | -------------- | ------------- | ------------- | ---------------------------------------------------------------------------- |
| run-1 | [Kaggle](https://www.kaggle.com/code/volekaikai/banking77)       | unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit | 4-bit        | 256        | 0.7622         | 0.7690         | 0.7450        | 0.8127        | Source: outputs/checkpoints/metrics.json (train=9002, valid=1001, eval=1000) |
| run-2 | [Kaggle](https://www.kaggle.com/code/volekaikai/banking77-gemma) | unsloth/gemma-4-e2b-it                 | 4-bit        | 512        | 0.7742         | 0.7741         | 0.7770        | 0.8251        | Source: outputs/checkpoints/metrics.json (train=9002, valid=1001, eval=1000) |

If your Kaggle run produces `outputs/checkpoints/metrics.json`, you can also report validation metrics from that file.

## Video demo

- Demo video: [YouTube](#) (Update later with actual link)

## Notes

- `configs/train.yaml` and `configs/inference.yaml` are the single source of truth for hyperparameters and inference settings.
- `outputs/checkpoints/final` contains the trained model and tokenizer.
