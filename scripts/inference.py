import argparse
import json
from pathlib import Path
import re

import torch
import yaml
from unsloth import FastLanguageModel


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Categorize the following banking query into its corresponding intent.\n\n"
    "### Input:\n"
    "{text}\n\n"
    "### Response:\n"
)


def normalize_prediction(raw: str):
    text = raw.strip().splitlines()[0].strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_?]", "", text)
    return text


class IntentClassification:
    def __init__(self, model_path):
        root_dir = Path(__file__).resolve().parent.parent
        config_path = root_dir / "configs" / "inference.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        model_cfg = config.get("model", {})
        self.max_length = int(model_cfg.get("max_length", 128))

        label_map_path = root_dir / model_cfg.get("label_map_path", "sample_data/label_map.json")
        if label_map_path.exists():
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
        else:
            self.labels = []

        device_cfg = str(model_cfg.get("device", "auto")).lower()
        if device_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_cfg

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.max_length,
            dtype=None,
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        )
        FastLanguageModel.for_inference(self.model)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, message):
        prompt = PROMPT_TEMPLATE.format(text=str(message))
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=8,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = normalize_prediction(generated)

        if self.labels and pred in set(self.labels):
            return pred
        return pred if pred else "unknown_intent"


def parse_args():
    parser = argparse.ArgumentParser(description="Run intent inference from a trained checkpoint.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/checkpoints/final",
        help="Path to the trained checkpoint directory.",
    )
    parser.add_argument(
        "--message",
        type=str,
        required=True,
        help="Input message to classify.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = IntentClassification(model_path=args.model_path)
    predicted_label = predictor(args.message)
    print(f"Input: {args.message}")
    print(f"Predicted intent: {predicted_label}")


if __name__ == "__main__":
    main()