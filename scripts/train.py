from unsloth import FastLanguageModel

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer


PROMPT_TEMPLATE = (
	"Below is an instruction that describes a task, paired with an input that provides further context. "
	"Write a response that appropriately completes the request.\n\n"
	"### Instruction:\n"
	"Categorize the following banking query into its corresponding intent. "
	"Respond with exactly one intent label in snake_case only.\n\n"
	"### Input:\n"
	"{text}\n\n"
	"### Response:\n"
)


def parse_args():
	parser = argparse.ArgumentParser(description="Train BANKING77 intent model with Unsloth.")
	parser.add_argument("--config", type=str, default="configs/train.yaml")
	parser.add_argument("--model_name", type=str, default=None)
	parser.add_argument("--max_length", type=int, default=None)
	return parser.parse_args()


def load_config(config_path: str):
	with open(config_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def load_labels(label_map_path: str):
	with open(label_map_path, "r", encoding="utf-8") as f:
		labels = json.load(f)
	return labels


def prepare_dataframe(df: pd.DataFrame, text_col: str, target_col: str, labels_set: set):
	if text_col not in df.columns or target_col not in df.columns:
		raise ValueError(f"Missing columns. Found={list(df.columns)}, required='{text_col}' and '{target_col}'.")

	out = df[[text_col, target_col]].copy()
	out = out.dropna().reset_index(drop=True)
	out[text_col] = out[text_col].astype(str)
	out[target_col] = out[target_col].astype(str)

	unknown = sorted(set(out[target_col]) - labels_set)
	if unknown:
		raise ValueError(f"Unknown labels found: {unknown[:10]}")
	return out


def split_train_validation(df: pd.DataFrame, test_size=0.1, random_state=42, target_col=None):
	train_df, valid_df = train_test_split(
		df,
		test_size=test_size,
		random_state=random_state,
		stratify=df[target_col] if target_col else None
	)
	return train_df, valid_df


def build_train_text(row, text_col: str, target_col: str):
	return PROMPT_TEMPLATE.format(text=row[text_col]) + row[target_col]


def normalize_prediction(raw: str):
	text = raw.strip().splitlines()[0].strip().lower().replace(" ", "_")
	text = re.sub(r"[^a-z0-9_?]", "", text)
	return text


def map_prediction_to_label(raw: str, labels):
	labels_set = set(labels)
	raw_norm = raw.strip().lower()
	candidate = normalize_prediction(raw_norm)

	if candidate in labels_set:
		return candidate

	search_space = re.sub(r"[^a-z0-9_ ]", " ", raw_norm).replace(" ", "_")
	for label in sorted(labels, key=len, reverse=True):
		if label in search_space:
			return label

	close = difflib.get_close_matches(candidate, labels, n=1, cutoff=0.75)
	if close:
		return close[0]

	return "__unknown__"


def evaluate_generation(model, tokenizer, test_df, labels, text_col: str, target_col: str, max_length: int):
	FastLanguageModel.for_inference(model)

	preds = []
	gts = []
	device = "cuda" if torch.cuda.is_available() else "cpu"

	for _, row in test_df.iterrows():
		prompt = PROMPT_TEMPLATE.format(text=row[text_col])
		encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
		encoded = {k: v.to(device) for k, v in encoded.items()}

		with torch.no_grad():
			outputs = model.generate(
				**encoded,
				max_new_tokens=24,
				do_sample=False,
				eos_token_id=tokenizer.eos_token_id,
				pad_token_id=tokenizer.eos_token_id,
			)

		generated = tokenizer.decode(outputs[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
		pred = map_prediction_to_label(generated, labels)

		preds.append(pred)
		gts.append(str(row[target_col]))

	known_labels = sorted(set(gts))
	accuracy = accuracy_score(gts, preds)
	f1_macro = f1_score(gts, preds, average="macro", labels=known_labels)
	return accuracy, f1_macro


def main():
	args = parse_args()
	cfg = load_config(args.config)

	data_cfg = cfg["data"]
	model_cfg = cfg["model"]
	lora_cfg = cfg["lora"]
	train_cfg = cfg["training"]

	if args.model_name:
		model_cfg["model_name"] = args.model_name
	if args.max_length is not None:
		model_cfg["max_length"] = int(args.max_length)

	train_csv = Path(data_cfg["train_csv"])
	test_csv = Path(data_cfg["test_csv"])
	label_map = Path(data_cfg["label_map"])
	text_col = data_cfg["text_column"]
	target_col = data_cfg["target_column"]

	labels = load_labels(str(label_map))
	labels_set = set(labels)

	train_df = pd.read_csv(train_csv)
	test_df = pd.read_csv(test_csv)
	train_df = prepare_dataframe(train_df, text_col, target_col, labels_set)
	test_df = prepare_dataframe(test_df, text_col, target_col, labels_set)

	train_df, valid_df = split_train_validation(
		train_df,
		test_size=0.1,
		random_state=int(train_cfg.get("seed", 42)),
		target_col=target_col
	)

	max_length = int(model_cfg.get("max_length", 256))
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name=model_cfg["model_name"],
		max_seq_length=max_length,
		dtype=None,
		load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
	)

	model = FastLanguageModel.get_peft_model(
		model,
		r=int(lora_cfg["r"]),
		target_modules=lora_cfg["target_modules"],
		lora_alpha=int(lora_cfg["lora_alpha"]),
		lora_dropout=float(lora_cfg["lora_dropout"]),
		bias=lora_cfg.get("bias", "none"),
		use_gradient_checkpointing=lora_cfg.get("gradient_checkpointing", "unsloth"),
		random_state=int(train_cfg.get("seed", 42)),
	)

	train_df = train_df.copy()
	train_df["text"] = train_df.apply(lambda r: build_train_text(r, text_col, target_col), axis=1)
	train_ds = Dataset.from_pandas(train_df[["text"]], preserve_index=False)

	training_args = SFTConfig(
		output_dir=train_cfg["output_dir"],
		per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
		gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
		learning_rate=float(train_cfg["learning_rate"]),
		num_train_epochs=float(train_cfg["num_train_epochs"]),
		weight_decay=float(train_cfg["weight_decay"]),
		warmup_ratio=float(train_cfg["warmup_ratio"]),
		logging_steps=int(train_cfg["logging_steps"]),
		save_strategy=train_cfg["save_strategy"],
		save_total_limit=int(train_cfg["save_total_limit"]),
		max_length=max_length,
		packing=True,
		packing_strategy="bfd",
		seed=int(train_cfg["seed"]),
		fp16=not torch.cuda.is_bf16_supported(),
		bf16=torch.cuda.is_bf16_supported(),
		report_to="none",
	)

	trainer = SFTTrainer(
		model=model,
		tokenizer=tokenizer,
		train_dataset=train_ds,
		dataset_text_field="text",
		args=training_args,
	)

	trainer.train()

	valid_accuracy, valid_f1_macro = evaluate_generation(
		model=model,
		tokenizer=tokenizer,
		test_df=valid_df,
		labels=labels,
		text_col=text_col,
		target_col=target_col,
		max_length=max_length,
	)

	eval_max_samples = int(train_cfg.get("eval_max_samples", 0))
	eval_df = test_df if eval_max_samples <= 0 else test_df.head(eval_max_samples)
	final_accuracy, final_f1_macro = evaluate_generation(
		model=model,
		tokenizer=tokenizer,
		test_df=eval_df,
		labels=labels,
		text_col=text_col,
		target_col=target_col,
		max_length=max_length,
	)

	output_dir = Path(train_cfg["output_dir"])
	final_dir = output_dir / "final"
	final_dir.mkdir(parents=True, exist_ok=True)
	model.save_pretrained(str(final_dir))
	tokenizer.save_pretrained(str(final_dir))

	metrics = {
		"valid_accuracy": float(valid_accuracy),
		"valid_f1_macro": float(valid_f1_macro),
		"valid_samples": int(len(valid_df)),
		
		"eval_accuracy": float(final_accuracy),
		"eval_f1_macro": float(final_f1_macro),
		"eval_samples": int(len(eval_df)),
		
		"train_samples": int(len(train_df)),
		"model_name": model_cfg["model_name"],
		"max_length": int(max_length),
	}

	metrics_path = output_dir / "metrics.json"
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)


if __name__ == "__main__":
	main()
