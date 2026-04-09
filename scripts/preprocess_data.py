import argparse
import pandas as pd
import json
import os
import requests
from pathlib import Path

DEFAULT_CONFIG = {
    "num_train_samples": 3850,
    "num_test_samples": 770,
    "random_seed": 42,
    "json_metadata_url": 'https://huggingface.co/datasets/PolyAI/banking77/resolve/main/dataset_infos.json',
    "prompt_template": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Categorize the following banking query into its corresponding intent.

### Input:
{}

### Response:
{}"""
}

def main(args=None):
    root_dir = Path(__file__).resolve().parent.parent
    sample_dir = root_dir / 'sample_data'
    sample_dir.mkdir(parents=True, exist_ok=True)

    if args is None:
        parser = argparse.ArgumentParser(description="Preprocess BANKING77 dataset for intent classification.")
        parser.add_argument("--num_train_samples", type=int, default=DEFAULT_CONFIG["num_train_samples"],
                          help=f"Number of train samples (default: {DEFAULT_CONFIG['num_train_samples']})")
        parser.add_argument("--num_test_samples", type=int, default=DEFAULT_CONFIG["num_test_samples"],
                          help=f"Number of test samples (default: {DEFAULT_CONFIG['num_test_samples']})")
        parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["random_seed"],
                          help=f"Random seed (default: {DEFAULT_CONFIG['random_seed']})")
        parser.add_argument("--use_all", action="store_true",
                          help="Use all data without sampling (overrides --num_train_samples and --num_test_samples)")
        args = parser.parse_args()

    CONFIG = DEFAULT_CONFIG.copy()
    if args.use_all:
        CONFIG["num_train_samples"] = -1
        CONFIG["num_test_samples"] = -1
    else:
        CONFIG["num_train_samples"] = args.num_train_samples
        CONFIG["num_test_samples"] = args.num_test_samples
    CONFIG["random_seed"] = args.seed

    use_all_text = "YES (using all data)" if args.use_all else "NO"
    print(f"Config: train_samples={CONFIG['num_train_samples']}, test_samples={CONFIG['num_test_samples']}, seed={CONFIG['random_seed']}, use_all={use_all_text}")
    print("1. Loading metadata and label mapping...")
    response = requests.get(CONFIG["json_metadata_url"], timeout=30)
    info = response.json()["default"]
    labels = info["features"]["label"]["names"]
    
    with open(sample_dir / 'label_map.json', 'w') as f:
        json.dump(labels, f)
    
    urls = list(info.get("download_checksums", {}).keys())
    train_url = [u for u in urls if 'train.csv' in u][0]
    test_url = [u for u in urls if 'test.csv' in u][0]

    print("2. Loading and standardizing data...")
    df_train = pd.read_csv(train_url)
    df_test = pd.read_csv(test_url)

    def standardize_label_column(df):
        possible_names = ['label', 'category', 'intent', 'class', 'target']
        for col_name in possible_names:
            if col_name in df.columns:
                if col_name != 'label':
                    df.rename(columns={col_name: 'label'}, inplace=True)
                return
        
        if len(df.columns) >= 2:
            actual_label_col = df.columns[1]
            df.rename(columns={actual_label_col: 'label'}, inplace=True)

    standardize_label_column(df_train)
    standardize_label_column(df_test)

    print("3. Performing stratified sampling...")
    if CONFIG["num_train_samples"] != -1:
        samples_per_label = CONFIG["num_train_samples"] // len(labels)
        sampled_parts = []
        for label_val in df_train['label'].unique():
            group = df_train[df_train['label'] == label_val]
            n_samples = min(len(group), samples_per_label)
            sampled_parts.append(group.sample(n=n_samples, random_state=CONFIG["random_seed"]))
        df_train = pd.concat(sampled_parts, ignore_index=True)
    
    if CONFIG["num_test_samples"] != -1:
        df_test = df_test.sample(n=min(len(df_test), CONFIG["num_test_samples"]), 
                                random_state=CONFIG["random_seed"]).reset_index(drop=True)

    print(f"   -> Dataset size: Train={len(df_train)}, Test={len(df_test)}")

    print("4. Mapping labels and creating prompts...")
    
    def process_df(df, is_train=True):
        if 'label' not in df.columns:
            raise ValueError(f"Label column not found. Available columns: {list(df.columns)}")
        
        def get_intent_name(val):
            if isinstance(val, str):
                return val
            else:
                try:
                    return labels[int(val)]
                except (ValueError, IndexError):
                    return str(val)
        
        df['target'] = df['label'].apply(get_intent_name)
        
        if is_train:
            df['prompt'] = df.apply(lambda r: CONFIG["prompt_template"].format(r['text'], r['target']), axis=1)
        else:
            df['prompt'] = df.apply(lambda r: CONFIG["prompt_template"].format(r['text'], ""), axis=1)
        return df[['text', 'target', 'prompt']]

    df_train_final = process_df(df_train.copy(), is_train=True)
    df_test_final = process_df(df_test.copy(), is_train=False)

    print("5. Saving files to sample_data/...")
    df_train_final.to_csv(sample_dir / 'train.csv', index=False)
    df_test_final.to_csv(sample_dir / 'test.csv', index=False)
    print("   Done!")

if __name__ == "__main__":
    main()