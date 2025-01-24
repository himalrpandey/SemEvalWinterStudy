import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOP_CUTOFF = 0.15 #top % for easy
BOTTOM_CUTOFF = 0.85 #1-bottom % for hard

def load_training_dynamics(jsonl_file):

    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"File not found: {jsonl_file}")

    #json file format eg {"id": "eng_train_track_a_00001", "logits": [[.], [.]], "gold": [1.0]}
    records = []
    with open(jsonl_file, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(record)
    return records

def compute_confidence_variability(training_dynamics):

    results = []
    for record in training_dynamics:
        instance_id = record["id"]
        logits_across_epochs = record["logits"]
        gold_label = record["gold"][0]

        # Convert each logit to prob for the gold label
        probs = []
        for logit_arr in logits_across_epochs:
            # logit_arr e.g. [5.2]
            logit = logit_arr[0]
            prob_pos = 1.0 / (1.0 + np.exp(-logit))  # sigmoid for positive class
            if gold_label == 1.0:
                probs.append(prob_pos)
            else:
                probs.append(1.0 - prob_pos)

        #confidence = mean probability of the gold label across all epochs
        confidence = float(np.mean(probs))
        variability = float(np.std(probs))
        results.append({"id": instance_id, "confidence": confidence, "variability": variability})
    return results

def categorize_data(data_map):
    
    #sort by confidence in a descending order
    sorted_data = sorted(data_map, key=lambda x: x["confidence"], reverse=True)
    n = len(sorted_data)
    top_cutoff = int(n * TOP_CUTOFF)
    bottom_cutoff = int(n * BOTTOM_CUTOFF)
    
    for i, item in enumerate(sorted_data):
        if i < top_cutoff:
            category = "easy-to-learn"
        elif i < bottom_cutoff:
            category = "ambiguous"
        else:
            category = "hard-to-learn"

        item["category"] = category

    return sorted_data


def plot_data_map(categorized_data, max_points=20000, emotion="", out_dir="cartography_output/cartography_graphs"):
    # If dataset is large, randomly sample for plotting clarity
    import random
    if len(categorized_data) > max_points:
        categorized_data = random.sample(categorized_data, max_points)

    category_colors = {
        "easy-to-learn": "green",
        "hard-to-learn": "red",
        "ambiguous": "blue",
    }

    x_vals = [d["variability"] for d in categorized_data]
    y_vals = [d["confidence"] for d in categorized_data]
    cats   = [d["category"]    for d in categorized_data]

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"data_map_{emotion}.png")
    
    plt.figure(figsize=(8, 6))
    for cat, color in category_colors.items():
        cat_x = [x for x, c in zip(x_vals, cats) if c == cat]
        cat_y = [y for y, c in zip(y_vals, cats) if c == cat]
        plt.scatter(cat_x, cat_y, s=10, c=color, alpha=0.7, label=cat)

    plt.xlabel("Variability (std of probability)")
    plt.ylabel("Confidence (mean probability of gold label)")
    plt.title(f"Data Map ({emotion})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", required=True)
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--emotion", type=str, default="")
    parser.add_argument("--max_points", type=int, default=20000)
    args = parser.parse_args()

    td_records = load_training_dynamics(args.jsonl_file)
    print(f"Loaded {len(td_records)} records from {args.jsonl_file}.")

    data_map = compute_confidence_variability(td_records)

    categorized = categorize_data(data_map)
    
    cat_counts = {}
    for item in categorized:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    print("Category distribution:", cat_counts)

    plot_data_map(categorized, max_points=args.max_points, emotion=args.emotion)
    
    outdir = "cartography_output"
    os.makedirs(outdir, exist_ok=True)
    
    df = pd.read_csv(args.csv_file)
    # Restrict to IDs in partial training dynamics
    partial_ids = set(item["id"] for item in categorized)
    original_len = len(df)
    df = df[df["id"].isin(partial_ids)].copy()
    print(f"Original CSV had {original_len} rows, after restricting => {len(df)} remain.")
    
    easy_ids = {item["id"] for item in categorized if item["category"]=="easy-to-learn"}
    ambi_ids = {item["id"] for item in categorized if item["category"]=="ambiguous"}
    hard_ids = {item["id"] for item in categorized if item["category"]=="hard-to-learn"}

    easy_out = os.path.join(outdir, f"{args.emotion}_easy-to-learn.csv")
    ambi_out = os.path.join(outdir, f"{args.emotion}_ambiguous.csv")
    hard_out = os.path.join(outdir, f"{args.emotion}_hard-to-learn.csv")

    df_easy = df[df["id"].isin(easy_ids)]
    df_ambi = df[df["id"].isin(ambi_ids)]
    df_hard = df[df["id"].isin(hard_ids)]
    
    easy_path = os.path.join(outdir, f"{args.emotion}_easy-to-learn.csv")
    ambi_path = os.path.join(outdir, f"{args.emotion}_ambiguous.csv")
    hard_path = os.path.join(outdir, f"{args.emotion}_hard-to-learn.csv")

    df_easy.to_csv(easy_path, index=False)
    df_ambi.to_csv(ambi_path, index=False)
    df_hard.to_csv(hard_path, index=False)

    print(f"Wrote {len(df_easy)} rows => {easy_out}")
    print(f"Wrote {len(df_ambi)} rows => {ambi_out}")
    print(f"Wrote {len(df_hard)} rows => {hard_out}")
    
if __name__ == "__main__":
    main()