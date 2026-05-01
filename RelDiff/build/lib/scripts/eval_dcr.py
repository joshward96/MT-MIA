import os
import json
import argparse

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from syntherela.data import load_tables
from syntherela.metadata import Metadata
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# Function to calculate distances in batches
def calculate_min_distances(syn_batch, data, batch_size_data):
    min_distances = torch.full(
        (syn_batch.size(0),), float("inf"), device=syn_batch.device
    )
    for start_idx in range(0, data.size(0), batch_size_data):
        end_idx = min(start_idx + batch_size_data, data.size(0))
        data_batch = data[start_idx:end_idx]
        distances = (syn_batch[:, None] - data_batch).abs().sum(dim=2)
        min_batch_distances, _ = distances.min(dim=1)
        min_distances = torch.min(min_distances, min_batch_distances)
    return min_distances


def transform_data(
    real_data: tuple[pd.DataFrame, pd.DataFrame],
    syn_data: tuple[pd.DataFrame, pd.DataFrame],
    test_data: tuple[pd.DataFrame, pd.DataFrame],
    num_scaler: MinMaxScaler | None = None,
    cat_encoder: OneHotEncoder | None = None,
):
    cat_real_data, num_real_data = real_data
    cat_syn_data, num_syn_data = syn_data
    cat_test_data, num_test_data = test_data

    if cat_encoder is not None:
        cat_real_data_oh = cat_encoder.transform(cat_real_data.to_numpy()).toarray()
        cat_syn_data_oh = cat_encoder.transform(cat_syn_data.to_numpy()).toarray()
        cat_test_data_oh = cat_encoder.transform(cat_test_data.to_numpy()).toarray()
    else:
        assert (
            cat_real_data.shape[1]
            == cat_syn_data.shape[1]
            == cat_test_data.shape[1]
            == 0
        )
        cat_real_data_oh = np.empty((cat_real_data.shape[0], 0))
        cat_syn_data_oh = np.empty((cat_syn_data.shape[0], 0))
        cat_test_data_oh = np.empty((cat_test_data.shape[0], 0))

    if num_scaler is not None:
        num_real_data_np = num_scaler.transform(num_real_data.to_numpy())
        num_syn_data_np = num_scaler.transform(num_syn_data.to_numpy())
        num_test_data_np = num_scaler.transform(num_test_data.to_numpy())

    real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
    syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
    test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)
    return real_data_np, syn_data_np, test_data_np


def eval_dcr(
    syn_data,
    real_data,
    test_data,
    metadata,
    dcr_batch_size=1000,
    n_repeats=1,
    device="cpu",
    subsample=None,
    save_path="",
):
    num_columns = metadata.get_column_names(sdtype="numerical")
    cat_columns = metadata.get_column_names(
        sdtype="categorical"
    ) + metadata.get_column_names(sdtype="boolean")

    scaler = MinMaxScaler()
    scaler.fit(real_data[num_columns].to_numpy())
    cat_encoder = None
    if len(cat_columns) > 0:
        cat_encoder = OneHotEncoder(
            handle_unknown="error"
        )  # the held-out and synthetic data should have the same attributes
        cat_encoder.fit(real_data[cat_columns].to_numpy())

    num_real_data = real_data[num_columns]
    cat_real_data = real_data[cat_columns]
    num_syn_data = syn_data[num_columns]
    cat_syn_data = syn_data[cat_columns]
    num_test_data = test_data[num_columns]
    cat_test_data = test_data[cat_columns]

    real_data_np_full, syn_data_np_full, test_data_np_full = transform_data(
        (cat_real_data, num_real_data),
        (cat_syn_data, num_syn_data),
        (cat_test_data, num_test_data),
        num_scaler=scaler,
        cat_encoder=cat_encoder,
    )

    num_real = real_data_np_full.shape[0]
    num_syn = syn_data_np_full.shape[0]
    if subsample is None:
        subsample = num_syn
    num_test = test_data_np_full.shape[0]
    # The held out data should be at least half the size of the real and synthetic sets
    assert num_real * 0.5 <= num_test
    print(f"Real data size: {num_real}")
    print(f"Synthetic data size: {num_syn}")
    print(f"Test data size: {num_test}")

    scores = []
    dcrs_real_full = []
    dcrs_test_full = []
    for repeat_i in range(n_repeats):
        idx_syn = np.random.choice(num_syn, size=min(subsample, num_syn), replace=False)

        real_data_np = real_data_np_full
        syn_data_np = syn_data_np_full[idx_syn]
        test_data_np = test_data_np_full

        real_data_th = torch.tensor(real_data_np).to(device)
        syn_data_th = torch.tensor(syn_data_np).to(device)
        test_data_th = torch.tensor(test_data_np).to(device)

        dcrs_real = []
        dcrs_test = []
        batch_size = dcr_batch_size

        for i in tqdm(
            range((syn_data_th.shape[0] // batch_size) + 1),
            desc=f"Calculating DCR ({repeat_i + 1}/{n_repeats})",
        ):
            if i != (syn_data_th.shape[0] // batch_size):
                batch_syn_data_th = syn_data_th[i * batch_size : (i + 1) * batch_size]
            else:
                batch_syn_data_th = syn_data_th[i * batch_size :]

            # Calculate distances for real and test data in smaller batches
            dcr_real = calculate_min_distances(
                batch_syn_data_th, real_data_th, batch_size
            )
            dcr_test = calculate_min_distances(
                batch_syn_data_th, test_data_th, batch_size
            )

            dcrs_real.append(dcr_real)
            dcrs_test.append(dcr_test)

        dcrs_real = torch.cat(dcrs_real)
        dcrs_test = torch.cat(dcrs_test)

        equal = (dcrs_real == dcrs_test) * 0.5
        per_sample_score = (dcrs_real < dcrs_test) * 1.0 + equal
        score = per_sample_score.mean().item()

        scores.append(score)
        dcrs_real_full.append(dcrs_real)
        dcrs_test_full.append(dcrs_test)

        # print("DCR Score, a value closer to 0.5 is better")
        print(f"Repetition {repeat_i}: DCR Score = {score}")
    # Calculate the average score
    score = np.mean(scores)
    score_std = np.std(scores)
    score_se = score_std / np.sqrt(len(scores))
    print(f"Average DCR Score: {score} ± {score_se}")
    dcrs_real = torch.cat(dcrs_real_full)
    dcrs_test = torch.cat(dcrs_test_full)

    torch.save(dcrs_real.cpu(), f"{save_path}dcrs_real.pt")
    torch.save(dcrs_test.cpu(), f"{save_path}dcrs_test.pt")
    return score, score_se, score_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DCR")
    parser.add_argument(
        "--database_name",
        type=str,
        help="Name of the database to process.",
        choices=["california_clava_dcr", "berka_clava_dcr"],
    )
    parser.add_argument("--method", type=str, help="Method to evaluate.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory where the data is stored.",
    )
    args = parser.parse_args()
    database_name = args.database_name
    method = args.method
    data_dir = args.data_dir

    if not os.path.exists(f"results/dcr/{method}_dcr.json"):
        print("No previous DCR results found. Creating a new file.")
        dcr_results = {}
    else:
        print("Loading DCR results from previous runs.")
        with open(f"results/dcr/{method}_dcr.json", "r") as f:
            dcr_results = json.load(f)

    dcr_results.setdefault(database_name, {})

    metadata = Metadata().load_from_json(
        f"{data_dir}/original/{database_name}/metadata.json"
    )
    tables = load_tables(f"{data_dir}/original/{database_name}", metadata)

    tables_real = load_tables(f"{data_dir}/original/{database_name}", metadata)
    tables_test = load_tables(f"{data_dir}/original/{database_name}_test", metadata)
    tables_syn = load_tables(
        f"{data_dir}/synthetic/{database_name}/{method}/1/sample1", metadata
    )
    metadata.validate_data(tables_real)
    metadata.validate_data(tables_test)
    metadata.validate_data(tables_syn)

    for table in metadata.get_tables():
        print(f"Evaluating table: {table}")
        syn_data = tables_syn[table].copy()
        real_data = tables_real[table].copy()
        test_data = tables_test[table].copy()

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"Using device: {device}")
        
        save_path = f"results/dcr/{table}_{method}_"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        score, se, score_std = eval_dcr(
            syn_data,
            real_data,
            test_data,
            metadata.get_table_meta(table, to_dict=False),
            device=device,
            save_path=save_path,
        )
        dcr_results[database_name][table] = {
            "score": score,
            "se": se,
            "std": score_std,
        }

    with open(f"results/dcr/{method}_dcr.json", "w") as f:
        json.dump(dcr_results, f, indent=4)
