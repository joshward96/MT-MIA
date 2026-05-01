import os
import pickle

import torch
import numpy as np
import pandas as pd
from syntherela.metadata import Metadata

from reldiff.data.utils import split_num_cat_target, recover_data


def convert_synthetic_tables(
    syn_tables: dict[str, torch.Tensor],
    edge_index: dict[tuple[str, str, str], torch.Tensor],
    metadata: Metadata,
    info_dict: dict[str, dict],
    sample_ids: bool = False,
) -> dict[str, pd.DataFrame]:
    synthetic_tables = dict()
    for table_name, syn_data in syn_tables.items():
        print(f"Shape of the generated sample = {syn_data.shape}")
        info = info_dict[table_name]

        num_all_zero_row = (syn_data.sum(dim=1) == 0).sum()
        if num_all_zero_row:
            print(f"The generated samples contain {num_all_zero_row} Nan instances!!!")
            print({"num_Nan_sample": num_all_zero_row})

        # Recover tables
        processed_data_path = info["processed_data_path"]

        with open(os.path.join(processed_data_path, "cat_transform.pkl"), "rb") as f:
            cat_transform = pickle.load(f)

        with open(os.path.join(processed_data_path, "num_transform.pkl"), "rb") as f:
            num_transform = pickle.load(f)

        def identity(x) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x

        if num_transform is not None:
            num_inverse = num_transform.inverse_transform
        else:
            num_inverse = identity
        int_inverse = identity
        if cat_transform is not None:
            cat_inverse = cat_transform.inverse_transform
        else:
            cat_inverse = identity

        info["task_type"] = None
        info["target_col_idx"] = []
        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, num_inverse, int_inverse, cat_inverse
        )

        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns=idx_name_mapping, inplace=True)
        synthetic_tables[table_name] = syn_df

        if sample_ids:
            pk = metadata.get_primary_key(table_name)
            if pk is not None:
                syn_df[pk] = np.arange(len(syn_df))
            # add foreign keys
            for parent in metadata.get_parents(table_name):
                for foreign_key in metadata.get_foreign_keys(parent, table_name):
                    # edge index is sorted by the child id
                    edges = edge_index[(parent, foreign_key, table_name)].cpu().numpy()
                    assert edges[1].tolist() == sorted(edges[1].tolist())
                    syn_df[foreign_key] = edges[0]

    return synthetic_tables
