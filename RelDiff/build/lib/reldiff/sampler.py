import os
import time
import pickle

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from syntherela.metadata import Metadata

from reldiff.utils import print_with_bar
from reldiff.data.utils import split_num_cat_target, recover_data
from reldiff.diffusion import MultiTableUnifiedCtimeDiffusion


class MultiTableSampler:
    def __init__(
        self,
        diffusion: MultiTableUnifiedCtimeDiffusion,
        dataset: HeteroData,
        metrics_dict,
        device=torch.device("cuda:1"),
        ckpt_path=None,
        dimension_tables: list | None = None,
        sampling_device=torch.device("cuda:0"),
        sampling_batch_size: int = 20000,
        **kwargs,
    ):
        self.diffusion = diffusion

        self.dataset = dataset
        self.dimension_tables = dimension_tables
        self.tables = dataset
        self.metrics_dict = metrics_dict

        self.device = device
        self.sampling_device = sampling_device
        self.sampling_batch_size = sampling_batch_size

        self.ckpt_path = ckpt_path
        if self.ckpt_path is not None:
            state_dicts = torch.load(self.ckpt_path, map_location=self.device)
            self.diffusion._denoise_fn.load_state_dict(state_dicts["denoise_fn"])
            self.diffusion.num_schedule.load_state_dict(state_dicts["num_schedule"])
            self.diffusion.cat_schedule.load_state_dict(state_dicts["cat_schedule"])
            print(f"Weights are loaded from {self.ckpt_path}")

    def sample_synthetic(self, sample_ids=False, metadata=None, batch_size=20000):
        print_with_bar("Starting Sampling")
        start_time = time.time()

        syn_tables = self.diffusion.sample_all(
            dataset=self.dataset, device=self.sampling_device, batch_size=batch_size
        )

        end_time = time.time()
        print_with_bar(
            f"Ending Sampling, total sampling time = {end_time - start_time}"
        )

        synthetic_tables = dict()
        for table_name, syn_data in syn_tables.items():
            print(f"Shape of the generated sample = {syn_data.shape}")
            metrics = self.metrics_dict[table_name]
            info = metrics.info

            num_all_zero_row = (syn_data.sum(dim=1) == 0).sum()
            if num_all_zero_row:
                print(
                    f"The generated samples contain {num_all_zero_row} Nan instances!!!"
                )

            # Recover tables
            processed_data_path = os.path.dirname(metrics.real_data_path)

            with open(
                os.path.join(processed_data_path, "cat_transform.pkl"), "rb"
            ) as f:
                cat_transform = pickle.load(f)

            with open(
                os.path.join(processed_data_path, "num_transform.pkl"), "rb"
            ) as f:
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
            idx_name_mapping = {
                int(key): value for key, value in idx_name_mapping.items()
            }

            syn_df.rename(columns=idx_name_mapping, inplace=True)

            if sample_ids:
                edge_index = self.dataset.edge_index_dict
                # add foreign keys
                for parent in metadata.get_parents(table_name):
                    for foreign_key in metadata.get_foreign_keys(parent, table_name):
                        # edge index is sorted by the child id
                        edges = (
                            edge_index[(table_name, foreign_key, parent)].cpu().numpy()
                        )
                        index = edges[0].tolist()
                        assert index == sorted(edges[0].tolist())
                        syn_df[foreign_key] = pd.Series(dtype="Int64")
                        syn_df.loc[index, foreign_key] = edges[1]
                # add primary key
                pk = metadata.get_primary_key(table_name)
                if pk is not None:
                    syn_df[pk] = np.arange(len(syn_df))

            synthetic_tables[table_name] = syn_df

        return synthetic_tables

    def sample_database(
        self,
        database_name: str,
        metadata: Metadata,
        dataset: HeteroData,
        mask_missing: bool = True,
        batch_size: int = 20000,
    ):
        synthetic_tables = self.sample_synthetic(
            sample_ids=True, metadata=metadata, batch_size=batch_size
        )

        # add constant columns and foreign keys
        for table in metadata.get_tables():
            with open(
                f"data/processed/{database_name}/{table}/constants.pkl", "rb"
            ) as f:
                constants = pickle.load(f)
            df = synthetic_tables.get(table, pd.DataFrame())
            # add foreign key only table
            if table not in synthetic_tables:
                df = None
                for edge_type in dataset.edge_types:
                    if edge_type[1].startswith(table):
                        edge_index = dataset[edge_type].edge_index
                        foreign_key = edge_type[1].replace(f"{table}_", "")
                        if df is None:
                            primary_key = metadata.get_primary_key(table)
                            df = pd.DataFrame()
                            df[primary_key] = np.arange(edge_index.shape[1])
                        df[foreign_key] = edge_index[0].numpy()
            # add constant columns
            # NOTE: a table will be treated as a foreign key only table if it only has
            # constant columns. Therefore we add those here (after foreign keys).
            for col, value in constants.items():
                df[col] = value
            synthetic_tables[table] = df

        for table, df in synthetic_tables.items():
            # handle missing values
            cat_columns = df.select_dtypes(include=["object"]).columns.to_list()
            for col in cat_columns:
                if "?" in df[col].unique():
                    df[col] = df[col].replace("?", np.nan)

            for col in df.columns:
                if col.endswith("_missing") and mask_missing:
                    imputed_column = col.split("_missing")[0]
                    missing_mask = df[col].astype(int).astype(bool)
                    df[imputed_column] = df[imputed_column].astype("float64")
                    df.loc[missing_mask, imputed_column] = np.nan
                    df = df.drop(columns=[col])

            # convert dates to datetime
            datetime_columns = metadata.get_column_names(table, sdtype="datetime")
            if len(datetime_columns) > 0:
                dates_df = pd.read_csv(
                    os.path.join(
                        f"data/processed/{database_name}/{table}", "dates.csv"
                    ),
                    parse_dates=datetime_columns,
                )
                min_date = dates_df[datetime_columns].min().min()
                # set the time for min_date to 0:0:0
                min_date = pd.to_datetime(min_date).replace(hour=0, minute=0, second=0)
                # In case of multiple datetime columns each is offset by the previous
                datetime_columns = (
                    dates_df[datetime_columns].mean().sort_values().index.tolist()
                )
            for i, col in enumerate(datetime_columns):
                timedelta = pd.to_timedelta(df[f"{col}_date"].round(), unit="D")
                if i > 0:
                    offset = df[datetime_columns[i - 1]] - min_date
                    timedelta += offset
                date_columns = [f"{col}_date"]
                if dates_df[col].dt.hour.sum() > 0:
                    seconds = pd.to_timedelta(df[f"{col}_time"].round(), unit="s")
                    timedelta += seconds
                    date_columns.append(f"{col}_time")
                df[col] = min_date + timedelta
                df = df.drop(columns=date_columns)
            synthetic_tables[table] = df

        metadata.validate_data(synthetic_tables)
        return synthetic_tables
