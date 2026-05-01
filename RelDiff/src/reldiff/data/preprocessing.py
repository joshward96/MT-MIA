import os
import json
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from reldiff.data.utils import (
    transform_datetime,
    encode_data,
    Normalization,
    get_decimals,
)

TYPE_TRANSFORM = {"float", np.float32, "str", str, "int", int}


def get_column_name_mapping(
    data_df: pd.DataFrame,
    num_col_idx: list,
    cat_col_idx: list,
    column_names: Optional[list] = None,
):
    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping = {}

    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def process_data(
    data: pd.DataFrame,
    name: str,
    metadata: SingleTableMetadata,
    data_path: str = "data",
    dataset_name: str = "",
    normalization: Normalization = "quantile",
    standardize: bool = False,
    sigma_data: float = 1.0,
):
    # Preprocessing
    save_dir = f"{data_path}/processed/{dataset_name}/{name}"
    eval_dir = f"{data_path}/eval/{dataset_name}/{name}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    categorical_columns = metadata.get_column_names(
        sdtype="categorical"
    ) + metadata.get_column_names(sdtype="boolean")
    datetime_columns = metadata.get_column_names(sdtype="datetime")
    numerical_columns = metadata.get_column_names(sdtype="numerical")
    id_columns = metadata.get_column_names(sdtype="id")

    for col in categorical_columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].cat.add_categories("?")
            data[col] = data[col].fillna("?")

    # store datetime columns
    if len(datetime_columns) > 0:
        data[datetime_columns].to_csv(
            f"{save_dir}/dates.csv",
            index=False,
        )
    # convert datetime column to int
    min_date = data[datetime_columns].min().min()
    # set the time for min_date to 0:0:0
    # In case of multiple datetime columns offset each column by the previous
    datecolumn_order = data[datetime_columns].mean().sort_values().index.tolist()

    min_date = pd.to_datetime(min_date).replace(hour=0, minute=0, second=0)
    offset = 0
    for i, col in enumerate(datecolumn_order):
        data, date_columns = transform_datetime(data, col, min_datetime=min_date)
        numerical_columns += date_columns
        if i > 0:
            data[f"{col}_date"] = data[f"{col}_date"] - offset
        offset += data[f"{col}_date"]

    # handle constant columns
    constants = dict()
    constant_columns = data.columns[data.nunique() == 1]
    for col in constant_columns:
        if col in id_columns:
            continue
        constants[col] = data[col].unique()[0]
        if col in categorical_columns:
            categorical_columns.remove(col)
        if col in numerical_columns:
            numerical_columns.remove(col)
    with open(f"{save_dir}/constants.pkl", "wb") as f:
        pickle.dump(constants, f)
    constant_columns = [col for col in constant_columns if col not in id_columns]
    data.drop(columns=constant_columns, inplace=True)

    # Store missing mask for numerical columns and add missing indicator columns
    missing_mask = pd.DataFrame(index=data.index, columns=numerical_columns)
    for col in numerical_columns:
        missing_mask[col] = data[col].isnull()
        col_mean = data[col].mean()
        if data[col].isnull().sum() > 0:
            data[f"{col}_missing"] = missing_mask[col].astype(int)
            categorical_columns.append(f"{col}_missing")
        data[col] = data[col].fillna(col_mean)

    # Move id columns to the end
    for col in id_columns:
        column = data.pop(col)
        data[col] = column

    cat_col_idx = sorted([data.columns.get_loc(c) for c in categorical_columns])
    num_col_idx = sorted([data.columns.get_loc(c) for c in numerical_columns])

    column_names = data.columns.tolist()

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data, num_col_idx, cat_col_idx, column_names
    )

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]

    print(name, data.shape)

    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]["type"] = "numerical"
        col_name = idx_name_mapping[col_idx]
        if col_name not in metadata.columns:
            original_col_name = "_".join(col_name.split("_")[:-1])
            if original_col_name in datetime_columns:
                subtype = "int"
                num_decimals = 0
            else:
                raise ValueError(f"Column {col_name} not found in metadata")
        else:
            col_meta = metadata.columns[col_name]
            if col_meta["sdtype"] == "numerical":
                if col_meta["computer_representation"] == "Int64":
                    subtype = "int"
                    num_decimals = 0
                elif col_meta["computer_representation"] == "Float":
                    subtype = "float"
                    if f"{col_name}_missing" in data.columns:
                        num_decimals = get_decimals(
                            data[col_name][~data[f"{col_name}_missing"].astype(bool)]
                        )
                    else:
                        num_decimals = get_decimals(data[col_name])
                else:
                    raise ValueError(
                        f"Unknown computer representation {col_meta['computer_representation']}"
                    )
        col_info[col_idx]["subtype"] = subtype
        col_info[col_idx]["max"] = float(data[col_name].max())
        col_info[col_idx]["min"] = float(data[col_name].min())
        col_info[col_idx]["decimals"] = num_decimals

    for col_idx in cat_col_idx:
        col_name = idx_name_mapping[col_idx]
        col_info[col_idx] = {}
        col_info[col_idx]["type"] = "categorical"
        col_info[col_idx]["subtype"] = "str"
        col_info[col_idx]["categorizes"] = data[col_name].unique().tolist()

    info = {
        "name": name,
        "column_info": col_info,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
    }

    data.rename(columns=idx_name_mapping, inplace=True)

    missing_mask = missing_mask[num_columns].to_numpy()
    np.save(f"{save_dir}/num_missing.npy", missing_mask)
    X_num = data[num_columns].to_numpy().astype(np.float64)
    X_cat = data[cat_columns].to_numpy()

    data[num_columns] = data[num_columns]

    data.to_csv(f"{save_dir}/data.csv", index=False)

    print("Numerical", X_num.shape)
    print("Categorical", X_cat.shape)

    info["column_names"] = column_names

    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    metadata = {"columns": {}}

    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    with open(f"{save_dir}/info.json", "w") as file:
        json.dump(info, file, indent=4)

    print(f"Processing and Saving {name} Successfully!")

    print(name)
    print("Num", info["num_col_idx"])
    print("Cat", info["cat_col_idx"])

    # Encoding
    X_num, X_cat, num_transform, cat_transform = encode_data(
        X_num,
        X_cat,
        missing_mask,
        normalization=normalization,
        cat_encoding="ordinal",
        standardize=standardize,
        sigma_data=sigma_data,
    )

    X_num = X_num.astype(np.float64)
    X_cat = X_cat.astype(np.int64)

    np.save(f"{save_dir}/X_num.npy", X_num)
    np.save(f"{save_dir}/X_cat.npy", X_cat)

    with open(f"{save_dir}/num_transform.pkl", "wb") as file:
        pickle.dump(num_transform, file)

    with open(f"{save_dir}/cat_transform.pkl", "wb") as file:
        pickle.dump(cat_transform, file)
