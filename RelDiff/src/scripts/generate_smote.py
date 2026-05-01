import os
import argparse

import numpy as np
import pandas as pd

from syntherela.metadata import Metadata
from syntherela.data import load_tables, save_tables

from imblearn.over_sampling import SMOTE, SMOTENC


# Argument parser
parser = argparse.ArgumentParser(description="Generate synthetic data using SMOTE.")
parser.add_argument("database_name", type=str, help="Name of the database to process.")
parser.add_argument(
    "--data_dir", type=str, default="data", help="Directory where the data is stored."
)
args = parser.parse_args()

database_name = args.database_name
data_dir = args.data_dir

metadata_path = os.path.join(data_dir, "original", database_name, "metadata.json")

# Load the metadata and tables
metadata = Metadata().load_from_json(metadata_path)
tables = load_tables(f"{data_dir}/original/{database_name}/", metadata)


tables_syn = {}
for table in metadata.get_tables():
    print(f"Table: {table}")
    print(f"  Number of rows: {len(tables[table])}")
    print(f"  Number of columns: {len(tables[table].columns)}")
    print(f"  Column names: {tables[table].columns.tolist()}")

    primary_key = metadata.get_primary_key(table)

    df = tables[table]
    cat_features: list = metadata.get_column_names(table, sdtype="categorical")
    bool_features: list = metadata.get_column_names(table, sdtype="boolean")
    cat_features.extend(bool_features)
    id_columns = metadata.get_column_names(table, sdtype="id")
    datetime_columns = metadata.get_column_names(table, sdtype="datetime")
    numerical_columns = metadata.get_column_names(table, sdtype="numerical")

    df = df.drop(columns=id_columns)

    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].astype(np.int64)
        numerical_columns.append(col)

    if len(cat_features) > 0:
        idx_cat_features = [df.columns.get_loc(col) for col in cat_features]
        sm = SMOTENC(categorical_features=idx_cat_features)
    else:
        sm = SMOTE()

    df[numerical_columns] = df[numerical_columns].fillna(0)

    dummy_y = np.random.randint(0, 2, size=len(df))
    X_smote, _ = sm.fit_resample(df.values, dummy_y)

    df_syn = pd.DataFrame(X_smote, columns=df.columns)
    for col in datetime_columns:
        df_syn[col] = pd.to_datetime(df_syn[col], unit="ns", errors="coerce")
    if primary_key is not None:
        df_syn[primary_key] = np.arange(len(df_syn))

    tables_syn[table] = df_syn


# Add foreign keys to ensure schema consistency
for table in tables_syn:
    parents = metadata.get_parents(table)
    for parent_table in parents:
        parent_primary_key = metadata.get_primary_key(parent_table)
        for foreign_key in metadata.get_foreign_keys(parent_table, table):
            tables_syn[table][foreign_key] = (
                tables_syn[parent_table][parent_primary_key]
                .sample(len(tables_syn[table]), replace=True)
                .values
            )


metadata.validate_data(tables_syn)


save_tables(
    tables_syn,
    f"{data_dir}/synthetic/{database_name}/SMOTE/1/sample1",
)
