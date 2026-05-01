"""Prepare MT-MIA datasets for RelDiff.

Converts data/{dataset}/split/mem/ (our format) into
RelDiff/data/original/{dataset}/ (SDV MultiTableMetadata format).
"""

import json
import os
import shutil

import pandas as pd

MT_MIA_DATA_DIR = "data"
RELDIFF_DATA_DIR = "RelDiff/data/original"

DATASETS = ["california", "airbnb", "airlines"]


def _computer_representation(series: pd.Series) -> str:
    """Return 'Int64' if all non-null values are whole numbers, else 'Float'."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return "Float"
    try:
        if (non_null == non_null.astype("int64")).all():
            return "Int64"
    except (ValueError, TypeError):
        pass
    return "Float"


def build_metadata(dataset_name):
    src_dir = os.path.join(MT_MIA_DATA_DIR, dataset_name, "split", "mem")

    with open(os.path.join(src_dir, "dataset_meta.json")) as f:
        meta = json.load(f)

    tables_sdv = {}
    for table_name, table_info in meta["tables"].items():
        primary_key = f"{table_name}_id"

        domain_path = os.path.join(src_dir, f"{table_name}_domain.json")
        with open(domain_path) as f:
            domain = json.load(f)

        # Load CSV to detect integer vs float for numerical columns
        csv_path = os.path.join(src_dir, f"{table_name}.csv")
        df = pd.read_csv(csv_path)

        columns = {primary_key: {"sdtype": "id"}}
        for col_name, col_info in domain.items():
            if col_info["type"] == "discrete":
                columns[col_name] = {"sdtype": "categorical"}
            else:
                comp_repr = _computer_representation(df[col_name])
                columns[col_name] = {
                    "sdtype": "numerical",
                    "computer_representation": comp_repr,
                }

        # Foreign key columns (parent_id carried in child table)
        for parent in table_info["parents"]:
            fk_col = f"{parent}_id"
            columns[fk_col] = {"sdtype": "id"}

        tables_sdv[table_name] = {
            "primary_key": primary_key,
            "columns": columns,
        }

    relationships = []
    for parent, child in meta["relation_order"]:
        if parent is None:
            continue
        relationships.append(
            {
                "parent_table_name": parent,
                "parent_primary_key": f"{parent}_id",
                "child_table_name": child,
                "child_foreign_key": f"{parent}_id",
            }
        )

    return {
        "METADATA_SPEC_VERSION": "V1",
        "tables": tables_sdv,
        "relationships": relationships,
    }, list(meta["tables"].keys())


def prepare_dataset(dataset_name):
    src_dir = os.path.join(MT_MIA_DATA_DIR, dataset_name, "split", "mem")
    dst_dir = os.path.join(RELDIFF_DATA_DIR, dataset_name)
    os.makedirs(dst_dir, exist_ok=True)

    sdv_meta, table_names = build_metadata(dataset_name)

    with open(os.path.join(dst_dir, "metadata.json"), "w") as f:
        json.dump(sdv_meta, f, indent=2)

    for table_name in table_names:
        src_csv = os.path.join(src_dir, f"{table_name}.csv")
        dst_csv = os.path.join(dst_dir, f"{table_name}.csv")
        shutil.copy(src_csv, dst_csv)
        print(f"  Copied {table_name}.csv")

    print(f"Prepared {dataset_name} → {dst_dir}")


if __name__ == "__main__":
    for dataset in DATASETS:
        prepare_dataset(dataset)
