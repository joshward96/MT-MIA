import argparse

from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns
from reldiff.data.preprocessing import process_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="rossmann_subsampled", type=str)
    parser.add_argument("--normalization", default="quantile", type=str)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--sigma-data", default=1.0, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    DATA_PATH = "data"

    args = parse_args()
    dataset_name = args.dataset_name
    normalization = args.normalization
    standardize = args.standardize
    sigma_data = args.sigma_data

    metadata = Metadata().load_from_json(
        f"{DATA_PATH}/original/{dataset_name}/metadata.json"
    )
    tables = load_tables(f"{DATA_PATH}/original/{dataset_name}/", metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)

    for table_name, table in tables.items():
        process_data(
            table,
            name=table_name,
            metadata=metadata.get_table_meta(table_name, to_dict=False),
            data_path=DATA_PATH,
            dataset_name=dataset_name,
            normalization=normalization,
            standardize=standardize,
            sigma_data=sigma_data,
        )
