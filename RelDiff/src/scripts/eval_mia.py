import os
import json
import argparse

from syntherela.data import load_tables
from syntherela.metadata import Metadata

from syntheval import SynthEval


def eval_mia(
    syn_data,
    real_data,
    test_data,
    metadata,
):
    """
    Adapted from https://github.com/jacobyhsi/TabRep/blob/main/eval/eval_privacy.py
    """
    for df in (real_data, test_data, syn_data):
        df.drop(
            columns=[metadata.primary_key], inplace=True
        )  # Remove primary key column
        for col_name in df.columns:  # Remove columns with 'id' in their name
            if "id" in col_name.lower():
                df.drop(columns=[col_name], inplace=True)
    for df in (real_data, test_data, syn_data):
        # find all categorical columns
        cat_cols = df.select_dtypes(include=["category"]).columns
        if len(cat_cols):
            # convert each one to its integer codes
            for col in cat_cols:
                df[col] = df[col].cat.codes
            print(f"Converted categorical columns to int codes: {list(cat_cols)}")

    S = SynthEval(real_data, holdout_dataframe=test_data)
    eval_df = S.evaluate(
        syn_data, None, "mia"
    )  # set the target column to the primary key of the table

    # Filter for rows with 'mia_recall' and 'mia_precision'
    filtered_rows = eval_df[eval_df["metric"].isin(["mia_recall", "mia_precision"])]

    # Extract values into variables
    mia_recall_val = filtered_rows.loc[
        filtered_rows["metric"] == "mia_recall", "val"
    ].values[0]
    mia_recall_err = filtered_rows.loc[
        filtered_rows["metric"] == "mia_recall", "err"
    ].values[0]
    mia_precision_val = filtered_rows.loc[
        filtered_rows["metric"] == "mia_precision", "val"
    ].values[0]
    mia_precision_err = filtered_rows.loc[
        filtered_rows["metric"] == "mia_precision", "err"
    ].values[0]

    # Print extracted variables

    print("mia_precision_val:", mia_precision_val)
    print("mia_precision_err:", mia_precision_err)
    print("mia_recall_val:", mia_recall_val)
    print("mia_recall_err:", mia_recall_err)

    return mia_precision_val, mia_precision_err, mia_recall_val, mia_recall_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MIA")
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

    if not os.path.exists(f"results/mia/{method}_mia.json"):
        print("No previous MIA results found. Creating a new file.")
        mia_results = {}
    else:
        print("Loading MIA results from previous runs.")
        with open(f"results/mia/{method}_mia.json", "r") as f:
            mia_results = json.load(f)

    mia_results.setdefault(database_name, {})

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

        save_path = f"results/mia/{table}_{method}_"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        mia_precision_val, mia_precision_err, mia_recall_val, mia_recall_err = eval_mia(
            syn_data,
            real_data,
            test_data,
            metadata.get_table_meta(table, to_dict=False),
        )
        mia_results[database_name][table] = {
            "precision": mia_precision_val,
            "precision_err": mia_precision_err,
            "recall": mia_recall_val,
            "recall_err": mia_recall_err,
        }

    with open(f"results/mia/{method}_mia.json", "w") as f:
        json.dump(mia_results, f, indent=4)
