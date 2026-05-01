import os
import pickle
import argparse

from torch_geometric.utils.convert import to_networkx
from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns

from reldiff.data import create_dataset, process_data


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_name", default="f1_subsampled", type=str)
argparser.add_argument("--data-path", default="data", type=str)
args = argparser.parse_args()

database_name = args.dataset_name
data_dir = args.data_path

has_multiedges = database_name == "CORA_v1"


metadata_path = os.path.join(data_dir, "original", database_name, "metadata.json")
metadata = Metadata().load_from_json(metadata_path)


tables = load_tables(f"{data_dir}/original/{database_name}/", metadata)
tables, metadata = remove_sdv_columns(tables, metadata)
for table_name, table in tables.items():
    process_data(
        table,
        name=table_name,
        metadata=metadata.get_table_meta(table_name, to_dict=False),
        data_path=data_dir,
        dataset_name=database_name,
    )

data_path = os.path.join(data_dir, "processed", database_name)


dataset = create_dataset(
    metadata,
    data_path,
    order_cols={},
    mask_missing=False,
    transform_fk_tables=False,
    add_reverse_edges=False,
)


G = to_networkx(dataset, to_multi=has_multiedges)


# Save the networkx graph
os.makedirs("data/structure", exist_ok=True)
with open(f"data/structure/{database_name}_graph.pkl", "wb") as f:
    pickle.dump(G, f)
