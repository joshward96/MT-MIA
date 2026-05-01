import os
import json
import argparse

import wandb
import torch
import numpy as np
from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns

from reldiff.metrics import TabMetrics
from reldiff.configs.utils import load_config
from reldiff.data import create_dataset, process_data
from reldiff.data.utils import get_category_proportions
from reldiff.data.dataloader import get_subgraph_dataloader
from reldiff.trainer import MultiTableTrainer
from reldiff.models import ModelJoint, GraphDiff
from reldiff.diffusion.unified_ctime_diffusion import MultiTableUnifiedCtimeDiffusion


argparser = argparse.ArgumentParser()
argparser.add_argument("dataset_name", default="rossmann_subsampled", type=str)
argparser.add_argument(
    "--num-epochs", default=10000, type=int, help="Number of epochs to train for"
)
argparser.add_argument("--batch-size", default=4096, type=int)
argparser.add_argument(
    "--sampling-batch-size", default=20000, type=int, help="Batch size for sampling"
)
argparser.add_argument("--run-id", default="", type=str)
argparser.add_argument("--no-wandb", action="store_true")
argparser.add_argument(
    "--config-path", default="src/reldiff/configs/reldiff_config.toml", type=str
)
argparser.add_argument("--sampling-device", default="cuda", type=str)
argparser.add_argument(
    "--device", default="cuda", type=str, help="Device to use for training"
)
argparser.add_argument("--use-ema", action="store_true", help="Use EMA for training")

args = argparser.parse_args()

run_id = args.run_id
database_name = args.dataset_name
num_epochs = args.num_epochs
batch_size = args.batch_size

# Load config
config = load_config(args.config_path)

n_hops_dataloader = config["graph"][
    "n_hops"
]  # This can be different for some disjoint datasets


DATA_DIR = "./data"
if args.device == "cuda" and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dimension_tables = []
order_cols = {}
if database_name == "rossmann_subsampled":
    is_disjoint = True
    order_cols = {"historical": "Date"}
elif database_name == "airbnb-simplified_subsampled":
    is_disjoint = True
elif database_name == "walmart_subsampled":
    is_disjoint = True
    order_cols = {"depts": "Date", "features": "Date"}
elif database_name == "california_clava":
    is_disjoint = True
elif database_name == "Biodegradability_v1":
    is_disjoint = True
    n_hops_dataloader = 3
elif database_name == "ccs_clava":
    is_disjoint = False
    dimension_tables = ["product"]  # TODO: get this from metadata
elif database_name == "berka_clava" or database_name == "Berka_subsampled":
    is_disjoint = False
    dimension_tables = ["district"]
elif database_name == "instacart_05_clava":
    is_disjoint = False
    dimension_tables = ["aisle", "department", "product"]
elif database_name == "CORA_v1":
    is_disjoint = False
    # The dataset has no numerical features
    config["diffusion_params"]["sampler_params"]["stochastic_sampler"] = False
    config["diffusion_params"]["sampler_params"]["second_order_correction"] = False
    config["diffusion_params"]["scheduler"] = "power_mean"
elif database_name == "f1_subsampled":
    is_disjoint = False
    dimension_tables = ["circuits"]
elif database_name == "california_clava_dcr" or database_name == "berka_clava_dcr":
    is_disjoint = True
elif database_name == "rel-hm":
    is_disjoint = False
    dimension_tables = ["article"]
elif database_name == "adventure_works":
    is_disjoint = False
    dimension_tables = [
        "PhoneNumberType",
        "ScrapReason",
        "EmailAddress",
        "Illustration",
        "BusinessEntity",
        "Shift",
        "UnitMeasure",
        "ProductPhoto",
        "Culture",
        "ContactType",
        "Currency",
        "PersonCreditCard",
        "CountryRegion",
        "AddressType",
        "ProductCategory",
        "ShipMethod",
    ]
else:
    is_disjoint = False

if dimension_tables is not None:
    print(
        f"Database {database_name} has the following dimension tables: {dimension_tables}"
    )
    config["diffusion_params"]["edm_params"]["net_conditioning"] = (
        "t"  # Condition on time as sigma is 0 for dimension tables
    )

metadata_path = os.path.join(DATA_DIR, "original", database_name, "metadata.json")
metadata = Metadata().load_from_json(metadata_path)

# Preprocess data
tables = load_tables(f"{DATA_DIR}/original/{database_name}/", metadata)
tables, metadata = remove_sdv_columns(tables, metadata)
for table_name, table in tables.items():
    process_data(
        table,
        name=table_name,
        metadata=metadata.get_table_meta(table_name, to_dict=False),
        data_path=DATA_DIR,
        dataset_name=database_name,
        normalization=config["data"]["normalization"],
        standardize=config["data"]["standardize"],
        sigma_data=config["diffusion_params"]["edm_params"]["sigma_data"],
    )

data_path = os.path.join(DATA_DIR, "processed", database_name)
model_save_path = os.path.join("ckpt", database_name, "multi" + run_id)
result_save_path = os.path.join("results", database_name, "multi" + run_id)

dataset = create_dataset(
    metadata,
    data_path,
    order_cols=order_cols,
)

root_table = sorted(metadata.get_root_tables())[-1]
gnn_params = {
    "node_types": dataset.node_types,
    "edge_types": dataset.edge_types,
    "aggr": config["gnn"]["aggr"],
    "num_layers": config["graph"]["n_hops"],
    "type": config["gnn"]["type"],
}

order_dict = None
if set(metadata.get_tables()).intersection(set(order_cols.keys())):
    order_dict = dataset.order_dict


metrics_dict = dict()
categories = dict()
proportions_dict = dict()
for table in metadata.get_tables():
    # Skip foreign key only tables
    if table not in dataset.node_types:
        continue
    real_data_path = f"data/processed/{database_name}/{table}/data.csv"
    test_data_path = f"data/eval/{database_name}/{table}/data.csv"
    val_data_path = f"data/eval/{database_name}/{table}/data.csv"
    info_path = f"data/processed/{database_name}/{table}/info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    metrics_dict[table] = TabMetrics(
        real_data_path,
        test_data_path,
        val_data_path,
        info,
        device,
        metric_list=[
            "density",
            "c2st",
        ],
    )
    num_classes = dataset.categories_dict[table]
    categories[table] = (np.array(num_classes) + 1).tolist()
    proportions_dict[table] = get_category_proportions(
        dataset[table].x_cat, num_classes, add_mask=True
    )

    if config["data"]["standardize"]:
        sigma_data = config["diffusion_params"]["edm_params"]["sigma_data"]
        # Check that the data is standardized
        assert np.isclose(dataset[table].x_num.mean(axis=0), 0, atol=1e-6).all()
        # Missing values can impact this
        assert np.bitwise_or(
            np.isclose(dataset[table].x_num.std(axis=0), sigma_data, rtol=0.5),
            dataset[table].x_num.std(axis=0).numpy() <= sigma_data,
        ).all()

    if is_disjoint:
        # Set all nodes as input (target) nodes when using disjoint subgraphs.
        dataset[table].input_id = torch.arange(dataset[table].num_nodes).long()

# Together with standardization this sets the expected numerical loss to 1.0
zero_init = config["data"]["standardize"]  # zero initialize if standardizing
backbone = GraphDiff(
    dataset.d_numerical_dict,
    categories,  # dataset.categories_dict + 1 for mask state
    gnn_params,
    **config["model"],
    zero_init=zero_init,
    proportions_dict=proportions_dict,
    order_enc=order_dict,
).cuda()


model = ModelJoint(
    denoise_fn=backbone,
    **config["diffusion_params"]["edm_params"],
)

assert n_hops_dataloader == config["graph"]["n_hops"] or is_disjoint

diffusion = MultiTableUnifiedCtimeDiffusion(
    num_classes=dataset.categories_dict,
    num_numerical_features=dataset.d_numerical_dict,
    denoise_fn=model,
    **config["diffusion_params"],
    timestep_sampling=config["train"]["timestep_sampling"],
    device=device,
    root_table=root_table,
    n_hops_dataloader=n_hops_dataloader,
    proportions_dict=proportions_dict,
    dequantize=config["data"]["dequantize"],
    dataset=dataset,
    is_disjoint=is_disjoint,
    num_neighbors=config["graph"]["num_neighbors"],
    dimension_tables=dimension_tables,
)

num_params = sum(p.numel() for p in diffusion.parameters())
print("The number of parameters = ", num_params)
diffusion.to(device)
diffusion.train()


dataloader = get_subgraph_dataloader(
    dataset,
    root_table=root_table,
    batch_size=batch_size,
    shuffle=True,
    n_hops=n_hops_dataloader,
    num_neighbors=config["graph"]["num_neighbors"],
    is_disjoint=is_disjoint,
    dimension_tables=dimension_tables,
    two_stage=config["graph"]["two_stage_sampling"],
    num_workers=0,  # Increase this for faster dataloading
)

## Enable Wandb
project_name = f"reldiff_{database_name}"
config["project_name"] = project_name
exp_name = f"multi_{config['model']['model_dim']}_{num_epochs}"
if run_id != "":
    exp_name += f"-{run_id}"

logger = wandb.init(
    project=config["project_name"],
    name=exp_name,
    config=config,
    mode="disabled" if args.no_wandb else "online",
)

trainer = MultiTableTrainer(
    diffusion,
    dataset,
    dataloader=dataloader,
    epochs=num_epochs,
    metrics_dict=metrics_dict,
    logger=logger,
    **config["train"],
    model_save_path=model_save_path,
    result_save_path=result_save_path,
    device=device,
    dimension_tables=dimension_tables,
    sampling_device=args.sampling_device,
    sampling_batch_size=args.sampling_batch_size,
    use_ema=args.use_ema,
)


trainer.run_loop()
