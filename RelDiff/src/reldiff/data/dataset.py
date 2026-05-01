import os

import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from syntherela.metadata import Metadata
from torch_geometric.data import HeteroData
from torch_geometric import transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix, sort_edge_index

from reldiff.data.utils import get_categories


def get_connected_components(data):
    homo = data.to_homogeneous()
    adj = to_scipy_sparse_matrix(homo.edge_index.cpu())

    num_components, component = sp.csgraph.connected_components(adj, connection="weak")
    components = dict()
    for i, key in enumerate(data.x_cat_dict.keys()):
        components[key] = component[homo.node_type.cpu() == i]

    return components, num_components


def transform_foreign_key_tables(data: HeteroData, metadata: Metadata):
    for table_name in data.node_types:
        # select foreign key tables
        if (
            data[table_name].x_num.shape[1] == 0
            and data[table_name].x_cat.shape[1] == 0
        ):
            assert len(metadata.get_children(table_name)) == 0
            parents = metadata.get_parents(table_name)
            if len(parents) == 1:
                parents = list(parents) * 2
                foreign_keys = metadata.get_foreign_keys(parents[0], table_name)
            else:
                foreign_keys = []
                for parent in parents:
                    fks = metadata.get_foreign_keys(parent, table_name)
                    foreign_keys.extend(list(fks))
            assert len(parents) == 2
            parent1, parent2 = parents
            fk1, fk2 = foreign_keys

            edge_index1 = data[(table_name, fk1, parent1)].edge_index
            edge_index2 = data[(table_name, fk2, parent2)].edge_index

            assert edge_index1.shape[1] == data[table_name].num_nodes
            assert edge_index2.shape[1] == data[table_name].num_nodes
            # The edge indexes should be the same as they are ordered by the child node
            assert (edge_index1[0] == edge_index2[0]).all().item()

            new_edge_index = torch.stack([edge_index1[1], edge_index2[1]])
            new_reverse_edge_index = torch.stack([edge_index2[1], edge_index1[1]])

            data[(parent1, f"{table_name}_{fk1}", parent2)].edge_index = new_edge_index
            data[
                (parent2, f"{table_name}_{fk2}", parent1)
            ].edge_index = new_reverse_edge_index
            # remove the node type
            del data[table_name]
            # remove the edge types
            for edge_type in data.edge_types:
                if edge_type[0] == table_name or edge_type[2] == table_name:
                    del data[edge_type]

    return data


def create_dataset(
    metadata: Metadata,
    data_path: str,
    order_cols: dict = {},
    mask_missing: bool = False,
    transform_fk_tables: bool = True,
    add_reverse_edges: bool = True,
    remove_isolated: bool = False,
) -> tuple[HeteroData, dict, int]:
    data = HeteroData()
    tables = dict()

    id_map = dict()
    for table_name in metadata.get_tables():
        table_dir = os.path.join(data_path, table_name)
        tables[table_name] = pd.read_csv(os.path.join(table_dir, "data.csv"))
        X_num = np.load(os.path.join(table_dir, "X_num.npy"))
        X_cat = np.load(os.path.join(table_dir, "X_cat.npy"))
        assert X_num.dtype == np.float64
        missing_mask = np.load(os.path.join(table_dir, "num_missing.npy"))
        data[table_name].x_cat = torch.from_numpy(X_cat).long()
        data[table_name].x_num = torch.from_numpy(X_num).float()
        if mask_missing:
            data[table_name].missing_mask = torch.from_numpy(missing_mask).bool()
        data[table_name].num_nodes = tables[table_name].shape[0]
        data[table_name].d_numerical = X_num.shape[1]
        data[table_name].categories = get_categories(X_cat)
        if table_name in order_cols:
            date_df = pd.read_csv(
                os.path.join(table_dir, "dates.csv"),
                parse_dates=[order_cols[table_name]],
            )
            values = date_df[order_cols[table_name]].values
            values = values - values.min()
            data[table_name].order = torch.tensor(
                values / np.timedelta64(1, "D")
            ).long()

        # store the id mappings to integers
        primary_key = metadata.get_primary_key(table_name)
        if primary_key is None:
            tables[table_name].reset_index(inplace=True)
            primary_key = "index"

        id_map.setdefault(table_name, dict())
        if primary_key not in id_map[table_name]:
            id_map[table_name][primary_key] = dict()
            idx = 0
            for primary_key_val in tables[table_name][primary_key].unique():
                id_map[table_name][primary_key][primary_key_val] = idx
                idx += 1

        for relationship in metadata.relationships:
            if relationship["parent_table_name"] != table_name:
                continue
            if relationship["child_table_name"] not in id_map:
                id_map[relationship["child_table_name"]] = {}

            id_map[relationship["child_table_name"]][
                relationship["child_foreign_key"]
            ] = id_map[table_name][relationship["parent_primary_key"]]

    # remap the ids
    for table_name in id_map.keys():
        for column_name in id_map[table_name].keys():
            if column_name not in tables[table_name].columns:
                raise ValueError(
                    f"Column {column_name} not found in table {table_name}"
                )
            tables[table_name][column_name] = tables[table_name][column_name].map(
                id_map[table_name][column_name]
            )

    # Set edges based on relationships.
    for relationship in metadata.relationships:
        parent_table = relationship["parent_table_name"]
        child_table = relationship["child_table_name"]
        foreign_key = relationship["child_foreign_key"]

        child_primary_key = metadata.get_primary_key(child_table)
        if child_primary_key is None:
            child_primary_key = "index"
        tables[child_table] = tables[child_table].dropna(subset=[child_primary_key])

        # some relationships can have missing foreign keys
        fks = tables[child_table][[foreign_key, child_primary_key]]
        fks = fks.dropna().astype("int64")
        assert not fks.empty, f"Empty foreign keys for {child_table} -> {parent_table}"
        edge_index = sort_edge_index(torch.tensor(fks.values.T).long())
        reverse_edge_index = sort_edge_index(
            torch.tensor(fks[[child_primary_key, foreign_key]].values.T).long()
        )

        data[parent_table, foreign_key, child_table].edge_index = edge_index
        if add_reverse_edges:
            data[child_table, foreign_key, parent_table].edge_index = reverse_edge_index

    if transform_fk_tables:
        data = transform_foreign_key_tables(data, metadata)

    if remove_isolated:
        remove_isolated_transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
        data = remove_isolated_transform(data)
    assert data.validate(), "Invalid graph"
    return data


def dataset_from_graph(
    G: nx.Graph,
    dataset: HeteroData,
    metadata: Metadata,
    add_reverse_edges: bool = True,
    transform_fk_tables: bool = True,
    dimension_tables: list = [],
) -> HeteroData:
    generated = HeteroData()
    add_ordering = False

    # Get node types
    node_types = nx.get_node_attributes(G, "type")
    nodes_by_table = {table: [] for table in set(node_types.values())}

    for node in G.nodes:
        nodes_by_table[node_types[node]].append(node)

    # Add nodes
    for table_name, nodes in nodes_by_table.items():
        cat_shape = dataset[table_name].x_cat.shape
        num_shape = dataset[table_name].x_num.shape
        num_nodes = len(nodes)
        generated[table_name].x_cat = torch.zeros(
            (num_nodes, cat_shape[1]), dtype=torch.long
        )
        generated[table_name].x_num = torch.zeros(
            (num_nodes, num_shape[1]), dtype=torch.float32
        )
        if table_name in dimension_tables:
            generated[table_name].x_num = dataset[table_name].x_num
            generated[table_name].x_cat = dataset[table_name].x_cat
        generated[table_name].num_nodes = num_nodes
        generated[table_name].d_numerical = dataset[table_name].d_numerical
        generated[table_name].categories = dataset[table_name].categories
        if "order" in dataset[table_name]:
            add_ordering = True

    # Get edge types
    edge_types = nx.get_edge_attributes(G, "type")
    edges_by_type = {edge_type: [] for edge_type in set(edge_types.values())}

    for edge in G.edges:
        edges_by_type[edge_types[edge]].append(edge)

    # Add edges
    for edge_type in dataset.edge_index_dict.keys():
        table1, key, table2 = edge_type
        edges = []
        if edge_type not in edges_by_type:
            continue
        for edge in edges_by_type[edge_type]:
            edges.append(list(edge))
        edges = torch.tensor(edges, dtype=torch.long)
        table1_nodes = nodes_by_table[table1]
        table2_nodes = nodes_by_table[table2]
        min_table1_id = min(table1_nodes)
        min_table2_id = min(table2_nodes)
        edge_index = edges[:, :2] - torch.tensor([min_table1_id, min_table2_id])
        # sort the edge index by the child node
        edge_index = sort_edge_index(edge_index.T)
        generated[edge_type].edge_index = edge_index
        # add reverse edge
        if add_reverse_edges:
            reverse_edge_index = sort_edge_index(edge_index[[1, 0], :])
            generated[(edge_type[2], key, edge_type[0])].edge_index = reverse_edge_index

    if transform_fk_tables:
        generated = transform_foreign_key_tables(generated, metadata)
    # orderings
    if add_ordering:
        for table, ordering in dataset.order_dict.items():
            parent = list(metadata.get_parents(table))[0]
            foreign_key = metadata.get_foreign_keys(parent, table)[0]
            edges = generated[(parent, foreign_key, table)].edge_index
            groups = edges[0, :].unique()
            original_ordering = dataset.order_dict[table]
            ordering = torch.zeros_like(original_ordering)
            pattern = original_ordering.unique()
            for group in groups:
                members = edges[0, :] == group
                # repeat the ordering pattern for each group
                order = pattern.repeat(members.sum() // pattern.shape[0] + 1)
                ordering[members] = order[: members.sum()]

            generated[table].order = ordering
    assert generated.validate(), "Invalid graph"
    return generated


if __name__ == "__main__":
    DATA_DIR = "./data"
    database_name = "rossmann_subsampled"
    metadata_path = os.path.join(DATA_DIR, "original", database_name, "metadata.json")
    metadata = Metadata().load_from_json(metadata_path)
    data_path = os.path.join(DATA_DIR, "processed", database_name)
    order_enc = {"historical": "Date"}
    dataset = create_dataset(metadata, data_path, order_enc=order_enc)
    print(dataset)
