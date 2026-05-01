import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput


# def get_dataloader(
#     dataset: HeteroData,
#     target_table: str,
#     num_layers: int = 2,
#     num_neighbors: int = -1,
#     batch_size: int = 64,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> NeighborLoader:
#     num_nodes = dataset[target_table].num_nodes
#     return NeighborLoader(
#         dataset,
#         num_neighbors=[num_neighbors for _ in range(num_layers)],
#         input_nodes=(target_table, torch.arange(num_nodes)),
#         batch_size=batch_size,
#         disjoint=False,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         persistent_workers=num_workers > 0,
#         # drop_last=num_nodes > batch_size,
#     )


# def get_dataloaders(
#     data: HeteroData,
#     batch_size: int = 1,
#     shuffle: bool = False,
#     table_to_n_hops: dict[str, int] | None = None,
#     table_to_num_neighbors: dict[str, list] | None = None,
#     n_hops: int = 2,
#     num_neighbors: int = -1,
#     num_workers: int = 0,
# ) -> dict[str, NeighborLoader]:
#     r"""Return dataloaders for a given database, one dataloader for each table in the database."""
#     if table_to_n_hops is None:
#         table_to_n_hops = {table_name: n_hops for table_name in data.node_types}
#     if table_to_num_neighbors is None:
#         table_to_num_neighbors = {
#             table_name: [num_neighbors] * table_to_n_hops[table_name]
#             for table_name in data.node_types
#         }
#     for table_name, n_hops in table_to_n_hops.items():
#         if len(table_to_num_neighbors[table_name]) != n_hops:
#             raise ValueError(
#                 f"Number of hops for table {table_name} does not match the number of neighbors per hop."
#             )

#     dataloaders = {}
#     for table_name in data.node_types:
#         dataloaders[table_name] = NeighborLoader(
#             data.cpu(),
#             num_neighbors=table_to_num_neighbors[table_name],
#             input_nodes=(table_name, None),
#             batch_size=batch_size,
#             shuffle=shuffle,
#             drop_last=False,
#             disjoint=True,  # will need to set this to False for some datasets
#             num_workers=num_workers,
#             persistent_workers=num_workers > 0,
#         )

#     return dataloaders


def get_subgraph_dataloader(
    data: HeteroData,
    root_table: str | None = None,
    batch_size: int = 1,
    shuffle: bool = False,
    n_hops: int = 2,
    num_neighbors: int = -1,
    num_workers: int = 8,
    is_disjoint: bool = True,
    dimension_tables: list[str] = [],
    num_seed_nodes: int | None = None,
    drop_last: bool = False,
    two_stage: bool = False,
) -> NeighborLoader:
    r"""Return dataloaders for a given database, one dataloader for each table in the database."""
    num_neighbors = [num_neighbors] * n_hops
    if is_disjoint:
        assert root_table is not None, (
            "Root table must be provided for disjoint subgraph dataloader."
        )
        input_nodes = (root_table, None)
        indices = torch.arange(data[root_table].num_nodes)
        if shuffle and data[root_table].num_nodes < batch_size:
            # Repeat the indices to fill the batch size
            indices = torch.cat([indices] * (batch_size // data[root_table].num_nodes))
            input_nodes = (root_table, indices)
        return NeighborLoader(
            data.cpu(),
            num_neighbors=num_neighbors,
            input_nodes=input_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            replace=False,
            disjoint=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            input_id=indices,
        )
    else:
        if two_stage:
            assert num_seed_nodes is not None, "Number of seed nodes must be provided."
            return HeteroTwoStageNeighborLoader(
                dataset=data.cpu(),
                n_hops=n_hops,
                replace=False,
                shuffle=shuffle,
                dimension_tables=dimension_tables,
                num_neighbors=num_neighbors,
                num_seed_nodes=num_seed_nodes,
            )
        else:
            return HeteroNeighborLoader(
                dataset=data.cpu(),
                batch_size=batch_size,
                n_hops=n_hops,
                replace=False,
                dimension_tables=dimension_tables,
                shuffle=shuffle,
                num_neighbors=num_neighbors,
                drop_last=drop_last,
                num_workers=num_workers,
            )


class HeteroTwoStageNeighborLoader(DataLoader):
    def __init__(
        self,
        dataset: HeteroData,
        n_hops: int = 2,
        replace: bool = False,
        dimension_tables: list = [],
        shuffle: bool = False,
        num_neighbors: int | list[int] = -1,
        num_seed_nodes: int = 10,
    ):
        self.data = dataset
        self.replace = replace
        self.shuffle = shuffle
        self.n_hops = n_hops
        self.dimension_tables = dimension_tables
        if not isinstance(num_neighbors, list):
            num_neighbors = [num_neighbors] * n_hops
        self.num_neighbors = num_neighbors
        self.num_seed_nodes_ = num_seed_nodes

        self.homogeneous = dataset.to_homogeneous(node_attrs=[], dummy_values=False)
        self.homogeneous.original_id = torch.zeros_like(self.homogeneous.node_type)

        self.dimension_mask = torch.zeros(self.homogeneous.num_nodes, dtype=torch.bool)
        for node_type_id, node_type in enumerate(dataset.node_types):
            mask = self.homogeneous.node_type == node_type_id
            self.homogeneous.original_id[mask] = torch.arange(mask.sum())
            if node_type in dimension_tables:
                self.dimension_mask[self.homogeneous.node_type == node_type_id] = True

        self.neighbor_sampler = NeighborSampler(
            self.homogeneous,
            num_neighbors=self.num_neighbors,
            replace=replace,
            subgraph_type="induced",
            disjoint=False,
        )

        self.seed_sampler = NeighborSampler(
            self.homogeneous,
            num_neighbors=[-1] * (n_hops * 2),
            replace=replace,
            subgraph_type="induced",
            disjoint=False,
        )

        num_batches = self.estimate_num_batches()
        self.sampled = torch.zeros(self.homogeneous.num_nodes, dtype=torch.bool)
        self.num_seed_nodes = self.num_seed_nodes_
        self.decreased = False

        self.iterator = range(num_batches)
        super().__init__(self.iterator, collate_fn=self.collate_fn)

    def reset_iterator(self):
        self.sampled = torch.zeros(self.homogeneous.num_nodes, dtype=torch.bool)
        self.num_seed_nodes = self.num_seed_nodes_
        self.decreased = False
        # print("Restarting iterator")
        self.seed_sampler = NeighborSampler(
            self.homogeneous,
            num_neighbors=[-1] * (self.n_hops * 2),
            replace=self.replace,
            subgraph_type="induced",
            disjoint=False,
        )

    def decrease_neighborhood_size(self):
        self.decreased = True
        # print("Decreasing neighborhood size and increasing batch size")
        self.num_seed_nodes = self.num_seed_nodes_ * 1000
        self.seed_sampler = NeighborSampler(
            self.homogeneous,
            num_neighbors=[-1],
            replace=self.replace,
            subgraph_type="induced",
            disjoint=False,
        )

    def sample_input_nodes(self):
        if self.sampled.all():
            self.reset_iterator()
            raise StopIteration("All nodes have been sampled")

        if self.sampled.float().mean() > 0.90 and not self.decreased:
            self.decrease_neighborhood_size()

        # Sample seed nodes from unvisited nodes
        indices = torch.arange(self.homogeneous.num_nodes)
        indices = indices[~(self.sampled | self.dimension_mask)]
        if self.shuffle:
            idx = torch.randperm(indices.shape[0])
            indices = indices[idx]
        # Select seed nodes
        seed_ids = indices[: self.num_seed_nodes]

        seed_data = NodeSamplerInput(
            input_id=None,
            node=seed_ids,
        )

        # Select the neighborhood of the seed nodes
        inputs = self.seed_sampler.sample_from_nodes(seed_data)
        # Select and mark sampled nodes
        input_ids = inputs.node
        self.sampled[input_ids] = True
        return input_ids

    def convert_to_heterodata(self, batch_ids, input_ids):
        node_ids = {
            table: torch.tensor([], dtype=torch.long) for table in self.data.node_types
        }
        input_nodes = {
            table: torch.tensor([], dtype=torch.long) for table in self.data.node_types
        }

        # Select input nodes
        input_mask = torch.isin(batch_ids, input_ids)
        original_ids = self.homogeneous.original_id[batch_ids]
        for node_type_id, node_type in enumerate(self.data.node_types):
            # Select nodes of the same type
            node_type_mask = self.homogeneous.node_type[batch_ids] == node_type_id

            node_ids[node_type] = original_ids[node_type_mask]
            input_nodes[node_type] = original_ids[node_type_mask & input_mask]

        subgraph = self.data.subgraph(node_ids)
        subgraph.set_value_dict("n_id", node_ids)
        subgraph.set_value_dict("input_id", input_nodes)
        return subgraph

    def collate_fn(self, input):
        if input[0] == len(self.iterator) - 1:
            assert self.sampled.all(), "Last batch should be empty"
        # 1) Sample the seed nodes
        # 2) and their neighborhood.
        input_ids = self.sample_input_nodes()

        # 3) Sample the neighborhood of the input nodes
        input_data = NodeSamplerInput(
            input_id=None,
            node=input_ids,
        )
        batch = self.neighbor_sampler.sample_from_nodes(input_data)

        return self.convert_to_heterodata(batch.node, input_ids)

    def estimate_num_batches(self, times=3):
        if not self.shuffle:
            times = 1
        num_batches = []
        for _ in range(times):
            batches = 0
            self.sampled = torch.zeros(self.homogeneous.num_nodes, dtype=torch.bool)
            self.num_seed_nodes = self.num_seed_nodes_
            self.decreased = False
            while True:
                batches += 1
                try:
                    self.sample_input_nodes()
                except StopIteration:
                    break
            num_batches.append(batches)
        if not self.shuffle:
            return batches - 1
        # print("Estimated number of batches", num_batches)
        return int(max(num_batches) * 1.5)


class HeteroNeighborLoader(DataLoader):
    def __init__(
        self,
        dataset: HeteroData,
        batch_size: int,
        n_hops: int = 2,
        replace: bool = False,
        dimension_tables: list = [],
        shuffle: bool = False,
        num_neighbors: int | list[int] = 64,
        drop_last: bool = False,
        num_workers: int = 0,
    ):
        self.data = dataset
        self.replace = replace
        self.shuffle = shuffle
        self.n_hops = n_hops
        self.batch_size = batch_size

        self.dimension_tables = dimension_tables
        if not isinstance(num_neighbors, list):
            num_neighbors = [num_neighbors] * n_hops

        self.num_neighbors = num_neighbors

        self.homogeneous = dataset.to_homogeneous(node_attrs=[], dummy_values=False)
        self.homogeneous.original_id = torch.zeros_like(self.homogeneous.node_type)

        self.dimension_mask = torch.zeros(self.homogeneous.num_nodes, dtype=torch.bool)
        for node_type_id, node_type in enumerate(dataset.node_types):
            mask = self.homogeneous.node_type == node_type_id
            self.homogeneous.original_id[mask] = torch.arange(mask.sum())
            if node_type in dimension_tables:
                self.dimension_mask[self.homogeneous.node_type == node_type_id] = True

        self.neighbor_sampler = NeighborSampler(
            self.homogeneous,
            num_neighbors=self.num_neighbors,
            replace=replace,
            subgraph_type="induced",
            disjoint=False,
        )

        indices = torch.arange(self.data.num_nodes)
        # Remove dimension table indices
        indices = indices[~self.dimension_mask].numpy().tolist()
        super().__init__(
            indices,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    def sample_nodes(self, seed_ids):
        seed_data = NodeSamplerInput(
            input_id=seed_ids,
            node=seed_ids,
        )

        # Select the neighborhood of the seed nodes
        batch = self.neighbor_sampler.sample_from_nodes(seed_data)
        return batch.node

    def convert_to_heterodata(self, batch_ids, input_ids):
        node_ids = {
            table: torch.tensor([], dtype=torch.long) for table in self.data.node_types
        }
        input_nodes = {
            table: torch.tensor([], dtype=torch.long) for table in self.data.node_types
        }

        # Select input nodes
        input_mask = torch.isin(batch_ids, input_ids)
        for node_type_id, node_type in enumerate(self.data.node_types):
            # Select nodes of the same type
            node_type_mask = self.homogeneous.node_type[batch_ids] == node_type_id

            original_ids = self.homogeneous.original_id[batch_ids]

            node_ids[node_type] = original_ids[node_type_mask]
            input_nodes[node_type] = original_ids[node_type_mask & input_mask]

        subgraph = self.data.subgraph(node_ids)
        subgraph.set_value_dict("n_id", node_ids)
        subgraph.set_value_dict("input_id", input_nodes)
        return subgraph

    def collate_fn(self, index):
        input_ids = torch.tensor(index, dtype=torch.long)
        node_ids = self.sample_nodes(input_ids)

        return self.convert_to_heterodata(node_ids, input_ids)
