import torch
import pandas as pd
import os
import json
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import OrdinalEncoder


def build_encoders(tables: dict):
    """Fit encoders on categorical columns from train split."""
    encoders = {}
    for table_name, df in tables.items():
        cat_cols = [c for c in df.columns 
                   if not c.endswith("_id") and not pd.api.types.is_numeric_dtype(df[c])]
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            enc.fit(df[cat_cols].astype(str))
            encoders[table_name] = (enc, cat_cols)
    return encoders


class EntityUnitDataset(Dataset):
    """Dataset that creates one HeteroData graph per entity unit."""
    
    def __init__(self, split_dir: str, schema: dict, entity_unit: str, encoders: dict = None):
        self.schema = schema
        self.entity_unit = entity_unit
        self.encoders = encoders or {}
        
        # Load all tables
        self.tables = {}
        for table_name in schema["tables"]:
            path = os.path.join(split_dir, f"{table_name}.csv")
            if os.path.exists(path):
                self.tables[table_name] = pd.read_csv(path)
        
        print(f"Loaded tables: {list(self.tables.keys())}")
        
        # Get unique entity unit IDs
        self.unit_ids = self.tables[entity_unit][f"{entity_unit}_id"].unique()
    
    def __len__(self):
        return len(self.unit_ids)
    
    def __getitem__(self, idx):
        unit_id = self.unit_ids[idx]
        return self._build_graph(unit_id)
    
    def _encode_features(self, table_name, df):
        """Encode features for a table, ensuring all outputs are numeric."""
        # 1. Identify feature columns (exclude IDs)
        feat_cols = [c for c in df.columns if not c.endswith("_id")]
        
        if not feat_cols:
            return torch.ones((len(df), 1))
        
        X = df[feat_cols].copy()
        
        # 2. Handle Categorical Encoding
        if table_name in self.encoders:
            enc, cat_cols = self.encoders[table_name]
            # Ensure we only try to transform columns that exist in this slice
            existing_cat_cols = [c for c in cat_cols if c in X.columns]
            if existing_cat_cols:
                X[existing_cat_cols] = enc.transform(X[existing_cat_cols].astype(str))
        
        # 3. Final Safety Check: Force everything to numeric
        # This converts any remaining strings/objects to NaN, then fills with 0
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        return torch.tensor(X.values, dtype=torch.float)
    
    def _build_graph(self, unit_id):
        """Build a HeteroData graph for one entity unit with padding for missing types."""
        data = HeteroData()
        node_maps = {}
        
        # 1. Add all nodes (including padded nodes for missing types)
        for table_name, table_info in self.schema["tables"].items():
            table_id_col = f"{table_name}_id"
            table_df = self.tables.get(table_name, pd.DataFrame())
            
            rows = pd.DataFrame()
            if table_name == self.entity_unit:
                rows = table_df[table_df[table_id_col] == unit_id]
            else:
                # Logic to find child rows connected to the unit_id
                parent_tables = table_info["parents"]
                if parent_tables:
                    mask = pd.Series([False] * len(table_df))
                    for parent in parent_tables:
                        parent_id_col = f"{parent}_id"
                        if parent_id_col in table_df.columns:
                            if parent == self.entity_unit:
                                mask |= table_df[parent_id_col] == unit_id
                            elif parent in node_maps:
                                parent_ids = list(node_maps[parent].keys())
                                mask |= table_df[parent_id_col].isin(parent_ids)
                    rows = table_df[mask]

            if len(rows) > 0:
                # Standard case: nodes exist
                data[table_name].x = self._encode_features(table_name, rows)
                node_maps[table_name] = {
                    row[table_id_col]: i 
                    for i, (_, row) in enumerate(rows.iterrows())
                }
            else:
                # PADDING CASE: create one dummy node of zeros
                # We determine feature dimension by running a dummy row through the encoder
                dummy_row = pd.DataFrame([[""] * len(table_df.columns)], columns=table_df.columns)
                dummy_feat = self._encode_features(table_name, dummy_row)
                data[table_name].x = torch.zeros_like(dummy_feat) # Shape [1, num_features]
                
                # Use a special ID (-99 or None) to map the dummy node
                node_maps[table_name] = {"dummy_id": 0}

        # 2. Add edges
        for table_name, table_info in self.schema["tables"].items():
            for parent_name in table_info["parents"]:
                # If both exist in node_maps (which they will now, due to padding)
                src_list, dst_list = [], []
                
                table_id_col = f"{table_name}_id"
                parent_id_col = f"{parent_name}_id"
                table_df = self.tables.get(table_name, pd.DataFrame())

                # Scenario A: Real connections
                child_ids = [k for k in node_maps[table_name].keys() if k != "dummy_id"]
                rows = table_df[table_df[table_id_col].isin(child_ids)]
                
                for _, row in rows.iterrows():
                    child_id, parent_id = row[table_id_col], row[parent_id_col]
                    if parent_id in node_maps[parent_name]:
                        src_list.append(node_maps[parent_name][parent_id])
                        dst_list.append(node_maps[table_name][child_id])

                # Scenario B: Padded connections (to keep the graph connected)
                # If the child is a dummy, connect it to the first available parent
                if "dummy_id" in node_maps[table_name]:
                    # Connect dummy child to index 0 of the parent (which might also be a dummy)
                    src_list.append(0) 
                    dst_list.append(node_maps[table_name]["dummy_id"])
                
                # Scenario C: Real child but parent is dummy
                elif "dummy_id" in node_maps[parent_name]:
                    for local_idx in node_maps[table_name].values():
                        src_list.append(0) # Map all children to the dummy parent
                        dst_list.append(local_idx)

                if src_list:
                    data[(parent_name, "to", table_name)].edge_index = \
                        torch.tensor([src_list, dst_list], dtype=torch.long)
                    data[(table_name, "to", parent_name)].edge_index = \
                        torch.tensor([dst_list, src_list], dtype=torch.long)
        
        return data


def load_custom_split_loaders(
    split_dirs: dict,
    schema_path: str,
    entity_unit: str,
    batch_size: int = 32,
    seed: int = 42,  # Add seed parameter
):
    """
    Load datasets and create DataLoaders for all splits.
    
    Args:
        split_dirs: Dict mapping split names to paths, e.g. 
                   {"train": "splits/train", "mem": "splits/mem", ...}
        schema_path: Path to schema JSON file
        entity_unit: Root table name for each graph
        batch_size: Batch size for DataLoaders
        seed: Random seed for reproducibility
    
    Returns:
        train_loader: DataLoader for training
        embed_loaders: Dict of DataLoaders for other splits
        encoders: Fitted encoders from training data
    """
    # Load schema
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    # Load train tables and fit encoders
    train_dir = split_dirs["train"]
    train_tables = {}
    for table_name in schema["tables"]:
        path = os.path.join(train_dir, f"{table_name}.csv")
        if os.path.exists(path):
            train_tables[table_name] = pd.read_csv(path)
    
    encoders = build_encoders(train_tables)
    
    # Setup seed worker function and generator
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np
        import random
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Create train dataset and loader with seeding
    train_dataset = EntityUnitDataset(train_dir, schema, entity_unit, encoders)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Create loaders for other splits (no shuffling needed for eval)
    embed_loaders = {}
    for split_name, split_dir in split_dirs.items():
        if split_name == "train":
            continue
        dataset = EntityUnitDataset(split_dir, schema, entity_unit, encoders)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        embed_loaders[split_name] = loader
    
    return train_loader, embed_loaders, encoders
class LabeledEntityDataset(torch.utils.data.Dataset):
    """
    Wraps an EntityUnitDataset and injects a binary label.
    """
    def __init__(self, base_dataset, label: int):
        self.base_dataset = base_dataset
        self.label = float(label)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        data.y = torch.tensor(self.label, dtype=torch.float)
        return data
