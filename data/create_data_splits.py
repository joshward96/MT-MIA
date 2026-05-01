import os
import json
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

def split_by_entity_unit(
    data_dir,
    out_dir,
    schema_path,
    entity_unit,
    N,
    n_splits=3,
    split_names=None,
    seed=42,
    ordinal_encode=False,
    encoding_map_path=None,
):
    """
    Split data by entity unit and optionally apply ordinal encoding.
    """
    rng = np.random.default_rng(seed)
    
    # Load schema
    with open(schema_path, "r") as f:
        schema = json.load(f)
    tables = schema["tables"]
    
    # Load all CSVs
    dfs = {}
    for t in tables:
        file_path = os.path.join(data_dir, f"{t}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path).fillna(0)
            dfs[t] = df
        else:
            print(f"Warning: {t}.csv not found in {data_dir}")

    unit_id_col = f"{entity_unit}_id"
    if entity_unit not in dfs:
        raise ValueError(f"Entity unit '{entity_unit}' table not found.")
    
    # Ordinal Encoding Setup
    encoding_maps = {}
    if ordinal_encode:
        for table, df in dfs.items():
            categorical_cols = [c for c in df.select_dtypes(include=['object']).columns 
                                if not c.endswith('_id')]
            if categorical_cols:
                encoding_maps[table] = {}
                for col in categorical_cols:
                    unique_vals = [v for v in df[col].unique() if v != 0 and v != '0']
                    encoding_maps[table][col] = {
                        val: idx + 1 for idx, val in enumerate(sorted(unique_vals))
                    }
    
    # Sample unit IDs
    unit_ids = dfs[entity_unit][unit_id_col].unique()
    rng.shuffle(unit_ids)
    
    if (n_splits * N) > len(unit_ids):
        raise ValueError(f"Not enough {entity_unit} IDs available.")
    
    unit_to_split = {}
    for s in range(n_splits):
        for uid in unit_ids[s*N : (s+1)*N]:
            unit_to_split[uid] = s
    
    # Assign rows and pull parents
    split_tables = [defaultdict(list) for _ in range(n_splits)]
    for table, df in dfs.items():
        if unit_id_col in df.columns:
            for row in df.to_dict("records"):
                uid = row[unit_id_col]
                if uid in unit_to_split:
                    split_tables[unit_to_split[uid]][table].append(row)
    
    # Referential Integrity
    for s in range(n_splits):
        added = True
        while added:
            added = False
            for table, rows in list(split_tables[s].items()):
                if table not in tables: continue
                for parent in tables[table].get("parents", []):
                    pid = f"{parent}_id"
                    if parent not in dfs: continue
                    needed = {r[pid] for r in rows if pid in r}
                    existing = {r[pid] for r in split_tables[s][parent] if pid in r}
                    missing = needed - existing
                    if missing:
                        new_rows = dfs[parent][dfs[parent][pid].isin(missing)].to_dict("records")
                        if new_rows:
                            split_tables[s][parent].extend(new_rows)
                            added = True
    
    # Write splits
    os.makedirs(out_dir, exist_ok=True)
    names = split_names or [f"split_{i}" for i in range(n_splits)]
    
    for s, name in enumerate(names):
        split_dir = os.path.join(out_dir, name)
        os.makedirs(split_dir, exist_ok=True)
        for table, rows in split_tables[s].items():
            if not rows: continue
            out_df = pd.DataFrame(rows).drop_duplicates()
            if ordinal_encode and table in encoding_maps:
                for col, mapping in encoding_maps[table].items():
                    if col in out_df.columns:
                        out_df[col] = out_df[col].map(mapping).fillna(0).astype(int)
            out_df.to_csv(os.path.join(split_dir, f"{table}.csv"), index=False)
    
    if ordinal_encode and encoding_map_path:
        with open(encoding_map_path, 'w') as f:
            json.dump(encoding_maps, f, indent=2)
    
    return encoding_maps

def main():
    parser = argparse.ArgumentParser(description="Split relational data by entity units.")
    parser.add_argument("--data_dir", type=str, default="airlines/cleaned/", help="Input CSV directory")
    parser.add_argument("--out_dir", type=str, default="airlines/split/", help="Output directory")
    parser.add_argument("--schema_path", type=str, default="airlines/dataset_meta.json", help="Path to schema JSON")
    parser.add_argument("--entity_unit", type=str, default="loyalty_history", help="Table name to split by")
    parser.add_argument("--N", type=int, default=1000, help="Units per split")
    parser.add_argument("--n_splits", type=int, default=3, help="Number of splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--encode", action="store_true", default=True, help="Enable ordinal encoding")
    parser.add_argument("--map_path", type=str, default="airlines/encoding_maps.json", help="Path for encoding map")

    args = parser.parse_args()

    # Define your specific split names
    custom_names = ["mem", "non_mem", "ref"] if args.n_splits == 3 else None

    split_by_entity_unit(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        schema_path=args.schema_path,
        entity_unit=args.entity_unit,
        N=args.N,
        n_splits=args.n_splits,
        split_names=custom_names,
        seed=args.seed,
        ordinal_encode=args.encode,
        encoding_map_path=args.map_path
    )
    print(f"Successfully created {args.n_splits} splits in {args.out_dir}")

if __name__ == "__main__":
    main()