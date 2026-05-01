import torch
import pandas as pd
import json
import os
from torch_geometric.loader import DataLoader
from src.synth_mia.attackers import *
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from src.utils import load_config
from src.synth_mia.attackers import gen_lra, dcr, dpi, logan, dcr_diff, domias, mc, density_estimate, local_neighborhood, classifier
from src.synth_mia.utils import tabular_preprocess
import numpy as np
import pandas as pd
import traceback
from sklearn.preprocessing import MinMaxScaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import pandas as pd
import json


def load_schema(schema_path):
    """Load schema from JSON file."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def infer_table_names(schema):
    """
    Infer parent and child table names from schema.
    Returns tuple: (parent_table, child_table)
    """
    tables = schema['tables']
    
    # Find parent table (has no parents)
    parent_table = None
    child_table = None
    
    for table_name, table_info in tables.items():
        if len(table_info['parents']) == 0:
            parent_table = table_name
        if len(table_info['parents']) > 0:
            child_table = table_name
    
    return parent_table, child_table

def load_tables_from_dir(directory, parent_table, child_table):
    """
    Load parent and child CSV files from a directory.
    Returns tuple: (parent_df, child_df)
    """
    parent_path = os.path.join(directory, f"{parent_table}.csv")
    child_path = os.path.join(directory, f"{child_table}.csv")

    if not os.path.exists(parent_path):
        raise FileNotFoundError(f"Parent table not found: {parent_path}")
    if not os.path.exists(child_path):
        raise FileNotFoundError(f"Child table not found: {child_path}")

    parent_df = pd.read_csv(parent_path)
    child_df = pd.read_csv(child_path)

    # Drop metadata columns
    # For parent table: drop parent_file_name_id if it exists
    parent_col_to_drop = f"{parent_table}_id"
    if parent_col_to_drop in parent_df.columns:
        parent_df = parent_df.drop(columns=[parent_col_to_drop])

    # For child table: drop both parent_file_name_id and child_file_name_id if they exist
    child_cols_to_drop = [f"{parent_table}_id", f"{child_table}_id"]
    existing_cols_to_drop = [col for col in child_cols_to_drop if col in child_df.columns]
    if existing_cols_to_drop:
        child_df = child_df.drop(columns=existing_cols_to_drop)

    return parent_df.astype(float).values, child_df.astype(float).values


def normalize_to_mem(mem, non_mem, ref, synth):
    """
    Fit a MinMaxScaler on mem and apply to all splits.
    Returns normalized (mem, non_mem, ref, synth).
    """
    scaler = MinMaxScaler()
    scaler.fit(mem)
    return (
        scaler.transform(mem),
        scaler.transform(non_mem),
        scaler.transform(ref),
        scaler.transform(synth),
    )


def main(config_path,
         train_path=None,
         interim_overwrite=None,
         schema_path=None,
         normalize=False):
    
    config = load_config(config_path)
    
    # --- APPLY OVERWRITES HERE ---
    if train_path:
        config['dataset_paths']['train'] = train_path
    
    if interim_overwrite:
        config['interim_output_file'] = interim_overwrite
    
    # Load schema to infer table names
    if schema_path is None:
        schema_path = config.get('schema_path', 'schema.json')
    
    schema = load_schema(schema_path)
    parent_table, child_table = infer_table_names(schema)

    print(f"Inferred tables - Parent: {parent_table}, Child: {child_table}")
    
    split_dirs = config['dataset_paths']
    interim_output_file = config.get('interim_output_file', 'black_box_ssl_mia_results.csv')
    print(split_dirs)
    output_dir = os.path.dirname(interim_output_file)
    if output_dir: # Only run if there is actually a directory in the path
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory ensured: {output_dir}")
    # Load tables from each directory
    mem_parent, mem_child = load_tables_from_dir(split_dirs['mem'], parent_table, child_table)
    non_mem_parent, non_mem_child = load_tables_from_dir(split_dirs['non_mem'], parent_table, child_table)
    ref_parent, ref_child = load_tables_from_dir(split_dirs['ref'], parent_table, child_table)
    synth_parent, synth_child = load_tables_from_dir(split_dirs['train'], parent_table, child_table)
    
    if normalize:
        mem_parent, non_mem_parent, ref_parent, synth_parent = normalize_to_mem(mem_parent, non_mem_parent, ref_parent, synth_parent)
        mem_child, non_mem_child, ref_child, synth_child = normalize_to_mem(mem_child, non_mem_child, ref_child, synth_child)

    # Store in dictionaries for easier access
    mem = {'parent': mem_parent, 'child': mem_child}
    non_mem = {'parent': non_mem_parent, 'child': non_mem_child}
    ref = {'parent': ref_parent, 'child': ref_child}
    synth = {'parent': synth_parent, 'child': synth_child}

    attackers = [
        dcr(),
        mc(),
        density_estimate(),
        dcr_diff(),
        #domias(),    
        classifier()
    ]

    parent_results = {}
    for attacker in attackers:
        try:
            # Run the attack
            scores, true_labels = attacker.attack(mem['parent'], 
                                                non_mem['parent'], 
                                                synth['parent'],
                                                ref['parent']
                                                )
            
            # Evaluate the attack
            eval_results = attacker.eval(true_labels, scores, metrics=['roc'])
            
            # Store results
            parent_results[attacker.name] = eval_results
            
        except Exception as e:
            print(f"Error running {attacker.name} on parent table: {e}")
            traceback.print_exc()
            parent_results[attacker.name] = None

    child_results = {}
    for attacker in attackers:
        try:
            # Run the attack
            scores, true_labels = attacker.attack(mem['child'], 
                                                non_mem['child'], 
                                                synth['child'],
                                                ref['child']
                                                )
            
            # Evaluate the attack
            eval_results = attacker.eval(true_labels, scores, metrics=['roc'])
            # Store results
            child_results[attacker.name] = eval_results
            
        except Exception as e:
            print(f"Error running {attacker.name} on child table: {e}")
            traceback.print_exc()
            child_results[attacker.name] = None

    
    # Save results to JSONL file
    with open(interim_output_file, 'w') as f:
        # Write parent results
        parent_line = {"parent_single_table": parent_results}
        f.write(json.dumps(parent_line) + '\n')
        
        # Write child results
        child_line = {"child_single_table": child_results}
        f.write(json.dumps(child_line) + '\n')

    print(f"Results saved to {interim_output_file}")


def load_config(config_path):
    """Placeholder - implement based on your config format (YAML/JSON)."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--train-path', help='Override train path')
    parser.add_argument('--interim-output-file', help='Override interim output file')
    parser.add_argument('--schema', help='Path to schema JSON file')
    parser.add_argument('--normalize', action='store_true', help='MinMax normalize features (fit on mem)')

    args = parser.parse_args()

    main(args.config, args.train_path, args.interim_output_file, args.schema, args.normalize)


