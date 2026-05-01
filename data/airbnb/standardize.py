import os
import pandas as pd
from pathlib import Path

def standardize_csv(file_path):
    """Read and scale numeric features to [0, 1] range."""
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Identify columns to scale (numeric columns that don't end with _id)
        cols_to_scale = []
        for col in df.columns:
            if not col.endswith('_id') and pd.api.types.is_numeric_dtype(df[col]):
                cols_to_scale.append(col)
        
        # Min-max scaling: (x - min) / (max - min)
        for col in cols_to_scale:
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Avoid division by zero for constant columns
            if col_max - col_min != 0:
                df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[col] = 0  # Set constant columns to 0
        
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_directory_structure(source_root, dest_root):
    """Process all CSVs in the directory structure."""
    source_path = Path(source_root)
    dest_path = Path(dest_root)
    
    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(source_path):
        # Get relative path from source root
        rel_path = Path(dirpath).relative_to(source_path)
        
        # Create corresponding directory in destination
        dest_dir = dest_path / rel_path
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all CSV files
        for filename in filenames:
            if filename.endswith('.csv'):
                source_file = Path(dirpath) / filename
                dest_file = dest_dir / filename
                
                print(f"Processing: {source_file}")
                
                # Standardize and save
                df = standardize_csv(source_file)
                if df is not None:
                    df.to_csv(dest_file, index=False)
                    print(f"  -> Saved to: {dest_file}")

if __name__ == "__main__":
    # Define base directories
    base_source = "split_1000"
    base_dest = "split_1000_STAN"
    
    # Process each subdirectory
    subdirs = ["mem", "non_mem", "ref", "synth"]
    
    print(f"Starting standardization...")
    print("-" * 50)
    
    for subdir in subdirs:
        source_dir = os.path.join(base_source, subdir)
        dest_dir = os.path.join(base_dest, subdir)
        
        if os.path.exists(source_dir):
            print(f"\nProcessing: {subdir}/")
            process_directory_structure(source_dir, dest_dir)
        else:
            print(f"\nSkipping {subdir}/ (not found)")
    
    print("-" * 50)
    print("Standardization complete!")