import pandas as pd
import json
import argparse
import sys

def generate_schema_json(csv_path, output_path):
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)
        
        # Filter out columns that contain "_id" (case-insensitive)
        cols_to_keep = [col for col in df.columns if "_id" not in col.lower()]
        df = df[cols_to_keep]
        
        schema = {}
        for col in df.columns:
            unique_count = int(df[col].nunique())
            # Rule: discrete if less than 15 unique values, else continuous
            col_type = "discrete" if unique_count < 15 else "continuous"
                
            schema[col] = {
                "size": unique_count,
                "type": col_type
            }
        
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=4)
        
        print(f"Success: {csv_path} -> {output_path}")

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON schema from CSV files based on unique value counts."
    )

    # Allow passing multiple pairs of (input, output)
    parser.add_argument(
        '-i', '--input', 
        nargs=2, 
        action='append', 
        metavar=('CSV_PATH', 'JSON_PATH'),
        help="Input CSV path and desired Output JSON path (can be used multiple times)"
    )

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    for csv_in, json_out in args.input:
        generate_schema_json(csv_in, json_out)

if __name__ == "__main__":
    main()
