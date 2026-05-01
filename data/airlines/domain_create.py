import pandas as pd
import json

def generate_schema_json(csv_path, output_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Filter out columns that contain "_id" (case-insensitive)
    cols_to_keep = [col for col in df.columns if "_id" not in col.lower()]
    df = df[cols_to_keep]
    
    schema = {}
    
    for col in df.columns:
        # Calculate number of unique values
        unique_count = int(df[col].nunique())
        
        # Rule: discrete if less than 15 unique values, else continuous
        col_type = "discrete" if unique_count < 15 else "continuous"
            
        schema[col] = {
            "size": unique_count,
            "type": col_type
        }
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=4)
    
    print(f"Schema successfully saved to {output_path}")

# Usage
generate_schema_json('split/mem/activity.csv', 'activity_domain.json')
generate_schema_json('split/mem/loyalty_history.csv', 'loyalty_history_domain.json')