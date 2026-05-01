import pandas as pd
import os

def add_id_to_csvs():
    # Get all files in the current directory
    files = [f for f in os.listdir('.') if f.startswith('t2') and f.endswith('.csv')]
    
    if not files:
        print("No files starting with 't2' were found.")
        return

    for file in files:
        try:
            # Load the CSV
            df = pd.read_csv(file)
            
            # Insert 't2_id' as the first column
            # range(1, len(df) + 1) creates a 1-based index
            df.insert(0, 't2_id', range(1, len(df) + 1))
            
            # Save back to the same filename
            df.to_csv(file, index=False)
            print(f"Successfully updated: {file}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    add_id_to_csvs()