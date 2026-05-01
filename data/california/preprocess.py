import pandas as pd
from pathlib import Path

import os

# 1. Get the absolute path to the directory containing THIS script
base_dir = os.path.dirname(os.path.abspath(__file__))

household_path = os.path.join(base_dir, 'original', 'household.csv')
individual_path = os.path.join(base_dir, 'original', 'individual.csv')
# Read the CSV files
household = pd.read_csv(household_path)
individual = pd.read_csv(individual_path)

Path(base_dir, 'cleaned').mkdir(parents=True, exist_ok=True)
household_path = os.path.join(base_dir, 'cleaned', 'household.csv')
individual_path = os.path.join(base_dir, 'cleaned', 'individual.csv')
# Save the modified dataframes
household.to_csv(household_path, index=False)
individual.to_csv(individual_path, index=False)