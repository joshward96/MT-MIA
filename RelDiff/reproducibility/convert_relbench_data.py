import os
from syntherela.data import save_tables
from syntherela.metadata import Metadata
from relbench.datasets import get_dataset

os.makedirs('data/original/rel-hm', exist_ok=True)

dataset = get_dataset("rel-hm")

db = dataset.get_db(upto_test_timestamp=True)


tables = dict()
for table_name, table in db.table_dict.items():
    # Convert each table to a DataFrame and save it as a CSV file
    df = table.df
    print(table_name, df.shape)
    if table_name == "customer":
        df.drop(columns=["postal_code"], inplace=True)
    print(df.head())
    tables[table_name] = df
    df.to_csv(f'data/original/rel-hm/{table_name}.csv', index=False)


metadata = Metadata()
metadata.detect_from_dataframes(tables)
metadata.update_column("transactions", "t_dat", datetime_format="%Y-%m-%d")

for table_name, table in tables.items():
    numerical_columns = metadata.get_column_names(table_name, sdtype="numerical")
    for column in numerical_columns:
        # get dtype of the column
        dtype = table[column].dtype
        if dtype == "int64":
            computer_representation = "Int64"
        elif dtype == "float64":
            computer_representation = "Float"
        else:
            raise ValueError(f"Unsupported dtype {dtype} for column {column} in table {table_name}")
        metadata.update_column(table_name, column, computer_representation=computer_representation)

metadata.validate()
metadata.validate_data(tables)
save_tables(tables, "data/original/rel-hm", metadata=metadata, save_metadata=True)