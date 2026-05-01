
import datetime

from syntherela.data import save_tables
from syntherela.metadata import Metadata
from redelex import datasets as ctu_datasets

dataset = ctu_datasets.AdventureWorks(
    cache_dir='data/tmp/cache/adventure_works'
)

db = dataset.get_db()

db.table_dict.pop('Password') # remove Password table as it is not needed for the experiments

tables = dict()
relationships = []
for table_name, table in db.table_dict.items():
    tables[table_name] = table.df
    for column, parent_table in table.fkey_col_to_pkey_table.items():
        relationship = {
            "parent_table_name": parent_table,
            "child_table_name": table_name,
            "parent_primary_key": db.table_dict[parent_table].pkey_col,
            "child_foreign_key": column
        }
        relationships.append(relationship)

# convert timedelta to integer
tables['Shift']['StartTime'] = tables['Shift']['StartTime'].dt.total_seconds().astype(int) // 3600
tables['Shift']['EndTime'] = tables['Shift']['EndTime'].dt.total_seconds().astype(int) // 3600

metadata = Metadata()
metadata.detect_from_dataframes(tables)


for table_name, table in db.table_dict.items():
    if table_name not in tables:
        continue

    id_columns = metadata.get_column_names(table_name, sdtype='id')
    for id_column in id_columns:
        if id_column != table.pkey_col:
            metadata.update_column(
                table_name,
                id_column,
                sdtype='numerical',
            )
    
    if metadata.get_primary_key(table_name) is not None:
        continue

    metadata.set_primary_key(
        table_name,
        table.pkey_col
    )

for relationship in relationships:
    child_table = relationship['child_table_name']
    parent_table = relationship['parent_table_name']
    primary_key = relationship['parent_primary_key']
    foreign_key = relationship['child_foreign_key']
    # ensure the primary and foreign key are of the same type
    pk_col = tables[parent_table][primary_key]
    fk_col = tables[child_table][foreign_key]
    

    assert pk_col.dtype == fk_col.dtype
    assert metadata.get_primary_key(parent_table) == primary_key

    metadata.update_column(
        child_table,
        foreign_key,
        sdtype='id',
    )

dimension_tables = []
# remove unsupported datatypes
for table_name, df in tables.items():
    table_metadata = metadata.get_table_metadata(table_name)
    all_unique_rows = True
    for column_name, column_metadata in table_metadata.columns.items():
        datatype = column_metadata['sdtype']
        if datatype not in ['id', 'numerical', 'categorical', 'datetime']:
            tables[table_name].drop(columns=[column_name], inplace=True)
            continue
        elif datatype == 'datetime':
            if tables[table_name][column_name].isnull().all():
                tables[table_name].drop(columns=[column_name], inplace=True)
            continue
        elif datatype == 'numerical' and 'id' in column_name.lower():
            # print(f"Dropping column {column_name} in table {table_name} as it is not a valid ID.")
            tables[table_name].drop(columns=[column_name], inplace=True)
            continue
        elif column_name == "rowguid":
            # check if all values are unique
            unique_values = tables[table_name][column_name].nunique()
            if unique_values == len(tables[table_name]):
                # print('Dropping rowguid column as it is not needed for the experiments.')
                tables[table_name].drop(columns=[column_name], inplace=True)
                continue
            else:
                print(f"Keeping {column_name} in table {table_name} as it has non-unique values.")
    
        unique_values = tables[table_name][column_name].nunique()
        if unique_values != len(tables[table_name]):
            all_unique_rows = False
    
    if all_unique_rows:
        # print(f"Table {table_name} is a dimension table.")
        dimension_tables.append(table_name)

# infer metadata again after dropping unsupported columns
metadata = Metadata()
metadata.detect_from_dataframes(tables)

for relationship in metadata.relationships.copy():
    metadata.remove_relationship(
        parent_table_name=relationship['parent_table_name'],
        child_table_name=relationship['child_table_name'],
    )


# add primary keys
for table_name, table in db.table_dict.items():

    id_columns = metadata.get_column_names(table_name, sdtype='id')
    for id_column in id_columns:
        if id_column != table.pkey_col:
            metadata.update_column(
                table_name,
                id_column,
                sdtype='numerical',
            )

    if metadata.get_primary_key(table_name) is not None:
        continue

    metadata.set_primary_key(
        table_name,
        table.pkey_col
    )


# add relationships
for relationship in relationships:
    child_table = relationship['child_table_name']
    parent_table = relationship['parent_table_name']
    primary_key = relationship['parent_primary_key']
    foreign_key = relationship['child_foreign_key']
    # ensure the primary and foreign key are of the same type
    pk_col = tables[relationship['parent_table_name']][relationship['parent_primary_key']]
    fk_col = tables[child_table][relationship['child_foreign_key']]
    assert pk_col.dtype == fk_col.dtype

    metadata.update_column(
        child_table,
        foreign_key,
        sdtype='id',
    )

    
    try:
        metadata.add_relationship(
            parent_table_name=parent_table,
            child_table_name=child_table,
            parent_primary_key=primary_key,
            child_foreign_key=foreign_key
        )
    except Exception as e:
        print(f"Error adding relationship {parent_table} -> {child_table}: {e}")

# add  data format information
for table_name, table in db.table_dict.items():
    table_metadata = metadata.get_table_metadata(table_name)
    all_unique_rows = True
    for column_name, column_metadata in table_metadata.columns.items():
        datatype = column_metadata['sdtype']
        if datatype == 'datetime':
            # infer datetime format
            sample_values = tables[table_name][column_name].dropna().head(10)
            if len(sample_values) > 0:
                # Try to infer format from sample values
                sample_str = str(sample_values.iloc[0])
                
                # Common datetime formats to try
                format_patterns = [
                    '%Y-%m-%d %H:%M:%S',      # 2023-01-01 12:30:45
                    '%Y-%m-%d %H:%M:%S.%f',   # 2023-01-01 12:30:45.123456
                    '%Y-%m-%d',               # 2023-01-01
                    '%m/%d/%Y',               # 01/01/2023
                    '%m/%d/%Y %H:%M:%S',      # 01/01/2023 12:30:45
                    '%d/%m/%Y',               # 01/01/2023 (European format)
                    '%Y/%m/%d',               # 2023/01/01
                    '%Y-%m-%dT%H:%M:%S',      # ISO format without Z
                    '%Y-%m-%dT%H:%M:%SZ',     # ISO format with Z
                ]
                
                detected_format = None
                for fmt in format_patterns:
                    try:
                        # Test if this format works for the sample
                        datetime.datetime.strptime(sample_str, fmt)
                        detected_format = fmt
                        break
                    except ValueError:
                        continue
                
                if detected_format:
                    # print(f"Detected datetime format for {table_name}.{column_name}: {detected_format}")
                    metadata.update_column(table_name, column_name, datetime_format=detected_format)
                else:
                    print(f"Could not detect datetime format for {table_name}.{column_name}, sample: {sample_str}")
            else:
                print(f"No sample values found for {table_name}.{column_name}, skipping datetime format detection.")
        elif datatype == 'numerical':
            dtype = tables[table_name][column_name].dtype
            if dtype == "int64":
                computer_representation = "Int64"
            elif dtype == "float64" or dtype == "Float64":
                computer_representation = "Float"
            elif dtype == 'Int32':
                computer_representation = "Int64"
                assert 'id' not in column_name.lower(), f"Column {column_name} in table {table_name} should not be an ID column."
                tables[table_name][column_name] = tables[table_name][column_name].astype('Int64')
            elif dtype == 'Int16':
                computer_representation = "Int64"
                # print(column_name, 'Int16')
                tables[table_name][column_name] = tables[table_name][column_name].astype('Int64')
            else:
                assert column_name in tables[table_name].columns
                # print(table_name, tables[table_name].head())
                raise ValueError(f"Unsupported dtype {dtype} for column {column} in table {table_name}")
            metadata.update_column(
                table_name,
                column_name,
                computer_representation=computer_representation
            )


metadata.visualize('data/tmp/adventure_works_metadata.png')
metadata.validate()
metadata.validate_data(tables)

print("Detected dimension tables:")
print(dimension_tables)

save_tables(tables, 'data/original/adventure_works', metadata=metadata, save_metadata=True)