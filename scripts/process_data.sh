python data/airbnb/preprocess.py
python data/airlines/preprocess.py
python data/california/preprocess.py

python data/create_data_splits.py --data_dir data/airlines/cleaned/ --out_dir data/airlines/split/ --schema_path data/airlines/dataset_meta.json --entity_unit loyalty_history
python data/create_data_splits.py --data_dir data/airbnb/cleaned/ --out_dir data/airbnb/split/ --schema_path data/airbnb/dataset_meta.json --entity_unit user
python data/create_data_splits.py --data_dir data/california/cleaned/ --out_dir data/california/split/ --schema_path data/california/dataset_meta.json --entity_unit household

python data/domain_create.py -i data/airlines/split/mem/activity.csv data/airlines/split/mem/activity_domain.json
python data/domain_create.py -i data/airlines/split/mem/loyalty_history.csv data/airlines/split/mem/loyalty_history_domain.json

python data/domain_create.py -i data/california/split/mem/household.csv data/california/split/mem/household_domain.json
python data/domain_create.py -i data/california/split/mem/individual.csv data/california/split/mem/individual_domain.json

python data/domain_create.py -i data/airbnb/split/mem/user.csv data/airbnb/split/mem/user_domain.json
python data/domain_create.py -i data/airbnb/split/mem/session.csv data/airbnb/split/mem/session_domain.json

python data_gen/move_data_to_clava.py

python data_gen/prepare_reldiff_data.py