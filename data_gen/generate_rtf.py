import os
from pathlib import Path
import pandas as pd
import torch
from realtabformer import REaLTabFormer

# -------------------------------
# 0. GPU Setup
# -------------------------------
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# -------------------------------
# 1. Dataset Configuration
# -------------------------------
datasets = {
    "california": {
        "metadata": {
            "relation_order": [[None, "household"], ["household", "individual"]],
            "tables": {
                "household": {"children": ["individual"], "parents": []},
                "individual": {"children": [], "parents": ["household"]}
            }
        },
        "data_dir": Path("./data/california/mem")
    },
    "airbnb": {
        "metadata": {
            "relation_order": [[None, "user"], ["user", "session"]],
            "tables": {
                "user": {"children": ["session"], "parents": []},
                "session": {"children": [], "parents": ["user"]}
            }
        },
        "data_dir": Path("./data/airbnb/mem")
    },
    "airlines": {
        "metadata": {
            "relation_order": [[None, "loyalty_history"], ["loyalty_history", "activity"]],
            "tables": {
                "loyalty_history": {"children": ["activity"], "parents": []},
                "activity": {"children": [], "parents": ["loyalty_history"]}
            }
        },
        "data_dir": Path("./data/airlines/mem")
    }
}

# -------------------------------
# 1b. Select which datasets to run
# -------------------------------
SELECTED_DATASETS = ["airbnb","california","airlines"]
# Examples: ["california"], ["airlines"], etc.

# -------------------------------
# 1c. Random Seeds and Paths
# -------------------------------
random_states = [42, 43, 44]
BASE_MODEL_DIR = Path("./models")
BASE_OUTPUT_DIR = Path("./synth_data/rtf_synth")

# -------------------------------
# 2. Main Execution Loop
# -------------------------------
for state in random_states:
    print(f"\n{'='*30}")
    print(f" RUNNING SEED: {state}")
    print(f"{'='*30}")

    for ds_name in SELECTED_DATASETS:
        print(f"\n>>> Dataset: {ds_name}")
        config = datasets[ds_name]
        metadata = config["metadata"]
        data_dir = config["data_dir"]

        # Directories include the seed for organization
        output_dir = BASE_OUTPUT_DIR / ds_name / f"seed_{state}"
        model_dir = BASE_MODEL_DIR / ds_name / f"seed_{state}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Load original data
        dfs = {table: pd.read_csv(data_dir / f"{table}.csv") for table in metadata["tables"]}
        synthetic_data = {}

        # -------------------------------
        # 3. Training & Generation Loop
        # -------------------------------
        for parent_name, child_name in metadata["relation_order"]:
            print(f"--- Processing: {child_name} ---")
            current_df = dfs[child_name]
            save_path = model_dir / child_name

            if parent_name is None:
                # ROOT TABLE
                join_on = f"{child_name}_id"
                model = REaLTabFormer(model_type="tabular", random_state=state,checkpoints_dir=f"{ds_name}_{state}_checkpoints")
                model.fit(current_df.drop(join_on, axis=1))
                model.save(save_path)

                print(f"Sampling {child_name}...")
                synth_df = model.sample(n_samples=len(current_df), gen_batch=64)
                synth_df.index.name = join_on
                synthetic_data[child_name] = synth_df.reset_index()

            else:
                # CHILD TABLE
                join_on = f"{parent_name}_id"

                # Fetch parent model path
                parent_model_root = model_dir / parent_name
                parent_model_p = sorted(
                    [p for p in parent_model_root.glob("id*") if p.is_dir()],
                    key=os.path.getmtime
                )[-1]

                model = REaLTabFormer(
                    model_type="relational",
                    parent_realtabformer_path=parent_model_p,
                    random_state=state,
                    epochs=500,
                    batch_size=4,checkpoints_dir=f"{ds_name}_{state}_checkpoints")
                
                model.fit(df=current_df, in_df=dfs[parent_name], join_on=join_on)
                model.save(save_path)

                print(f"Sampling {child_name}...")
                synth_parent = synthetic_data[parent_name]
                synth_child = model.sample(
                    input_unique_ids=synth_parent[join_on],
                    input_df=synth_parent,
                    gen_batch=32
                )
                synth_child.index.name = join_on
                synthetic_data[child_name] = synth_child.reset_index()

        # -------------------------------
        # 4. Save Final CSVs
        # -------------------------------
        for name, df in synthetic_data.items():
            final_path = output_dir / f"{name}.csv"
            df.to_csv(final_path, index=False)
            print(f"Saved to: {final_path}")

print("\nProcessing complete for all seeds and selected datasets.")
