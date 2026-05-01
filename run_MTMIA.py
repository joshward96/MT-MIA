import torch
import json
import os
from torch_geometric.loader import DataLoader
from src.synth_mia.attackers import * 
import torch.nn.functional as F
from src.model import generate_ssl_embeddings, compute_reconstruction_losses, sslModel
from src.utils import run_dcr, run_recon_loss_attack, set_seed, load_config
from src.data_loader import load_custom_split_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def materialize(dataset):
    return [dataset[i] for i in range(len(dataset))]

def main(config_path,  
         train_path=None, interim_overwrite=None):
    
    config = load_config(config_path)

    # --- APPLY OVERWRITES HERE ---
    if train_path:
        config['dataset_paths']['train'] = train_path
    
    if interim_overwrite:
        config['interim_output_file'] = interim_overwrite    
   
    seed = config.get('seed', 42)
    set_seed(seed)
    worker_seed = torch.initial_seed() % 2**32

    # Extract config
    LEARNING_RATE = float(config['LEARNING_RATE'])
    EPOCHS = config['EPOCHS']
    BATCH_SIZE = config['BATCH_SIZE']
    PARENT_LOSS_WEIGHT = config.get('PARENT_LOSS_WEIGHT', 0.5)
    CHILD_LOSS_WEIGHT = config.get('CHILD_LOSS_WEIGHT', 0.5)
    HIDDEN_DIM = config.get('HIDDEN_DIM', 1024)
    entity_unit = config['entity_unit']

    split_dirs = config['dataset_paths']
    interim_output_file = config.get('interim_output_file', 'black_box_ssl_mia_results.csv')
    print(config)
    print(split_dirs)
    print("Loading datasets...")
    output_dir = os.path.dirname(interim_output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory ensured: {output_dir}")

    train_loader, embed_loaders, encoders = load_custom_split_loaders(
        split_dirs=split_dirs,
        schema_path=config['schema_path'],
        entity_unit=entity_unit,
        batch_size=BATCH_SIZE,
        seed=42
    )
    
    print("Materializing datasets...")
    train_graphs = materialize(train_loader.dataset)
    print(f"Materialized {len(train_graphs)} train graphs")
    
    ref_graphs = materialize(embed_loaders["ref"].dataset)
    print(f"Materialized {len(ref_graphs)} ref graphs")
    
    mem_graphs = materialize(embed_loaders["mem"].dataset)
    print(f"Materialized {len(mem_graphs)} mem graphs")
    
    non_mem_graphs = materialize(embed_loaders["non_mem"].dataset)
    print(f"Materialized {len(non_mem_graphs)} non_mem graphs")


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np
        import random
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)


    train_loader = DataLoader(
        train_graphs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=seed_worker,  # Add this
        generator=g  # Add this
    )

    print(f"Train batches: {len(train_loader)}")
    for name, loader in embed_loaders.items():
        print(f"{name} batches: {len(loader)}")

    # Initialize model
    sample_batch = train_loader.dataset[0]
    node_types = sample_batch.node_types
    edge_types = list(sample_batch.edge_index_dict.keys())
    metadata = (node_types, edge_types)
    
    model = sslModel(
        hidden_channels=HIDDEN_DIM,
        metadata=metadata,
        target_node_type=entity_unit,
        sample_batch=sample_batch
    ).to(device)

    # Initialize lazy modules with a dummy forward pass
    print("Initializing model with dummy batch...")
    # Use multiple samples to satisfy BatchNorm requirements
    dummy_samples = [train_loader.dataset[i] for i in range(min(4, len(train_loader.dataset)))]
    dummy_loader = DataLoader(dummy_samples, batch_size=len(dummy_samples))
    dummy_batch = next(iter(dummy_loader)).to(device)
    model.eval() 
    with torch.no_grad():
        _ = model(dummy_batch)
    model.train() 
    print("Model initialized!")


        # Test script
    model.train()
    out, z_t, z_c,_ = model(dummy_batch)
    loss = out.sum() # Simple dummy loss
    loss.backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    for epoch in range(1, EPOCHS + 1):
        model.train()

        epoch_recon_target_loss = 0
        epoch_recon_context_loss = 0    
        epoch_recon_loss = 0
        epoch_total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # Get embeddings for the clean graph
            z_final, _, _, x_dict = model(batch)

            # A. Target Reconstruction (Self)
            z_target_recon = model.target_decoder(z_final)
            true_target_feat = batch[entity_unit].x
            loss_recon_target = F.mse_loss(z_target_recon, true_target_feat)

            # B. Context Reconstruction (Neighborhood)
            z_context_recon = model.context_decoder(z_final)
            target_edges = [et for et in batch.edge_types if et[0] == entity_unit]

            if target_edges:
                edge_type = target_edges[0]
                dst_type = edge_type[2]
                edge_index = batch[edge_type].edge_index
                
                # Calculate the actual sum of item features for each target node
                item_features = batch[dst_type].x
                true_item_sum = torch.zeros(z_final.size(0), item_features.size(1), device=device)
                true_item_sum.index_add_(0, edge_index[0], item_features[edge_index[1]])
                
                loss_recon_context = F.mse_loss(z_context_recon, true_item_sum)
            else:
                loss_recon_context = 0
                
            loss = (PARENT_LOSS_WEIGHT)*loss_recon_target + (CHILD_LOSS_WEIGHT)*loss_recon_context     
            loss.backward()
            optimizer.step()
            


            epoch_recon_loss += loss.item()
            epoch_recon_target_loss += loss_recon_target.item()
            epoch_recon_context_loss += loss_recon_context.item()
            epoch_total_loss += loss.item()

        # Average metrics for the epoch
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_recon_target = epoch_recon_target_loss / len(train_loader)
        avg_recon_context = epoch_recon_context_loss / len(train_loader)
        avg_total = epoch_total_loss / len(train_loader)

        print(f"Epoch {epoch:03d} | Total: {avg_total:.4f}  | Recon: {avg_recon:.4f} (Target: {avg_recon_target:.4f}, Context: {avg_recon_context:.4f})")
        if epoch % 5 == 0:
            print(f"\nEvaluating at epoch {epoch}...")
            model.eval()
            with torch.no_grad():
                # Generate embeddings
                z_mem, z_target_mem, z_context_mem = generate_ssl_embeddings(model, mem_graphs, BATCH_SIZE)
                z_non_mem, z_target_non_mem, z_context_non_mem = generate_ssl_embeddings(model, non_mem_graphs, BATCH_SIZE)
                z_synth, z_target_synth, z_context_synth = generate_ssl_embeddings(model, train_graphs, BATCH_SIZE)
                z_ref, z_target_ref, z_context_ref = generate_ssl_embeddings(model, ref_graphs, BATCH_SIZE)
            print('Final Embeddings MIA Evaluation:')
            full_eval_results,full_scores, _ = run_dcr(z_mem, z_non_mem, z_synth,z_ref)

            print('Parent Embeddings MIA Evaluation:')
            target_eval_results, target_scores, _ = run_dcr(z_target_mem, z_target_non_mem, z_target_synth,z_target_ref)

            print('Context Embeddings MIA Evaluation:')
            context_eval_results, context_scores, _ = run_dcr(z_context_mem, z_context_non_mem, z_context_synth,z_context_ref)


            # Structure the current epoch's data
            current_data = {
                "epoch": epoch,
                "full": {"results": full_eval_results, "scores": full_scores.tolist()},
                "target": {"results": target_eval_results, "scores": target_scores.tolist()},
                "context": {"results": context_eval_results, "scores": context_scores.tolist()},

            }

            # Append to the single file
            with open(interim_output_file, 'a') as f:
                f.write(json.dumps(current_data) + '\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SSL MIA experiments")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML/JSON config")
    parser.add_argument('--train-path', type=str, help="Overwrite dataset_paths: train")
    parser.add_argument('--interim-output-file', type=str, help="Overwrite interim_output_file")
    args = parser.parse_args()

    main(
        args.config,
        train_path=args.train_path,                
        interim_overwrite=args.interim_output_file
    )