from src.synth_mia.attackers import *
from src.synth_mia.base import BaseAttacker
import pandas as pd
import numpy as np
import torch
import os
import yaml
import json
def run_dcr(mem,non_mem,synth,ref):
    attackers = [
    dcr(),
]
    results = {}

    for attacker in attackers:
        # Run the attack
        scores, true_labels = attacker.attack(mem, 
                                            non_mem, 
                                            synth,
                                            ref
                                            )
        # Evaluate the attack
        eval_results = attacker.eval(true_labels, scores, metrics=['roc'])
        
        # Store results
        results[attacker.name] = eval_results
    print(pd.DataFrame(results).T)

    return eval_results, scores, true_labels

def run_dcr_diff(mem,non_mem,synth,ref):
    attackers = [
    dcr_diff(),
]
    
    results = {}

    for attacker in attackers:
        # Run the attack
        scores, true_labels = attacker.attack(mem, 
                                            non_mem, 
                                            synth,
                                            ref)
        # Evaluate the attack
        eval_results = attacker.eval(true_labels, scores, metrics=['roc'])
        
        # Store results
        results[attacker.name] = eval_results
    print(pd.DataFrame(results).T)
    df = pd.DataFrame(scores, columns=["scores"])
    df["true_label"] = true_labels
    return eval_results

def run_recon_loss_attack(mem_losses, non_mem_losses):
    """
    Run a reconstruction-loss MIA. Lower loss → more likely member, so scores are
    negated losses (higher score = more likely member, consistent with DCR convention).
    Does not require synthetic or reference data.
    """
    scores = np.concatenate([-mem_losses, -non_mem_losses])
    true_labels = np.concatenate([np.ones(len(mem_losses)), np.zeros(len(non_mem_losses))])
    attacker = BaseAttacker()
    eval_results = attacker.eval(true_labels, scores, metrics=['roc'])
    print(pd.DataFrame({'ReconLoss': eval_results}).T)
    return eval_results, scores, true_labels


def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    import os

    # 1. Standard seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 2. Force Deterministic CUDA algorithms
    os.environ["PYTHONHASHSEED"] = str(seed)
    # This environment variable is required for many CUDA operations to be deterministic
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 3. Enable PyTorch deterministic mode
    # Warning: This may make some operations slower or throw errors if no deterministic version exists
    torch.use_deterministic_algorithms(True, warn_only=True) 

    print(f"Deterministic seed set to: {seed}")

def save_checkpoint(model, path="mia_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

def load_checkpoint(model, path="mia_model.pth"):
    if os.path.exists(path):
        # map_location handles CPU/GPU transitions automatically
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Successfully loaded weights from {path}")
        return True
    else:
        print("No checkpoint found. Starting training from scratch.")
        return False
    
def load_config(config_path):
    """Load configuration from YAML or JSON"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.endswith((".yaml", ".yml")):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be YAML or JSON")