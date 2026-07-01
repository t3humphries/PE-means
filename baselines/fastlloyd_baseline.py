# NOTE use FastLloyd/env.yml
import os
import sys
import pickle
import numpy as np
from src.helpers import *

# --- FastLloyd Setup ---
project_root = os.getcwd() 
submodule_path = os.path.join(project_root, 'FastLloyd')
if submodule_path not in sys.path:
    sys.path.append(submodule_path)
    print(f"Successfully added: {submodule_path}")

from FastLloyd.configs import Params
from FastLloyd.utils.protocols import local_proto

def fastlloyd(dataset, k, eps, linf, seed):
    config = {
        "k": k,
        "num_clients": 1,
        "dp": "gaussiananalytic", 
        "eps": eps,
        "method": "diagonal_then_frac",
        "post": "fold",           
        "delay": 0,
        "alpha": 0.8,
        "seed": seed,
        "fixed": False            
    }

    params = Params(
        num_clients=config["num_clients"],
        k=config["k"],
        dim=dataset.shape[1],
        data_size=dataset.shape[0],
        dp=config["dp"],
        eps=config["eps"],
        method=config["method"],
        post=config["post"],
        delay=config["delay"],
        linf_bound=linf
    )
    params.alpha = config["alpha"]
    params.seed = config["seed"]
    params.fixed = config["fixed"]

    # Wrap data for the protocol
    client_data_list = [dataset]

    # Run the protocol
    centroids, unassigned = local_proto(client_data_list, params, method="unmasked")
    return centroids

# --- Configuration ---
datasets, k_dict, r_dict, l_inf_dict = get_grid_datasets_final()
FOLDER = 'result_final/fastlloyd_baseline_50' # Updated folder name
NUM_RUNS = 50
os.makedirs(FOLDER, exist_ok=True)

# Assuming EPSILONS is defined in helpers or globally
for dataset_name, dataset in datasets.items():
    k = k_dict[dataset_name]
    L_inf = l_inf_dict[dataset_name]
    
    centers = {}
    loss = {}

    for eps in EPSILONS:
        cents = []
        losses = []
        
        for run in range(NUM_RUNS):
            z_centers = fastlloyd(dataset, k, eps, L_inf, seed=run)
            
            # Calculate loss and store
            current_loss = simple_loss(dataset, z_centers)
            losses.append(current_loss)
            cents.append(z_centers)
            
        centers[str(eps)] = cents
        loss[str(eps)] = losses

    # Save results for this dataset
    results = {'centers': centers, 'loss': loss}
    output_path = f'{FOLDER}/fastlloyd_{dataset_name}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Saved FastLloyd results to {output_path}")