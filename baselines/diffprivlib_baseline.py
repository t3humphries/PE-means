# NOTE: use FastLloyd/env.yml
from src.helpers import *
import os
import pickle
from diffprivlib.models import KMeans

def diffprivlib_baseline(data, k, epsilon, l_inf):
    # diffprivlib bounds need to be (min, max) for each feature/dimension
    dim = data.shape[1]
    bounds = ([-l_inf] * dim, [l_inf] * dim)
    
    # Initialize and fit
    dp_kmeans = KMeans(n_clusters=k, epsilon=epsilon, bounds=bounds)
    dp_kmeans.fit(data)
    
    return dp_kmeans.cluster_centers_

# --- Configuration ---
datasets, k_dict, r_dict, l_inf_dict = get_grid_datasets_final()
FOLDER = 'result_final/diffprivlib_baseline_50'
NUM_RUNS = 50
os.makedirs(FOLDER, exist_ok=True)

for dataset_name, dataset in datasets.items():
    k = k_dict[dataset_name]
    L_inf = l_inf_dict[dataset_name]
    
    centers = {}
    loss = {}

    for eps in EPSILONS:
        cents = []
        losses = []
        
        for run in range(NUM_RUNS):
            # Generate centers using the diffprivlib wrapper
            z_centers = diffprivlib_baseline(dataset, k, eps, L_inf)
            
            # Calculate loss and store
            current_loss = simple_loss(dataset, z_centers)
            losses.append(current_loss)
            cents.append(z_centers)
            
        centers[str(eps)] = cents
        loss[str(eps)] = losses

    # Save results for this dataset
    results = {'centers': centers, 'loss': loss}
    with open(f'{FOLDER}/diffpriv_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Saved diffprivlib baseline for {dataset_name}")