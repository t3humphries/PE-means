from src.helpers import *
from clustering import clustering_params
from src.pe_means_main import hdpe_means
from src.DP_comp import get_noise_multiplier
import numpy as np
import pickle
import ray
import os

datasets, k_dict, r_dict, l_inf_dict = get_grid_datasets_final()


# Initialize Ray
if not ray.is_initialized():
    total_cores = os.cpu_count() or 1
    ray_cpus = max(1, total_cores - 2)
    print(f"Initializing Ray with {ray_cpus} CPUs (total cores: {total_cores})")
    ray.init(num_cpus=ray_cpus)

NUM_REPEATS = 50  # Number of times to run each configuration
# Define remote function to run a single configuration
@ray.remote
def run_ea_config(dataset_ref, k, this_config, privacy_param):
    dataset = dataset_ref
    print(f"Running EA clustering with config: {this_config}")
    loss, centers, convergence = hdpe_means(k, dataset, this_config, privacy_param, visualize=False)
    return (this_config, loss, centers, convergence)

# Process each dataset: create tasks, collect results, and save immediately
for dataset_name, dataset in datasets.items():
    dataset_object = clustering_params.Data(
        datapoints=dataset,
        radius=r_dict[dataset_name]
    )
    # Store dataset in Ray's object store
    dataset_ref = ray.put(dataset_object)
    k = k_dict[dataset_name]  
    N = dataset.shape[0]
    dim = dataset.shape[1]
    if dim < 32:
        continue
    # Create tasks for all parameter combinations for this dataset
    dataset_tasks = []

    for eps in EPSILONS:
        base_gens = int(4 * np.sqrt(dim))
        if eps>1.0:
            gens = max(int(eps * base_gens), 1)
        else:
            gens = max(base_gens, 1)

        privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=eps, delta=1/(N**1.1))
            
        base_L = int(max(N/(5*k), 4))
    
        for repeat in range(NUM_REPEATS):
            this_config = PE_PARAMS.copy()
            this_config['dataset_name'] = dataset_name
            this_config['k'] = k
            this_config['eps'] = eps
            
            
            #loop setting
            this_config['num_gen'] = gens
            this_config['L'] = base_L
            this_config['run_id'] = repeat
            task = run_ea_config.remote(
                dataset_ref, k, this_config, privacy_param
            )
            dataset_tasks.append(task)
    
    # Collect results for this dataset
    dataset_results = []
    for task in dataset_tasks:
        result = ray.get(task)
        dataset_results.append(result)
    
    FOLDER = 'result_final/ours_dim_reduced_may14'
    os.makedirs(FOLDER, exist_ok=True)
    with open(f'{FOLDER}/grid_results_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(dataset_results, f)
    print(f"Saved results for {dataset_name}")

# Shutdown Ray
ray.shutdown()