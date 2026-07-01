from src.helpers import *
from clustering import clustering_params
from src.pe_means_main import pe_means
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

NUM_REPEATS = 10  # Number of times to run each configuration
# Define remote function to run a single configuration
@ray.remote
def run_ea_config(dataset_ref, k, this_config, privacy_param):
    dataset = dataset_ref
    print(f"Running EA clustering with config: {this_config}")
    loss, centers, convergence = pe_means(k, dataset, this_config, privacy_param, visualize=False)
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
    
    privacy_param = clustering_params.DifferentialPrivacyParam(
    epsilon=1.0, delta=1/(N**1.1))
    l_values = [int(max(N/(x*k), 4)) for x in [2, 5, 10, 15, 20, 50, 100] if int(max(N/(x), 4)) < 20000]
    l_tags = [2, 5, 10, 15, 20, 50, 100]
    gens = [max(int(x * np.sqrt(dim)), 1) for x in [0.5, 1, 2, 3, 4, 5, 7, 10]]
    gen_tags = [0.5, 1, 2, 3, 4, 5, 7, 10]


    # Create tasks for all parameter combinations for this dataset
    dataset_tasks = []

    for eps in [1.0]:
        for gen, gen_tag in zip(gens, gen_tags):
            for L, l_tag in zip(l_values, l_tags):
                for repeat in range(NUM_REPEATS):
                    this_config = PE_PARAMS.copy()
                    this_config['dataset_name'] = dataset_name
                    this_config['k'] = k
                    this_config['eps'] = eps
                    
                    #loop setting
                    this_config['num_gen'] = gen
                    this_config['gen_tag'] = gen_tag
                    this_config['L_tag'] = l_tag
                    this_config['L'] = L
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
    
    FOLDER = 'result_final/grid_L_gen_eps1_May12'
    os.makedirs(FOLDER, exist_ok=True)
    with open(f'{FOLDER}/grid_results_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(dataset_results, f)
    print(f"Saved results for {dataset_name}")

# Shutdown Ray
ray.shutdown()