# NOTE: use envs/icml17.yml
import os
import pickle
import multiprocessing
from oct2py import Oct2Py
from src.helpers import *

# --- GLOBALS FOR WORKER PROCESSES ---
# This variable will hold the persistent Octave instance for each worker
oc_instance = None

def init_worker():
    """
    This function runs once per worker process when the Pool is created.
    It initializes a single, long-running Octave instance.
    """
    global oc_instance
    # Initialize Octave and suppress terminal output to save PTY resources
    oc_instance = Oct2Py()
    
    # Pre-load necessary packages and paths
    oc_instance.eval('pkg load statistics')
    oc_instance.addpath('icml17_matlab')
    oc_instance.eval('global range; global side_length;')

def run_single_iteration(task_args):
    """
    Worker function to run a single (epsilon, run) pair using the 
    already-initialized oc_instance.
    """
    (dataset, dataset_name, N, dim, k, eps, delta, radius, side_length_val, run_id) = task_args
    
    global oc_instance
    
    try:
        # Set the global variables in Octave for this specific task
        oc_instance.push('range', radius)
        oc_instance.push('side_length', side_length_val)
        
        # Execute the clustering algorithm
        # nout=5 matches the 5 return values from the Matlab/Octave function
        z_centers, clusters, u_centers, c_candidates, L_loss = oc_instance.clustering(
            dataset.T, N, dim, k, eps, delta, nout=5
        )
        
        # Calculate loss and prepare return data
        loss_val = simple_loss(dataset, z_centers.T)
        center_val = z_centers.T
        
        return eps, center_val, loss_val

    except Exception as e:
        print(f"Error in Worker Process (Dataset: {dataset_name}, EPS: {eps}, Run: {run_id}): {e}")
        return eps, None, None

if __name__ == '__main__':
    datasets, k_dict, r_dict, l_inf_dict = get_grid_datasets_final()

    FOLDER = 'result_final/icml_baseline_50_jun24'
    NUM_RUNS = 50
    os.makedirs(FOLDER, exist_ok=True)

    num_cpus = min(multiprocessing.cpu_count(), 150)
    print(f"Starting parallel processing with {num_cpus} workers...")

    for dataset_name, dataset in datasets.items():
        k = k_dict[dataset_name]  
        N = dataset.shape[0]
        dim = dataset.shape[1]
        delta = 1 / (N**1.1)
        radius = r_dict[dataset_name]
        L_inf = l_inf_dict[dataset_name]
        
        side_length_val = 2.0 * L_inf + 1e-6

        # Prepare task list for all epsilons and runs
        tasks = []
        for eps in EPSILONS:
            for run_id in range(NUM_RUNS):
                tasks.append((dataset, dataset_name, N, dim, k, eps, delta, radius, side_length_val, run_id))

        print(f"Processing dataset: {dataset_name} ({len(tasks)} total tasks)")

        with multiprocessing.Pool(processes=num_cpus, initializer=init_worker) as pool:
            results_list = pool.map(run_single_iteration, tasks)

        centers = {str(eps): [] for eps in EPSILONS}
        loss = {str(eps): [] for eps in EPSILONS}

        for eps, center_val, loss_val in results_list:
            if center_val is not None:
                centers[str(eps)].append(center_val)
                loss[str(eps)].append(loss_val)

        # Save results
        final_output = {'centers': centers, 'loss': loss}
        output_path = os.path.join(FOLDER, f'baseline_{dataset_name}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(final_output, f)
            
        print(f"Successfully saved baseline for {dataset_name} to {output_path}")