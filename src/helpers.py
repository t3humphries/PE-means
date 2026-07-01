
import numpy as np

#Defaults 
EPSILONS = [0.25, 0.5, 1.0, 2.0, 4.0]
PE_PARAMS = {
    'init_mode': 'sphere_packing', # 'packing' or 'random'
    'select_mode': 'k_means', # 'k_means' or 'top_k
    'var_scale': 0.01, 
    'levy_beta' : 1.75,
    'L_reduce_threshold' : 1.0,
    'L_reduce_factor' : 0.5,
    'L': 20, 
    'num_gen': 10
}
def load_txt(path: str):
    """
    From https://github.com/D-Diaa/FastLloyd/blob/v0.0.1/data_io/data_handler.py
    Load numerical data from a text file, skipping lines containing 'x'.

    Args:
        path (str): Path to the text file containing numerical data

    Returns:
        np.ndarray: Array containing the loaded numerical values

    Example:
        >>> values = load_txt("data.txt")
        >>> print(values.shape)
        (100, 5)  # If file contains 100 rows of 5 numbers each
    """
    values_list = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "x" in line:
                continue
            values_list.append([float(x) for x in line.split()])
    values_arr = np.array(values_list)
    return values_arr

def dataset_preprocess(datapoints):
    # 1. Center
    X_centered = datapoints - np.mean(datapoints, axis=0)

    # 2. Scale to Unit Sphere (for simplicity)
    global_max_norm = np.max(np.linalg.norm(X_centered, axis=1))
    X_normed = X_centered / global_max_norm

    radius = np.max(np.linalg.norm(X_normed, axis=1))
    #For Su and icml17 side length 
    L_inf = np.max(np.abs(X_normed))
    
    return X_normed, radius, L_inf
    
    
def get_grid_datasets_final():
    dataset_paths = {
        'birch2': "datasets/real/birch2.txt",
        'iris' : "datasets/real/iris.txt",
        'adult' : "datasets/real/adult.txt",
        'mnist' : "datasets/real/mnist_lenet_full.txt",
        'letter' : "datasets/real/letter_recognition_full.txt",
        'gas' : "datasets/real/gas_turbine_full.txt"}

    k_dict = {
        'birch2': 100,
        'adult' : 3,
        'iris' : 3,
        'mnist' : 10,
        'letter' : 26,
        'gas' : 6
    }
    for d in [4, 16, 64, 128]:
        dataset_paths[f'g2_{d}'] = f"datasets/g2/g2-{d}-50.txt"
        k_dict[f'g2_{d}'] = 2
    for k in [4, 16, 64]:
        for d in [4, 16, 64, 128]:
            dataset_paths[f'scale_{k}_{d}'] = f"datasets/scale/SynthNew_{k}_{d}_1.txt"
            k_dict[f'scale_{k}_{d}'] = k
            
    for dimension in [4, 16, 64, 128]:
        for k in [4, 16, 64]:
            dataset_paths[f"sklearn_synthetic_{k}_{dimension}"] = f"datasets/sklearn/sklearn_{k}_{dimension}.txt"
            k_dict[f"sklearn_synthetic_{k}_{dimension}"] = k
            
    datasets= {}
    radius_dict = {}
    L_inf_dict = {}
    for dataset_name, path in dataset_paths.items():
        datapoints = load_txt(path)
        dataset, radius, L_inf = dataset_preprocess(datapoints)
        datasets[dataset_name] = dataset
        radius_dict[dataset_name] = radius
        L_inf_dict[dataset_name] = L_inf
    return datasets, k_dict, radius_dict, L_inf_dict


def simple_loss(datapoints, centers):
    '''Extracted the part of clustering_algorithm.ClusteringResult that computes the loss to avoid a dependency other than numpy'''
    def closest_center(datapoint: np.ndarray):
      """Returns closest center to data point and the squared distance from it.

      Args:
        datapoint: 1D np.ndarray containing a single datapoint
      """
      squared_distances = np.sum((centers - datapoint)**2, axis=1)
      min_index = np.argmin(squared_distances)
      return (min_index, squared_distances[min_index])

    
    result = [closest_center(datapoint) for datapoint in datapoints]
    return sum([res[1] for res in result])


def cluster_label_accuracy(datapoints, labels, centers):
    # Assign each point to its nearest center
    sq_dists = np.sum((datapoints[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (n, k)
    assignments = np.argmin(sq_dists, axis=1)  # (n,)

    # For each cluster, take the majority ground-truth label and assign it to every point in that cluster
    predicted = np.empty_like(labels)
    for cluster_idx in range(len(centers)):
        mask = assignments == cluster_idx
        if not np.any(mask):
            continue  # empty cluster — no points to label
        majority = np.bincount(labels[mask]).argmax()
        predicted[mask] = majority

    return np.mean(predicted == labels)


def get_label_set():
    dataset_paths = {
        'mnist'  : "datasets/real/mnist_lenet_full.txt",
        'letter' : "datasets/real/letter_recognition_full.txt",
        'iris'   : "datasets/real/iris.txt",
        'birch2' : "datasets/real/birch2.txt",
    }
    label_paths = {
        'mnist'  : "datasets/real/mnist_lenet_labels.txt",
        'letter' : "datasets/real/letter_recognition_labels.txt",
        'iris'   : "datasets/real/iris.data.txt",
        'birch2' : "datasets/real/b2-gt.txt",
    }

    k_dict = {
        'mnist'  : 10,
        'letter' : 26,
        'iris'   : 3,
        'birch2' : 100,
    }
    for d in [16, 128]:
        dataset_paths[f'g2_{d}'] = f"datasets/g2/g2-{d}-50.txt"
        label_paths[f'g2_{d}'] = f"datasets/g2/labels/g2-{d}-50-gt.txt"
        k_dict[f'g2_{d}'] = 2

    datasets = {}
    labels_dict = {}
    radius_dict = {}
    L_inf_dict = {}
    for dataset_name, path in dataset_paths.items():
        datapoints = load_txt(path)
        dataset, radius, L_inf = dataset_preprocess(datapoints)
        datasets[dataset_name] = dataset
        radius_dict[dataset_name] = radius
        L_inf_dict[dataset_name] = L_inf

        if dataset_name in ('mnist', 'letter'):
            labels_dict[dataset_name] = load_txt(label_paths[dataset_name]).astype(int).ravel()
        elif dataset_name == 'iris':
            # iris.data.txt is CSV; last column is the species string
            species_order = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            s2i = {s: i for i, s in enumerate(species_order)}
            rows = [ln.strip() for ln in open(label_paths['iris']) if ln.strip()]
            labels_dict['iris'] = np.array([s2i[r.split(',')[-1]] for r in rows], dtype=int)
        else:
            # g2 / birch2: gt file contains cluster centers; assign each point to nearest
            centers = load_txt(label_paths[dataset_name])  # shape (k, d)
            sq_dists = np.array([np.sum((dataset - c) ** 2, axis=1) for c in centers])  # (k, n)
            labels_dict[dataset_name] = np.argmin(sq_dists, axis=0).astype(int)

    return datasets, k_dict, radius_dict, L_inf_dict, labels_dict