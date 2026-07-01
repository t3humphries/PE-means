from src.helpers import *
from sklearn.cluster import KMeans
from clustering import clustering_algorithm
from clustering import clustering_params
import os
import pickle
datasets, k_dict, r_dict, l_inf_dict = get_grid_datasets_final()

FOLDER = 'result_final/google_baseline_50'
NUM_RUNS = 50
os.makedirs(FOLDER, exist_ok=True)
for dataset_name, dataset in datasets.items():
    k = k_dict[dataset_name]  
    N = dataset.shape[0]
    dim = dataset.shape[1]
    dataset_object = clustering_params.Data(
        datapoints=dataset,
        radius=r_dict[dataset_name]
    )
    centers = {}
    loss = {}
    for eps in EPSILONS:
        privacy_param = clustering_params.DifferentialPrivacyParam(epsilon=eps, delta=1/(N**1.1))
        cents = []
        losses = []
        for run in range(NUM_RUNS):
            result = clustering_algorithm.private_lsh_clustering(k, dataset_object, privacy_param)
            cents.append(result.centers)
            losses.append(result.loss)
        centers[str(eps)] = cents
        loss[str(eps)] = losses
    
    #Inf case
    cents = []
    losses = []
    for run in range(NUM_RUNS):
        kmeans = KMeans(n_clusters=k, n_init=1)
        kmeans.fit(dataset)
        cents.append(kmeans.cluster_centers_)
        losses.append(simple_loss(dataset, kmeans.cluster_centers_))
    centers['inf'] = cents
    loss['inf'] = losses
    
    results = {'centers':centers, 'loss':loss}
    with open(f'{FOLDER}/baseline_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved baseline for {dataset_name}")