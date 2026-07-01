import os
from threadpoolctl import threadpool_limits
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["SKLEARN_ENABLE_CUBLAS"] = "0"  # Disable cuBLAS in scikit-learn to prevent it from using multiple threads
import faiss
faiss.omp_set_num_threads(1)  # Limit FAISS threads to 1 for better performance in this test
#stop scikit-learn from using multiple threads, which can interfere with our Ray parallelism


import time
import numpy as np
from src.pe_means_main import pe_means
from clustering import clustering_params
from sklearn.datasets import make_blobs
from src.helpers import *
import os


def main():
    print("Generating synthetic dataset...")
    data, _ = make_blobs(n_samples=5000, n_features=10, centers=16, random_state=1)
    dataset, r, _ = dataset_preprocess(data)
    dataset_object = clustering_params.Data(
        datapoints=dataset,
        radius=r
    )
    k = 16

    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=1.0, delta=1/(dataset.shape[0]**1.1))
    print("Running EA clustering...")
    start = time.time()
    # We pass visualize=False so it doesn't try to plot
    this_config = PE_PARAMS.copy()
    this_config['run_id'] = 0
    with threadpool_limits(limits=1, user_api='blas'):
        result = pe_means(k, dataset_object, this_config, privacy_param=privacy_param, visualize=False, verbose=True)
    duration = time.time() - start
    print(f"Clustering complete in {duration:.2f} seconds.")

if __name__ == '__main__':
    main()