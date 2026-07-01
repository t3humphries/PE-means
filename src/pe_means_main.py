import faiss
import numpy as np
from collections import Counter
from clustering import clustering_params, clustering_algorithm
from scipy.special import gamma
from sklearn.cluster import KMeans
from src.DP_comp import get_noise_multiplier
from src.helpers import *

def get_labels_memory_efficient(data, centers):
    num_points = data.shape[0]
    num_centers = centers.shape[0]
    
    # Initialize labels and a vector to track the minimum distance found so far
    labels = np.zeros(num_points, dtype=int)
    min_dists = np.full(num_points, np.inf)
    
    for i in range(num_centers):
        # Compute squared Euclidean distance to the i-th center
        # (Using squared distance avoids the expensive sqrt and yields the same argmin)
        dist_sq = np.sum((data - centers[i])**2, axis=1)
        
        # Where this center is closer than previous ones, update labels and min_dists
        closer_mask = dist_sq < min_dists
        min_dists[closer_mask] = dist_sq[closer_mask]
        labels[closer_mask] = i
        
    return labels

def levy_mutation(centroids, L, radius=1.0, beta=1.5, step_scale=0.01):
    K, D = centroids.shape
    total_mutants = K * L
    targets = np.repeat(centroids, L, axis=0)
    if beta >= 2.0:
        # Pure Gaussian mutation
        steps = np.random.normal(0, 1, (total_mutants, D))
    else:
        # Calculate the step length based on Mantegna's algorithm
        num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))
        sigma_u = (num / den)**(1 / beta)
        
        u = np.random.normal(0, sigma_u, (total_mutants, D))
        v = np.random.normal(0, 1, (total_mutants, D))
        
        steps = u / (np.abs(v)**(1 / beta))
    
    mutated_samples = targets + (step_scale * radius * steps)
    
    # Project back to hypersphere
    norms = np.linalg.norm(mutated_samples, axis=1, keepdims=True)
    mutated_samples = np.where(norms > radius, (mutated_samples / norms) * radius, mutated_samples)
    
    return mutated_samples

def nn_utility(
    population: np.ndarray,
    data: clustering_params.Data,
    noise_multiplier: float,
    debug: bool = False
) -> np.ndarray:
    
    index = faiss.IndexFlatL2(population.shape[1])
    index.add(population)
    distance, ids = index.search(data.datapoints, k=1)
    valid_votes = ids.flatten()
    counter = Counter(list(valid_votes))
    # shape of the synthetic samples
    count = np.zeros(shape=population.shape[0])
    for k in counter:
        count[k % population.shape[0]] += counter[k]
    count = np.asarray(count)
    # clean_count = count.copy()
    if noise_multiplier > 0:
        count += (np.random.normal(size=len(count)) * noise_multiplier)
    assert len(count) == population.shape[0], f"Utility length {len(count)} does not match population size {population.shape[0]}"
    if debug:
        return count, valid_votes, distance.flatten()
    else:
        return count
    
def find_threshold(utilitys, dataset_size):
    arr = np.sort(utilitys)
    rev_cumsum = np.cumsum(arr[::-1])
    idx = np.searchsorted(rev_cumsum, dataset_size)
    
    if idx >= len(arr):
        return 0
    threshold = arr[len(arr) - 1 - idx]
    return threshold

def init_population(
    pop_size: int,
    dim: int,
    radius: float) -> np.ndarray:

    # Step 1: Generate points uniformly on the surface of a unit d-dimensional sphere
    normal_deviates = np.random.randn(pop_size, dim)
    norms = np.linalg.norm(normal_deviates, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    unit_sphere_points = normal_deviates / norms
    
    # Step 2: Scale the points to be uniformly distributed within the interior volume
    uniform_deviates = np.random.rand(pop_size, 1)
    radii = radius * (uniform_deviates ** (1.0 / dim))
    
    # Step 3: Combine direction and radius
    return unit_sphere_points * radii

def pack_in_sphere(num_points, dims, container_radius, max_retries=100, seed=None):
    '''Adapted to use a radius boundry from _init_centers in https://github.com/IBM/differential-privacy-library/blob/5d9c9d873c99295f14b0cfb61866fafbf3cc4684/diffprivlib/models/k_means.py'''
    
    k = num_points
    generator = np.random.RandomState(seed)
    
    # We start by trying to pack the largest spheres possible
    # A good starting point is the container_radius / 2
    cluster_proximity = container_radius / 2.0
    
    while cluster_proximity > 0:
        centers = np.zeros(shape=(k, dims))
        cluster, retry = 0, 0
        
        # The center of the small sphere must be within this distance 
        # from the origin to stay inside the container
        max_center_dist = container_radius - cluster_proximity
        
        while retry < max_retries:
            if cluster >= k:
                break
            
            # 1. Generate a uniform random point inside a hypersphere
            # Method: Normal distribution + Normalization + Radial Scaling
            vec = generator.normal(0, 1, dims)
            mag = np.linalg.norm(vec)
            # Scale by U^(1/d) to ensure uniform volume distribution
            u = generator.random() ** (1.0 / dims)
            temp_center = (vec / mag) * u * max_center_dist
            
            if cluster == 0:
                centers[0, :] = temp_center
                cluster += 1
                continue
            
            # 2. Check distance to existing spheres
            # (Distances are between centers, so they must be >= 2 * radius)
            min_distance_sq = ((centers[:cluster, :] - temp_center) ** 2).sum(axis=1).min()
            
            if np.sqrt(min_distance_sq) >= 2 * cluster_proximity:
                centers[cluster, :] = temp_center
                cluster += 1
                retry = 0
            else:
                retry += 1
                
        if cluster >= k:
            return centers
        
        # If we couldn't fit them, shrink the spheres and try again
        cluster_proximity /= 2.0
        
    return None

def k_means_selection(population, utility, k, assignments=False):
    # Initialize the model with your desired number of final clusters 'k'
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)

    # Fit the model using your potential centroids and their respective weights
    kmeans.fit(X=population, sample_weight=utility)

    # The approximated optimal centroids are now available
    final_centroids = kmeans.cluster_centers_
    if assignments:
        assignments = kmeans.labels_
        return final_centroids, assignments
    return final_centroids

class GaussianProjector:
    def __init__(self, input_dim, latent_dim, seed=42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initialize the random projection matrix
        # Scale by 1/sqrt(latent_dim) to preserve the approximate norm of vectors
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((input_dim, latent_dim)) / np.sqrt(latent_dim)
        

    def project(self, x):
        """Map high-dim (128) to latent (32/16)"""
        # x shape: (n_samples, input_dim)
        return np.dot(x, self.W)


    
def pe_means(
    k: int,
    data: clustering_params.Data,
    ea_params: dict,
    privacy_param: clustering_params.DifferentialPrivacyParam | None = None,
    short_description: str = "Default",
    visualize: bool = True,
    verbose: bool = False,
    reduce: bool = False
) -> tuple:
    """Clusters data into k clusters with generation-by-generation visualization.
    
    Returns:
    (ClusteringResult, history) where history contains population, utility, and selected indices for each generation.
    """
    ##--------------------SETUP---------------------##
    np.random.seed(ea_params['run_id'])
    if privacy_param is not None:
        gen_total = ea_params['num_gen'] + 2 if reduce else ea_params['num_gen']
        noise_multiplier = get_noise_multiplier(
            epsilon=privacy_param.epsilon,
            num_steps=gen_total,
            delta=privacy_param.delta)
        if verbose:
            print(f'Using noise multiplier: {noise_multiplier:.4f} for epsilon={privacy_param.epsilon}, delta={privacy_param.delta}, num_steps={gen_total}')
    else:
        noise_multiplier = 0.0
        if verbose:
            print("No differential privacy required.")

    size_per_generation = (ea_params['L'] +1)*k

    
    
    ##--------------------INIT---------------------##
    if ea_params['init_mode'] == 'random':
        population = init_population(
            pop_size=int(size_per_generation),
            dim=data.dim,
            radius=data.radius)
    
    elif ea_params['init_mode'] == 'sphere_packing':
        population = pack_in_sphere(
            num_points=int(size_per_generation),
            dims=data.dim,
            container_radius=data.radius
        )

    else:
        raise ValueError(f"Unsupported initialization mode: {ea_params['init_mode']}")
    
    ##--------------------MAIN LOOP---------------------##
    history = []

    for gen in range(ea_params['num_gen']):
        utility = nn_utility(population, data, noise_multiplier)
        threshold = find_threshold(utility, dataset_size=data.datapoints.shape[0])
        utility = np.where(utility >= threshold, utility, 0)
        
        if ea_params['L_reduce_threshold'] is not None and noise_multiplier > 0:
            sum_square_util = np.sum(utility**2)
            noise = len(utility) * (noise_multiplier**2)
            signal_noise_ratio = sum_square_util/noise
            if verbose:
                print("signal to noise ratio:", signal_noise_ratio)
            if signal_noise_ratio < ea_params['L_reduce_threshold']:
                ea_params['L'] = max(int(ea_params['L_reduce_factor'] * ea_params['L']), 4)

        if ea_params['select_mode'] == 'k_means':
            top_samples = k_means_selection(population, utility, k)
        elif ea_params['select_mode'] == 'top_k':
            top_indices = np.argsort(utility)[-k:]
            top_samples = population[top_indices]
        else:
            raise ValueError(f"Unsupported selection mode: {ea_params['select_mode']}")

        # Store history
        loss = clustering_algorithm.ClusteringResult(data, top_samples).loss
        if visualize:
            history.append({
                'generation': gen,
                'population': population.copy(),
                'utility': utility.copy(),
                'top_indices': np.argsort(utility)[-k:][::-1].copy(),
                'top_samples': top_samples.copy(),
                'loss': loss
            })
        else:
            history.append(loss)
        if verbose:
            print(f'Generation {gen+1}/{ea_params["num_gen"]} for {short_description}, loss: {loss:.8f}')
        
        if gen == ea_params['num_gen'] - 1: #Last generation
            break
        
        # Variation
        mutated_samples = levy_mutation(top_samples, L=ea_params['L'], radius=data.radius, step_scale=ea_params['var_scale'],beta=ea_params['levy_beta'])
        
        
        # Combine and create new population
        population = np.vstack((top_samples, mutated_samples))
        
        if gen != 0 and ea_params['L_reduce_threshold'] is None: #First gen is larger.
            assert population.shape[0] == size_per_generation, f"Population size {population.shape[0]} does not match expected {size_per_generation}"
        
    return clustering_algorithm.ClusteringResult(data, top_samples).loss, top_samples, history


def hdpe_means(
    k: int,
    data: clustering_params.Data,
    ea_params: dict,
    privacy_param: clustering_params.DifferentialPrivacyParam | None = None,
    short_description: str = "Default",
    visualize: bool = True,
    verbose: bool = False,
    debug: bool = False,
    new_dim: int = 16
):
    np.random.seed(ea_params['run_id'])
    if data.dim > 32:
        projector = GaussianProjector(data.dim, new_dim)
        latent_features = projector.project(data.datapoints)
        new_data = clustering_params.Data(
            datapoints=latent_features,
            radius=data.radius
        )
        ea_params['num_gen'] = int(4 * np.sqrt(new_dim)) - 2
    else:
        new_data = data
    

    loss, centers, history = pe_means(
        k=k,
        data=new_data,
        ea_params=ea_params,
        privacy_param=privacy_param,
        short_description=short_description,
        visualize=visualize,
        verbose=verbose,
        reduce=True
    )
    if data.dim <= 32:
        return loss, centers, history

    #Project back
    if privacy_param is not None:
        noise_multiplier = get_noise_multiplier(
            epsilon=privacy_param.epsilon,
            num_steps=ea_params['num_gen'],
            delta=privacy_param.delta)
    else:
        noise_multiplier = 0.0
    labels = get_labels_memory_efficient(latent_features, centers)
    new_centers = []
    for i in range(k):
    # Mask the raw data for points assigned to cluster i
        cluster_points = data.datapoints[labels == i]
        if len(cluster_points) > 0:
            sum = np.sum(cluster_points, axis=0) + (np.random.normal() * noise_multiplier * data.radius)
            count = len(cluster_points)+ (np.random.normal() * noise_multiplier)
            new_centers.append(sum/count)
    return clustering_algorithm.ClusteringResult(data, np.array(new_centers)).loss, new_centers, None
    