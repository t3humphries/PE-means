# PE-means
Artifact for the paper *PE-means: Improved Differentially Private $k$-means Clustering through Private Evolution*.

## Project structure

```
PE-means/
├── src/           # core algorithm (helpers, pe_means_main, DP_comp)
├── baselines/     # run scripts for competitor implementations (diffprivlib, FastLloyd, Google, ICML17)
├── envs/          # conda environments 
├── experiments/   # grid-search and run scripts for PE-means
├── notebooks/     # analysis and plotting notebooks
├── setup/         # scripts to download submodules and data
├── icml17_matlab/ # git submodule
└── FastLloyd/     # git submodule
```

## Dependencies
The main env for running most files is ```envs/pe_means.yml```. There are also two other environments for running the baselines as well as plotting results (due to compatibility). We specify the files that don't use ```envs/pe_means.yml``` (and also include a note in those files).

- ```FastLloyd/env.yml``` is used for ```baselines/fastlloyd_baseline.py```, ```baselines/diffprivlib_baseline.py```, and for any notebooks starting with ```paper_```. The only modification to this env is to install ipykernel (or whatever you need to run notebooks in your setup).
- ```envs/icml17.yml``` is used for ```baselines/icml_baseline.py```

## Project Setup

After cloning this repo, the following steps are required to ensure all code and datasets are present.

1. Pull all submodules and apply a patch to FastLloyd (to use the correct privacy parameters) by running ``` bash setup/initial_setup.sh```
2. Download datasets with ```bash setup/download_datasets.sh```
3. Create the remaining dataset by running all cells in ```setup/generate_datasets.ipynb```

## Running experiments
### Note on PATH
We recommend running all python files from the project root while specifying the python path with `PYTHONPATH=.`
To avoid typing `PYTHONPATH=.` every time, export it for your shell session:
```bash
export PYTHONPATH=.
# then run normally:
python experiments/run_ours.py
```
### Baselines
To get the baselines, run all python files in the baselines directory using the correct conda env (specified above) e.g.:
```
PYTHONPATH=. python baselines/google_baseline.py
PYTHONPATH=. python baselines/diffprivlib_baseline.py
# etc.
```
Result logs will be outputted to ```result_final/<baseline>```
### Experiments
To reproduce the results for PE-means, run all python files in the experiments directory e.g.:

```
PYTHONPATH=. python experiments/run_ours.py
PYTHONPATH=. python experiments/grid_private.py
#etc.
```
Result logs will be outputted to ```result_final/<experiment_name>```

### Single run
To conduct a single run of PE-means (rather than reproduce all runs) we include ```experiments/test_single_run.py```. By default this runs the algorithm on a small synthetic dataset, but this file can be edited to run any dataset or parameter configuration.

## Edits to Balcan et al.'s code

We uncovered the following three inconsistencies in the privacy analysis between the paper of [Balcan et al.](https://proceedings.mlr.press/v70/balcan17a.html) and the [source code](https://github.com/mouwenlong/dp-clustering-icml17/tree/master) which we include in `icml17_matlab/`:

**1. Budget overrun in Lloyd refinement (`clustering.m`)**

The code runs an extra Lloyd refinement step not in the paper. With `T=1` the overall budget of `clustering.m` breaks down as: candidate `2ε/3`, localsearch `ε/12`, recover `ε/6`, `nLloyd` steps of Lloyd at `ε/(2·nLloyd)` each  — totalling `17ε/12`. 
**Fix:** use `ε/(12·nLloyd)` for the Lloyd recover calls so everything sums to `ε`.

**2. Exponential mechanism sensitivity off by 2× (`localsearch.m`)**

The code uses sensitivity `Lambda^2 = (2·range)^2` but the paper uses `8·Λ² = 8·range^2`. 
**Fix:** change `exp(-ε·gains / (Lambda^2 ·(T+1)))` to `exp(-ε·gains / (2·Lambda^2·(T+1)))` (both the per-iteration and output-selection calls).

**3. Laplace noise scale missing `√d` and diameter factor (`recover.m`)**

The code adds Laplace noise with scale `range/(ε·n)` where `range` is the L₂ radius. Two issues: (a) Laplace needs an L₁ sensitivity bound, which for an L₂ ball of radius `range` in `d` dimensions is `√d·range`; (b) `range` is a radius so the max per-point change is `2·range`. 
**Fix**: change scale to `(2·√d·range)/(ε·n)`.

## Edits to Google-LSH
In the following [fork](https://github.com/t3humphries/dp_clustering.git) of Google's repo, we edit the code to use the ground-truth dataset size since we assume the dataset size is public in our work. This fork is installed as a package in ```envs/pe_means.yml```.

## Edits to FastLloyd
In [FastLloyd](https://github.com/D-Diaa/FastLloyd), all datasets are assumed to be in a hypercube with each dimension in $[-1,1]$, we modify the code to accept our tighter $\ell_\infty$ bound that is computed non-privately for all related work. We also adjust the hard-coded $\delta$ to match the one in our work. These edits are applied by a patch when checking out the submodule using ```setup/initial_setup.sh```.

## Contact

For questions or issues, please contact [Thomas Humphries](https://t3humphries.github.io).