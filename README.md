# Bayesian SEM

## Setup

Initial install:

    $ conda env create -f environment.yml
    $ source activate bayes-sem

Update:

    $ conda env update -f environment.yml

Deactivate Environment:

    $ source deactivate

Remove:

    $ conda remove -n bayes-sem --all


Convert `a.ipynb` to `a.py`:

    $ jupyter nbconvert --to script a.ipynb


## How to use:

#### Continuous Data

* To run the Muthen model run `run_muthen_exp1.py`. The command runs as follows

  ```
  run_muthen_exp1.py <num_warmup> <num_samples> <model_code>
  ```

  with optional flags for `-th` (`--task_handle`), `-pm` (`--print_model`), (`--num_chains`), 
  (`--gender`) and
  `-xdir` (`--existing_directory`). The results are saved in
  `src/log/<date-time>_<task_handle>`.
  
  To choose which model to run use the `<model_code>` option as follows:
  "0:benchmark saturated model,1 CFA/4 EFA:exact zeros no u's, 2 CFA/5 EFA: full factor model"
  
  If an existing directory is given then the script looks for an existing compiled
  stan model to load and run with the new number of iterations.
  
  The results are processed using `model-results.py`

* The same model but replacing 10% of the data with 10's `run_muthen_extreme_exp.py`.
The command runs as follows

    ```
    run_muthen_extreme_exp.py <num_warmup> <num_samples> <num_chains> <men/women>
    ```

* Script `run_muthen_kfold.py` runs a kfold version of the `run_muthen_exp1`
script. It works as follows

    ```
    run_muthen_kfold.py <num_warmup> <num_samples> <num_chains> <men/women> <model_code>
    ```

  with optional flags for `-th` (`--task_handle`), `-pm` (`--print_model`), `-nfl`
  (`n_splits`) and `-xdir` (`--existing_directory`). The results are saved in
  `src/log/<date-time>_<task_handle>`.
  
  To choose which model to run use the `<model_code>` option as follows:
  "0:benchmark saturated model,1 CFA/4 EFA:exact zeros no u's, 2 CFA/5 EFA: full factor model"
  
  If an existing directory is given then the script looks for an existing compiled
  stan model to load and run with the new number of iterations.
  
  The results are processed using `kfold_results.py` by passing the two `log/<dir>`
  of the two models to be compared. The order matters, as follows.
    
    ```
    kfold_results.py <log_dir_benchmark_model> <log_dir_proposed_model> 
    ```
    

* To run a 2-factor and 6 manifest variables  model simulation run `run_sim_exp1.py`. The command runs as follows

  ```
  run_sim_exp1.py <num_warmup> <num_samples> <data_scenario> <model_code>
  ```

  with optional flags for `-th` (`--task_handle`), `-pm` (`--print_model`),`-num_chains` (`--num_chains`), `-seed` (`--random_seed`), `-nd` (`--nsim_data`)  and
  `-xdir` (`--existing_directory`). The results are saved in
  `src/log/<date-time>_<task_handle>`.
  
  To choose which model to run use the `<model_code>` option as follows:
  "0:benchmark saturated model,1 CFA/4 EFA:exact zeros no u's, 2 CFA/5 EFA: full factor model"
  
  If an existing directory is given then the script looks for an existing compiled
  stan model to load and run with the new number of iterations.
  
  The results are processed using `model-results.py`

* To run a 2-factor and 6 manifest variables  model simulation with kfold splits run `run_kfold_sim.py` with similar options as above. 

* For versions of the script that run the experiment multiple times with different seeds look for the `_bstr` suffix.


#### Discrete Data

 * Run simulation with `run_sim_bin.py` and `run_kfold_sim_binary.py`. Get the PPP results in notebooks such `9.1 Simulation 2 Factor Binary - PPP.ipynb` which uses functions from `modelresultsbinary.py` and `modelresultsbinary_4chains.py`. Kfold results are run in notebooks such `9.4 Simulation Factor Binary - Kfold-Pattern Index.ipynb`
 
 
### Real World binary data

#### FND Data

We used our methodology on the on the real dataset FND (Fagerstrom Nicotine Dependence). There are 6 questions (J=6) that are binary (either originally or made to be). We used two different factor structures: 1 factor (K=1) loading on all items, and 2 factors (K=2) where the first factor loads to questions 1,2,3 and the second factor loads to 4,5,6.  

The model structure and related literature of the dataset and other factor analyses papers written on the dataset is described in *"A confirmatory factor analysis of the Fagerstrom Test for Nicotine Dependence"* by
Chris G. Richardson, Pamela A. Ratner.  

* Raw data is saved in `dat/real_data/FND.csv` and the data descriptions are in `dat/real_data/FTND Description.xlsx`. Data is processed with `src/codebase/data_FND.py`. There is also a notebook presenting some descriptive statistics of the dataset `src/10. Real Data Setup - FND.ipynb`

* Run the FND data experiment with `run_FND_bin.py`. To choose which model to run use the `<model_code>` option as follows:  "1: Model1 (2 factor), 2:Model 2 (2 factor), 3: Model 1.5 (2 factor), 4: 1 Factor Model, 5: EFA no u's, 6: EFA with u's". Results are saved in `log/<time_task_handle>` directories. By default it runs the 'PPP' experiment, pass `-cv cv` to run the Cross Validation experiment. 

* Compute PPP values with `src/compute_ppp_bin.py` and cross validation values with `src/compute_cv_bin.py`

### On Fabian

We installed pystan version 2.19 which should run the same scripts as local correctly.

1. Remove `apps/anaconda3/5.0.0`   
2. Activate `apps/anaconda3/4.4.0` 
3. Activate env `pystan-dev`    


    ```
    module del apps/anaconda3/5.0.0
    module add apps/anaconda3/5.0.0
    source activate pystan-dev
    ```

### On local system  

To save storage space you can remove old pickled files with 

    find . type f -name '*.p' -delete


## Results:

* To run the experiment of Muthen data `run_muthen_exp1.py`. Usually we run this
on the server, collect the results and visualize them locally with
`1.1.Muthen-women` or `1.2.Muthen-men`. 

* Compared the PPP values using `model-results.py` of model 1 and model 2.

* Compared the kfold-PPP values using `kfold_results.py` of model 1 and model 2 as it compares to model 0.

* The results of experiment of replacing 10% is in notebooks
`2.1.Muthen-women-experiment` and `2.2.Muthen-men-experiment` but are outdated now, as we have not run them using the latest models.


## To Do

* ~~Compare PPP values for `model1` and `model2`~~
* ~~Compare k-fold PPP for `model1` and `model2`~~
* ~~Run EFA (model 1) for all simulated scenarios including kfold~~
* ~~Run EFA (model 2) for all simulated scenarios including kfold~~
* ~~Clean data from Irini to run binary model~~
* ~~Run 1 factor model for the processed data of Irini (running)~~
* ~~Undersand why adding approximate zeros to `beta` in Binary model fails. (Try 8 variables and 2 factors)~~
* Create PPP for 1 factor models
* ~~Fix hard code issue of 100 n_sim PPP calculation~~
* ~~Write run scripts for FND data PPP and CV Index~~
* ~~Get PPP results for EFA Model 2 on FND data~~
* ~~Get CV Index results for Models 1 and 2 on FND data~~
* ~~Plot and see if PPP results make sense~~
* ~~Plot and see if CV results make sense~~
* Move vis display code to `altair`
* Change code for modelresultsbinary to recognize when to use 1 chain and when to use 4. Maybe do not squeeze results of one chain
* Re-run continuous model code for experiments and clean up code and save results