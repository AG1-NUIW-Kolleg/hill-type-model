# Replication Data for: COUPLED SIMULATIONS AND PARAMETER INVERSION FOR NEURAL SYSTEM AND ELECTROPHYSIOLOGICAL MUSCLE MODELS

This dataset allows to reproduce the results from the paper "COUPLED SIMULATIONS AND PARAMETER INVERSION FOR NEURAL SYSTEM AND ELECTROPHYSIOLOGICAL MUSCLE MODELS" submitted to GAMM Mitteilungen in September 2023.

# BayesianInference

Markov Chain Monte Carlo methods used for the use-case of a a Hill-type muscle model. We compare the results of a Metropolis-Hastings Markov Cahin Monte Carlo method with an Hamiltonian Monte Carlo method. The investigated muscle setup is a two-muscle-one tendon system.

## Structure

List of files provided:

- **BayesianInference-main.tar.gz**: This are the case files and results for the Bayesian Inference methods. A python scriptfor plotting is included.

## How to run the Bayesian Inference cases

Here we provide a quick guide for Ubuntu 22.04 users that want to run the Markov Chain Monte Carlo and Hamiltonian Monte Carlo method to invert for the prestretch in the "AMI" case

1. **Install Python requirements**

    - Extract requirements.txt from **BayesianInference-main.tar.gz**.
    - Run `pip install -r requirements.txt`

2. **Run the Hamiltonian Monte Carlo (HMC) method**

    - Extract the code in the folder **HMC** from **BayesianInference-main.tar.gz**.
    - Run `python NUTS_HMC_Hill_Type_Model.py` to create the HMC results
    - Run `python Plot_Results.py` to plot the results
    - Run `python Plot_Results_NUTS_Infos.py` to plot informations about the HMC simulation
    
3. **Run the Markov Chain Monte Carlo (MCMC) method**

    - Extract the code in the folder **MCMC** from **BayesianInference-main.tar.gz**.
    - Run `python MCMC_Hill_Type_Model.py` to create the MCMC results
    - Run `python Effective_Sample_Size_MCMC_Bulk.py` to plot the results






