import sys
import timeit
import csv
import random
import numpy as np 
import jax as jx
from jax import jit
from jax import grad
from jax.config import config
import jax.numpy as jnp
import jax.random as random_jax
from jax.scipy.stats import beta
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import statistics 
#from scipy.stats import lognorm, gamma, t, beta, invgamma
from scipy.signal import find_peaks
import AD_Hill_System_HMC_Py as Hill_Solution

start = timeit.default_timer()

# Enable double precision
jx.config.update("jax_enable_x64", True)
# Uncomment this line to force using the CPU
jx.config.update('jax_platform_name', 'cpu')

# Define observed data, in this case expected change of muscle length

lobs_m1 = 15.0 # Observed prestretched length of muscle 1 [cm]
lobs_m2 = 15.0 # Observed prestretched length of muscle 2 [cm]
lobs_td = 5.0 # Observed prestretched length of tendon [cm]
extmaxobs_muscle_1 = 18.3 # Maximal extension of muscle 1
extminobs_muscle_1 = 8.7 # Minimal extension of muscle 1
extmaxobs_muscle_2 = 21.00 # Maximal extension of muscle 2
extminobs_muscle_2 = 11.5 # Minimal extension of muscle 2

observed_data = [extmaxobs_muscle_1,extminobs_muscle_1,extmaxobs_muscle_2,extminobs_muscle_2]

# Parameters

ode_solver = 1 # 1 = Thelen, 2 = Van Soest, 3 = Silva, 4 = Hyperelastic

# Define range of possible model input parameters 

lslack_muscle_1_input = [9.0,17.0] # Stress-free length of muscle 1
lslack_muscle_2_input = [9.0,17.0] # Stress-free length of muscle 2
#lslack_tendon_input = [4.95,7.0] # Stress-free length of tendon

# Simulation parameters
Tstart_input = 0.0
Tend_input = 6.0

# MCMC parameters
    
NUM_BINS = 2 #Numbers of blocks in Histogramm
NUM_DRAWS = 1000 #Number of HMC draws
burn_in = 500 #Number of iterations for the burn-in of the HMC

min_sample_boundary = [lslack_muscle_1_input[0],lslack_muscle_2_input[0]]
max_sample_boundary = [lslack_muscle_1_input[1],lslack_muscle_2_input[1]]

# Allow number of different tries to choose a valid start sample for the HMC algorithm
maximum_start_counter = 5

# Define a priori parameters of distributions 
STD_DEVIATION_PROP_DIST = [0.1,0.1] #Standard deviation of proposal distribution
STD_DEVIATION_PRIOR_DIST = [6.0,6.0] #Standard deviation of prior distribution
EXP_VALUE_PRIOR_DIST = [13.5,13.5] #Expected value of prior distribution
STD_DEVIATION_DATA = [1.0,1.0,1.0,1.0] #Standard deviation of data

# Model parameters
model_parameters = {'Obs_Length_M1': lobs_m1,
          'Obs_Length_M2': lobs_m2,
          'Obs_Length_Td': lobs_td,
          'Obs_Ext_Max_M1': extmaxobs_muscle_1,
          'Obs_Ext_Min_M1': extminobs_muscle_1,
          'Obs_Ext_Max_M2': extmaxobs_muscle_2,
          'Obs_Ext_Min_M2': extminobs_muscle_2,
          'Std_Data': STD_DEVIATION_DATA,
          'Exp_Prior': EXP_VALUE_PRIOR_DIST,
          'Std_Prior': STD_DEVIATION_PRIOR_DIST,
          'Mass_M1': 1.0,
          'Mass_M2': 1.0,
          'Force_M1': 0.0,
          'Force_M2': 0.0,
          'Force_Ref_M0': 6820.0,
          'Force_Ref_Max': 1.6}


# HMC Parameter

L_LF = 20 # Number of Leapfrog-Steps
#epsilon_range = [0.1*1.0e-1,0.3*1.0e-1] # Range of Leapfrog-Step-length for HMC-Step
epsilon_range = [0.01,0.015] # Range of Leapfrog-Step-length for HMC-Step
delta_max = 300.0 # Large value, that it does not interfere the algorithm

STD_DEVIATION_MOMENTUM = [1.0,1.0] # Standard deviation of artificial Hamiltonian momentum
EXP_VALUE_MOMENTUM = [0.0,0.0] # Expected value of artificial Hamiltonian momentum

# Plot of prior Distribution
LEFT_LIM = [lslack_muscle_1_input[0],lslack_muscle_2_input[0]]
RIGHT_LIM = [lslack_muscle_1_input[1],lslack_muscle_2_input[1]]

# Plot parameters
rc('font', **{'size':12})#, 'family':'serif', 'serif':['Computer Modern Roman']}) 
rc('text', usetex=True)

# Plot the Simulation Results
PATH1 = 'compare_results.pdf'
PATH2 = 'hmc_walk_par1.pdf'
PATH3 = 'hmc_walk_par2.pdf'

# Define probability functions for automatic differentiation

# Draw from normalized prior distribution
def normal_prior_dist_start_length_m1_beta_jax(theta):
    l_m1_max = 16.5
    location = 0.0
    sca = l_m1_max
    alpha_var = 15.0
    beta_var = 5.0
    theta_true = (theta-location)/sca
    #if theta_true >= 0.0 and theta_true <= 1.0:
    normal_prior_prob = beta.pdf(theta_true,alpha_var,beta_var) # Beta distribution
    #else:
    #    normal_prior_prob = 0.0
    return normal_prior_prob

# Draw from normalized prior distribution
def normal_prior_dist_start_length_m2_beta_jax(theta):
    l_m2_max = 16.5
    location = 0.0
    sca = l_m2_max
    alpha_var = 15.0
    beta_var = 5.0
    theta_true = (theta-location)/sca
    #if theta_true >= 0.0 and theta_true <= 1.0:
    normal_prior_prob = beta.pdf(theta_true,alpha_var,beta_var) # Beta distribution
    #else:
    #    normal_prior_prob = 0.0
    return normal_prior_prob

# Draw from normalized prior distribution
def normal_prior_dist_prestret_start_length_m1_normal_jax(theta,exp_val_1d,std_data_1d):
    normal_prior_prob = (1/(std_data_1d*jnp.sqrt(2*math.pi)))*jnp.exp(-0.5*jnp.power(((theta-exp_val_1d)/std_data_1d), 2)) # White noise prior L2
    return normal_prior_prob

# Draw from normalized prior distribution
def normal_prior_dist_prestret_start_length_m2_normal_jax(theta,exp_val_1d,std_data_1d):
    normal_prior_prob = (1/(std_data_1d*jnp.sqrt(2*math.pi)))*jnp.exp(-0.5*jnp.power(((theta-exp_val_1d)/std_data_1d), 2)) # White noise prior L2
    return normal_prior_prob

# Draw from prior distribution
def prior_dist_jax(theta,exp_val_prior,std_prior):
    prior_prob = (normal_prior_dist_prestret_start_length_m1_normal_jax(theta[0],exp_val_prior[0],std_prior[0])) * (normal_prior_dist_prestret_start_length_m2_normal_jax(theta[1],exp_val_prior[1],std_prior[1]))
    #prior_prob = (normal_prior_dist_start_length_m1_beta_jax(theta[0])) * (normal_prior_dist_start_length_m2_beta_jax(theta[1]))
    #prior_prob = (normal_prior_dist_m1_invgamma(theta[0])) * (normal_prior_dist_m2_invgamma(theta[1])) * (normal_prior_dist_td_invgamma(theta[2]))
    # TODO uniform dist?
    return prior_prob 

# Calculate likelihood exponent  
def likelihood_dist_exp_jax(obs,calc,theta,std_data):
    likelihood_prob = -0.5*jnp.sum(jnp.power(((jnp.asarray(calc)-jnp.asarray(obs))/jnp.asarray(std_data)), 2)) 
    return likelihood_prob 

# Calculate likelihood     
def likelihood_dist_jax(obs,calc,theta,std_data):
    likelihood_prob = jnp.exp(-0.5*jnp.sum(jnp.power(((jnp.asarray(calc)-jnp.asarray(obs))/jnp.asarray(std_data)), 2))) 
    return likelihood_prob 

# Draw from prosterior distribution
def non_normalized_posterior_dist_jax(obs,calc,theta,std_data,exp_val_prior,std_prior):
    likeli = likelihood_dist_jax(obs,calc,theta,std_data)
    priori = prior_dist_jax(theta,exp_val_prior,std_prior)
    posterior_prob = likeli*priori
    return posterior_prob 

# Calculate potential energy U of the Hamiltonian
def HMC_U(obs,calc,theta,std_data,exp_val_prior,std_prior):
    Ham_U = -np.log(non_normalized_posterior_dist_jax(obs,calc,theta,std_data,exp_val_prior,std_prior))
    return Ham_U

# Calculate kinetic energy K of the Hamiltonian
def HMC_K(mom,diagkovm):
    Ham_K = np.sum(np.power(mom,2)/(2*np.array(diagkovm)))
    return Ham_K

# Calculate non normalized posterior distribution for given input values
def posterior_hill_type_function_jax(input_value,params):
    # Compute Hill-type model simulation output from input values
    calculated_data_m1_raw_jax, calculated_data_m2_raw_jax = Hill_Solution.Hill_System_ODE_Solve(input_value,params)        

    # Calculate output of the forward problem
    value_maximal_length_m1_jax = jnp.amax(calculated_data_m1_raw_jax)
    value_minimal_length_m1_jax = jnp.amin(calculated_data_m1_raw_jax)
    value_maximal_length_m2_jax = jnp.amax(calculated_data_m2_raw_jax)
    value_minimal_length_m2_jax = jnp.amin(calculated_data_m2_raw_jax)

    calculated_data_blackbox_jax = jnp.asarray([value_maximal_length_m1_jax,value_minimal_length_m1_jax,value_maximal_length_m2_jax,value_minimal_length_m2_jax])    
    
    # Initialize observed data
    observed_data_jax = jnp.asarray([params['Obs_Ext_Max_M1'],params['Obs_Ext_Min_M1'],params['Obs_Ext_Max_M2'],params['Obs_Ext_Min_M2']])
    
    # Standard deviation of observed data
    std_data_jax = jnp.asarray(params['Std_Data'])
    
    # Specify a priori distribution
    exp_value_prior_jax = jnp.asarray(params['Exp_Prior'])
    std_prior_jax = jnp.asarray(params['Std_Prior'])
    
    # Calculate posterior distribution for given input value
    posterior_distribution_jax = non_normalized_posterior_dist_jax(observed_data_jax,calculated_data_blackbox_jax,input_value,std_data_jax,exp_value_prior_jax,std_prior_jax)
    
    return posterior_distribution_jax

# Compute Hill-type simulation output from input values
def black_box_simulation(input_value,params):
    calculated_data_m1_raw_jax, calculated_data_m2_raw_jax = Hill_Solution.Hill_System_ODE_Solve(input_value,params)

    output_data_1 = np.asarray(calculated_data_m1_raw_jax, dtype = 'float')
    output_data_2 = np.asarray(calculated_data_m2_raw_jax, dtype = 'float')
    
    output_data = [output_data_1,output_data_2]
    return output_data

# Compute forward simulation output from input values
def observe_blackbox_simulation(sample_input,params):    
    calculated_data_m1_raw, calculated_data_m2_raw = black_box_simulation(sample_input,params)        

    value_maximal_length_m1 = np.amax(calculated_data_m1_raw)
    value_minimal_length_m1 = np.amin(calculated_data_m1_raw)
    value_maximal_length_m2 = np.amax(calculated_data_m2_raw)
    value_minimal_length_m2 = np.amin(calculated_data_m2_raw)

    calculated_data_blackbox = [value_maximal_length_m1,value_minimal_length_m1,value_maximal_length_m2,value_minimal_length_m2]
    
    return calculated_data_blackbox

# Compute one Leap-Frog step
def LeapFrog(Theta,momentum,epsilon,grad_U,model_parameters):
    momentum = momentum + 0.5 * epsilon * np.asarray(grad_U(jnp.asarray(Theta),model_parameters))
    Theta = Theta + epsilon * momentum   
    # Calculate log probability and grad(log probability)
    logprob = - HMC_U(observed_data,observe_blackbox_simulation(Theta,model_parameters),Theta,jnp.asarray(model_parameters['Std_Data']),jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior']))
    gradlogprob = - np.asarray(grad_U(jnp.asarray(Theta),model_parameters))
    # Half step for momentum again
    momentum = momentum + 0.5 * epsilon * gradlogprob 
    return Theta,momentum,gradlogprob,logprob

# Adaptive algorithm to find a step size epsilon for the Leap-Frog step
def FindReasonableEpsilon(Theta,grad_U,model_parameters):
    print('Start epsilon search')
    logprob_old = - HMC_U(observed_data,observe_blackbox_simulation(Theta,model_parameters),Theta,jnp.asarray(model_parameters['Std_Data']),jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior']))
    counter_max = 100
    counter = 1
    eps_start = 1.
    mom_start = np.random.normal(EXP_VALUE_MOMENTUM,STD_DEVIATION_MOMENTUM,len(STD_DEVIATION_PRIOR_DIST))
    _,mom_start_prop,grad_prop,logprob_prop = LeapFrog(Theta,mom_start,eps_start,grad_U,model_parameters)
     
    k = 1. 
    while np.isinf(logprob_prop) or np.isinf(grad_prop).any():
        k *= 0.5
        _,mom_start_prop,grad_prop,logprob_prop = LeapFrog(Theta,mom_start,(eps_start*k),grad_U,model_parameters)
        
    eps_start = 0.5 * k * eps_start
    
    logacceptprob = logprob_prop - logprob_old +HMC_K(mom_start,STD_DEVIATION_MOMENTUM)-HMC_K(mom_start_prop,STD_DEVIATION_MOMENTUM)
    
    alpha = 1. if logacceptprob > np.log(0.5) else - 1.
    #HMC_U_prop = HMC_U(observed_data,observe_blackbox_simulation(Theta_prop,model_parameters),Theta_prop,jnp.asarray(model_parameters['Std_Data']),jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior'])) 
    #P_quotient = np.exp(HMC_U_old-HMC_U_prop+HMC_K(mom_start,STD_DEVIATION_MOMENTUM)-HMC_K(mom_start_prop,STD_DEVIATION_MOMENTUM))
    #alpha = 2*(1.0 if P_quotient > 0.5 else 0.0) - 1.0
    #print('P_quotient',P_quotient)
    print('alpha',alpha)
    while (alpha * logacceptprob > - alpha * np.log(2.)) and counter <= counter_max: 
        eps_start = np.power(2.,alpha)*eps_start
        _,mom_start_prop,grad_prop,logprob_prop = LeapFrog(Theta,mom_start,eps_start,grad_U,model_parameters)
        logacceptprob = logprob_prop - logprob_old +HMC_K(mom_start,STD_DEVIATION_MOMENTUM)-HMC_K(mom_start_prop,STD_DEVIATION_MOMENTUM)
        #HMC_U_prop = HMC_U(observed_data,observe_blackbox_simulation(Theta_prop,model_parameters),Theta_prop,jnp.asarray(model_parameters['Std_Data']),jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior'])) 
        #P_quotient = np.exp(HMC_U_old-HMC_U_prop+HMC_K(mom_start,STD_DEVIATION_MOMENTUM)-HMC_K(mom_start_prop,STD_DEVIATION_MOMENTUM))
        #print('P_quotient',P_quotient)
        print('Counter find epsilon start', counter)
        counter = counter + 1
    print('Find epsilon result=',eps_start)
    print('End epsilon search')
    return eps_start

def determination_criterion(Theta_minus,Theta_plus,mom_minus,mom_plus):
    delta_theta = Theta_plus - Theta_minus
    return (np.dot(delta_theta,mom_minus.T) >= 0) & (np.dot(delta_theta,mom_plus.T) >= 0)

# Build Tree for No-U-Turn sampler for an improved adaptive HMC method (adaptivly change number of Leap-Frog steps)
def BuildTree(Theta,mom,logu,v,j,epsilon,Theta_0,r_0,grad_U,model_parameters):
    #grad_U_ausgewertet = np.asarray(grad_U(jnp.asarray(theta_proposal),model_parameters))
    HMC_U_0 = HMC_U(observed_data,observe_blackbox_simulation(Theta_0,model_parameters),Theta_0,jnp.asarray(model_parameters['Std_Data']),jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior'])) 
    HMC_K_0 = HMC_K(r_0,STD_DEVIATION_MOMENTUM)
    Hamiltonian_0 = - HMC_U_0 - HMC_K_0
    if (j == 0):
        Theta_prop,mom_prop,grad_prop,logprob_prop = LeapFrog(Theta,mom,v*epsilon,grad_U,model_parameters)
        #HMC_U_prop = HMC_U(observed_data,observe_blackbox_simulation(Theta_prop_if,model_parameters),Theta_prop_if,jnp.asarray(model_parameters['Std_Data']),jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior'])) #HMC_U_prop=logprob_prop
        HMC_K_prop = HMC_K(mom_prop,STD_DEVIATION_MOMENTUM)
        Hamiltonian_prop = logprob_prop - HMC_K_prop
        
        n_prop = int(logu < Hamiltonian_prop) #new point in the slice
        s_prop = int((logu - 1000.) < Hamiltonian_prop) #simulation widely inaccurate?
        
        Theta_minus = Theta_prop[:]
        Theta_plus = Theta_prop[:]
        mom_minus = mom_prop[:]
        mom_plus = mom_prop[:]
        grad_minus = grad_prop[:]
        grad_plus = grad_prop[:]
        
        alpha_prop = min(1.,np.exp(Hamiltonian_prop-Hamiltonian_0))
        n_alpha_prop = 1
        #n_prop_if = (1.0 if u <= np.exp(Hamiltonian_prop) else 0.0) 
        #s_prop_if = (1.0 if u < np.exp(delta_max + Hamiltonian_prop) else 0.0) 
        #return Theta_prop_if,mom_prop_if,Theta_prop_if,mom_prop_if,Theta_prop_if,n_prop_if,s_prop_if,np.min([1.0,np.exp(Hamiltonian_prop-Hamiltonian_0)]),1 
    else:        
        Theta_minus,mom_minus,grad_minus,Theta_plus,mom_plus,grad_plus,Theta_prop,grad_prop,logprob_prop,n_prop,s_prop,alpha_prop,n_alpha_prop = BuildTree(Theta,mom,logu,v,j-1,epsilon,Theta_0,r_0,grad_U,model_parameters)
        if (s_prop == 1):
            if (v == -1):
                Theta_minus,mom_minus,grad_minus,_,_,_,Theta_prop_2,grad_prop_2,logprob_prop_2,n_prop_2,s_prop_2,alpha_prop_2,n_alpha_prop_2 = BuildTree(Theta_minus,mom_minus,logu,v,j-1,epsilon,Theta_0,r_0,grad_U,model_parameters)
            else:
                _,_,_,Theta_plus,mom_plus,grad_plus,Theta_prop_2,grad_prop_2,logprob_prop_2,n_prop_2,s_prop_2,alpha_prop_2,n_alpha_prop_2 = BuildTree(Theta_plus,mom_plus,logu,v,j-1,epsilon,Theta_0,r_0,grad_U,model_parameters)
            
            if (np.random.uniform() < (float(n_prop_2) / max(float(int(n_prop) + int(n_prop_2)),1.))):
                Theta_prop = Theta_prop_2[:]
                grad_prop = grad_prop_2[:]
                logprob_prop = logprob_prop_2
            
            n_prop = int(n_prop) + int(n_prop_2)
            s_prop = int(s_prop and s_prop_2 and determination_criterion(Theta_minus,Theta_plus,mom_minus,mom_plus))
            alpha_prop = alpha_prop + alpha_prop_2
            n_alpha_prop = n_alpha_prop + n_alpha_prop_2
            #uniform_sample = np.random.uniform(0,1) 
            #if (n_prop+n_prop_2) > 0:
            #    if uniform_sample < min(1,(n_prop_2/(n_prop+n_prop_2))):
                    #Theta_prop = Theta_prop_2 
            #a_prop = a_prop + a_prop_2 
            #na_prop = na_prop + na_prop_2
            #s_prop = s_prop_2*(1 if (np.dot((Theta_plus-Theta_minus),mom_minus) >= 0.0) else 0)*(1 if (np.dot((Theta_plus-Theta_minus),mom_plus) >= 0.0) else 0)
            #n_prop = n_prop + n_prop_2 

    return Theta_minus,mom_minus,grad_minus,Theta_plus,mom_plus,grad_plus,Theta_prop,grad_prop,logprob_prop,n_prop,s_prop,alpha_prop,n_alpha_prop 

# Compute one samples of the HMC method with the No-U-Turn sampler method
def NUTS_HMC(start_sample,observed_data,grad_posterior_distribution,number_iterations,max_NUTS_iterations,model_parameters,timestep_adaptive=True,reasonable_epsilon=True,epsilon_range=[0.1*1.0e-1,0.3*1.0e-1],epsilon_change=[-0.1*1.0e-1,0.1*1.0e-1]):
    lss = len(start_sample)
    obsle = len(observed_data)
        
    samples = np.empty((lss,number_iterations),dtype=float) 
    lnprob = np.empty((1,number_iterations),dtype=float)
    epsilon_0_iterations = np.empty(number_iterations-1,dtype=float)
    counter_NUTS_iterations = np.empty(number_iterations-1,dtype=float)
    
    samples[0:lss,0] = start_sample 
    lnprob[0,0] = - HMC_U(observed_data,observe_blackbox_simulation(start_sample,model_parameters),start_sample,jnp.asarray(model_parameters['Std_Data']),jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior']))
    
    # Vector to save samples just from accepted iterations
    accepted_runs = 1
    
    # Choose a reasonable epsilon (Leapfrog-Step length)
    if reasonable_epsilon==True:
        #TODO test reasonable epsilon
        epsilon_0 = FindReasonableEpsilon(start_sample,grad_posterior_distribution,model_parameters)
    else:
        epsilon_0 = np.random.uniform(epsilon_range[0],epsilon_range[1])
    # Set NUTS parameters 
    delta = 0.3 #0.6 #Dual Averaging parameter 
    M_adapt = 500 #number_iterations
    mu_NUTS = np.log(10.0*epsilon_0)
    epsilon_0_bar = 1
    H_0_bar = 0
    gamma_NUTS = 0.05
    t_0_NUTS = 10
    kappa_NUTS = 0.75
    
    #Theta_old = Theta_0
    #H_bar_old = H_0_bar
    #epsilon_NUTS_old = epsilon_0
    #H_bar = H_0_bar
    #epsilon_bar = epsilon_0_bar
    
    #v_j_set = (-1,1)
    
    #Start MCMC iteration (HMC)
    for i in range(0, (number_iterations-1)):            
        print("Run: ", 1 , "Iteration", i+1)
        
        # NUTS (No-U-Turn Hamiltonian Monte Carlo)
        
        #momentum_0 = np.random.normal(EXP_VALUE_MOMENTUM,STD_DEVIATION_MOMENTUM,lss)
        momentum_0 = np.random.normal(0,1,lss)
        #epsilon = np.random.uniform(epsilon_range[0],epsilon_range[1])
        momentum_old = momentum_0
        
        theta_proposal = samples[0:lss,i]
        theta_proposal_old = theta_proposal
        
        HMC_U_NUTS = HMC_U(observed_data,observe_blackbox_simulation(theta_proposal_old,model_parameters),theta_proposal_old,jnp.asarray(model_parameters['Std_Data']),jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior'])) 
        HMC_K_NUTS = HMC_K(momentum_0,STD_DEVIATION_MOMENTUM)
        Hamiltonian_NUTS = - HMC_U_NUTS - HMC_K_NUTS
        Theta_minus = theta_proposal
        Theta_plus = theta_proposal
        mom_minus = momentum_0
        mom_plus = momentum_0
        logu = float(Hamiltonian_NUTS - np.random.exponential(1,size=1))
        #u = np.random.uniform(0,np.exp(Hamiltonian_NUTS))
        j_NUTS = 0
        n_NUTS = 1
        s_NUTS = 1
        counter_NUTS = 0
        accepted_NUTS = 0
        
        while (s_NUTS == 1.0) and (counter_NUTS < max_NUTS_iterations):
            print('Start while NUTS with counter:',counter_NUTS)
            #v_j = np.random.choice(v_j_set)
            v_j = int(2 * (np.random.uniform() < 0.5) - 1)
            print('NUTS v_j:',v_j)
            if (v_j == -1):
                Theta_minus,mom_minus,grad_minus,_,_,_,theta_proposal,grad_prop,logprob_prop, n_prop,s_prop,alpha_NUTS,n_alpha_NUTS = BuildTree(Theta_minus,mom_minus,logu,v_j,j_NUTS,epsilon_0,theta_proposal_old,momentum_0,grad_posterior_distribution,model_parameters)
                
            else:
                _,_,_,Theta_plus,mom_plus,grad_plus,theta_proposal,grad_prop,logprob_prop, n_prop,s_prop,alpha_NUTS,n_alpha_NUTS = BuildTree(Theta_plus,mom_plus,logu,v_j,j_NUTS,epsilon_0,theta_proposal_old,momentum_0,grad_posterior_distribution,model_parameters)
            #print('s_prop:',s_prop)
            
            _tmp = min(1,float(n_prop)/float(n_NUTS))
            if (s_prop == 1) and (np.random.uniform() < _tmp):
                # Compute simulation output from theta proposal
                samples[0:lss,(i+1)] = theta_proposal
                lnprob[0,(i+1)] = logprob_prop
                
                #uniform_sample = np.random.uniform(0,1) #Random uniform varible to set acceptance level 

                #if uniform_sample < min(1,(n_prop/n_NUTS)):
                    #samples[(j*lss):(j*lss+lss),(i+1)] = theta_proposal
                    #if accepted_NUTS == 0:
                        #accepted_runs += 1
                        #accepted_NUTS = 1
                    #print('accepted')
                #else:
                    #samples[(j*lss):(j*lss+lss),(i+1)] = samples[(j*lss):(j*lss+lss),i]
                    #print('rejected')
            
            n_NUTS += n_prop
            s_NUTS = s_prop and determination_criterion(Theta_minus,Theta_plus,mom_minus,mom_plus)
            j_NUTS += 1
            #n_NUTS = n_NUTS + n_prop
            #s_NUTS = s_prop*(1 if (np.dot((Theta_plus-Theta_minus),mom_minus) >= 0.0) else 0)*(1 if (np.dot((Theta_plus-Theta_minus),mom_plus) >= 0.0) else 0) 
            #j_NUTS = j_NUTS + 1
            counter_NUTS = counter_NUTS + 1
            print('End counter_NUTS',counter_NUTS)
            #print('s_NUTS',s_NUTS)
            #print('counter_NUTS',counter_NUTS)
        
        # Dual averaging to adaptivley change step-size of NUTS
        epsilon_0_iterations[i] = epsilon_0
        counter_NUTS_iterations[i] = counter_NUTS
        
        if timestep_adaptive==True:
            # Dual averaging choosing epsilon
            #print('Start dual averaging')
            eta = 1./float((i+1) + t_0_NUTS)
            H_0_bar = (1. - eta) * H_0_bar + eta * (delta - alpha_NUTS / float(n_alpha_NUTS))
            if (i+1) <= M_adapt:
                epsilon_0 = np.exp(mu_NUTS - np.sqrt((i+1)) /gamma_NUTS * H_0_bar)
                eta = (i+1) ** -kappa_NUTS
                epsilon_0_bar = np.exp((1. - eta) * np.log(epsilon_0_bar) + eta * np.log(epsilon_0))
                #H_0_bar = (1-(1/((i+1)+t_0_NUTS)))* H_0_bar + (1/((i+1)+t_0_NUTS))*(delta - (a_NUTS/na_NUTS))
                #epsilon_NUTS = np.exp(mu_NUTS - (np.sqrt((i+1))/gamma_NUTS) * H_0_bar)
                #epsilon_bar = np.exp(np.power((i+1),-kappa_NUTS)*np.log(epsilon_NUTS) + (1 - np.power((i+1),-kappa_NUTS)) * np.log(epsilon_bar))                                     
            else:
                epsilon_0 = epsilon_0_bar   
            #print('End dual averaging')
        else:
            #epsilon_0 = epsilon_0 + np.random.uniform(epsilon_change[0],epsilon_change[1])
            epsilon_0 = np.random.uniform(epsilon_range[0],epsilon_range[1])
        print('epsilon_0 Update',epsilon_0)
    #print('Samples',samples)
    return samples,accepted_runs,epsilon_0_iterations,counter_NUTS_iterations
# Visualization of the calculated results
def visualization(samples,expected_input_value,std_deviation_input_vaule,expected_observed_data,acceptance_ratio,all_start_samples,epsilon_0_iterations,counter_NUTS_iterations):    
    """
    Plot staistics and comparison of HMC runs
    """
    
    fig, ax = plt.subplots()

    plt.text(0.02, 0.95, 'Exp.-In Len M1= %.3f' % expected_input_value[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.90, 'Stdv.-In Len M1= %.3f' % std_deviation_input_vaule[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.85, 'Exp.-In Len M2= %.3f' % expected_input_value[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.80, 'Stdv.-In Len M2= %.3f' % std_deviation_input_vaule[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.75, 'Out Max M1= %.3f' % expected_observed_data[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.70, 'Out Min M1= %.3f' % expected_observed_data[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.65, 'Out Max M2= %.3f' % expected_observed_data[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.60, 'Out Min M2= %.3f' % expected_observed_data[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.55, 'Obs. Max M1= %.3f' % observed_data[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.50, 'Obs. Min M1= %.3f' % observed_data[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.45, 'Obs. Max M2= %.3f' % observed_data[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.40, 'Obs. Min M2= %.3f' % observed_data[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.35, 'Acc.-Ratio= %.3f' % acceptance_ratio[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.35, 'Start Len M1= %.3f' % all_start_samples[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.30, 'Start Len M2= %.3f' % all_start_samples[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.15, 'Error Count 1= %.3f' % error_counter_1[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.11, 'Error Count 2= %.3f' % error_counter_2[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.07, 'Error Count 3= %.3f' % error_counter_3[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.03, 'Error Count 4= %.3f' % error_counter_4[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)

    plt.title(r'Run results')

    plt.savefig(PATH1, bbox_inches='tight')
    plt.close()
    
    """
    Results for all iterations stress free length of muscle 1
    """
    
    # Plot of sample values in a row (plot the walk)
    iteration_number = np.linspace(0,NUM_DRAWS, num=NUM_DRAWS, endpoint=False)
    
    plt.plot(iteration_number[:(burn_in+1)],samples[0][:(burn_in+1)], color='r', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[0][burn_in:], color='b', linestyle='-', linewidth=2)

    plt.xlabel(r'Iteration')
    plt.ylabel(r'Sample')
    plt.title(r'Stress free length muscle 1')

    plt.savefig(PATH2, bbox_inches='tight')
    plt.close()
    
    """
    Results for all iterations stress free length of muscle 2
    """
    
    # Plot of sample values in a row (plot the walk)
    iteration_number = np.linspace(0,NUM_DRAWS, num=NUM_DRAWS, endpoint=False)
    
    plt.plot(iteration_number[:(burn_in+1)],samples[1][:(burn_in+1)], color='r', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[1][burn_in:], color='b', linestyle='-', linewidth=2)

    plt.xlabel(r'Iteration')
    plt.ylabel(r'Sample')
    plt.title(r'Stress free length muscle 2')

    plt.savefig(PATH3, bbox_inches='tight')
    plt.close()
    
    # Write simulation results in csv-files 
    
    # Open the file in the write mode for calculated samples
    with open('run_samples.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerows(samples)
    
    len1 = len(STD_DEVIATION_PROP_DIST)
    len2 = len(observed_data)
    len3 = NUM_DRAWS-1
    run_observed_data = observed_data
    data_input1 = np.reshape(expected_input_value,(len1,1),order='F')
    data_input2 = np.reshape(std_deviation_input_vaule,(len1,1),order='F')
    data_input3 = np.reshape(all_start_samples,(len1,1),order='F')
    data_input4 = np.reshape(expected_observed_data,(len2,1),order='F')
    data_input5 = np.reshape(run_observed_data,(len2,1),order='F') 
    data_input6 = np.reshape(epsilon_0_iterations,(1,len3),order='F')
    data_input7 = np.reshape(counter_NUTS_iterations,(1,len3),order='F')
        
    #pre_data = [acceptance_ratio]#,error_counter_1,error_counter_2,error_counter_3,error_counter_4]
    
    #run_data = np.concatenate((data_input3,data_input1,data_input2,data_input5,data_input4,pre_data))
    run_data = np.concatenate((data_input3,data_input1,data_input2,data_input5,data_input4))
        
    # Open the file in the write mode for additional data and information about HMC runs
    with open('run_data.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerows(run_data)
        
    # Open the file in the write mode for calculated samples
    with open('NUTS_infos.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerows([epsilon_0_iterations,counter_NUTS_iterations])

# Start Hamiltonian Monte Carlo simulation   
def main():  
    obsle = len(observed_data)
    expected_input_value = np.empty(len(STD_DEVIATION_PRIOR_DIST),dtype=float)
    std_deviation_input_vaule = np.empty(len(STD_DEVIATION_PRIOR_DIST),dtype=float)
    expected_observed_data = np.empty(obsle,dtype=float)
    acceptance_ratio = np.empty(1,dtype=float) 
    all_start_samples = np.empty(len(STD_DEVIATION_PRIOR_DIST),dtype=float)
    
    #grad_posterior_distribution = jit(grad(posterior_hill_type_function_jax))

    # Create start sample without likelihood and prior /= 0.0
    possible_start_sample = [14.0,14.0] # np.multiply([15.0,15.0,7.2],(1-(0.1*j)))
    print("Possible Input:", possible_start_sample)
    
    non_normalized_prior_dist_old = prior_dist_jax(possible_start_sample,jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior']))
    
    grad_posterior_distribution = jit(grad(posterior_hill_type_function_jax))
    
    if non_normalized_prior_dist_old == 0.0 or math.isnan(non_normalized_prior_dist_old):
        calculated_data = [0.0,0.0,0.0,0.0]
        likelihood_dist_exponent_old = 0.0
    else:
        # Compute simulation output from theta start
        calculated_data = observe_blackbox_simulation(possible_start_sample,model_parameters)           
        likelihood_dist_exponent_old = likelihood_dist_exp_jax(observed_data,calculated_data,possible_start_sample,jnp.asarray(model_parameters['Std_Data']))
    
    start_counter = 1.0

    while non_normalized_prior_dist_old == 0.0 or math.isnan(non_normalized_prior_dist_old):
        if start_counter > maximum_start_counter:
            sys.exit("no start sample found") 
        
        # Draw new start sample uniformly distributed between min ans max start boundaries
        possible_start_sample = np.random.uniform(min_sample_boundary, max_sample_boundary) 
        print("Possible Input extra start sample:", possible_start_sample)
        
        non_normalized_prior_dist_old = prior_dist_jax(possible_start_sample,jnp.asarray(model_parameters['Exp_Prior']),jnp.asarray(model_parameters['Std_Prior']))
        
        if non_normalized_prior_dist_old == 0.0 or math.isnan(non_normalized_prior_dist_old):
            calculated_data = [0.0,0.0,0.0,0.0]
            likelihood_dist_exponent_old = 0.0
        else:
            # Compute simulation output from theta start
            calculated_data = observe_blackbox_simulation(possible_start_sample,model_parameters)
            likelihood_dist_exponent_old = likelihood_dist_exp_jax(observed_data,calculated_data,possible_start_sample,jnp.asarray(model_parameters['Std_Data']))
        
        start_counter += 1.0

    # Vector to save sample from each iteration
    start_sample = possible_start_sample
    lss = len(start_sample)
    all_start_samples[0:lss] = start_sample
    
    if len(np.shape(start_sample)) > 1:
        raise ValueError('start sample should be a 1-D array')
    
    # Execute NUTS algorithm to sample HMC results
    samples,accepted_runs,epsilon_0_iterations,counter_NUTS_iterations = NUTS_HMC(start_sample,observed_data,grad_posterior_distribution,NUM_DRAWS,12,model_parameters,timestep_adaptive=False,reasonable_epsilon=False)
    
    # Calculate expected input value and standard deviation without burn-in iterations from HMC samples 
    for i in range(len(start_sample)):        
        expected_input_value[i] = statistics.mean(samples[lss*0+i][burn_in:]) #Expected value for input to get the observed data
        std_deviation_input_vaule[i] = statistics.stdev(samples[lss*0+i][burn_in:]) #Standard deviation of the expected value for input to get the observed data
        #var_input_vaule = statistics.variance(samples[burn_in:]) #Variance of the expected value for input to get the observed data

    # Compute simulation output from expected value    
    calculated_expected_observed_data = observe_blackbox_simulation(expected_input_value[0:lss],model_parameters)
    expected_observed_data[0:obsle] = calculated_expected_observed_data
    
    # Calculations for samples from all accepted iterations        
    #expected_input_value_clean = statistics.mean(samples_clean) #Expected value for input to get the observed data
    #std_deviation_input_vaule_clean = statistics.stdev(samples_clean) #Standard deviation of the expected value for input to get the observed data
    #var_input_vaule = statistics.variance(samples) #Variance of the expected value for input to get the observed data
    
    # Compute simulation output from expected value
    #expected_observed_data_clean = black_box_simulation(expected_input_value_clean)
    
    # Calculate how many of the MCMC draws got accepted
    acceptance_ratio[0] = accepted_runs/NUM_DRAWS 
    
    # Visualize calculated data and write output files
    visualization(samples,expected_input_value,std_deviation_input_vaule,expected_observed_data,acceptance_ratio,all_start_samples,epsilon_0_iterations,counter_NUTS_iterations)
    
    
if __name__ == "__main__":
    main()
    
stop = timeit.default_timer()

print("Time HMC Simulation: ", stop - start) 

