import sys
import subprocess
import shlex
import timeit
import csv
import random
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
from matplotlib import rc
import math
import statistics 
from scipy.stats import lognorm, gamma, t, beta, invgamma
from scipy.signal import find_peaks
from HMC.hmc_constants import FILEPATH_DATA

start = timeit.default_timer()

#define observed data, expected change of muscle length in this case

lobs_m1 = 15.0 # Observed prestretched length of muscle 1 [cm]
lobs_m2 = 15.0 # Observed prestretched length of muscle 2 [cm]
lobs_td = 5.0 # Observed prestretched length of tendon [cm]
extmaxobs_muscle_1 = 18.3 # Maximal extension of muscle 1
extminobs_muscle_1 = 8.7 # Minimal extension of muscle 1
extmaxobs_muscle_2 = 21.00 # Maximal extension of muscle 2
extminobs_muscle_2 = 11.5 # Minimal extension of muscle 2
#rangeobs = 0.7 # Range of motion
#tobs = 5.52 # Time duration if one period of the between maximal and minimal extension of the system

observed_data = [extmaxobs_muscle_1,extminobs_muscle_1,extmaxobs_muscle_2,extminobs_muscle_2]

# Parameters

# Model Parameters

#solver_input = 2 # 1 = explicit Euler, 2 = Heun

ode_solver = 1 # 1 = Thelen, 2 = Van Soest, 3 = Silva, 4 = Hyperelastic

# Model Parameters Mooney 

lslack_muscle_1_input = [9.0,17.0] # Stress-free length of muscle 1
lslack_muscle_2_input = [9.0,17.0] # Stress-free length of muscle 2
#lslack_tendon_input = [4.95,7.0] # Stress-free length of tendon

# Simulation parameters

Tstart_input = 0.0
Tend_input = 6.0

# MCMC parameters

number_runs = 4 #Number of different runs
    
NUM_BINS = 2 #Numbers of blocks in Histogramm
NUM_DRAWS = 1000 #Number of MCMC draws
burn_in = 500 #Number of iterations for the burn-in of the MCMC

#min_sample_boundary = [lslack_muscle_1_input[0],lslack_muscle_2_input[0],lslack_tendon_input[0]]
#max_sample_boundary = [lslack_muscle_1_input[1],lslack_muscle_2_input[1],lslack_tendon_input[1]]
min_sample_boundary = [lslack_muscle_1_input[0],lslack_muscle_2_input[0]]
max_sample_boundary = [lslack_muscle_1_input[1],lslack_muscle_2_input[1]]

maximum_start_counter = 5

#STD_DEVIATION_PROP_DIST = [0.1,0.1,0.01] #Standard deviation of proposal distribution
#STD_DEVIATION_PRIOR_DIST = [1.5,1.5,0.5] #Standard deviation of prior distribution
#EXP_VALUE_PRIOR_DIST = [13.5,13.5,5.5] #Expected value of prior distribution
STD_DEVIATION_PROP_DIST = [0.1,0.1] #Standard deviation of proposal distribution
STD_DEVIATION_PRIOR_DIST = [1.5,1.5] #Standard deviation of prior distribution
EXP_VALUE_PRIOR_DIST = [13.5,13.5] #Expected value of prior distribution
STD_DEVIATION_DATA = [0.2,0.2,0.2,0.2] #Standard deviation of data

#Plot of prior Distribution
ACC_TARGET_DIST = 30001
#LEFT_LIM = [lslack_muscle_1_input[0],lslack_muscle_2_input[0],lslack_tendon_input[0]]
#RIGHT_LIM = [lslack_muscle_1_input[1],lslack_muscle_2_input[1],lslack_tendon_input[1]]
LEFT_LIM = [lslack_muscle_1_input[0],lslack_muscle_2_input[0]]
RIGHT_LIM = [lslack_muscle_1_input[1],lslack_muscle_2_input[1]]

#Plot parameters
rc('font', **{'size':12})#, 'family':'serif', 'serif':['Computer Modern Roman']}) 
rc('text', usetex=True)

# Plot the Simulation Results
PATH1 = 'compare_results.pdf'

PATH3 = 'metropolis_walk_par1.pdf'

PATH6 = 'metropolis_walk_par2.pdf'

PATH9 = 'metropolis_walk_par3.pdf'

#Draw from proposal distribution
def draw_sample(theta):
    return np.random.normal(theta, STD_DEVIATION_PROP_DIST)

#Draw from normalized prior distribution
def normal_prior_dist_prestret_start_length_m1_student(theta):
    if theta > 0.0:
        nu = 1.0
        normal_prior_prob = t.pdf(theta,nu,loc=16.0,scale=1.0) # Student t distribution
    else: 
        normal_prior_prob = 0.0
    return normal_prior_prob

#Draw from normalized prior distribution
def normal_prior_dist_prestret_start_length_m1_normal(theta):
    normal_prior_prob = (1/(STD_DEVIATION_PRIOR_DIST[0]*np.sqrt(2*math.pi)))*np.exp(-0.5*np.power(((theta-EXP_VALUE_PRIOR_DIST[0])/STD_DEVIATION_PRIOR_DIST[0]), 2)) # White noise prior L2
    return normal_prior_prob

#Draw from normalized prior distribution
def normal_prior_dist_prestret_start_length_m2_normal(theta):
    normal_prior_prob = (1/(STD_DEVIATION_PRIOR_DIST[1]*np.sqrt(2*math.pi)))*np.exp(-0.5*np.power(((theta-EXP_VALUE_PRIOR_DIST[1])/STD_DEVIATION_PRIOR_DIST[1]), 2)) # White noise prior L2
    return normal_prior_prob

#Draw from normalized prior distribution
#def normal_prior_dist_prestret_start_length_td_normal(theta):
#    normal_prior_prob = (1/(STD_DEVIATION_PRIOR_DIST[2]*np.sqrt(2*math.pi)))*np.exp(-0.5*np.power(((theta-EXP_VALUE_PRIOR_DIST[2])/STD_DEVIATION_PRIOR_DIST[2]), 2)) # White noise prior L2
#    return normal_prior_prob

#Draw from normalized prior distribution
def normal_prior_dist_m1_invgamma(theta):
    a_invgamma = 69.88
    b_invgamma = 831.87
    normal_prior_prob = invgamma.pdf(theta/b_invgamma,a_invgamma)/b_invgamma
    return normal_prior_prob

#Draw from normalized prior distribution
def normal_prior_dist_m2_invgamma(theta):
    a_invgamma = 69.88
    b_invgamma = 831.87
    normal_prior_prob = invgamma.pdf(theta/b_invgamma,a_invgamma)/b_invgamma
    return normal_prior_prob

#Draw from normalized prior distribution
#def normal_prior_dist_td_invgamma(theta):
#    a_invgamma = 1622.91
#    b_invgamma = 8418.68
#    normal_prior_prob = invgamma.pdf(theta/b_invgamma,a_invgamma)/b_invgamma
#    return normal_prior_prob

#Draw from normalized prior distribution
def normal_prior_dist_start_length_m1_beta(theta):
    location = 0.0
    sca = lobs_m1
    alpha_var = 70.0
    beta_var = 15.0
    theta_true = (theta-location)/sca
    if theta_true >= 0.0 and theta_true <= 1.0:
        normal_prior_prob = beta.pdf(theta_true,alpha_var,beta_var) # Beta distribution
    else:
        normal_prior_prob = 0.0
    return normal_prior_prob

#Draw from normalized prior distribution
def normal_prior_dist_start_length_m2_beta(theta):
    location = 0.0
    sca = lobs_m2
    alpha_var = 70.0
    beta_var = 15.0
    theta_true = (theta-location)/sca
    if theta_true >= 0.0 and theta_true <= 1.0:
        normal_prior_prob = beta.pdf(theta_true,alpha_var,beta_var) # Beta distribution
    else:
        normal_prior_prob = 0.0
    return normal_prior_prob

#Draw from normalized prior distribution
#def normal_prior_dist_start_length_td_beta(theta):
#    location = 4.8
#    sca = lobs_td
#    alpha_var = 5.0
#    beta_var = 70.0
#    theta_true = (theta-location)/sca
#    if theta_true >= 0.0 and theta_true <= 1.0:
#        normal_prior_prob = beta.pdf(theta_true,alpha_var,beta_var) # Beta distribution
#    else:
#        normal_prior_prob = 0.0
#    return normal_prior_prob

#Draw from normalized prior distribution
def normal_prior_dist_m1_gamma(theta):
    if theta > 0.0:
        alpha = 20.0
        normal_prior_prob = gamma.pdf(theta,alpha,loc=0.0,scale=1.0) # Gamma distribution
    else:
        normal_prior_prob = 0.0
    return normal_prior_prob

#Draw from normalized prior distribution
def normal_prior_dist_m2_gamma(theta):
    if theta > 0.0:
        alpha = 20.0
        normal_prior_prob = gamma.pdf(theta,alpha,loc=0.0,scale=1.0) # Gamma distribution
    else:
        normal_prior_prob = 0.0
    return normal_prior_prob

#Draw from prior distribution
def prior_dist(theta):
    #prior_prob = (normal_prior_dist_prestret_start_length_m1_normal(theta[0])) * (normal_prior_dist_prestret_start_length_m2_normal(theta[1])) * (normal_prior_dist_prestret_start_length_td_normal(theta[2]))
    
    #prior_prob = (normal_prior_dist_m1_invgamma(theta[0])) * (normal_prior_dist_m2_invgamma(theta[1])) * (normal_prior_dist_td_invgamma(theta[2]))
    prior_prob = (normal_prior_dist_m1_invgamma(theta[0])) * (normal_prior_dist_m2_invgamma(theta[1]))
    return prior_prob 

#Calculate likelihood     
def likelihood_dist(obs,calc,theta):
    #print("The type is : ", type(calc))
    #print("The type is : ", type(obs))
    #print("The type is : ", type(STD_DEVIATION_PROP_DIST))
    likelihood_prob = np.exp(-0.5*np.sum(np.power(((np.array(calc)-np.array(obs))/np.array(STD_DEVIATION_DATA)), 2))) 
    return likelihood_prob 

#Calculate exponent of likelihood     
def likelihood_dist_exponent(obs,calc,theta):
    likelihood_exponent = -0.5*np.sum(np.power(((np.array(calc)-np.array(obs))/np.array(STD_DEVIATION_DATA)), 2)) 
    return likelihood_exponent

#Draw from prosterior distribution
def non_normalized_posterior_dist(obs,calc,theta):
    posterior_prob = likelihood_dist(obs,calc,theta)*prior_dist(theta)
    return posterior_prob 

def black_box_simulation(input_value):
    
    sim_process_input = shlex.split(f"python3 Hill_System_MCMC_Py.py --lslack_muscle1={input_value[0]} --lslack_muscle2={input_value[1]} --lslack_tendon={lobs_td} --Tstart={Tstart_input} --Tend={Tend_input} --ode_solver={ode_solver} --lobs_muscle1={lobs_m1} --lobs_muscle2={lobs_m2} --lobs_tendon={lobs_td}")

    subprocess.run(sim_process_input)
        
    with open('length_muscle_1.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            len_muscle_1 = row

    with open('length_muscle_2.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            len_muscle_2 = row
            
    #with open('length_tendon.csv', newline='') as f:
        #reader = csv.reader(f)
        #for row in reader:
            #len_tendon = row

    output_data_1 = np.array(len_muscle_1, dtype = 'float')
    output_data_2 = np.array(len_muscle_2, dtype = 'float')
    #output_data_3 = np.array(len_tendon, dtype = 'float')
    output_data = [output_data_1,output_data_2]
    return output_data

#Visualization of the calculated results
def visualization(samples,expected_input_value,std_deviation_input_vaule,expected_observed_data,acceptance_ratio,all_start_samples,error_counter_1,error_counter_2,error_counter_3,error_counter_4):
    
    fig, ax = plt.subplots()

    plt.text(0.02, 0.95, 'Exp.-In Len M1= %.3f' % expected_input_value[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.90, 'Stdv.-In Len M1= %.3f' % std_deviation_input_vaule[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.85, 'Exp.-In Len M2= %.3f' % expected_input_value[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.80, 'Stdv.-In Len M2= %.3f' % std_deviation_input_vaule[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.75, 'Exp.-In Len Td= %.3f' % expected_input_value[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.70, 'Stdv.-In Len Td= %.3f' % std_deviation_input_vaule[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.65, 'Out Max M1= %.3f' % expected_observed_data[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.60, 'Out Min M1= %.3f' % expected_observed_data[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.55, 'Out Max M2= %.3f' % expected_observed_data[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.50, 'Out Min M2= %.3f' % expected_observed_data[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.45, 'Obs. Max M1= %.3f' % observed_data[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.40, 'Obs. Min M1= %.3f' % observed_data[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.35, 'Obs. Max M2= %.3f' % observed_data[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.30, 'Obs. Min M2= %.3f' % observed_data[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.25, 'Acc.-Ratio= %.3f' % acceptance_ratio[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.20, 'Start Len M1= %.3f' % all_start_samples[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.02, 0.15, 'Start Len M2= %.3f' % all_start_samples[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.10, 'Start Len Td= %.3f' % all_start_samples[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.15, 'Error Count 1= %.3f' % error_counter_1[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.11, 'Error Count 2= %.3f' % error_counter_2[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.07, 'Error Count 3= %.3f' % error_counter_3[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.02, 0.03, 'Error Count 4= %.3f' % error_counter_4[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    
    plt.text(0.26, 0.95, 'Exp.-In Len M1= %.3f' % expected_input_value[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.90, 'Stdv.-In Len M1= %.3f' % std_deviation_input_vaule[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.85, 'Exp.-In Len M2= %.3f' % expected_input_value[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.80, 'Stdv.-In Len M2= %.3f' % std_deviation_input_vaule[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.26, 0.75, 'Exp.-In Len Td= %.3f' % expected_input_value[5], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.26, 0.70, 'Stdv.-In Len Td= %.3f' % std_deviation_input_vaule[5], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.65, 'Out Max M1= %.3f' % expected_observed_data[4], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.60, 'Out Min M1= %.3f' % expected_observed_data[5], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.55, 'Out Max M2= %.3f' % expected_observed_data[6], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.50, 'Out Min M2= %.3f' % expected_observed_data[7], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.45, 'Obs. Max M1= %.3f' % observed_data[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.40, 'Obs. Min M1= %.3f' % observed_data[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.35, 'Obs. Max M2= %.3f' % observed_data[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.30, 'Obs. Min M2= %.3f' % observed_data[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.25, 'Acc.-Ratio= %.3f' % acceptance_ratio[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.20, 'Start Len M1= %.3f' % all_start_samples[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.26, 0.15, 'Start Len M2= %.3f' % all_start_samples[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.26, 0.10, 'Start Len Td= %.3f' % all_start_samples[5], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.26, 0.15, 'Error Count 1= %.3f' % error_counter_1[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.26, 0.11, 'Error Count 2= %.3f' % error_counter_2[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.26, 0.07, 'Error Count 3= %.3f' % error_counter_3[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.26, 0.03, 'Error Count 4= %.3f' % error_counter_4[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    
    plt.text(0.50, 0.95, 'Exp.-In Len M1= %.3f' % expected_input_value[4], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.90, 'Stdv.-In Len M1= %.3f' % std_deviation_input_vaule[4], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.85, 'Exp.-In Len M2= %.3f' % expected_input_value[5], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.80, 'Stdv.-In Len M2= %.3f' % std_deviation_input_vaule[5], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.50, 0.75, 'Exp.-In Len Td= %.3f' % expected_input_value[8], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.50, 0.70, 'Stdv.-In Len Td= %.3f' % std_deviation_input_vaule[8], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.65, 'Out Max M1= %.3f' % expected_observed_data[8], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.60, 'Out Min M1= %.3f' % expected_observed_data[9], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.55, 'Out Max M2= %.3f' % expected_observed_data[10], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.50, 'Out Min M2= %.3f' % expected_observed_data[11], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.45, 'Obs. Max M1= %.3f' % observed_data[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.40, 'Obs. Min M1= %.3f' % observed_data[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.35, 'Obs. Max M2= %.3f' % observed_data[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.30, 'Obs. Min M2= %.3f' % observed_data[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.25, 'Acc.-Ratio= %.3f' % acceptance_ratio[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.20, 'Start Len M1= %.3f' % all_start_samples[4], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.50, 0.15, 'Start Len M2= %.3f' % all_start_samples[5], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.50, 0.10, 'Start Len Td= %.3f' % all_start_samples[8], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.50, 0.15, 'Error Count 1= %.3f' % error_counter_1[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.50, 0.11, 'Error Count 2= %.3f' % error_counter_2[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.50, 0.07, 'Error Count 3= %.3f' % error_counter_3[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.50, 0.03, 'Error Count 4= %.3f' % error_counter_4[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    
    plt.text(0.74, 0.95, 'Exp.-In Len M1= %.3f' % expected_input_value[6], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.90, 'Stdv.-In Len M1= %.3f' % std_deviation_input_vaule[6], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.85, 'Exp.-In Len M2= %.3f' % expected_input_value[7], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.80, 'Stdv.-In Len M2= %.3f' % std_deviation_input_vaule[7], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.74, 0.75, 'Exp.-In Len Td= %.3f' % expected_input_value[11], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.74, 0.70, 'Stdv.-In Len Td= %.3f' % std_deviation_input_vaule[11], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.65, 'Out Max M1= %.3f' % expected_observed_data[12], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.60, 'Out Min M1= %.3f' % expected_observed_data[13], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.55, 'Out Max M2= %.3f' % expected_observed_data[14], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.50, 'Out Min M2= %.3f' % expected_observed_data[15], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.45, 'Obs. Max M1= %.3f' % observed_data[0], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.40, 'Obs. Min M1= %.3f' % observed_data[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.35, 'Obs. Max M2= %.3f' % observed_data[2], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.30, 'Obs. Min M2= %.3f' % observed_data[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.25, 'Acc.-Ratio= %.3f' % acceptance_ratio[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.20, 'Start Len M1= %.3f' % all_start_samples[6], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.text(0.74, 0.15, 'Start Len M2= %.3f' % all_start_samples[7], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.74, 0.10, 'Start Len Td= %.3f' % all_start_samples[11], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.74, 0.15, 'Error Count 1= %.3f' % error_counter_1[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.74, 0.11, 'Error Count 2= %.3f' % error_counter_2[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.74, 0.07, 'Error Count 3= %.3f' % error_counter_3[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    #plt.text(0.74, 0.03, 'Error Count 4= %.3f' % error_counter_4[3], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)

    plt.title(r'Compare run results')

    plt.savefig(PATH1, bbox_inches='tight')
    plt.close()
    
    """
    Results for all iterations start length stress free muscle 1
    """
    
    #Plot of sample values in a row (plot the walk)
    iteration_number = np.linspace(0,NUM_DRAWS, num=NUM_DRAWS, endpoint=False)
    
    #plt.plot(iteration_number,samples, color='k', linestyle='-', linewidth=2)
    
    plt.plot(iteration_number[:(burn_in+1)],samples[0][:(burn_in+1)], color='r', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[0][burn_in:], color='b', linestyle='-', linewidth=2)
    plt.plot(iteration_number[:(burn_in+1)],samples[2][:(burn_in+1)], color='chocolate', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[2][burn_in:], color='purple', linestyle='-', linewidth=2)
    plt.plot(iteration_number[:(burn_in+1)],samples[4][:(burn_in+1)], color='sandybrown', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[4][burn_in:], color='fuchsia', linestyle='-', linewidth=2)
    plt.plot(iteration_number[:(burn_in+1)],samples[6][:(burn_in+1)], color='saddlebrown', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[6][burn_in:], color='indigo', linestyle='-', linewidth=2)

    plt.xlabel(r'Iteration')
    plt.ylabel(r'Sample')
    plt.title(r'Stress free length muscle 1')

    plt.savefig(PATH3, bbox_inches='tight')
    plt.close()
    
    """
    Results for all iterations start length stress free muscle 2
    """
    
    #Plot of sample values in a row (plot the walk)
    iteration_number = np.linspace(0,NUM_DRAWS, num=NUM_DRAWS, endpoint=False)
    
    #plt.plot(iteration_number,samples, color='k', linestyle='-', linewidth=2)
    
    plt.plot(iteration_number[:(burn_in+1)],samples[1][:(burn_in+1)], color='r', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[1][burn_in:], color='b', linestyle='-', linewidth=2)
    plt.plot(iteration_number[:(burn_in+1)],samples[3][:(burn_in+1)], color='chocolate', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[3][burn_in:], color='purple', linestyle='-', linewidth=2)
    plt.plot(iteration_number[:(burn_in+1)],samples[5][:(burn_in+1)], color='sandybrown', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[5][burn_in:], color='fuchsia', linestyle='-', linewidth=2)
    plt.plot(iteration_number[:(burn_in+1)],samples[7][:(burn_in+1)], color='saddlebrown', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[7][burn_in:], color='indigo', linestyle='-', linewidth=2)

    plt.xlabel(r'Iteration')
    plt.ylabel(r'Sample')
    plt.title(r'Stress free length muscle 2')

    plt.savefig(PATH6, bbox_inches='tight')
    plt.close()
    
    
    """
    Results for all iterations start length stress free tendon
    """
    
    """
    #Plot of sample values in a row (plot the walk)
    iteration_number = np.linspace(0,NUM_DRAWS, num=NUM_DRAWS, endpoint=False)
    
    #plt.plot(iteration_number,samples, color='k', linestyle='-', linewidth=2)
    
    plt.plot(iteration_number[:(burn_in+1)],samples[2][:(burn_in+1)], color='r', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[2][burn_in:], color='b', linestyle='-', linewidth=2)
    plt.plot(iteration_number[:(burn_in+1)],samples[5][:(burn_in+1)], color='chocolate', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[5][burn_in:], color='purple', linestyle='-', linewidth=2)
    plt.plot(iteration_number[:(burn_in+1)],samples[8][:(burn_in+1)], color='sandybrown', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[8][burn_in:], color='fuchsia', linestyle='-', linewidth=2)
    plt.plot(iteration_number[:(burn_in+1)],samples[11][:(burn_in+1)], color='saddlebrown', linestyle='-', linewidth=2)
    plt.plot(iteration_number[burn_in:],samples[11][burn_in:], color='indigo', linestyle='-', linewidth=2)

    plt.xlabel(r'Iteration')
    plt.ylabel(r'Sample')
    plt.title(r'Stress free length tendon')

    plt.savefig(PATH9, bbox_inches='tight')
    plt.close()
    """
    
    # open the file in the write mode
    with open(f'{FILEPATH_DATA}/run_samples.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerows(samples)
    
    len1 = len(STD_DEVIATION_PROP_DIST)
    len2 = len(observed_data)
    run_observed_data = number_runs * observed_data
    data_input1 = np.reshape(expected_input_value,(len1,number_runs),order='F')
    data_input2 = np.reshape(std_deviation_input_vaule,(len1,number_runs),order='F')
    data_input3 = np.reshape(all_start_samples,(len1,number_runs),order='F')
    data_input4 = np.reshape(expected_observed_data,(len2,number_runs),order='F')
    data_input5 = np.reshape(run_observed_data,(len2,number_runs),order='F')    
        
    pre_data = [acceptance_ratio,error_counter_1,error_counter_2,error_counter_3,error_counter_4]
    
    run_data = np.concatenate((data_input3,data_input1,data_input2,data_input5,data_input4,pre_data))
        
    # open the file in the write mode
    with open(f'{FILEPATH_DATA}/run_data.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerows(run_data)

def observe_blackbox_simulation(sample_input):
    #Compute simulation output from theta start
    calculated_data_m1_raw, calculated_data_m2_raw = black_box_simulation(sample_input)        

    value_maximal_length_m1 = np.amax(calculated_data_m1_raw)

    value_minimal_length_m1 = np.amin(calculated_data_m1_raw)
    
    value_maximal_length_m2 = np.amax(calculated_data_m2_raw)

    value_minimal_length_m2 = np.amin(calculated_data_m2_raw)

    calculated_data_blackbox = [value_maximal_length_m1,value_minimal_length_m1,value_maximal_length_m2,value_minimal_length_m2]
    print("Output:", calculated_data_blackbox)
    
    return calculated_data_blackbox
    
def main():
    
    lss = len(STD_DEVIATION_PROP_DIST)
    obsle = len(observed_data)
        
    samples = np.zeros((len(STD_DEVIATION_PROP_DIST)*number_runs,NUM_DRAWS))    
    
    expected_input_value = np.zeros(len(STD_DEVIATION_PROP_DIST)*number_runs)
    std_deviation_input_vaule = np.zeros(len(STD_DEVIATION_PROP_DIST)*number_runs)
    expected_observed_data = np.zeros(obsle*number_runs)
    acceptance_ratio = np.zeros(number_runs) 
    error_counter_1 = np.zeros(number_runs)
    error_counter_2 = np.zeros(number_runs)
    error_counter_3 = np.zeros(number_runs)
    error_counter_4 = np.zeros(number_runs)
    all_start_samples = np.zeros(len(STD_DEVIATION_PROP_DIST)*number_runs)
    
    for j in range(0,number_runs):
        #Create start sample without likelihood and prior /= 0.0
        possible_start_sample = [15.5*(1-(0.1*j)),15.5*(1-(0.1*j))] # np.multiply([15.0,15.0,7.2],(1-(0.1*j)))
        print("Possible Input:", possible_start_sample)
        
        #non_normalized_posterior_dist_old = non_normalized_posterior_dist(observed_data,calculated_data,samples[(j*lss):(j*lss+3),0])
        
        non_normalized_prior_dist_old = prior_dist(possible_start_sample)
        
        if non_normalized_prior_dist_old == 0.0 or math.isnan(non_normalized_prior_dist_old):
            calculated_data = [0.0,0.0,0.0,0.0]
            likelihood_dist_exponent_old = 0.0
        else:
            #Compute simulation output from theta start
            calculated_data = observe_blackbox_simulation(possible_start_sample)
            
            likelihood_dist_exponent_old = likelihood_dist_exponent(observed_data,calculated_data,possible_start_sample)
            #likelihood_dist_old = np.exp(likelihood_dist_exponent_old)
        
        start_counter = 1.0

        while non_normalized_prior_dist_old == 0.0 or math.isnan(non_normalized_prior_dist_old):

            if start_counter > maximum_start_counter:
                sys.exit("no start sample found") 
            
            #draw new start sample uniformly distributed between min ans max start boundaries
            possible_start_sample = np.random.uniform(min_sample_boundary, max_sample_boundary) 
            print("Possible Input extra start sample:", possible_start_sample)
            
            non_normalized_prior_dist_old = prior_dist(possible_start_sample)
            
            if non_normalized_prior_dist_old == 0.0 or math.isnan(non_normalized_prior_dist_old):
                calculated_data = [0.0,0.0,0.0,0.0]
                likelihood_dist_exponent_old = 0.0
            else:
                #Compute simulation output from theta start
                calculated_data = observe_blackbox_simulation(possible_start_sample)
                
                likelihood_dist_exponent_old = likelihood_dist_exponent(observed_data,calculated_data,possible_start_sample)
                #likelihood_dist_old = np.exp(likelihood_dist_exponent_old)
            
            start_counter += 1.0
    
        #Vector to save sample from each iteration
        start_sample = possible_start_sample
        all_start_samples[(j*lss):(j*lss+lss)] = start_sample
        samples[(j*lss):(j*lss+lss),0] = start_sample 
        
        #Vector to save samples just from accepted iterations
        #samples_clean = [start_sample]
        accepted_runs = 1

        #Compute simulation output from theta start
        #calculated_data = black_box_simulation(samples[(j*lss):(j*lss+3),0])
            
        #non_normalized_posterior_dist_old = non_normalized_posterior_dist(observed_data,calculated_data,samples[(j*lss):(j*lss+3),0])
        
        #Start MCMC iteration
        for i in range(0, (NUM_DRAWS-1)):

            theta_proposal = draw_sample(samples[(j*lss):(j*lss+lss),i]) #Proposed sample 
            print("Input Proposal:", theta_proposal)
            #Compute simulation output from theta proposal
            
            non_normalized_prior_dist_prop = prior_dist(theta_proposal)
            if non_normalized_prior_dist_prop == 0.0 or math.isnan(non_normalized_prior_dist_prop):
                calculated_data = [0.0,0.0,0.0,0.0]
                likelihood_dist_exponent_prop = 0.0
                likelihood_dist_prop_old = 0.0
            else:
                calculated_data = observe_blackbox_simulation(theta_proposal)

                #non_normalized_posterior_dist_prop = non_normalized_posterior_dist(observed_data,calculated_data,theta_proposal)
                
                likelihood_dist_exponent_prop = likelihood_dist_exponent(observed_data,calculated_data,theta_proposal)
                likelihood_dist_prop_old = np.exp(likelihood_dist_exponent_prop-likelihood_dist_exponent_old)

            #Test if proposed sample is accepted or not
            if likelihood_dist_prop_old > 0.0 and non_normalized_prior_dist_prop > 0.0 and math.isnan(likelihood_dist_prop_old) == False and math.isnan(non_normalized_prior_dist_prop) == False:
                #acceptance_rate = min(1, ((non_normalized_posterior_dist_prop)/(non_normalized_posterior_dist_old)))
                acceptance_rate = min(1, (likelihood_dist_prop_old*(non_normalized_prior_dist_prop)/(non_normalized_prior_dist_old)))
            else:
                acceptance_rate = 0.0

            uniform_sample = np.random.uniform(0,1) #Random uniform varible to set acceptance level

            if uniform_sample < acceptance_rate:
                samples[(j*lss):(j*lss+lss),(i+1)] = theta_proposal
                #samples_clean.append(theta_proposal)
                #non_normalized_posterior_dist_old = non_normalized_posterior_dist_prop
                non_normalized_prior_dist_old = non_normalized_prior_dist_prop
                likelihood_dist_exponent_old = likelihood_dist_exponent_prop
                accepted_runs += 1
            else:
                samples[(j*lss):(j*lss+lss),(i+1)] = samples[(j*lss):(j*lss+lss),i]
                if acceptance_rate == 0.0:
                    #STD_DEVIATION_PROP_DIST = np.multiply(STD_DEVIATION_PROP_DIST,1.1)
                    if non_normalized_prior_dist_prop == 0.0:
                        error_counter_2[j] += 1
                    elif math.isnan(non_normalized_prior_dist_prop) == True:
                        error_counter_4[j] += 1
                    elif math.isnan(likelihood_dist_prop_old) == True:
                        error_counter_3[j] += 1
                    elif likelihood_dist_prop_old == 0.0:
                        error_counter_1[j] += 1

            print("Run: ", j+1 , "Iteration", i+1)
        
        #Calculations for samples from all iterations
        
        #expected_input_value = statistics.mean(samples) #Expected value for input to get the observed data
        #std_deviation_input_vaule = statistics.stdev(samples) #Standard deviation of the expected value for input to get the observed data
        #var_input_vaule = statistics.variance(samples) #Variance of the expected value for input to get the observed data
        
        #Without burn-in Iterations
        
        for i in range(len(start_sample)):
        
            expected_input_value[lss*j+i] = statistics.mean(samples[lss*j+i][burn_in:]) #Expected value for input to get the observed data
            std_deviation_input_vaule[lss*j+i] = statistics.stdev(samples[lss*j+i][burn_in:]) #Standard deviation of the expected value for input to get the observed data
            #var_input_vaule = statistics.variance(samples[burn_in:]) #Variance of the expected value for input to get the observed data
        
        calculated_expected_observed_data = observe_blackbox_simulation(expected_input_value[(j*lss):(j*lss+lss)])
        
        #Compute simulation output from expected value
        expected_observed_data[(j*obsle):(j*obsle+obsle)] = calculated_expected_observed_data
        
        #Calculations for samples from all accepted iterations
        
        #expected_input_value_clean = statistics.mean(samples_clean) #Expected value for input to get the observed data
        #std_deviation_input_vaule_clean = statistics.stdev(samples_clean) #Standard deviation of the expected value for input to get the observed data
        #var_input_vaule = statistics.variance(samples) #Variance of the expected value for input to get the observed data
        
        #Compute simuulation output from expected value
        #expected_observed_data_clean = black_box_simulation(expected_input_value_clean)
        
        #Calculate how many of the MCMC draws got accepted
        acceptance_ratio[j] = accepted_runs/NUM_DRAWS 
        
        #print(expected_input_value)
        #print(std_deviation_input_vaule)
        #print(var_input_vaule) 
        #print(acceptance_ratio)
        #print(expected_observed_data)
    
    #Visualize calculated data
    visualization(samples,expected_input_value,std_deviation_input_vaule,expected_observed_data,acceptance_ratio,all_start_samples,error_counter_1,error_counter_2,error_counter_3,error_counter_4)
    
    
if __name__ == "__main__":
    main()
    
stop = timeit.default_timer()

print("Time MCMC Simulation: ", stop - start) 
