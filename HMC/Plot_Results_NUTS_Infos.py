import sys
import csv
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
from matplotlib import rc
from matplotlib import cm
import math 
import scipy.stats as st
import statistics

from hmc_constants import FILEPATH_DATA
 
with open(f'{FILEPATH_DATA}/NUTS_infos.csv', newline='') as f:
    reader = csv.reader(f)
    infos = [np.array(row, dtype = 'float') for row in reader]

## open the file in the write mode
#with open(f'{FILEPATH_DATA}/run_samples.csv', 'w', encoding='UTF8', newline='') as f:
    ## create the csv writer
    #writer = csv.writer(f)
    ## write a row to the csv file
    #writer.writerows(samples)
    
NUM_BINS = 40 #Numbers of blocks in Histogramm
burn_in = 20 #Number of iterations for the burn-in of the HMC

len1 = len(infos[0][:])
len2 = len(infos[1][:])

iteration_number_1 = np.linspace(0,len1, num=len1, endpoint=False)
iteration_number_2 = np.linspace(0,len2, num=len2, endpoint=False)

#plt.hist(x, bins=bins);

#Plot parameters
#rc('font', **{'size':12})#, 'family':'serif', 'serif':['Computer Modern Roman']}) 
#rc('text', usetex=True)

# Plot the Simulation Results
PATH1 = f'{FILEPATH_PLOTS}/NUTS_Steps.pdf'
PATH2 = f'{FILEPATH_PLOTS}/Epsilon_0_Adaptive.pdf'


# Funktionsplot
# Scatter plot detailed informations
fig, ax =plt.subplots(figsize=(6,6))
ax.set_title('NUTS Steps per Iteration',color="black",fontsize=14)
#plt.plot(samples[0][burn_in:], samples[1][burn_in:], 'xb-')
#plt.plot(samples[0][burn_in:], samples[1][burn_in:],zorder=1,color="mediumpurple",linewidth="0.5") 
#ax.plot(iteration_number_1[:],infos[1][:],zorder=1, color='r', linestyle='-', linewidth=1)
ax.scatter(iteration_number_1[:], infos[1][:],zorder=2,color="blue",marker="x",linewidths=0.5,alpha=0.5,s=10)
ax.set_xlabel(r'Iteration',color="black",fontsize=12)
ax.set_ylabel(r'#(NUTS Steps)',color="black",fontsize=12)#,rotation=True)
#ax.legend(loc='best')
plt.savefig(PATH1, bbox_inches='tight')
plt.close()

# Scatter plot detailed informations
fig, ax =plt.subplots(figsize=(6,6))
ax.set_title('Adaptive change of HMC Step-Size',color="black",fontsize=14)
#plt.plot(samples[0][burn_in:], samples[1][burn_in:], 'xb-')
#plt.plot(samples[0][burn_in:], samples[1][burn_in:],zorder=1,color="mediumpurple",linewidth="0.5") 
#ax.plot(iteration_number_2[:],infos[0][:],zorder=1, color='r', linestyle='-', linewidth=1)
ax.scatter(iteration_number_2[:], infos[0][:],zorder=2,color="blue",marker="x",linewidths=0.5,alpha=0.5,s=10)
ax.set_xlabel(r'Iteration',color="black",fontsize=12)
ax.set_ylabel(r'Step-Size $\epsilon_0$',color="black",fontsize=12)#,rotation=True)
#ax.legend(loc='best')
plt.savefig(PATH2, bbox_inches='tight')
plt.close()
