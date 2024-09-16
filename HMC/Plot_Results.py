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

 
with open('run_samples.csv', newline='') as f:
    reader = csv.reader(f)
    samples = [np.array(row, dtype = 'float') for row in reader]

## open the file in the write mode
#with open('run_samples.csv', 'w', encoding='UTF8', newline='') as f:
    ## create the csv writer
    #writer = csv.writer(f)
    ## write a row to the csv file
    #writer.writerows(samples)
    
NUM_BINS = 40 #Numbers of blocks in Histogramm
BURN_IN = 1 #Number of iterations for the burn-in of the HMC

exact_input_value = [12.5,14.5]

expected_input_value = np.empty(len(samples),dtype=float)
std_deviation_input_vaule = np.empty(len(samples),dtype=float)

# Informations about parameters
for i in range(len(samples)):    
    expected_input_value[i] = statistics.mean(samples[i][BURN_IN:]) #Expected value for input to get the observed data
    std_deviation_input_vaule[i] = statistics.stdev(samples[i][BURN_IN:]) #Standard deviation of the expected value for input to get the observed data

print('Expected Input values',expected_input_value)
print('Standard deviation Input values',std_deviation_input_vaule)
print('Numpy correlation',np.corrcoef(samples[0][BURN_IN:],samples[1][BURN_IN:]))
print('Scipy Pearson',st.pearsonr(samples[0][BURN_IN:],samples[1][BURN_IN:])[0])
print('Scipy Spearmanr',st.spearmanr(samples[0][BURN_IN:],samples[1][BURN_IN:])[0])
print('Scipy Kendalltau',st.kendalltau(samples[0][BURN_IN:],samples[1][BURN_IN:])[0])

#plt.hist(x, bins=bins);

#Plot parameters
#rc('font', **{'size':12})#, 'family':'serif', 'serif':['Computer Modern Roman']}) 
#rc('text', usetex=True)

# Plot the Simulation Results
PATH1 = 'plots/Histogramm_M1_Result.pdf'
PATH2 = 'plots/Histogramm_M2_Result.pdf'
PATH3 = 'plots/Joint_Plot_M1_M2_Result.pdf'
PATH4 = 'plots/Scatter_Plot_M1_M2_Sample_Generation_Result.pdf'
PATH6 = 'plots/Scatter_Plot_M1_M2_More_Infos_Result.pdf'
PATH5 = 'plots/Histogramm_3D_Plot_M1_M2_Result.pdf'
PATH7 = 'plots/HMC_walk_M1.pdf'
PATH8 = 'plots/HMC_walk_M2.pdf'

# Funktionsplot
x = samples[0][BURN_IN:]

q25, q75 = np.percentile(x, [25, 75])
bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
bins_M1 = round((x.max() - x.min()) / bin_width)
print("Freedman–Diaconis number of bins:", bins_M1)

fig, ax =plt.subplots(figsize=(6,6))
#ax.set_title('Stress free length muscle 1',color="black",fontsize=14)
ax.hist(x, bins=bins_M1, color='magenta', histtype='bar', ec='black', density=True, label="Data")
mn, mx = plt.xlim()
plt.xlim(mn, mx)
kde_xs = np.linspace(mn, mx, 300)
kde = st.gaussian_kde(x)
plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
#ax.legend(loc="upper right")
#ax.set_xlabel(r'Drawn samples (Muscle length [$cm$])',color="black",fontsize=12)
ax.set_xlabel(r'Stress-free length muscle 1 [$cm$]',color="black",fontsize=12)
ax.set_ylabel(r'Bin counts',color="black",fontsize=12)#,rotation=True)
#ax.set_ylabel(r'length [$cm$]',color="black",fontsize=12)#,rotation=True)
ax.legend(loc='best')
#ax.set_xticks(np.arange(0,11,1))
#ax.set_yticks(np.arange(0,110,10))
#ax.grid(True)
#plt.show()
#plt.style.use('ggplot')
plt.savefig(PATH1, bbox_inches='tight')
plt.close()

# Funktionsplot
x = samples[1][BURN_IN:]

q25, q75 = np.percentile(x, [25, 75])
bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
bins_M2 = round((x.max() - x.min()) / bin_width)
print("Freedman–Diaconis number of bins:", bins_M2)

fig, ax =plt.subplots(figsize=(6,6))
#ax.set_title('Stress free length muscle 2',color="black",fontsize=14)
ax.hist(x, bins=bins_M2, color='magenta', histtype='bar', ec='black', density=True, label="Data")
mn, mx = plt.xlim()
plt.xlim(mn, mx)
kde_xs = np.linspace(mn, mx, 300)
kde = st.gaussian_kde(x)
plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
#ax.legend(loc="upper right")
#ax.set_xlabel(r'Drawn samples (Muscle length [$cm$])',color="black",fontsize=12)
ax.set_xlabel(r'Stress-free length muscle 2 [$cm$]',color="black",fontsize=12)
ax.set_ylabel(r'Bin counts',color="black",fontsize=12)#,rotation=True)
#ax.set_ylabel(r'length [$cm$]',color="black",fontsize=12)#,rotation=True)
ax.legend(loc='best')
#ax.set_xticks(np.arange(0,11,1))
#ax.set_yticks(np.arange(0,110,10))
#ax.grid(True)
#plt.show()
#plt.style.use('ggplot')
plt.savefig(PATH2, bbox_inches='tight')
plt.close()

# Joint plot of stress free length and prestretch traction
g = sns.jointplot(x=samples[0][BURN_IN:],y=samples[1][BURN_IN:])
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.rugplot, color="r",clip_on=False)
#g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
g.set_axis_labels(xlabel=r'Stress-free length muscle 1 [$cm$]', ylabel=r'Stress-free length muscle 2 [$cm$]')
g.savefig(PATH3, bbox_inches='tight')
plt.close()

# Scatter plot sample generation
fig, ax =plt.subplots(figsize=(6,6))
#ax.set_title('Muscle stress-free lengths',color="black",fontsize=14)
#plt.plot(samples[0][BURN_IN:], samples[1][BURN_IN:], 'xb-')
plt.plot(samples[0][:], samples[1][:],zorder=1,color="mediumpurple",linewidth="0.5") 
plt.scatter(samples[0][:], samples[1][:],zorder=2,color="blue",marker="x",linewidths=0.5)
plt.scatter(samples[0][0], samples[1][0],zorder=3,color="red",marker="x",linewidths=2) 
ax.set_xlabel(r'Stress free length muscle 1 [$cm$]',color="black",fontsize=12)
ax.set_ylabel(r'Stress free length muscle 2 [$cm$]',color="black",fontsize=12)#,rotation=True)
plt.savefig(PATH4, bbox_inches='tight')
plt.close()

# Scatter plot detailed informations
fig, ax =plt.subplots(figsize=(6,6))
#ax.set_title('Stress free muscle-lengths',color="black",fontsize=14)
#plt.plot(samples[0][BURN_IN:], samples[1][BURN_IN:], 'xb-')
#plt.plot(samples[0][BURN_IN:], samples[1][BURN_IN:],zorder=1,color="mediumpurple",linewidth="0.5") 
ax.scatter(samples[0][BURN_IN:], samples[1][BURN_IN:],zorder=3,color="blue",marker="o",linewidths=0.5,label='HMC-Samples',alpha=0.5,s=10)
ax.scatter(expected_input_value[0], expected_input_value[1],zorder=4,color="yellow",edgecolor="red",marker="X",linewidths=0.5,label='Expected Input',alpha=0.9,s=60) 
ax.scatter(exact_input_value[0], exact_input_value[1],zorder=5,color="lime",edgecolor="darkgreen",marker="*",linewidths=0.5,label='Exact Input',alpha=0.9,s=100)
ax.scatter(14.0, 14.0,zorder=6,color="darkturquoise",edgecolor="indigo",marker="d",linewidths=0.5,label='Initial Sample',alpha=0.9,s=50) 
ax.axvspan(expected_input_value[0]-std_deviation_input_vaule[0],expected_input_value[0]+std_deviation_input_vaule[0],zorder=1,alpha=0.5,facecolor='red',edgecolor='blue',lw=1.,label='SF-length M1')
ax.axhspan(expected_input_value[1]-std_deviation_input_vaule[1],expected_input_value[1]+std_deviation_input_vaule[1],zorder=2,alpha=0.5,facecolor='violet',edgecolor='blue',lw=1.,label='SF-length M2')
ax.set_xlabel(r'Stress-free length muscle 1 [$cm$]',color="black",fontsize=12)
ax.set_ylabel(r'Stress-free length muscle 2 [$cm$]',color="black",fontsize=12)#,rotation=True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1),ncol=3)
plt.savefig(PATH6, bbox_inches='tight')
plt.close()

# 3D Histogramm plot
xAmplitudes = samples[0][BURN_IN:]
yAmplitudes = samples[1][BURN_IN:]

x = np.array(xAmplitudes)   #turn x,y data into numpy arrays
y = np.array(yAmplitudes)

fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
ax = fig.add_subplot(111, projection='3d')

#make histogram stuff - set bins - I choose 20x20 because I have a lot of data
hist, xedges, yedges = np.histogram2d(x, y, bins=(20,20))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz] 

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
#plt.title('Stress free muscle-lengths',color="black",fontsize=14)
plt.xlabel(r'Stress-free length muscle 1 [$cm$]',color="black",fontsize=12)
plt.ylabel(r'Stress-free length muscle 2 [$cm$]',color="black",fontsize=12)
plt.savefig(PATH5, bbox_inches='tight')
plt.show()

NUM_DRAWS = len(samples[0][:])

"""
Results for all iterations stress free length of muscle 1
"""

# Plot of sample values in a row (plot the walk)
iteration_number = np.linspace(0,NUM_DRAWS, num=NUM_DRAWS, endpoint=False)
fig, ax =plt.subplots(figsize=(6,6))
plt.plot(iteration_number[:(BURN_IN+1)],samples[0][:(BURN_IN+1)], color='SkyBlue', linestyle='-', linewidth=1)
plt.plot(iteration_number[BURN_IN:],samples[0][BURN_IN:], color='magenta', linestyle='-', linewidth=1)

ax.set_xlabel(r'Iteration',color="black",fontsize=12)
ax.set_ylabel(r'Stress-free length muscle 1 [$cm$]',color="black",fontsize=12)
#plt.title(r'Stress-free length muscle 1')

plt.savefig(PATH7, bbox_inches='tight')
plt.close()

"""
Results for all iterations stress free length of muscle 2
"""

# Plot of sample values in a row (plot the walk)
iteration_number = np.linspace(0,NUM_DRAWS, num=NUM_DRAWS, endpoint=False)
fig, ax =plt.subplots(figsize=(6,6))
plt.plot(iteration_number[:(BURN_IN+1)],samples[1][:(BURN_IN+1)], color='SkyBlue', linestyle='-', linewidth=1)
plt.plot(iteration_number[BURN_IN:],samples[1][BURN_IN:], color='magenta', linestyle='-', linewidth=1)

ax.set_xlabel(r'Iteration',color="black",fontsize=12)
ax.set_ylabel(r'Stress-free length muscle 2 [$cm$]',color="black",fontsize=12)
#plt.title(r'Stress-free length muscle 2')

plt.savefig(PATH8, bbox_inches='tight')
plt.close()
