
import numpy as np
#import pandas as pd
import arviz as az
import arviz.labels as azl
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import statistics

# Model parameters
burn_in = 50 #Number of iterations for the burn-in of the HMC
exact_input_value = [12.5,14.5]

# Import samples from MCMC 
with open(f'{FILEPATH_DATA}/run_samples.csv', newline='') as f:
    reader = csv.reader(f)
    samples_all = np.array([np.array(row, dtype = 'float') for row in reader])

samples_3 = samples_all[0:2,:]
samples_2 = samples_all[2:4,:]
samples = samples_all[4:6,:]
samples_4 = samples_all[6:8,:]

#data = np.transpose(np.array([samples]),(0,2,1))
data = np.transpose(np.array([samples[:,0:1000]]),(0,2,1))
datadict = {
    "Stress-free muscle length": data
    }
coords = {"SFML": ["Muscle 1","Muscle 2"]}
dims = {"Stress-free muscle length": ["SFML"]}
idata = az.convert_to_inference_data(datadict,dims=dims,coords=coords)

data_2 = np.transpose(np.array([samples_2[:,0:1000]]),(0,2,1))
datadict_2 = {
    "Stress-free muscle length": data_2
    }
coords = {"SFML": ["Muscle 1","Muscle 2"]}
dims = {"Stress-free muscle length": ["SFML"]}
idata_2 = az.convert_to_inference_data(datadict_2,dims=dims,coords=coords)

data_3 = np.transpose(np.array([samples_3[:,0:1000]]),(0,2,1))
datadict_3 = {
    "Stress-free muscle length": data_3
    }
coords = {"SFML": ["Muscle 1","Muscle 2"]}
dims = {"Stress-free muscle length": ["SFML"]}
idata_3 = az.convert_to_inference_data(datadict_3,dims=dims,coords=coords)

data_4 = np.transpose(np.array([samples_4[:,0:1000]]),(0,2,1))
datadict_4 = {
    "Stress-free muscle length": data_4
    }
coords = {"SFML": ["Muscle 1","Muscle 2"]}
dims = {"Stress-free muscle length": ["SFML"]}
idata_4 = az.convert_to_inference_data(datadict_4,dims=dims,coords=coords)

## Informations about parameters
#expected_input_value = np.empty(len(samples),dtype=float)
#std_deviation_input_vaule = np.empty(len(samples),dtype=float)

## Informations about parameters
#for i in range(len(samples)):    
    #expected_input_value[i] = statistics.mean(samples[i][burn_in:]) #Expected value for input to get the observed data
    #std_deviation_input_vaule[i] = statistics.stdev(samples[i][burn_in:]) #Standard deviation of the expected value for input to get the observed data

#data = np.transpose(np.array([samples]),(0,2,1))
#idata = az.convert_to_inference_data(data)

# Combine samples from multiple runs
data_compare = np.concatenate((data,data_2,data_3,data_4),axis = 0)
idata_compare = az.convert_to_inference_data(data_compare)

# Compare/combine samples from multiple runs
print('Summary compare')
print(az.summary(idata_compare))

# Calculate effective sample size (method: tail)
ess_arviz_compare = az.ess(idata_compare,method="tail",relative=True)
print('ess arviz compare=',ess_arviz_compare)

# Create information about one run of MCMC
print('Summary 1')
print(az.summary(idata))
print('Posterior 1')
print(idata.posterior)

print('Summary 2')
print(az.summary(idata_2))
print('Posterior 2')
print(idata_2.posterior)

print('Summary 3')
print(az.summary(idata_3))
print('Posterior 3')
print(idata_3.posterior)

print('Summary 4')
print(az.summary(idata_4))
print('Posterior 4')
print(idata_4.posterior)

# Calculate effective sample size (method: tail)
ess_arviz = az.ess(idata,method="tail",relative=True)
print('ess arviz =',ess_arviz)

# Plot the Simulation Results with arviz
PATH1 = 'ESS_Trace.pdf'
PATH2 = 'ESS_Posterior.pdf'
PATH3 = 'ESS_Dist.pdf'
PATH4 = 'ESS_2D_KDE.pdf'
PATH5 = 'ESS_Joint.pdf'
PATH6 = 'ESS_Quantiles.pdf'
PATH7 = 'ESS_Point_Estimate.pdf'
PATH8_a = 'ESS_Autocorrelation_a.pdf'
PATH8_b = 'ESS_Autocorrelation_b.pdf'
PATH8_c = 'ESS_Autocorrelation_c.pdf'
PATH8_d = 'ESS_Autocorrelation_d.pdf'
PATH9_a = 'ESS_ESS_Evolution_a.pdf'
PATH9_b = 'ESS_ESS_Evolution_b.pdf'
PATH9_c = 'ESS_ESS_Evolution_c.pdf'
PATH9_d = 'ESS_ESS_Evolution_d.pdf'
PATH10 = 'ESS_ESS_Local.pdf'
PATH11 = 'ESS_MCSE.pdf'

az.style.use("arviz-darkgrid")

# Plot trace
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title('Trace Plot',color="black",fontsize=14)
az.plot_trace(idata_compare)
#plt.plot(samples[0][burn_in:], samples[1][burn_in:], 'xb-')
#plt.plot(samples[0][:], samples[1][:],zorder=1,color="mediumpurple",linewidth="0.5") 
#plt.scatter(samples[0][:], samples[1][:],zorder=2,color="blue",marker="x",linewidths=0.5)
#ax.set_xlabel(r'stress free length muscle 1 [$cm$]',color="black",fontsize=12)
#ax.set_ylabel(r'stress free length muscle 2 [$cm$]',color="black",fontsize=12)#,rotation=True)
plt.savefig(PATH1, bbox_inches='tight')
plt.close()

# Plot posterior
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title('Posterior Plot',color="black",fontsize=14)
az.plot_posterior(idata)
plt.savefig(PATH2, bbox_inches='tight')
plt.close()

# Dist Plot
az.style.use("arviz-doc")
fig, ax = plt.subplots(1, 2)
fig.suptitle("Distributions")

ax[0].set_title("Muscle 1")
az.plot_dist(data[0,:,0], color="C1", label="P1", ax=ax[0])

ax[1].set_title("Muscle 2")
az.plot_dist(data[0,:,1], color="C2", label="P2", ax=ax[1])
plt.savefig(PATH3, bbox_inches='tight')
plt.close()

# 2D KDE Plot
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")

az.plot_kde(
    data[0,:,0],
    data[0,:,1],
    contour_kwargs={"colors": None, "cmap": plt.cm.viridis, "levels": 30},
    contourf_kwargs={"alpha": 0.5, "levels": 30},
)

plt.savefig(PATH4, bbox_inches='tight')
plt.close()

# Joint Plot
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")

az.plot_pair(
    idata,
    kind="kde", # kind: hexbin, scatter, kde
    #gridsize=30, # only for hexbin
    marginals=True,
    figsize=(11.5, 11.5),
)

plt.savefig(PATH5, bbox_inches='tight')
plt.close()

# Quantiles Plot
az.style.use("arviz-doc")
fig, ax = plt.subplots(1, 2)
#fig.suptitle("Distributions")

ax[0].set_title("Muscle 1")
az.plot_kde(data[0,:,0], quantiles=[0.25, 0.5, 0.75], ax=ax[0])

ax[1].set_title("Muscle 2")
az.plot_kde(data[0,:,1], quantiles=[0.25, 0.5, 0.75], ax=ax[1])

plt.savefig(PATH6, bbox_inches='tight')
plt.close()

# Point Estimate Pair Plot
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")

ax = az.plot_pair(
    data,
    #var_names=["mu", "theta"],
    kind=["scatter", "kde"],
    kde_kwargs={"fill_last": False},
    marginals=True,
    #coords=coords,
    point_estimate="median", # mean, mode, median
    figsize=(11.5, 5),
)

plt.savefig(PATH7, bbox_inches='tight')
plt.close()

# Inference Diagnostics

# Autocorrelation Plot
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_autocorr(idata)
plt.savefig(PATH8_a, bbox_inches='tight')
plt.close()

az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_autocorr(idata_2)
plt.savefig(PATH8_b, bbox_inches='tight')
plt.close()

az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_autocorr(idata_3)
plt.savefig(PATH8_c, bbox_inches='tight')
plt.close()

az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_autocorr(idata_4)
plt.savefig(PATH8_d, bbox_inches='tight')
plt.close()

# ESS Evolution Plot
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_ess(idata, kind="evolution",min_ess=0)
plt.savefig(PATH9_a, bbox_inches='tight')
plt.close()

az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_ess(idata_2, kind="evolution",min_ess=0)
plt.savefig(PATH9_b, bbox_inches='tight')
plt.close()

az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_ess(idata_3, kind="evolution",min_ess=0)
plt.savefig(PATH9_c, bbox_inches='tight')
plt.close()

az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_ess(idata_4, kind="evolution",min_ess=0)
plt.savefig(PATH9_d, bbox_inches='tight')
plt.close()

# ESS Local Plot
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_ess(idata, kind="local") # kind: local, quantile
plt.savefig(PATH10, bbox_inches='tight')
plt.close()

# ESS Monte Carlo Standard Error
az.style.use("arviz-doc")
fig, ax = plt.subplots(figsize=(6,6))
#fig.suptitle("Distributions")
az.plot_mcse(idata, extra_methods=True) #, rug=True
plt.savefig(PATH11, bbox_inches='tight')
plt.close()


#def my_ESS(x):
    #""" Compute the effective sample size of estimand of interest. Vectorised implementation. """
    #m_chains, n_iters = x.shape

    #print('m_chains', m_chains)
    #print('n_iters', n_iters)

    #variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    #post_var = my_gelman_rubin(x)

    #t = 1
    #rho = np.ones(n_iters)
    #negative_autocorr = False

    ## Iterate until the sum of consecutive estimates of autocorrelation is negative
    #while not negative_autocorr and (t < n_iters):
        #rho[t] = 1 - variogram(t) / (2 * post_var)

        #if not t % 2:
            #negative_autocorr = sum(rho[t-1:t+1]) < 0

        #t += 1

    #return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

#def my_gelman_rubin(x):
    #""" Estimate the marginal posterior variance. Vectorised implementation. """
    #m_chains, n_iters = x.shape

    ## Calculate between-chain variance
    #B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    ## Calculate within-chain variances
    #W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))

    ## (over) estimate of variance
    #s2 = W * (n_iters - 1) / n_iters + B_over_n

    #return s2


#effective_sample_size = my_ESS(np.array([samples_5[0,:],samples_6[0,0:100]]))
##effective_sample_size_arviz = az.ess(np.array([samples_5[0,:],samples_6[0,0:100]]))
#idata = az.convert_to_inference_data(np.expand_dims(np.array([samples_5,samples_6[:,0:100]]).T, 0))
#idata_5 = az.convert_to_inference_data(np.expand_dims(samples_5.T, 0))
#idata_6 = az.convert_to_inference_data(np.expand_dims(samples_6.T, 0))
#print('Data', idata.posterior)
##effective_sample_size_arviz = az.ess(np.array([samples_5[0,:]]))
#effective_sample_size_arviz_5 = az.ess(idata_5)
#effective_sample_size_arviz_6 = az.ess(idata_6)
#effective_sample_size_arviz = az.ess(idata)
#print('Effective sample size =',effective_sample_size)
#print('Effective sample size arviz 5 =',effective_sample_size_arviz_5)
#print('Effective sample size arviz 6 =',effective_sample_size_arviz_6)
#print('Effective sample size arviz compare=',effective_sample_size_arviz)
#print('Summary',az.summary(np.array([samples_5,samples_6[:,0:100]]).T))
