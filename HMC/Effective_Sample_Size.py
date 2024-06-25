import numpy as np
import pandas as pd
import arviz as az
import csv

with open('run_samples.csv', newline='') as f:
    reader = csv.reader(f)
    samples = np.array([np.array(row, dtype = 'float') for row in reader])

print('samples',samples)
burn_in = 50 #Number of iterations for the burn-in of the HMC

def my_ESS(x):
    """ Compute the effective sample size of estimand of interest. Vectorised implementation. """
    m_chains, n_iters = x.shape

    print('m_chains', m_chains)
    print('n_iters', n_iters)

    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    post_var = my_gelman_rubin(x)

    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0

        t += 1

    return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

def my_gelman_rubin(x):
    """ Estimate the marginal posterior variance. Vectorised implementation. """
    m_chains, n_iters = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))

    # (over) estimate of variance
    s2 = W * (n_iters - 1) / n_iters + B_over_n

    return s2


#effective_sample_size = my_ESS(np.array([samples[0,:]]))
#print('Effective sample size =',effective_sample_size)
