"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from scipy.stats import norm


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    K = mixture.mu.shape[0]
    d = X.shape[1]
    gprob = lambda x, m, s: (1/(2*np.pi*s)**(d/2)) * np.exp( -((x-m)**2).sum(axis=1)/(2*s) )
    soft_counts = np.empty((0,K))
    likelihoods = np.empty((0,K))

    for i in range(X.shape[0]):
        prob = gprob(np.tile(X[i], (K,1)), mixture.mu, mixture.var)
        prob = prob.reshape(1, K)
        likelihoods = np.append( likelihoods, prob, axis=0)
        
        fin_prob = (prob*mixture.p)/(prob*mixture.p).sum()
        soft_counts = np.append(soft_counts, fin_prob, axis=0)
        

    log_likelihood = np.log( (mixture.p*likelihoods).sum(axis=1) ).sum()
    return soft_counts, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    ftrs = X.shape[1]

    new_p = post.sum(axis=0) / post.sum()

    new_mu = np.empty((post.shape[1], 0))
    for d in range(ftrs):
        comp_avgs = (X[:,d].reshape(-1,1) * post).sum(axis=0) / post.sum(axis=0)
        new_mu = np.column_stack( (new_mu, comp_avgs) )

    new_var = np.empty( (0) )
    for k in range(new_mu.shape[0]):
        var = (
                ( post[:,k].reshape(-1,1) * ( (X - new_mu[k,:])**2 )  ).sum() / 
                ((post[:,k]).sum(axis=0) * ftrs)
              )
        new_var = np.append( new_var, var )

    return GaussianMixture( mu=new_mu, var=new_var, p=new_p )



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_ll = None
    new_ll = None
    while (prev_ll is None or (new_ll-prev_ll > 1e-6*abs(new_ll)) ):
        prev_ll = new_ll
        post, new_ll = estep(X=X, mixture=mixture)
        mixture = mstep(X=X, post=post)
    return mixture, post, new_ll





