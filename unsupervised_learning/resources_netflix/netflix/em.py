"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    K = len(mixture.mu)
    
    def log_gprob(x, m, s, d):
        comp1 = np.log(1) - (d/2) * ( np.log(2) + np.log(np.pi) + np.log(s) )
        comp2 = ( -((x-m)**2).sum(axis=1)/(2*s) )
        return comp1 + comp2

    gprob = lambda x, m, s, d: (1/(2*np.pi*s)**(d/2)) * np.exp( -((x-m)**2).sum(axis=1)/(2*s) )
    logprob = lambda cluster_p, gprobval: np.log(cluster_p + 1e-16) + np.log(gprobval + 1e-16)
    soft_counts = np.empty((0,K))
    likelihoods = np.empty((0,K))

    fltr = X!=0

    for i in range(X.shape[0]):
        f = fltr[i]
        Cu = X[i][f]
        Cmu = mixture.mu[:,f]
        d = len(Cu)

        repeated_Cu = np.tile(Cu, (K,1))
        prob = np.exp( log_gprob(repeated_Cu, Cmu, mixture.var, d) )
        prob = prob.reshape(1, K)
        likelihoods = np.append( likelihoods, prob, axis=0)

        f_uj = logprob(mixture.p, prob)
        l_ju = f_uj - logsumexp(f_uj)
        soft_counts = np.append(soft_counts, np.exp(l_ju), axis=0)


    log_likelihood = np.log( (mixture.p*likelihoods).sum(axis=1) + 1e-16).sum()
    return soft_counts, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    ftrs = X.shape[1]

    new_p = post.sum(axis=0) / post.sum()

    Cu = (X != 0)

    new_mu = np.empty((post.shape[1], 0))
    for d in range(ftrs):
        
        Cud = Cu[:,d].reshape(-1,1)
        denominator = (post * Cu[:,d].reshape(-1,1)).sum(axis=0)
        comp_avgs = (X[:,d].reshape(-1,1) * post).sum(axis=0) / denominator
        mu_d = mixture.mu[:,d]
        final_mu_d = np.where(denominator>=1, comp_avgs, mu_d)
        new_mu = np.column_stack( (new_mu, final_mu_d) )

    new_var = np.empty( (0) )
    for k in range(new_mu.shape[0]):
        C_mu = np.tile( new_mu[k,:], (X.shape[0], 1) ) * Cu
        var = (
                ( post[:,k].reshape(-1,1) * ( (X - C_mu)**2 )  ).sum(axis=1).sum(0) / 
                ( post[:,k].reshape(-1,1) * Cu.sum(1).reshape(-1,1) ).sum()
              )
        new_var = np.append( new_var, var )
    new_var = np.where(new_var>=0.25, new_var, min_variance)

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
        mixture = mstep(X=X, post=post, mixture=mixture)
    return mixture, post, new_ll



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    K = len(mixture.mu)
    
    def log_gprob(x, m, s, d):
        comp1 = np.log(1) - (d/2) * ( np.log(2) + np.log(np.pi) + np.log(s) )
        comp2 = ( -((x-m)**2).sum(axis=1)/(2*s) )
        return comp1 + comp2

    logprob = lambda cluster_p, gprobval: np.log(cluster_p + 1e-16) + np.log(gprobval + 1e-16)
    soft_counts = np.empty((0,K))

    fltr = X!=0

    for i in range(X.shape[0]):
        f = fltr[i]
        Cu = X[i][f]
        Cmu = mixture.mu[:,f]
        d = len(Cu)

        repeated_Cu = np.tile(Cu, (K,1))
        prob = np.exp( log_gprob(repeated_Cu, Cmu, mixture.var, d) )
        prob = prob.reshape(1, K)

        f_uj = logprob(mixture.p, prob)
        l_ju = f_uj - logsumexp(f_uj)
        soft_counts = np.append(soft_counts, np.exp(l_ju), axis=0)

    max_post_indices = np.argmax(soft_counts, axis=1)
    corres_mus = mixture.mu[max_post_indices]
    return np.where(X==0, corres_mus, X)

