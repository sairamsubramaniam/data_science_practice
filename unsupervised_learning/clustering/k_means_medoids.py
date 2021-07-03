
from collections import defaultdict

import numpy as np
from scipy.spatial import distance_matrix


def kmeans_iter(xs, centroids, method="l2"):

    clusters = defaultdict(list)
    fin_cent = {}

    if method == "l1":
        def dist_func(point, cntrd):
            return sum( abs(np.array(point) - np.array(cntrd)) )
    elif method == "l2":
        def dist_func(point, cntrd):
            return sum( (np.array(point) - np.array(cntrd))**2 )

    total_cost = 0
    for x in xs:
        distances = np.array([dist_func(x,c) for c in centroids])
        cluster = np.argmin( distances )
        clusters[cluster].append(x)
        total_cost += distances.min()

    for clust, mems in clusters.items():
        fin_cent[clust] = np.median( np.array(mems, np.float64), axis=0 )

    final_centroids = [0] * len(fin_cent)
    for k, v in fin_cent.items():
        final_centroids[k] = v

    return total_cost, final_centroids, clusters



def kmedoids_iter(xs, centroids, method="l2"):

    clusters = defaultdict(list)
    fin_cent = {}

    if method == "l1":
        def dist_func(point, cntrd):
            return sum( abs(np.array(point) - np.array(cntrd)) )
    elif method == "l2":
        def dist_func(point, cntrd):
            return sum( (np.array(point) - np.array(cntrd))**2 )

    total_cost = 0
    for x in xs:
        distances = np.array([dist_func(x,c) for c in centroids])
        cluster = np.argmin( distances )
        clusters[cluster].append(x)
        total_cost += distances.min()

    for clust, mems in clusters.items():
        ds = distance_matrix(mems, mems).sum(axis=1)
        new_cent = mems[ np.argmin(ds) ]
        fin_cent[clust] = new_cent

    final_centroids = [0] * len(fin_cent)
    for k, v in fin_cent.items():
        final_centroids[k] = v

    return total_cost, final_centroids, clusters




xs = [(0,-6),(4,4),(0,0),(-5,2)]
centroids = [(-5,2),(0,-6)]

for iteration in range(10):
    print()
    print("-----------------------")
    print( "ITER: ", iteration)
    print(centroids)
    cost, centroids, clusters = kmeans_iter(xs=xs, centroids=centroids, method="l2")
    print( cost, centroids, clusters )
    print("-----------------------")



# ===================================================================

# Gaussian Mixture Models:

import numpy as np
from scipy.stats import norm
def hw1_1(p1, p2, x, mu1, mu2, s1, s2):

    """
    p1 and p2 are the prior probabilities
    x is the x value
    mu1/2 are the means 1 and 2 given in the expercise
    s1/2 are the variances 1 and 2 given in the expercise
    """
    n1 = (norm(mu1, s1).pdf(x))*p1
    n2 = (norm(mu2, s2).pdf(x))*p2
    return n1 + n2

x0 = np.log(hw1_1(0.5, 0.5, -1, 6, 7, 1, 2))
x1 = np.log(hw1_1(0.5, 0.5, 0, 6, 7, 1, 2))
x2 = np.log(hw1_1(0.5, 0.5, 4, 6, 7, 1, 2))
x3 = np.log(hw1_1(0.5, 0.5, 5, 6, 7, 1, 2))
x4 = np.log(hw1_1(0.5, 0.5, 6, 6, 7, 1, 2))

al_oce = x0 + x1 + x2 + x3+ x4




