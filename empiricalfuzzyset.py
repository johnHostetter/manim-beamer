# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:52:22 2020

@author: jhost
"""

import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import Counter
from statistics import stdev 

numerator = None
distances = powers = np.array([])

def unimodalDensity(X, i, dist):
    """ Calculate the unimodal density of a particular ith data sample
    from the set of observations, X, with the distance metric, dist. """
    global numerator, distances, powers
    K = len(X)
#    numerator = 0
    denominator = 0
    
    if len(distances) == 0:
        print('Building matrices...')
        distances = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
        powers = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
        print('Done.')
        print('Standby for initialization...')
    
    if numerator == None:
        numerator = 0
        for k in range(K):
            for j in range(K):
                numerator += pow(dist(X[k], X[j]), 2)
    for j in range(K):
        if distances[i, j] != float('inf') or distances[j, i] != float('inf'):
            denominator += distances[i, j]
        else:    
            distances[i, j] = dist(X[i], X[j])
            powers[i, j] = pow(distances[i, j], 2)
            distances[j, i] = distances[i, j]
            powers[j, i] = powers[i, j]
            denominator += powers[i, j]
        
    denominator *= 2 * K
    return numerator / denominator

def unimodalDensity1(X, x, dist):
    """ Calculate the unimodal density of a particular ith data sample
    from the set of observations, X, with the distance metric, dist. """
    global numerator, distances, powers
    K = len(X)
#    numerator = 0
    denominator = 0
    
    if len(distances) == 0:
        print('Building matrices...')
        distances = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
        powers = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
        print('Done.')
        print('Standby for initialization...')
    
    if numerator == None:
        numerator = 0
        for k in range(K):
            for j in range(K):
                numerator += pow(dist(X[k], X[j]), 2)
    for j in range(K):
        denominator += pow(dist(x, X[j]), 2)
        
    denominator *= 2 * K
    return numerator / denominator

def multimodalDensity(X, U, F, i, dist):
    """ Calculate the multimodal density of a particular ith data sample
    from the set of observations, X, with the distance metric, dist, and
    using the set of frequencies, F. """
#    idx = X.index(U[i])
    idx = i
    return F[i] * unimodalDensity(X, idx, dist)

def multimodalDensity1(X, x, dist):
    """ Calculate the multimodal density of a particular ith data sample
    from the set of observations, X, with the distance metric, dist, and
    using the set of frequencies, F. """
#    idx = X.index(U[i])
    return unimodalDensity1(X, x, dist)

def unique(X):
    counter = Counter(X)
    U = list(counter.keys()) # unique observations
    F = list(counter.values()) # frequencies
    return (U, F)

def plotDistribution(X, Y):
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.title('Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
#    plt.plot(X, Y, 'o', color='blue')
    plt.plot(X, Y, 'o', color='blue')
    plt.legend()
    plt.show()

X = []

env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(250):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    observation[0] /= 1
    observation[1] = (observation[1] - (-2)) / (2 - (-2))
    observation[3] = (observation[3] - (-2)) / (2 - (-2))
    observation[2] = observation[2] * 2 * math.pi / 360
    print(observation)
    X.append(observation[:4])
    
    if done:
        observation = env.reset()

env.close()

#X.sort()
#U, F = unique(X)
U = X
F = np.array([1]*len(X))

x_lst = []
y_lst = []
densities = {}

# step 1

for i in range(len(U)):
    mm = multimodalDensity(X, U, F, i, distance.euclidean)
    print('%s/%s: %s' % (i, len(U), mm))
    densities[mm] = i # add the multimodal density to the set
    x_lst.append(U[i])
    y_lst.append(mm)
    
# step 2
    
maximum_multimodal_density = max(densities.keys()) # find the maximum multimodal density
idx = densities[maximum_multimodal_density] # find the index of the unique data sample with the maximum multimodal density
u1_star = U[idx] # find the unique data sample with the maximum multimodal density

# step 3

#U.pop(idx) # remove from the set of unique observations
DMM = []
ULstar = []
uR = u1_star

ctr = 0
visited = {}
previousLeftIndex = 0
previousLeftValue = 0
localMaximaIndexes = []
prototypes = []
while len(visited.keys()) < len(U):
    # step 4
    srted = sorted(distances[idx])
    for uRidx in range(len(srted)):
        item = srted[uRidx]
        if uRidx == idx:
            continue
        if uRidx in visited.keys():
            continue
        else:
            visited[uRidx] = ctr
            ctr += 1
            ULstar.append(U[uRidx])
            idx = uRidx # step 5, now go back to step 4
            mm = multimodalDensity(X, U, F, uRidx, distance.euclidean)
            DMM.append(mm) # step 6
            # step 7 is both if & else statements
            if mm > previousLeftValue:
                previousLeftIndex = uRidx
                previousLeftValue = mm
            else:
                localMaximaIndexes.append(uRidx - 1)
                prototypes.append(U[uRidx - 1])
                previousLeftIndex = uRidx
                previousLeftValue = mm
            break
#    row = idx
#    col = 0
#    U = np.delete(U, row, col)

# step 8
clouds = {} # each element is a list that is indexed by a prototype index
labels = [] # a direct labeling where the ith element of X has an ith label
for x in X:
    min_p = None
    min_idx = float('inf')
    min_dist = float('inf')
    for prototype_idx in range(len(prototypes)):
        prototype = prototypes[prototype_idx]
        dist = distance.euclidean(x, prototype)
        if dist < min_dist:
            min_p = prototype
            min_idx = prototype_idx
            min_dist = dist
    labels.append(min_p)
    try:
        clouds[min_idx].append(x)
    except KeyError:
        clouds[min_idx] = []
        clouds[min_idx].append(x)
    print(min_p)

# step 9
p0 = {} # the centers of the prototypes
for prototype_idx in clouds.keys():
    elements = clouds[prototype_idx]
    center = sum(elements) / len(elements)
    p0[prototype_idx] = center

# step 10
dmm_p0 = {}
for prototype_idx in p0.keys():
    dmm_p0[prototype_idx] = multimodalDensity1(X, p0[prototype_idx], distance.euclidean)
    
# step 11
runAgain = True
iteration = 0
while(runAgain):
    n = 0 # the number of unique pairs
    eta = 0.0
    ds = []
    sigma = 0.0
    for i in p0.keys():
        for j in p0.keys():
            if i > j:
                d = distance.euclidean(p0[i], p0[j])
                ds.append(d)
                eta += d
                n += 1
    eta /= n
    sigma = stdev(ds)
    R = sigma * (1 - (sigma / eta))
    piN = {}
    
    for i in p0.keys():
        for j in p0.keys():
            d = distance.euclidean(p0[j], p0[i])
            if d < R:
                try:
                    piN[i].append(p0[j])
                except KeyError:
                    piN[i] = []
                    piN[i].append(p0[j])
    
    p1 = {}
    for i in p0.keys():
        pi = p0[i]
        max_val = dmm_p0[i]
        member = True
        for q in piN[i]:
            if multimodalDensity1(X, q, distance.euclidean) > max_val:
                member = False
        if member:
            p1[i] = pi
            
    print(iteration)
    iteration += 1
    print('%s vs %s' % (len(p1.keys()), len(p0.keys())))
    runAgain = len(p1.keys()) < len(p0.keys()) # step 13
    p0 = p1 # step 12
    
# step 14
clouds = {} # each element is a list that is indexed by a prototype index
labels = [] # a direct labeling where the ith element of X has an ith label
for x in X:
    min_p = None
    min_idx = float('inf')
    min_dist = float('inf')
    for prototype_idx in p0.keys():
        prototype = p0[prototype_idx]
        dist = distance.euclidean(x, prototype)
        if dist < min_dist:
            min_p = prototype
            min_idx = prototype_idx
            min_dist = dist
    labels.append(min_p)
    try:
        clouds[min_idx].append(x)
    except KeyError:
        clouds[min_idx] = []
        clouds[min_idx].append(x)
    print(min_p)
    
def gaussianMembership(x, center, sigma):
    numerator = (-1) * pow(x - center, 2)
    denominator = 2 * pow(sigma, 2)
    return pow(math.e, numerator / denominator)

x_lst = []
mu_lst = []
for p_idx in p0.keys():
    for x in X:
        c = p0[p_idx][0]
        x = x[0]
        mu = gaussianMembership(x, c, sigma)
        x_lst.append(x)
        mu_lst.append(mu)
    plotDistribution(x_lst, mu_lst)
        
#x_lst = np.array(range(len(U)))
#print('star: %s' % u1_star)