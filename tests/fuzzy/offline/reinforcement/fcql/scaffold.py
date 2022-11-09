import time

"""
The first algorithm we need to define is **Categorical Learning Induced Partitioning (CLIP)**. 
CLIP is used to generate the Gaussian membership functions that describe fuzzy sets. 
By running CLIP over the entire training data, we will be able to create a global language 
(i.e. linguistic terms) for interpretation.
"""

from tests.fuzzy.online.unsupervised.granulation.clip import CLIP

"""
Evolving Clustering Method is required for fuzzy logic rule generation. 
Below we define its necessary functions.
"""

from ecm import ECM

"""
Next, after ECM, is the Wang-Mendel method for fuzzy logic rule generation. 
It comes with a few extras that go beyond just the Wang-Mendel method as described in the paper 
because it is a general implementation. Specifically, it includes certainty factor calculations as 
described in the HyFIS paper. These certainty factors are ignored in our current procedure.
"""

from wang_mendel import rule_creation


def unsupervised(train_X, eps=0.2, kappa=0.6, ecm=True, Dthr=1e-3, verbose=False):
    """
    Applies CLIP, ECM and Wang-Mendel method to produce the fuzzy sets, candidates and fuzzy logic rules, respectively.

    Parameters
    ----------
    train_X : 2-D Numpy array
        The input vector, has a shape of (number of observations, number of inputs/attributes).
    eps : float, optional
        The 'alpha' parameter used in CLIP.
    kappa : float, optional
        The 'beta' parameter used in CLIP.
    ecm : boolean, optional
        This boolean controls whether to enable the ECM algorithm for candidate rule generation. The default is True.
    Dthr : float, optional
        The distance threshold for the ECM algorithm; only matters if ECM is enabled. The default is 1e-3.
    verbose : boolean, optional
        If enabled (True), the execution of this function will print out step-by-step to show progress. The default is False.

    Returns
    -------
    The fuzzy logic rules, their corresponding weights (unused), and the antecedent terms they are defined over.

    """
    print('The shape of the training data is: (%d, %d)\n' %
          (train_X.shape[0], train_X.shape[1]))
    train_X_mins = train_X.min(axis=0)
    train_X_maxes = train_X.max(axis=0)

    if verbose:
        print('Creating/updating the membership functions...')

    start = time.time()
    import numpy as np
    np.save('clip_input', train_X)
    antecedents = CLIP(train_X, train_X_mins, train_X_maxes,
                       [], eps=eps, kappa=kappa)
    end = time.time()
    if verbose:
        print('membership functions for the antecedents generated in %.2f seconds.' % (
                end - start))

    if ecm:
        if verbose:
            print('\nReducing the data observations to clusters using ECM...')
        start = time.time()
        clusters = ECM(train_X, [], Dthr)
        if verbose:
            print('%d clusters were found with ECM from %d observations...' % (
                len(clusters), train_X.shape[0]))
        reduced_X = [cluster.center for cluster in clusters]
        end = time.time()
        if verbose:
            print('done; the ECM algorithm completed in %.2f seconds.' %
                  (end - start))
    else:
        reduced_X = train_X

    if verbose:
        print('\nCreating/updating the fuzzy logic rules...')
    start = time.time()
    antecedents, rules, weights = rule_creation(reduced_X, antecedents, [], [],
                                                consistency_check=False)

    K = len(rules)
    end = time.time()
    if verbose:
        print('%d fuzzy logic rules created/updated in %.2f seconds.' %
              (K, end - start))
    return rules, weights, antecedents
