import numpy as np


def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))


def R(sigma_1, sigma_2):
    # regulator function
    return (1 / 2) * (sigma_1 + sigma_2)


def CLIP(X, mins, maxes, terms=[], eps=0.2, kappa=0.6, theta=1e-8):
    # theta is a parameter I add to accomodate for the instance in which an observation has values that are the minimum/maximum
    # otherwise, when determining the Gaussian membership, a division by zero will occur
    # it essentially acts as an error tolerance
    antecedents = terms
    min_values_per_feature_in_X = mins
    max_values_per_feature_in_X = maxes
    for idx, x in enumerate(X):
        if not terms:
            # no fuzzy clusters yet, create the first fuzzy cluster
            for p in range(len(x)):
                c_1p = x[p]
                min_p = min_values_per_feature_in_X[p]
                max_p = max_values_per_feature_in_X[p]
                left_width = np.sqrt(-1.0 * (np.power((min_p - x[p]) + theta, 2) / np.log(eps)))
                right_width = np.sqrt(-1.0 * (np.power((max_p - x[p]) + theta, 2) / np.log(eps)))
                sigma_1p = R(left_width, right_width)
                terms.append([{'center': c_1p, 'sigma': sigma_1p, 'support': 1}])
        else:
            # calculate the similarity between the input and existing fuzzy clusters
            for p in range(len(x)):
                SM_jps = []
                for j, A_jp in enumerate(terms[p]):
                    SM_jp = gaussian(x[p], A_jp['center'], A_jp['sigma'])
                    SM_jps.append(SM_jp)
                j_star_p = np.argmax(SM_jps)
                if np.max(SM_jps) > kappa:
                    # the best matched cluster is deemed as being able to give satisfactory description of the presented value
                    A_j_star_p = terms[p][j_star_p]
                    A_j_star_p['support'] += 1
                else:
                    # a new cluster is created in the input dimension based on the presented value
                    jL_p = None
                    jR_p = None
                    jL_p_differences = []
                    jR_p_differences = []
                    for j, A_jp in enumerate(terms[p]):
                        c_jp = A_jp['center']
                        if c_jp >= x[p]:
                            continue  # the newly created cluster has no immediate left neighbor
                        else:
                            jL_p_differences.append(np.abs(c_jp - x[p]))
                    try:
                        jL_p = np.argmin(jL_p_differences)
                    except ValueError:
                        jL_p = None

                    for j, A_jp in enumerate(terms[p]):
                        c_jp = A_jp['center']
                        if c_jp <= x[p]:
                            continue  # the newly created cluster has no immediate right neighbor
                        else:
                            jR_p_differences.append(np.abs(c_jp - x[p]))
                    try:
                        jR_p = np.argmin(jR_p_differences)
                    except ValueError:
                        jR_p = None

                    new_c = x[p]
                    new_sigma = None

                    # --- this new fuzzy set has no left or right neighbor ---
                    if jL_p is None and jR_p is None:
                        continue

                    # --- there is BOTH a left and a right neighbor to this fuzzy set ---
                    if jR_p is not None and jL_p is not None:
                        cR_jp = terms[p][jR_p]['center']
                        sigma_R_jp = terms[p][jR_p]['sigma']
                        left_sigma_R = np.sqrt(-1.0 * (np.power(cR_jp - x[p], 2) / np.log(eps)))
                        sigma_R = R(left_sigma_R, sigma_R_jp)

                        cL_jp = terms[p][jL_p]['center']
                        sigma_L_jp = terms[p][jL_p]['sigma']
                        left_sigma_L = np.sqrt(-1.0 * (np.power(cL_jp - x[p], 2) / np.log(eps)))
                        sigma_L = R(left_sigma_L, sigma_L_jp)

                        new_sigma = R(sigma_R, sigma_L)
                        # update the existing terms to make room for the new term
                        terms[p][jR_p]['sigma'] = terms[p][jL_p]['sigma'] = new_sigma

                    # --- there is no left neighbor to this new fuzzy set ---
                    elif jL_p is None:
                        cR_jp = terms[p][jR_p]['center']
                        sigma_R_jp = terms[p][jR_p]['sigma']
                        left_sigma_R = np.sqrt(-1.0 * (np.power(cR_jp - x[p], 2) / np.log(eps)))
                        sigma_R = R(left_sigma_R, sigma_R_jp)

                        new_sigma = sigma_R
                        # update the existing term to make room for the new term
                        terms[p][jR_p]['sigma'] = new_sigma

                    # --- there is no right neighbor to this new fuzzy set ---
                    elif jR_p is None:
                        cL_jp = terms[p][jL_p]['center']
                        sigma_L_jp = terms[p][jL_p]['sigma']
                        left_sigma_L = np.sqrt(-1.0 * (np.power(cL_jp - x[p], 2) / np.log(eps)))
                        sigma_L = R(left_sigma_L, sigma_L_jp)

                        new_sigma = sigma_L
                        # update the existing term to make room for the new term
                        terms[p][jL_p]['sigma'] = new_sigma
                    terms[p].append({'center': new_c, 'sigma': new_sigma, 'support': 1})
    return terms
