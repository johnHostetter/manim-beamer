import time
import numpy as np

from tests.fuzzy.online.unsupervised.granulation.clip import gaussian


def rule_creation(X, antecedents, existing_rules=[], existing_weights=[], consistency_check=True):
    start = time.time()
    rules = existing_rules
    weights = existing_weights
    for x in X:
        CF = 1.0  # certainty factor of this rule
        # this block of code is for antecedents
        A_star_js = []
        for p in range(len(x)):
            SM_jps = []
            for j, A_jp in enumerate(antecedents[p]):
                SM_jp = gaussian(x[p], A_jp['center'], A_jp['sigma'])
                SM_jps.append(SM_jp)
            CF *= np.max(SM_jps)
            j_star_p = np.argmax(SM_jps)
            A_star_js.append(j_star_p)

        # with some work, you can remove the 'C' key-value here in the rule, it stands for 'consequent(s)'
        # however, later on, in the neuro-fuzzy Q-network, it will expect this to be here
        R_star = {'A': A_star_js, 'C': [0], 'CF': CF, 'time_added': start}

        if not rules:  # no rules in knowledge base yet
            rules.append(R_star)
            weights.append(1.0)
        else:  # there are rules in the knowledge base, so check for uniqueness (i.e., this new rule you made --
            # R_star -- is it needed?)
            add_new_rule = True
            for k, rule in enumerate(rules):
                try:
                    if (rule['A'] == R_star['A']) and (rule['C'] == R_star['C']):
                        # the generated rule is not unique, it already exists, enhance this rule's weight
                        weights[k] += 1.0
                        rule['CF'] = min(rule['CF'], R_star['CF'])
                        add_new_rule = False
                        break
                except ValueError:  # this happens because R_star['A'] and R_star['C'] are Numpy arrays
                    if all(rule['A'] == list(R_star['A'])) and all(rule['C'] == list(R_star['C'])):
                        # the generated rule is not unique, it already exists, enhance this rule's weight
                        weights[k] += 1.0
                        rule['CF'] = min(rule['CF'], R_star['CF'])
                        add_new_rule = False
                        break
                    elif all(rule['A'] == list(R_star['A'])):  # my own custom else-if statement
                        if rule['CF'] <= R_star['CF']:
                            add_new_rule = False
            if add_new_rule:
                rules.append(R_star)
                weights.append(1.0)

    # check for consistency
    if consistency_check:
        all_antecedents = [rule['A'] for rule in rules]

        repeated_rule_indices = set()
        for k in range(len(rules)):
            indices = np.where(np.all(all_antecedents == np.array(rules[k]['A']), axis=1))[0]
            if len(indices) > 1:
                if len(repeated_rule_indices) == 0:  # this can be combined with the following elif-statement
                    repeated_rule_indices.add(tuple(indices))
                elif len(repeated_rule_indices) > 0:  # this can be combined with the above if-statement
                    repeated_rule_indices.add(tuple(indices))

        for indices in repeated_rule_indices:
            weights_to_compare = [weights[idx] for idx in indices]
            strongest_rule_index = indices[
                np.argmax(weights_to_compare)]  # keep the rule with the greatest weight to it
            for index in indices:
                if index != strongest_rule_index:
                    rules[index] = None
                    weights[index] = None
        rules = [rules[k] for k, rule in enumerate(rules) if rules[k] is not None]
        weights = [weights[k] for k, weight in enumerate(weights) if weights[k] is not None]

        # need to check that no antecedent terms are "orphaned" (i.e., they go unused)

        all_antecedents = [rule['A'] for rule in rules]
        all_antecedents = np.array(all_antecedents)
        for p in range(len(x)):
            if len(antecedents[p]) == len(np.unique(all_antecedents[:, p])):
                continue
            else:
                # orphaned antecedent term
                indices_for_antecedents_that_are_used = set(all_antecedents[:, p])
                updated_indices_to_map_to = list(range(len(indices_for_antecedents_that_are_used)))
                antecedents[p] = [antecedents[p][index] for index in indices_for_antecedents_that_are_used]

                paired_indices = list(zip(indices_for_antecedents_that_are_used, updated_indices_to_map_to))
                for index_pair in paired_indices:  # the paired indices are sorted w.r.t. the original indices
                    original_index = index_pair[0]  # so, when we updated the original index to its new index
                    new_index = index_pair[1]  # we are guaranteed not to overwrite the last updated index
                    all_antecedents[:, p][all_antecedents[:, p] == original_index] = new_index

        # update the rules in case any orphaned terms occurred
        for idx, rule in enumerate(rules):
            rule['A'] = all_antecedents[idx]

    return antecedents, rules, weights