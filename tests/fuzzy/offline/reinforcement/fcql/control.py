import torch
import numpy as np
import torch.nn as nn

from torch import optim
from torch.nn.parameter import Parameter

"""
A Gaussian activation function is then defined where it can have different centers 
and widths depending on which input variable and input term it is describing. 
For example, the i'th fuzzy set (across all input dimensions) would be defined 
by its corresponding i'th center and sigma.
"""


class Gaussian(nn.Module):
    """
    Helpful reference: https://towardsdatascience.com/extending-pytorch-with-custom-activation-functions-2d8b065ef2fa

    Implementation of the Gaussian membership function.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - centers: trainable parameter
        - sigmas: trainable parameter
    Examples:
        # >>> a1 = gaussian(256)
        # >>> x = torch.randn(256)
        # >>> x = a1(x)
    """

    def __init__(self, in_features, centers=None, sigmas=None, trainable=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            centers and sigmas are initialized randomly by default,
            but sigmas must be > 0
        """
        super(Gaussian, self).__init__()
        self.in_features = in_features

        # initialize centers
        if centers is None:
            self.centers = Parameter(torch.randn(self.in_features))
        else:
            self.centers = torch.tensor(centers)

        # initialize sigmas
        if sigmas is None:
            self.sigmas = Parameter(torch.abs(torch.randn(self.in_features)))
        else:
            # make sure the sigma values are positive
            self.sigmas = torch.abs(torch.tensor(sigmas))

        self.centers.requires_grad = trainable
        self.sigmas.requiresGrad = trainable
        self.centers.grad = None
        self.sigmas.grad = None

    def forward(self, x):
        """
        Forward pass of the function. Applies the function to the input elementwise.
        """

        return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(self.sigmas, 2)))


class FLC(nn.Module):
    """
    Helpful reference: https://towardsdatascience.com/extending-pytorch-with-custom-activation-functions-2d8b065ef2fa

    Implementation of the Fuzzy Logic Controller.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - centers: trainable parameter
        - sigmas: trainable parameter
    Examples:
        # >>> antecedents = [[{'type': 'gaussian', 'parameters': {'center': 1.2, 'sigma': 0.1}},
                            {'type': 'gaussian', 'parameters': {'center': 3.0, 'sigma': 0.4}}],
                            [{'type': 'gaussian', 'parameters': {'center': 0.2, 'sigma': 0.4}}]]
        # consequences are not required, default is None
        # >>> consequences = [[{'type': 'gaussian', 'parameters': {'center': 0.1, 'sigma': 0.7}},
                            {'type': 'gaussian', 'parameters': {'center': 0.4, 'sigma': 0.41}}],
                            [{'type': 'gaussian', 'parameters': {'center': 0.9, 'sigma': 0.32}}]]
        # if consequences are not to be specified, leave the key-value out
        # >>> rules = [{'antecedents':[0, 0], 'consequences':[0]}, {'antecedents':[1, 0], 'consequences':[1]}]
        # >>> n_input = len(antecedents)  # the length of antecedents should be equal to number of inputs
        # >>> n_output = len(consequences)  # the length of antecedents should be equal to number of inputs
        # >>> flc = FLC(n_input, n_output, antecedents, rules, consequences)
        # >>> x = torch.randn(n_input)
        # >>> y = flc(x)
    """

    def __init__(self, in_features, out_features, antecedents, rules, consequences=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - consequences: (optional) trainable parameter
            consequences are initialized randomly by default,
            but sigmas must be > 0
        """
        super(FLC, self).__init__()
        self.in_features = in_features

        # find the number of antecedents per input variable
        num_of_antecedents = np.zeros(in_features).astype('int32')
        unique_id = 0
        gaussians = {'centers': [], 'sigmas': []}  # currently, we assume only Gaussians are used
        self.input_variable_ids = []
        self.transformed_x_length = 0
        for input_variable_idx in range(in_features):
            num_of_antecedents[input_variable_idx] = len(antecedents[input_variable_idx])
            self.input_variable_ids.append(set())
            for term_idx, antecedent in enumerate(antecedents[input_variable_idx]):
                try:
                    gaussians['centers'].append(antecedent['parameters']['center'])
                    gaussians['sigmas'].append(antecedent['parameters']['sigma'])
                except KeyError:
                    gaussians['centers'].append(antecedent['center'])
                    gaussians['sigmas'].append(antecedent['sigma'])
                antecedent['id'] = unique_id
                self.input_variable_ids[-1].add(unique_id)
                unique_id += 1
        self.transformed_x_length = unique_id

        # find the total number of antecedents across all input variables
        self.n_rules = len(rules)
        self.links_between_antecedents_and_rules = np.zeros((num_of_antecedents.sum(), self.n_rules))

        for rule_idx, rule in enumerate(rules):
            try:
                for input_variable_idx, term_idx in enumerate(rule['antecedents']):
                    new_term_idx = antecedents[input_variable_idx][term_idx]['id']
                    self.links_between_antecedents_and_rules[new_term_idx, rule_idx] = 1
            except KeyError:
                for input_variable_idx, term_idx in enumerate(rule['A']):
                    new_term_idx = antecedents[input_variable_idx][term_idx]['id']
                    self.links_between_antecedents_and_rules[new_term_idx, rule_idx] = 1

        print(self.links_between_antecedents_and_rules)

        # begin creating the model's layers
        self.input_terms = Gaussian(in_features=self.in_features, centers=gaussians['centers'],
                                    sigmas=gaussians['sigmas'], trainable=False)

        # initialize consequences
        if consequences is None:
            num_of_consequent_terms = self.n_rules
            self.consequences = Parameter(torch.zeros(num_of_consequent_terms, out_features))
        else:
            self.consequences = Parameter(torch.tensor(consequences))

        self.consequences.requires_grad = True

    def __transform(self, X):
        """
        Transforms the given 'X' to make it compatible with the first layer.

        The shape of 'X' is (num. of observations, num. of features).
        """
        shape = X.shape
        n_observations = shape[0]  # number of observations
        new_X = np.zeros((n_observations, self.transformed_x_length))
        for input_variable_idx, indices_to_repeat_for in enumerate(self.input_variable_ids):
            min_column_idx = min(indices_to_repeat_for)
            max_column_idx = max(indices_to_repeat_for) + 1
            copies = len(indices_to_repeat_for)  # how many copies we should make of this column
            new_X[:, min_column_idx:max_column_idx] = np.repeat(X[:, input_variable_idx], copies).reshape(
                (n_observations, copies))
        return torch.tensor(new_X)  # the shape of new_X should now be (num. of observations, num. of antecedent terms)

    def forward(self, X):
        """
        Forward pass of the function. Applies the function to the input elementwise.

        The shape of 'X' is (num. of observations, num. of features).
        """
        # we need to make the given 'X' compatible with our first layer,
        # which means repeating it for some entries
        antecedents_memberships = self.input_terms(self.__transform(X))
        terms_to_rules = antecedents_memberships[:, :, None] * torch.tensor(self.links_between_antecedents_and_rules)
        terms_to_rules[terms_to_rules == 0] = 1.0  # ignore zeroes, this is from the weights between terms and rules
        # the shape of terms_to_rules is (num of observations, num of ALL terms, num of rules)
        rules_applicability = terms_to_rules.prod(dim=1)
        numerator = (rules_applicability * self.consequences.T[0]).sum(dim=1)  # MISO
        denominator = rules_applicability.sum(dim=1)
        denominator[denominator == 0] = 1e-300  # add a minimum rule applicability to avoid division by zero error
        return numerator / denominator  # the dim=1 is taking product across ALL terms, now shape (num of observations, num of rules), MISO

    def predict(self, X):
        """
        A wrapper function, for calls that prefer the usage of 'predict'.

        The shape of 'X' is (num. of observations, num. of features).
        """
        return self.forward(X)


"""
The above created FLC class can only handle multi-input-single-output (MISO). 
In other words, we could only calculate the Q-value of a single possible action. 
However, to extend this to multiple actions (i.e., multiple-output), we use the 
well-known fact that a collection of multiple FLCs is commonly used 
for producing multiple outputs.
"""


class MultiFLC:
    """
    A multi-input-multi-output (MIMO) Neuro-Fuzzy Network is a collection of
    multiple multi-input-single-output (MISO) Neuro-Fuzzy Networks (Klir, 1992).

    Essentially, for each possible action, we create a MISO Neuro-Fuzzy Q-Network
    to learn that action's Q-values across all the fuzzy logic rules.
    """

    def __init__(self, n_inputs, n_outputs, antecedents, rules, learning_rate=3e-4, cql_alpha=0.5):
        """
        Build the MIMO Neuro-Fuzzy Q-Network with an Adam optimizer for each individual FLC (per action).
        """
        self.flcs = []
        self.optimizers = []
        self.cql_alpha = cql_alpha
        for flc_idx in range(n_outputs):
            flc = FLC(n_inputs, 1, antecedents, rules)
            self.flcs.append(flc)
            # https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
            self.optimizers.append(optim.Adam(flc.parameters(), lr=learning_rate))

    def predict(self, X):
        """
        Using all Neuro-Fuzzy networks, output all Q-values using the entire MIMO Neuro-Fuzzy Q-network.

        The shape of 'X' is (num. of observations, num. of features).
        """
        output = []
        for flc in self.flcs:
            output.append(list(flc.predict(X).detach().numpy()))
        return torch.tensor(output).T

    def train(self, mode):
        """
        Disable training for all Neuro-Fuzzy networks used in this MIMO Neuro-Fuzzy Q-network.
        """
        for flc in self.flcs:
            flc.train(mode)

    def zero_grad(self):
        """
        Zeroes the gradient for each Neuro-Fuzzy network that creates this MIMO Neuro-Fuzzy Q-network.
        """
        for flc in self.flcs:
            flc.zero_grad()

    def fcql_loss_function(self, all_q_values, pred_qvalues, target_qvalues, action_indices, flc_idx):
        """
        The Fuzzy Conservative Q-Learning loss function.
        To ensure correct implementation, this code is an adaptation of the loss function provided from:

        https://colab.research.google.com/drive/1oJOYlAIOl9d1JjlutPY66KmfPkwPCgEE?usp=sharing
        """
        logsumexp_qvalues = torch.logsumexp(all_q_values, dim=-1)

        tmp_pred_qvalues = all_q_values.gather(
            1, action_indices.reshape(-1, 1)).squeeze()
        cql_loss = logsumexp_qvalues - tmp_pred_qvalues

        new_targets = target_qvalues[:, flc_idx]
        loss = torch.mean((pred_qvalues - new_targets) ** 2)
        return loss + self.cql_alpha * torch.mean(cql_loss)

    def offline_update(self, states, target_q_values, action_indices):
        """
        Updates each individual FLC's Q-values.
        """
        self.train(True)  # make sure training is enabled
        avg_loss = 0.
        all_q_values = self.predict(states)
        # the loss must be computed for each individual FLC
        for flc_idx, flc in enumerate(self.flcs):
            # compute the loss and its gradients
            q_values = flc.predict(states)  # get the Q-values for this action using its corresponding FLC
            loss = self.fcql_loss_function(all_q_values, q_values, target_q_values, action_indices, flc_idx)
            loss.backward()

            # apply the update
            self.optimizers[flc_idx].step()
            avg_loss += loss.item()
        self.train(False)  # when not training, turn it off
        return avg_loss / len(self.flcs)

    def compute_loss(self, states, target_q_values, action_indices):
        """
        Only compute the loss, but do not apply an update.
        """
        self.train(False)  # just in case, disable training
        avg_loss = 0.
        all_q_values = self.predict(states)
        for flc_idx, flc in enumerate(self.flcs):
            q_values = flc.predict(states)  # get the Q-values for this action using its corresponding FLC
            loss = self.fcql_loss_function(all_q_values, q_values, target_q_values, action_indices, flc_idx)
            avg_loss += loss.item()
        self.train(True)  # activate training
        return avg_loss / len(self.flcs)
