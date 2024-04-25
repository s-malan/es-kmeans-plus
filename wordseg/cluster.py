"""
Main functions and class for eskmeans clustering (k-means clustering).

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: April 2024
"""

import numpy as np
import random

class KMeans_Herman():
    """
    The K-means model class.

    If a component is emptied out, its mean is set to a a random data vector.

    Parameters
    ----------
    X : NxD matrix
        A matrix of N data vectors, each of dimension D. (!! N utterance segments, D features after downsampled)
    K_max : int
        Maximum number of components. (!! Compenents = clusters)
    assignments : None or str or vector of int
        If vector of int, this gives the initial component assignments. The
        vector should have N entries between 0 and `K_max`. A value of -1
        indicates that a data vector does not belong to any component. If
        `assignments` is None, then all data vectors are left unassigned with
        -1. Alternatively, `assignments` can take one of the following values:
        "rand" assigns data vectors randomly; "each-in-own" assigns each data
        point to a component of its own; and "spread" makes an attempt to
        spread data vectors evenly over the components.

    Attributes
    ----------
    mean_numerators : matrix (K_max, D)
        The sum of all the data vectors assigned to each component, i.e. the
        component means without normalization.
    means : matrix (K_max, D)
        component means, i.e. with normalization.
    counts : vector of int
        Counts for each of the `K_max` components.
    assignments : vector of int
        component assignments for each of the `N` data vectors. !! the cluster to which each utterance (segment) belongs
    """

    def __init__(self, X, k_max, assignments): #assignments="rand"
        """
        """

        # Attributes from parameters
        self.X = X
        self.N, self.D = X.shape
        self.K_max = k_max
        self.K = 0

        # Attributes
        self.mean_numerators = np.zeros((self.K_max, self.D), np.float64)
        self.counts = np.zeros(self.K_max, np.int32)
        self.assignments = -1*np.ones(self.N, dtype=np.int32)
        self.setup_random_means()
        self.means = self.random_means.copy()

        # Assignments
        if isinstance(assignments, str) and assignments == "rand":
            assignments = np.random.randint(0, self.K_max, self.N)
        elif isinstance(assignments, str) and assignments == "spread":
            assignment_list = (
                list(range(self.K_max))*int(np.ceil(float(self.N)/self.K_max))
                )[:self.N]
            random.shuffle(assignment_list)
            assignments = np.array(assignment_list)
        else:
            assignments = np.asarray(assignments, np.int32)
        print('assignments random:', assignments, assignments.shape)

        # Make sure we have consequetive values ??? make sure we have some segments assigned to each component
        for k in range(assignments.max()): # [0, max)
            print('k:', k, np.nonzero(assignments == k)[0])
            while len(np.nonzero(assignments == k)[0]) == 0:
                assignments[np.where(assignments > k)] -= 1
                print('new assignments:', assignments, assignments.shape)
            if assignments.max() == k:
                print('max has been reached k =', k)
                break

        print('\n')
        # Add the data vectors (!! only has an impact if the values were not consequetive as above)
        for k in range(assignments.max() + 1): # [0, max]
            print('k:', k, np.where(assignments == k)[0])
            for i in np.where(assignments == k)[0]: # !! i is the segment that belongs to component k
                self.add_item(i, k) # !! Add data vector `X[i]` (segment i) to component `k`
        print('assignments:', self.assignments, self.assignments.shape)
    
    def setup_random_means(self):
        self.random_means = self.X[np.random.choice(range(self.N), self.K_max, replace=True), :]
    
    def neg_sqrd_norm(self, i):
        """
        Return the vector of the negative squared distances of `X[i]` to the
        mean of each of the components.
        """
        
        deltas = self.means - self.X[i]
        return -(deltas*deltas).sum(axis=1)  # equavalent to np.linalg.norm(deltas, axis=1)**2
    
    def max_neg_sqrd_norm_i(self, i):
        return np.max(self.neg_sqrd_norm(i))
    
    def argmax_neg_sqrd_norm_i(self, i):
        return np.argmax(self.neg_sqrd_norm(i))
    
    def sum_neg_sqrd_norm(self):
        """
        Return the k-means maximization objective: the sum of the negative
        squared norms of all the items.
        """
        objective = 0
        for k in range(self.K):
            X = self.X[np.where(self.assignments == k)]
            mean = self.mean_numerators[k, :]/self.counts[k]
            deltas = mean - X
            objective += -np.sum(deltas*deltas)
        return objective
    
    def get_max_assignments(self, list_of_i):
        """
        Return a vector of the best assignments for the data vector indices in
        `list_of_i`.
        """
        return [self.argmax_neg_sqrd_norm_i(i) for i in list_of_i]
    
    def get_assignments(self, list_of_i):
        """
        Return a vector of the current assignments for the data vector indices
        in `list_of_i`.
        """
        return self.assignments[np.asarray(list_of_i)]
    
    def add_item(self, i, k):
        """
        Add data vector `X[i]` to component `k`.

        If `k` is `K`, then a new component is added. No checks are performed
        to make sure that `X[i]` is not already assigned to another component.
        """

        assert not i == -1
        assert self.assignments[i] == -1

        if k > self.K:
            k = self.K
        if k == self.K:
            self.K += 1

        self.mean_numerators[k, :] += self.X[i]
        self.counts[k] += 1
        self.means[k, :] = self.mean_numerators[k, :] / self.counts[k]
        print('++ Adding segment', i, 'to component', k, 'with K =', self.K)
        self.assignments[i] = k
    
    def del_item(self, i):
        """Remove data vector `X[i]` from its component."""
        
        assert not i == -1
        k = self.assignments[i]

        # Only do something if the data vector has been assigned
        if k != -1:
            self.counts[k] -= 1
            self.assignments[i] = -1
            self.mean_numerators[k, :] -= self.X[i]
            if self.counts[k] != 0:
                self.means[k, :] = self.mean_numerators[k, :] / self.counts[k]
            else:
                self.del_component(k)

    def del_component(self, k):
        """Remove component `k`."""

        assert k < self.K

        self.K -= 1
        if k != self.K:
            print('Move component', self.K, 'to', k)
            # Put stats from last component into place of the one being removed
            self.mean_numerators[k] = self.mean_numerators[self.K]
            self.counts[k] = self.counts[self.K]
            self.means[k, :] = self.mean_numerators[self.K, :] / self.counts[self.K]
            self.assignments[np.where(self.assignments == self.K)] = k

        # Empty out stats for last component
        self.mean_numerators[self.K].fill(0.)
        self.counts[self.K] = 0
        self.means[self.K] = self.random_means[self.K]

    def clean_components(self):
        """Remove all empty components."""
        for k in np.where(self.counts[:self.K] == 0)[0][::-1]:
            self.del_component(k)
            print('!! Deleting component', k, 'new K =', self.K)

    def fit(self, n_iterations, unasigned=False):
        """
        """
        record_dict = {}
        record_dict["sum_neg_sqrd_norm"] = []
        record_dict["K"] = []
        record_dict["n_mean_updates"] = []

        for i_iteration in range(n_iterations):
            # List of tuples (i, k) where i is the data item and k is the new
            # component to which it should be assigned
            mean_numerator_updates = [] # !! keeps track of the new assignments to delete old and add new

            print(f'\n\t\t~~~~~ Iteration {i_iteration} ~~~~~')
            print(f'\tAssignments: {self.assignments}, K = {self.K}\n')

            for i in range(self.N): # !! i is the segment

                print(f'--- Segment {i}, current component {self.assignments[i]} ---')

                # Keep track of old value in case we do not have to update
                k_old = self.assignments[i] # !! k_old is the component to which the segment currently belongs
                if not unasigned and k_old == -1:
                    continue

                # Pick the new component
                scores = self.neg_sqrd_norm(i) # !! negative squared distances of `X[i]` (segment) to the mean of each of the components
                k = np.argmax(scores) # !! new closest component for segment i
                if k != k_old: # !! a new assignment for segment i
                    print(f'New component {k} for segment {i}')
                    mean_numerator_updates.append((i, k))
                
            # Update means
            print('Updates:', mean_numerator_updates, 'but K =', self.K)
            for i, k in mean_numerator_updates:
                self.del_item(i) # !! delete old history of segment i's component
                self.add_item(i, k) # !! add segment i to new component k

            print(f'\n\tAssignments: {self.assignments}, new K = {self.K}\n')
            
            # Remove empty components
            self.clean_components()

            # Update records
            record_dict["sum_neg_sqrd_norm"].append(self.sum_neg_sqrd_norm())
            record_dict["K"].append(self.K)
            record_dict["n_mean_updates"].append(len(mean_numerator_updates))

            if len(mean_numerator_updates) == 0:
                break
        
        return record_dict
    
def main():

    random.seed(42)
    np.random.seed(42)

    # Data parameters
    D = 2           # dimensions (!! number of features after downsampling)
    N = 10          # number of points to generate (!! number of utterance segments)
    K_true = 4      # the true number of components

    # Model parameters
    K = 6           # maximum number of components
    n_iter = 10

    # Generate data
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    print(X, X.shape)

    # Setup K-means model
    model = KMeans_Herman(X, K) #, "rand"

    print(model.fit(n_iter, unasigned=False))

if __name__ == "__main__":
    main()