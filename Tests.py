import unittest
import numpy as np
import scipy.linalg.interpolative
from Stage_A import *
from Stage_B import *


class Test_Stage_A(unittest.TestCase):
    def test_randomized_range_finder(self):
        """ Check equation (1.9) 
        """
        m = 100
        n = 150
        l = 50

        A = np.random.randn(m, n)
        Q = randomized_range_finder(A, l)
        singular_value = np.linalg.svd(A, full_matrices=True)[1]
        self.assertTrue(np.linalg.norm(A-np.linalg.multi_dot([Q,Q.T,A])) <= (1 + (11 * (l * min(m, n)) ** (1 / 2))) * singular_value[l + 1])

    def test_randomized_power_iteration(self):
        """ Check equation (1.9) 
        """
        m = 100
        n = 150
        l = 50
        q = 5

        A = np.random.randn(m, n)
        Q = randomized_power_iteration(A, l, 5)
        singular_value = np.linalg.svd(A, full_matrices=True)[1]
        self.assertTrue(
            np.linalg.norm(A - np.linalg.multi_dot([Q, Q.T, A])) <= (1 + (11 * (l * min(m, n)) ** (1 / 2))) *
            singular_value[l + 1])

    def randomized_subspace_iteration(self):
        """ Check equation (1.9) 
        """
        m = 100
        n = 150
        l = 50
        q = 5

        A = np.random.randn(m, n)
        Q = randomized_subspace_iteration(A, l, q)
        singular_value = np.linalg.svd(A, full_matrices=True)[1]
        self.assertTrue(
            np.linalg.norm(A - np.linalg.multi_dot([Q, Q.T, A])) <= (1 + (11 * (l * min(m, n)) ** (1 / 2))) *
            singular_value[l + 1])


    def randomized_range_finder_Gossian_distribution(self):
        """ Check equation (1.9) 
        """
        m = 100
        n = 150
        l = 50

        A = np.random.randn(m, n)
        Q = randomized_range_finder_Gossian_distribution(A, l)
        singular_value = np.linalg.svd(A, full_matrices=True)[1]
        self.assertTrue(
            np.linalg.norm(A - np.linalg.multi_dot([Q, Q.T, A])) <= (1 + (11 * (l * min(m, n)) ** (1 / 2))) *
            singular_value[l + 1])

class Test_Stage_B(unittest.TestCase):

    def test_SVD_error_from_true_range(self):
        """ Check equation (5.2) using true left singular vectors as matrix Q
        """
        m = 47
        n = 53

        A = np.random.randn(m, n)
        u_true, S, vh_true = np.linalg.svd(A)
        u_est, s_est, vh_est = direct_svd(u_true, A)

        m = u_true.shape[1]
        n = vh_true.shape[0]
        s_true = np.zeros((m, n))
        for i in range(len(S)):
            s_true[i, i] = S[i]

        eps = np.linalg.norm(A - u_true @ s_true @ vh_true)
        err = np.linalg.norm(A - u_est @ s_est @ vh_est)
        try:
            self.assertLessEqual(err, eps)
        except AssertionError:
            np.testing.assert_almost_equal(err - eps, 0)

    def test_SVD_error_from_Q(self):
        """ Check equation (5.2) using matrix Q estimated with randomized
        range finder.
        """
        m = 100
        n = 150
        l = 10

        A = np.random.randn(m, n)
        Q = randomized_range_finder(A, l)
        u_est, s_est, vh_est = direct_svd(Q, A)

        eps = np.linalg.norm(A - Q @ Q.T.conj() @ A)
        err = np.linalg.norm(A - u_est @ s_est @ vh_est)
        try:
            self.assertLessEqual(err, eps)
        except AssertionError:
            np.testing.assert_almost_equal(err - eps, 0)
