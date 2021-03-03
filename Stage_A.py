
import numpy as np

def randomized_range_finder(A,l) :
    """
    :param A: An m*n matrix
    :param l: An integer
    :return: An m*l orthogonal matrix Q whose range approximate the range of A
    """
    n = np.shape(A)[1]
    Omega = np.random.randn(n,l)
    Y = np.dot(A,Omega)
    Q = np.linalg.qr(Y)[0]
    return Q

def randomized_power_iteration(A,l,q) :
    """
    This algorithm is a modified scheme for matrix whose singular values decay slowly
    :param A: An m*n matrix
    :param l: An integer
    :param q: The number of iterations
    :return: An m*l orthogonal matrix Q whose range approximate the range of A
    """
    n = np.shape(A)[1]
    Omega = np.random.randn(n, l)
    Y = np.linalg.multi_dot([(np.dot(A,A.T))**q,A,Omega])
    Q = np.linalg.qr(Y)[0]
    return Q

def randomized_subspace_iteration(A,l,q) :
    """
    This algorithm correct the rounding errors of the randomized power iteration algorithm
    :param A: An m*n matrix
    :param l: An integer
    :param q: The number of iterations
    :return: An m*l orthogonal matrix Q whose range approximate the range of A
    """
    n = np.shape(A)[1]
    Omega = np.random.randn(n, l)
    Y = np.dot(A,Omega)
    Q = np.linalg.qr(Y)[0]
    for i in range(q) :
        Y_tild = np.dot(A.T,Q)
        Q_tild = np.linalg.qr(Y_tild)[0]
        Y = np.dot(A,Q_tild)
        Q = np.linalg.qr(Y)[0]
    return Q
