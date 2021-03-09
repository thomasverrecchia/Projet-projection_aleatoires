import numpy as np
import scipy.linalg.interpolative
from Stage_A import*


def direct_svd(q, a):
    """
        param q: orthonormal matrix whose range captures the
                action of the input matrix a
        
        param a: input matrix

        returns: an array with the svd of a
    """

    b = q.T.dot(a)
    ub, s, vh = np.linalg.svd(b, full_matrices=False)
    s = np.diag(s)

    return np.array([q.dot(ub), s, vh])


def svd_row_extraction(q, a, l):
    idx, x = scipy.linalg.interpolative.interp_decomp(q.T, l)
    print(q)
    print(np.dot(x, q.T[idx, :]))

    # aj = a[idx, :]
    # rh, wh = np.linalg.qr(aj)[0, 1]

    # z = np.dot(proj, rh)


