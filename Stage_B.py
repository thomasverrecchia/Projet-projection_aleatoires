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
    ub, S, vh = np.linalg.svd(b, full_matrices=True)
    m = ub.shape[1]
    n = vh.shape[0]
    s = np.zeros((m, n))
    for i in range(len(S)):
        s[i, i] = S[i]
    return np.array([q.dot(ub), s, vh])


# def svd_row_extraction(q, a, l):
#     idx, x = scipy.linalg.interpolative.interp_decomp(q.T, l)
#     print(q)
#     print(np.dot(x, q.T[idx, :]))

    # aj = a[idx, :]
    # rh, wh = np.linalg.qr(aj)[0, 1]

    # z = np.dot(proj, rh)


a = np.array([[1, 1, 1, 1], [1, 5, 0, 8], [1, 1, 0, 6]])
q = randomized_range_finder(a, 3)
b = np.linalg.multi_dot(direct_svd(q, a))
print(b)