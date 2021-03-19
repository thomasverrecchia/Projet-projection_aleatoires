import numpy as np

def eigenvalue(A, v):

    Av = A.dot(v)

    return v.dot(Av)

def power_iteration(A):

    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)

    ev = eigenvalue(A, v)

    while True:

        Av = A.dot(v)

        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)

        if np.abs(ev - ev_new) < 0.01:

            break

        v = v_new

        ev = ev_new

    return ev_new, v_new


def simultaneous_power_iteration(A, k):

    n, m = A.shape

    Q = np.random.rand(n, k)

    Q, _ = np.linalg.qr(Q)

    Q_prev = Q

    for i in range(1000):

        Z = A.dot(Q)

        Q, R = np.linalg.qr(Z)

        # can use other stopping criteria as well
        err = ((Q - Q_prev) ** 2).sum()

        if i % 10 == 0:

            print(i, err)

        Q_prev = Q

        if err < 1e-3:

            break

    return np.diag(R), Q