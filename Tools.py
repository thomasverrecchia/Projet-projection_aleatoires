import numpy as np
from Stage_B import*

# Generation of a matrix where who can control the distance between singular values
####################################################################################

def espaced(x):
    return np.exp(2*x**2) - 1

def medium(x) :
    return x

def close(x) :
    return 1 - 0.1*x

def generate_singular_values(n,fct) :
    """
    :param n: The number of singular value you want to generate
    :param fct: The fonction defined on [0,1] --> [0,1]
    :return: A list of n values between 0 and 1 using the function.
    """
    output = [0 for i in range(n)]
    for k in range(n):
        step = (k+1)*(1/n)
        output[k] = fct(step)
    return output


def generate_singular_values_matrix(n,fct,m) :
    """
    :param n: The number of singular values
    :param fct: The function to generate the singular values
    :param m: The size of the matrix
    :return: A diagonal matrix
    """
    singular_values = generate_singular_values(n,fct)
    S = np.zeros([m,m])
    for k in range(n) :
        S[k][k] = singular_values[k]
    A = np.random.randn(m,m)
    B = np.random.randn(m,m)
    U = randomized_range_finder(A,m)
    V = randomized_range_finder(B,m)
    return np.linalg.multi_dot([U,S,V])

####################################################################################