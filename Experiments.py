from Stage_B import*
import matplotlib.pyplot as plt
from time import*



# Calculation time difference between stage A methods
####################################################################################################

def calculate_time_between_range_finder_and_power_iteration(m,n,l,q=3) :
    A = np.random.randn(m,n)
    starting_range_finder = time()
    Q = randomized_range_finder(A,l)
    ending_range_finder = time() - starting_range_finder
    starting_power_iteration = time()
    Q = randomized_power_iteration(A,l,q)
    ending_power_iteration = time() - starting_power_iteration
    return ending_range_finder - ending_power_iteration



def display_time_between_range_finder_and_power_iteration(min,max,q=3) :
    timing = []
    for i in range(min,max) :
        print(i)
        timing += [calculate_time_between_range_finder_and_power_iteration(i,i,int(i/2))]
    plt.plot(timing)
    plt.axhline(y=0,color="red")
    plt.title("time of range finder - time of power iteration")
    plt.xlabel("size of the matrix")
    plt.ylabel("time in (s)")
    plt.savefig("Time_between_power_iteration_and_range_finder.png")
    plt.show()

# Calculation time difference between randomized svd and numpy svd.
####################################################################################################

def calculate_time_between_numpy_svd_and_randomized_svd(m,n,l,q=3) :
    A = np.random.randn(m, n)
    starting_numpy_svd = time()
    Q = np.linalg.svd(A)
    ending_numpy_svd = time() - starting_numpy_svd
    starting_randomized_svd = time()
    Q = randomized_power_iteration(A, l, q)
    direct = direct_svd(Q,A)
    ending_randomized_svd = time() - starting_randomized_svd
    return ending_numpy_svd - ending_randomized_svd

def display_time_between_numpy_svd_and_randomized_svd(min,max,q=3) :
    timing = []
    for i in range(min,max) :
        print(i)
        timing += [calculate_time_between_numpy_svd_and_randomized_svd(i,i,int(i/2))]
    plt.plot(timing)
    plt.axhline(y=0,color="red")
    plt.title("time of numpy svd - time of randomized svd")
    plt.xlabel("size of the matrix")
    plt.ylabel("time in (s)")
    plt.savefig("Time_between_numpy_svd_and_randomized_svd.png")
    plt.show()

#display_time_between_range_finder_and_power_iteration(10,1000)

# Calculation of error between the original matrix and the randomized one ( i.e. between A and Q ).
####################################################################################################

def error_calculation_between_A_and_Q(A,l) :
    """
    :param A: An m*n matrix.
    :param l: An integer
    :return: The error between the original matrix and the randomized matrix Q.
    """
    Q = randomized_range_finder(A,l)
    X = np.linalg.multi_dot([Q,Q.T,A])
    return np.linalg.norm(A-X)

def display_error_calculation_between_A_and_Q(m,n) :
    """
    :param m: An integer, the number of row of the matrix A.
    :param n: An integer, the number of columns of the matrix A.
    :return: Display a graph of the error depending on l.
    """
    A = np.random.randn(m,n)
    error = []
    for l in range(1,n+1) :
        error += [error_calculation_between_A_and_Q(A,l)]
    print(error)
    plt.plot(error)
    plt.show()

def display_expectation_of_error_calculation_between_A_and_Q(m, n, N, p) :
    """
    :param m: An integer, the number of row of the matrix A.
    :param n: An integer, the number of columns of the matrix A.
    :param N: An integer, the number of matrix created by value of l.
    :param p:  oversampling parameter (p ≥ 2)
    :return: Display a graph of the expectation of error depending on l and a list of valus whis not respect Theorem 1.1.
    """
    A = np.random.randn(m, n)
    svdA = np.linalg.svd(A, full_matrices=True)[1]
    strange_values = []
    expectation_of_error = []
    therical_bounds = []

    for k in range(0, min(m, n) - p + 1):
        l = k + p
        error = []
        for i in range(N):
            error += [error_calculation_between_A_and_Q(A, l)]

        expectation_of_error += [(1 / N) * sum(error)]
        therical_bounds += [(1 + (4 * (l * min(m, n)) ** (1 / 2)) / (p - 1)) * svdA[k + 1]]
        if expectation_of_error[-1] > therical_bounds[-1]:
            strange_values += [expectation_of_error[-1], therical_bounds[-1], k, p]

    plt.plot(expectation_of_error)
    plt.plot(therical_bounds)
    plt.show()

    return strange_values

def error_calculation_between_Clasical_SVD_and_Randomized_SVD(A,l) :
    """
    :param A: An m*n matrix.
    :param l: An integer
    :return: The error between the original matrix and the randomized matrix Q.
    """
    Q = randomized_range_finder(A,l)
    Ub, S, Vh = direct_svd(Q, A)
    X = np.linalg.multi_dot([Ub,S,Vh])
    return np.linalg.norm(A-X)

def display_error_calculation_between_Clasical_SVD_and_Randomized_SVD(m,n) :
    """
    :param m: An integer, the number of row of the matrix A.
    :param n: An integer, the number of columns of the matrix A.
    :return: Display a graph of the error depending on l.
    """
    A = np.random.randn(m, n)
    error = []
    for l in range(1, n+1) :
        error += [error_calculation_between_Clasical_SVD_and_Randomized_SVD(A, l)]
    print(error)
    plt.plot(error)
    plt.show()

def display_expectation_of_error_calculation_between_Clasical_SVD_and_Randomized_SVD(m, n, N, q = 1) :
    """
    :param m: An integer, the number of row of the matrix A.
    :param n: An integer, the number of columns of the matrix A.
    :param N: An integer, the number of matrix created by value of l.
    :param q:
    :return: Display a graph of the expectation of error depending on l and a list of valus whis not respect Theorem 1.1.
    """
    A = np.random.randn(m, n)
    svdA = np.linalg.svd(A, full_matrices=True)[1]
    strange_values = []
    expectation_of_error = []
    therical_bounds = []
    for k in range(2, int(min(m, n) / 2) + 1):
        l = 2 * k
        error = []
        for i in range(N):
            error += [error_calculation_between_Clasical_SVD_and_Randomized_SVD(A, l)]

        expectation_of_error += [(1 / N) * sum(error)]
        therical_bounds += [((1 + (4 * ((2 * min(m, n)) / (k - 1)) ** (1 / 2))) ** (1 / (2 * q + 1))) * svdA[k + 1]]
        if expectation_of_error[-1] > therical_bounds[-1]:
            strange_values += [[expectation_of_error[-1], therical_bounds[-1], k]]
    print(len(strange_values))
    plt.plot(expectation_of_error)
    plt.plot(therical_bounds)
    plt.show()

    return strange_values


#print(display_expectation_of_error_calculation_between_Clasical_SVD_and_Randomized_SVD(100, 10, 20, 2))
#print(display_error_calculation_between_A_and_Q(100, 100))
#print(display_expectation_of_error_calculation_between_A_and_Q(100, 100, 20, 2))
