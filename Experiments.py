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

def display_error_calculation_between_A_and_Q(m,n, p) :
    """
    :param m: An integer, the number of row of the matrix A.
    :param n: An integer, the number of columns of the matrix A.
    :param p:  oversampling parameter (p ≥ 2)
    :return: Display a graph of the error, singular valus and theorical bounds depending on k.
    """
    A = np.random.randn(m, n)
    singular_value = np.linalg.svd(A, full_matrices=True)[1]
    error = []
    theorical_bounds = []
    strange_values = []

    for k in range(1, min(m, n) - p + 1) :
        l = k + p
        error += [error_calculation_between_A_and_Q(A, l)]
        theorical_bounds += [(1 + (11 * (l * min(m, n)) ** (1 / 2))) * singular_value[k + 1]]

        if error[-1] > theorical_bounds[-1]:
            strange_values += [error[-1], theorical_bounds[-1], k, p]

    plt.plot(error)
    plt.plot(theorical_bounds)
    plt.plot(singular_value[2 : min(m, n) - p + 2])
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title("Actual error for calculation between A and Q")
    plt.legend(["Actual error", "Theoretical bounds (maximal error)", "Singular values (minimal error)"], loc = "upper right")
    plt.savefig("Actual error, singular valus (minimal error) and theorical bounds (maximal error) for calculation_between_A_and_Q.png")
    plt.show()
    return strange_values

def display_expectation_of_error_calculation_between_A_and_Q(m, n, N, p) :
    """
    :param m: An integer, the number of row of the matrix A.
    :param n: An integer, the number of columns of the matrix A.
    :param N: An integer, the number of matrix created by value of l.
    :param p:  oversampling parameter (p ≥ 2)
    :return: Display a graph of the expectation of error, singular valus and theorical bounds depending on k and a list of valus whis not respect Theorem 1.1.
    """
    A = np.random.randn(m, n)
    singular_value = np.linalg.svd(A, full_matrices=True)[1]
    strange_values = []
    expectation_of_error = []
    theorical_bounds = []

    for k in range(0, min(m, n) - p + 1):
        l = k + p
        error = []
        for i in range(N):
            error += [error_calculation_between_A_and_Q(A, l)]

        expectation_of_error += [(1 / N) * sum(error)]
        theorical_bounds += [(1 + (4 * ((l * min(m, n)) ** (1 / 2)) / (p - 1))) * singular_value[k + 1]]
        if expectation_of_error[-1] > theorical_bounds[-1]:
            strange_values += [expectation_of_error[-1], theorical_bounds[-1], k, p]
    #print(len(expectation_of_error), len(theorical_bounds), len(singular_value))
    plt.plot(expectation_of_error)
    plt.plot(theorical_bounds)
    plt.plot(singular_value[2 : min(m, n) - p + 3])
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title("Actual expectation of error for calculation between A and Q")
    plt.legend(["Actual expectation of error", "Theoretical bounds (maximal error)", "Singular values (minimal error)"], loc = "upper right")
    plt.savefig("Actual error, singular valus (minimal error) and theorical bounds (maximal error) for expetation of calculation_between_A_and_Q.png")
    plt.show()

    return strange_values

def error_calculation_between_Clasical_SVD_and_Randomized_SVD(A,l, q) :
    """
    :param A: An m*n matrix.
    :param l: An integer
    :return: The error between the original matrix and the randomized matrix Q.
    """
    Q = randomized_power_iteration(A, l, q)
    Ub, S, Vh = direct_svd(Q, A)
    X = np.linalg.multi_dot([Ub,S,Vh])
    return np.linalg.norm(A-X)

def display_error_calculation_between_Clasical_SVD_and_Randomized_SVD(m, n, q) :
    """
    :param m: An integer, the number of row of the matrix A.
    :param n: An integer, the number of columns of the matrix A.
    :return: Display a graph of the error depending on l.
    """
    A = np.random.randn(m, n)
    error = []
    singular_value = np.linalg.svd(A, full_matrices=True)[1]

    for l in range(1, min(m, n) + 1) :
        error += [error_calculation_between_Clasical_SVD_and_Randomized_SVD(A, l, q)]
    #print(error)
    plt.plot(error)
    plt.plot(singular_value[2 : min(m, n) + 1])
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title("Actual error for calculation between Clasical SVD and Randomized SVD")
    plt.legend(["Actual error", "Singular values (minimal error)"], loc="upper right")
    plt.savefig("Actual error, singular values (minimal error) for calculation between A and Q.png")
    plt.show()

def display_expectation_of_error_calculation_between_Clasical_SVD_and_Randomized_SVD(m, n, N, q) :
    """
    :param m: An integer, the number of row of the matrix A.
    :param n: An integer, the number of columns of the matrix A.
    :param N: An integer, the number of matrix created by value of l.
    :param q:
    :return: Display a graph of the expectation of error, singular values and theoretical bounds depending on k and a list of valus whis not respect Theorem 1.10.
    """
    A = np.random.randn(m, n)
    singular_value = np.linalg.svd(A, full_matrices=True)[1]
    strange_values = []
    expectation_of_error = []
    theorical_bounds = []
    for k in range(2, int(min(m, n) / 2) + 1):
        l = 2 * k
        error = []
        for i in range(N):
            error += [error_calculation_between_Clasical_SVD_and_Randomized_SVD(A, l, q)]

        expectation_of_error += [(1 / N) * sum(error)]
        theorical_bounds += [((1 + (4 * ((2 * min(m, n)) / (k - 1)) ** (1 / 2))) ** (1 / (2 * q + 1))) * singular_value[k + 1]]
        if expectation_of_error[-1] > theorical_bounds[-1]:
            strange_values += [[expectation_of_error[-1], theorical_bounds[-1], k]]
            #print(singular_value[k + 1])
    plt.plot(expectation_of_error)
    plt.plot(theorical_bounds)
    plt.plot(singular_value[2 : int(min(m, n) / 2) + 1])
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title("Error betwen Clasical SVD and Randomized SVD")
    plt.legend(["Actual expectation of error", "Theoretical bounds (maximal error)", "Singular values (minimal error)"], loc = "upper right")
    plt.savefig("error_Clasical_SVD_and_Randomized_SVD.png")
    plt.show()

    return strange_values

def display_expectation_of_error_calculation_between_Clasical_SVD_and_Randomized_SVD_bis(m, n, N, q) :
    """
    :param m: An integer, the number of row of the matrix A.
    :param n: An integer, the number of columns of the matrix A.
    :param N: An integer, the number of matrix created by value of l.
    :param q:
    :return: Display a graph of the expectation of error, singular values and theoretical bounds depending on k and a list of valus whis not respect Theorem 1.11.
    """
    A = np.random.randn(m, n)
    singular_value = np.linalg.svd(A, full_matrices=True)[1]
    strange_values = []
    expectation_of_error = []
    theorical_bounds = []
    for k in range(2, int(min(m, n) / 2) + 1):
        l = k
        error = []
        for i in range(N):
            error += [error_calculation_between_Clasical_SVD_and_Randomized_SVD(A, l, q)]

        expectation_of_error += [(1 / N) * sum(error)]
        theorical_bounds += [((1 + (4 * ((2 * min(m, n)) / (k - 1)) ** (1 / 2))) ** (1 / (2 * q + 1))) * singular_value[k + 1] + singular_value[k + 1]]
        if expectation_of_error[-1] > theorical_bounds[-1]:
            strange_values += [[expectation_of_error[-1], theorical_bounds[-1], k]]
            #print(singular_value[k + 1])
    plt.plot(expectation_of_error)
    plt.plot(theorical_bounds)
    plt.plot(singular_value[2 : int(min(m, n) / 2) + 1])
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title("Error betwen Clasical SVD and Randomized SVD")
    plt.legend(["Actual expectation of error", "Theoretical bounds (maximal error)", "Singular values (minimal error)"], loc = "upper right")
    plt.savefig("error_Clasical_SVD_and_Randomized_SVD_bis.png")
    plt.show()

    return strange_values

#print(display_expectation_of_error_calculation_between_Clasical_SVD_and_Randomized_SVD(100, 1000, 20, 2))
#print(display_expectation_of_error_calculation_between_Clasical_SVD_and_Randomized_SVD_bis(100, 1000, 20, 2))
#print(display_error_calculation_between_A_and_Q(100, 1000, 2))
#print(display_expectation_of_error_calculation_between_A_and_Q(100, 1000, 20, 2))
