import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
import seaborn as sns
sns.set_style("darkgrid")


def Lambda_BSM(m, r_min, r_max, numden, energy):

    def sigmav(p):
        # Lets consider the unsuppressed operator O4 from pp-annihilation paper
        return (m**2 + p**2) * np.sqrt(p**2 + m**2) / (4 * np.pi * m)

    # Averaging cross-section over the Fermi Sea
    def avg_sigmav(n):
        return scipy.integrate.quad(sigmav, 0, (3 * np.pi ** 2 * n) ** (1/3))[0] / (3 * np.pi ** 2 * n) ** (1/3) 
    
    def integrand(r):
        return 4 * np.pi * r ** 2 * numden(r) ** 2 * energy(r) * avg_sigmav(numden(r))

    Rate = scipy.integrate.quad(integrand, r_min, r_max)[0]

    def num_lambda(r):
        return 4 *  np.pi * numden(r) * r ** 2

    Num_of_lambda = scipy.integrate.quadrature(num_lambda, r_min, r_max)[0]
    # Rate of energy lost by coldest neutron star
    rate_neutron_star = 4 * np.pi * (11 * 10 ** 3) **2 * 5.67 * 10 ** (-8) * (42000) ** 4 * 6.242 * 10 ** 9 * 6.58 * 10 ** (-25) # In  GeV^2

    # Defining Lambda 
    Lambda =  (Rate / rate_neutron_star) ** (1/4)

    return Num_of_lambda, Lambda