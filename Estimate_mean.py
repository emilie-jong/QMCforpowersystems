## Author: Emilie Jong // Technical University of Denmark, based on Qiskit credit risk analysis: https://qiskit.org/ecosystem/finance/tutorials/09_credit_risk_analysis.html
import importlib
import Functions
importlib.reload(Functions)
from Functions import *

from Distributions import UniformDistribution, WeibullDistribution
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
import matplotlib.pyplot as plt
from ClassicalMonteCarlo import *

import sys



## If qc = True, run algorithms on quantum computer
qc = False
computer = 'ibm_kyoto'
# IBMQ.save_account(token='', overwrite=True)
# IBMQ.load_account()


# sys.path.append('./qcinpowersystems')

# IBMQ.save_account(token='', overwrite=True)

## Uncomment the probability distribution that you are interested in
mu = 0.1
sigma = 0.01
variance = sigma**2 #variance needs to be the input for the normal distribution. 
num_qubits = 4

## Normal distribution interval
a = 0.05
b = 0.15
bounds =  (a,b)

u = NormalDistribution(num_qubits, mu = mu, sigma = variance, bounds = (a, b))
# u = load_probability_distribution(num_qubits, 'normal', params = (mu, variance), bounds = (a,b))

# # Loading the probability distribution with the interval [a, b]
# a = 0.05
# b = 0.15
# u = NormalDistribution(num_qubits, mu = mu, sigma = variance, bounds = (a, b))

# a = -0.2
# b = 3.2
# c = 1.8
# u = WeibullDistribution(num_qubits, c, bounds = [-0.2,3.2])

# b = 1
# a = 0
# u = UniformDistribution(num_qubits, a = a, b = b, bounds = (a, b))

u.draw(output='mpl')
# plt.show()
# Histogram on simulator
probs = u.probabilities
import numpy as np
print(u.values) #x = np.linspace(bounds[0], bounds[1], num = 2**num_qubits)
scaling = u.values[1]-u.values[0]
print("Scaling factor =", scaling)

from Functions import plothist
hist, results = plothist(u)
hist

state_preparation_mean, qr_state, objective, state_draw = prepare_state_mean(u)
state_draw


#### Estimating the mean using the simulator
problem_mean = estimate_problem(state_preparation_mean, qr_state, objective)

classical_mean, classical_var_level, classical_cvar_level = get_mean_VaR_CVaR('normal', params=[mu, sigma], value_at_risk_level=0.95)
epsilon = 1*10**(-3)
alpha = 0.05
shots = 100
seed = 2718
eval = 3
max_iter = 3
print('## Estimating the mean ##')
result_mean, ae_mean, scaled_estimate_mean, conf_int_mean, scaled_conf_int_mean, rel_error_mean, oracles_mean = perform_IAE(problem_mean, alpha, epsilon, shots, seed, scaling, bounds, classical_mean)
print('- IQAE -')
print_results_QAE(classical_mean, result_mean, bounds, scaling, scaled_conf_int_mean, scaled_estimate_mean, rel_error_mean)
result_mean_MLAE, ae_mean_MLAE, scaled_estimate_mean_MLAE, conf_int_mean_MLAE, scaled_conf_int_mean_MLAE, rel_error_mean_MLAE, oracles_mean_MLAE = perform_MLAE(problem_mean, eval, shots, seed, scaling, bounds, classical_mean)
print('- MLAE -')
print_results_QAE(classical_mean, result_mean_MLAE, bounds, scaling, scaled_conf_int_mean_MLAE, scaled_estimate_mean_MLAE, rel_error_mean_MLAE)

result_mean_FAE, ae_mean_FAE, scaled_estimate_mean_FAE, conf_int_mean_FAE, scaled_conf_int_mean_FAE, rel_error_mean_FAE, oracles_mean_FAE = perform_FAE(problem_mean, alpha, max_iter, shots, seed, scaling, bounds, classical_mean)
print('- FAE -')
print_results_QAE(classical_mean, result_mean_FAE, bounds, scaling, scaled_conf_int_mean_FAE, scaled_estimate_mean_FAE, rel_error_mean_FAE)

depth_Grover_operators, oracle_queries = get_depth_for_Grover_operators(result_mean, ae_mean, problem_mean, 3)

num_iterations = 10
maxiter = 3
eval = 3
print('## Running IQAE ##')
result_mean_IAE, ae_mean_IAE, scaled_estimate_mean_IAE, conf_int_mean_IAE, scaled_conf_int_mean_IAE, rel_error_mean_IAE, oracles_mean_IAE = run_IAE_iter(problem_mean, alpha, epsilon, shots, seed, scaling, bounds, classical_mean, num_iterations)
print('## Running FAE ##')
result_mean_FAE, ae_mean_FAE, scaled_estimate_mean_FAE, conf_int_mean_FAE, scaled_conf_int_mean_FAE, rel_error_mean_FAE, oracles_mean_FAE = run_FAE_iter(problem_mean, alpha, maxiter, shots, seed, scaling, bounds, classical_mean, num_iterations)
print('## Running MLAE ##')
result_mean_MLAE, ae_mean_MLAE, scaled_estimate_mean_MLAE, conf_int_mean_MLAE, scaled_conf_int_mean_MLAE, rel_error_mean_MLAE, oracles_mean_MLAE = run_MLAE_iter(problem_mean, eval, shots, seed, scaling, bounds, classical_mean, num_iterations)



## Print results running QAE algorithm 10 times


print('## Results IQAE ##')
stats_iter(result_mean_IAE, rel_error_mean_IAE, scaled_estimate_mean_IAE, scaled_conf_int_mean_IAE, classical_mean, oracles_mean_IAE, bounds, scaling)

print('## Result FAE ##')
stats_iter(result_mean_FAE, rel_error_mean_FAE, scaled_estimate_mean_FAE, scaled_conf_int_mean_FAE, classical_mean, oracles_mean_FAE, bounds, scaling)

print('## Result MLAE ##')
stats_iter(result_mean_MLAE, rel_error_mean_MLAE, scaled_estimate_mean_MLAE, scaled_conf_int_mean_MLAE, classical_mean, oracles_mean_MLAE, bounds, scaling)




plt.show()

