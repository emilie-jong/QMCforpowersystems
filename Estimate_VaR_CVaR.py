## Author: Emilie Jong // Technical University of Denmark, based on Qiskit credit risk analysis: https://qiskit.org/ecosystem/finance/tutorials/09_credit_risk_analysis.html
import importlib
import Functions
importlib.reload(Functions)
from Functions import *
import numpy as np
from ClassicalMonteCarlo import *

from Distributions import UniformDistribution, WeibullDistribution
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
import matplotlib.pyplot as plt

import sys

# sys.path.append('./qcinpowersystems')


## If qc = True, run algorithms on quantum computer
qc = False
computer = 'ibm_kyoto'
# IBMQ.save_account(token='', overwrite=True)
# IBMQ.load_account()

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
# u = WeibullDistribution(num_qubits, c, bounds = [a,b])

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
# print("Scaling factor =", scaling)

from Functions import plothist
hist, results = plothist(u)
hist

alpha = 0.05
high_level = 2**u.num_qubits-1
low_level = 0

num_iterations = 10
var, val, oracles = bisection_search_IAE_iter(u, alpha, num_iterations)
var, val, oracles = bisection_search_MLAE_iter(u, alpha, num_iterations)
var, val, oracles = bisection_search_FAE_iter(u, alpha, num_iterations)
value_at_risk = bounds[0]+np.mean(var)*scaling
print(f'The value at risk at {np.mean(val)*100:.2f}% is {value_at_risk:.2f}')
total_oracles = [np.sum(oracles[i]) for i in range(len(oracles))]
print(f'Average total amount of oracles needed for the complete bisection search: {np.ceil(np.mean(total_oracles))}')
flattened_oracles = sum(oracles, [])
print(f'Average amount of oracles needed for one iteration in the bisection search: {np.ceil(np.mean(flattened_oracles))}')

classical_mean, classical_var_95, classical_cvar_95 = get_mean_VaR_CVaR('normal', params=[mu, sigma], value_at_risk_level=0.95)

classical_mean, classical_var_level, classical_cvar_level = get_mean_VaR_CVaR('normal', params=[mu, sigma], value_at_risk_level=np.mean(val))

print(f'The average relative error for the VaR estimate at {np.mean(val)*100} is {abs(classical_var_level-value_at_risk)/classical_var_level*100}%')

#### CVaR #####
print('## Estimating CVaR ##')
state_preparation_cvar, qr_state_cvar, cvar_objective = prepare_state_cvar(np.mean(var), u)
problem_cvar = estimate_problem(state_preparation_cvar, qr_state_cvar, cvar_objective)

epsilon = 10**(-3)
shots = 100
seed = 2718
max_iter = 3
result_cvar, ae_cvar, scaled_estimate_cvar, conf_int_cvar, scaled_conf_int_cvar, rel_error_cvar, oracles_cvar = perform_IAE(problem_cvar, alpha, epsilon, shots, seed, scaling, bounds, classical_cvar_level)
print('- IQAE -')
print_results_QAE(classical_cvar_level, result_cvar, bounds, scaling, scaled_conf_int_cvar, scaled_estimate_cvar, rel_error_cvar)

result_cvar_MLAE, ae_cvar_MLAE, scaled_estimate_cvar_MLAE, conf_int_cvar_MLAE, scaled_conf_int_cvar_MLAE, rel_error_cvar_MLAE, oracles_cvar_MLAE = perform_MLAE(problem_cvar, eval, shots, seed, scaling, bounds, classical_cvar_level)
print('- MLAE -')
print_results_QAE(classical_cvar_level, result_cvar_MLAE, bounds, scaling, scaled_conf_int_cvar_MLAE, scaled_estimate_cvar_MLAE, rel_error_cvar_MLAE)

result_cvar_FAE, ae_cvar_FAE, scaled_estimate_cvar_FAE, conf_int_cvar_FAE, scaled_conf_int_cvar_FAE, rel_error_cvar_FAE, oracles_cvar_FAE = perform_FAE(problem_cvar, alpha, max_iter, shots, seed, scaling, bounds, classical_cvar_level)
print('- FAE -')
print_results_QAE(classical_cvar_level, result_cvar_FAE, bounds, scaling, scaled_conf_int_cvar_FAE, scaled_estimate_cvar_FAE, rel_error_cvar_FAE)


num_iterations = 10
maxiter = 3
eval = 3
print('## Running IQAE ##')
result_cvar_IAE, ae_cvar_IAE, scaled_estimate_cvar_IAE, conf_int_cvar_IAE, scaled_conf_int_cvar_IAE, rel_error_cvar_IAE, oracles_cvar_IAE = run_IAE_iter(problem_cvar, alpha, epsilon, shots, seed, scaling, bounds, classical_cvar_level, num_iterations)
print('## Running FAE ##')
result_cvar_FAE, ae_cvar_FAE, scaled_estimate_cvar_FAE, conf_int_cvar_FAE, scaled_conf_int_cvar_FAE, rel_error_cvar_FAE, oracles_cvar_FAE = run_FAE_iter(problem_cvar, alpha, maxiter, shots, seed, scaling, bounds, classical_cvar_level, num_iterations)
print('## Running MLAE ##')
result_cvar_MLAE, ae_cvar_MLAE, scaled_estimate_cvar_MLAE, conf_int_cvar_MLAE, scaled_conf_int_cvar_MLAE, rel_error_cvar_MLAE, oracles_cvar_MLAE = run_MLAE_iter(problem_cvar, eval, shots, seed, scaling, bounds, classical_cvar_level, num_iterations)



## Print results running QAE algorithm 10 times


print('## Results IQAE ##')
stats_iter(result_cvar_IAE, rel_error_cvar_IAE, scaled_estimate_cvar_IAE, scaled_conf_int_cvar_IAE, classical_cvar_level, oracles_cvar_IAE, bounds, scaling)

print('## Result FAE ##')
stats_iter(result_cvar_FAE, rel_error_cvar_FAE, scaled_estimate_cvar_FAE, scaled_conf_int_cvar_FAE, classical_cvar_level, oracles_cvar_FAE, bounds, scaling)

print('## Result MLAE ##')
stats_iter(result_cvar_MLAE, rel_error_cvar_MLAE, scaled_estimate_cvar_MLAE, scaled_conf_int_cvar_MLAE, classical_cvar_level, oracles_cvar_MLAE, bounds, scaling)