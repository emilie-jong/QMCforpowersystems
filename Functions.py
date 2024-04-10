
from Distributions import UniformDistribution, WeibullDistribution
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit, Aer, execute, QuantumRegister, transpile
from qiskit.utils import QuantumInstance
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram  
from qiskit import IBMQ
from scipy.stats import norm, weibull_min, uniform
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem, MaximumLikelihoodAmplitudeEstimation, FasterAmplitudeEstimation
import numpy as np
import random

def get_mean_VaR_CVaR(distribution, params, value_at_risk_level):
        value = value_at_risk_level
        if (distribution == 'normal'):
            mu=params[0]
            sigma=params[1]
            mean, var, skew, kurt = norm.stats(loc = mu, scale = sigma, moments = 'mvsk')
            VaR = norm.ppf(value, loc = mu, scale = sigma)
            tail_loss = norm.expect(lambda x: x, loc = mu, scale = sigma, lb = VaR) #lb is lower bound
            CVaR = (1 / (1 - value)) * tail_loss
            print("Normal distribution: Mean = %.10f, VaR = %.10f, CVaR = %.10f" % (mean, VaR, CVaR))
            return mean, VaR, CVaR
        elif distribution == 'weibull':
            c = params
            mean, var, skew, kurt = weibull_min.stats(c, moments='mvsk')
            VaR = weibull_min.ppf(value, c)
            tail_loss = weibull_min.expect(lambda x: x, args=(c,), lb = VaR) #lb is lower bound
            CVaR = (1 / (1 - value)) * tail_loss
            print("Weibull distribution: Mean = %.10f, VaR = %.10f, CVaR = %.10f" % (mean, VaR,CVaR))
            return mean, VaR, CVaR
        elif distribution == 'uniform':
            a=params[0]
            b=params[1]
            mean, var, skew, kurt = uniform.stats(loc = a, scale = b, moments='mvsk')
            VaR= uniform.ppf(value, loc = a, scale = b)
            tail_loss = uniform.expect(lambda x: x,loc = a, scale = b, lb = VaR) #lb is lower bound
            CVaR = (1 / (1 - value)) * tail_loss
            print("Uniform distribution: Mean = %.10f, VaR = %.10f, CVaR = %.10f" % (mean, VaR, CVaR))
            return mean, VaR, CVaR


def load_probability_distribution(num_qubits, distribution, params, bounds):
    #     """
    # Load a probability distribution based on the input parameter.

    # Parameters:
    #     distribution (str): Name of the distribution. Supported values: 'normal','uniform', 'weibull'.
    #     params (list): List of parameters for the distribution.
    #     bounds: interval of the distribution

    # Returns:
    #     quantum circuit for corresponding probability distribution
    # """
        if (distribution == 'normal'):
            if len(params) != 2:
                 raise ValueError("Normal distribution requires two parameters: 'loc' and 'scale'.")
            else:
                u = NormalDistribution(num_qubits, mu = params[0], sigma = params[1], bounds = (bounds[0], bounds[1]))
                scaling = u.values[1]-u.values[0]
                return u, scaling
        elif distribution == 'weibull':
            if len(params) != 1:
               raise ValueError("Weibull distribution requires one parameter: 'c'")
            else:
                u = WeibullDistribution(num_qubits, c = params[0], bounds=(bounds[0], bounds[1]))
                scaling = u.values[1]-u.values[0]
                return u, scaling
        elif distribution == 'uniform':
            if len(params) != 2:
                 raise ValueError("Uniform distribution requires two parameters: 'loc' and 'scale'.")
            else:
                u = UniformDistribution(num_qubits, a = params[0], b = params[1], bounds = (bounds[0], bounds[1]))
                scaling = scaling = u.values[1]-u.values[0]
                return u, scaling
           #raise ValueError("Unsupported distribution. Supported distributions: 'normal','uniform', 'weibull'.")
        
def plothist(circuit):
    results = execute(circuit, Aer.get_backend('statevector_simulator')).result().get_counts()
    return plot_histogram(results), results

def execute_circuit_on_qc(circuit):
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibmq_quito')
    job = execute(circuit, backend, shots=100)
    job_monitor(job)
    return plot_histogram(job.result().get_counts()), job

def prepare_state_mean(circuit):
    # define linear objective function
    breakpoints = [0]
    slopes = [1]
    offsets = [0]
    f_min = 0
    f_max = 2**circuit.num_qubits-1
    c_approx = 0.25

    objective = LinearAmplitudeFunction(
        circuit.num_qubits,
        slope=slopes,
        offset=offsets,
        # max value that can be reached by the qubit register (will not always be reached)
        domain=(0, 2**circuit.num_qubits-1), #amount of qubits
        image=(f_min, f_max), #image -> the mapping of MW onto qubits
        rescaling_factor=c_approx,
        breakpoints=breakpoints,
    )

    # define the registers for convenience and readability
    qr_state = QuantumRegister(circuit.num_qubits, "state")
    qr_obj = QuantumRegister(1, "objective")

    # define the circuit
    state_preparation = QuantumCircuit(qr_state, qr_obj, name="A")
    state_preparation.append(circuit.to_gate(), qr_state)


    # linear objective function
    state_preparation.append(objective.to_gate(), qr_state[:] + qr_obj[:])
    return state_preparation, qr_state, objective, state_preparation.draw(output='mpl')

def get_circuit_depth(circuit):
    backend = Aer.get_backend("aer_simulator")
    depth_circ = transpile(circuit, backend = backend, basis_gates=['id', 'rz', 'sx', 'x', 'cx'], optimization_level=3)
    print('Circuit depth (simulator):', depth_circ.depth() )
    return depth_circ

def estimate_problem(state_preparation, qr_state, objective):
    # Define Estimation Problem
    problem = EstimationProblem(
    state_preparation=state_preparation,
    objective_qubits=[len(qr_state)],
    post_processing=objective.post_processing)
    return problem

def perform_IAE(problem, alpha, epsilon, shots, seed, scaling, bounds, exact_mean, qc=False, computer=None):
    if (qc==True):
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.get_backend(computer)
        qi = QuantumInstance(backend, shots=shots)
    else:
        qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=shots, seed_simulator=seed)
    ae = IterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)
    result = ae.estimate(problem)
    scaled_estimate = bounds[0] + result.estimation_processed*scaling
    conf_int = np.array(result.confidence_interval_processed)
    scaled_conf_int = bounds[0]+np.array(conf_int)*scaling
    rel_error = abs(scaled_estimate-exact_mean)/exact_mean*100
    oracles = result.num_oracle_queries

    return result, ae, scaled_estimate, conf_int, scaled_conf_int, rel_error, oracles

def print_results_QAE(exact_mean, result, bounds, scaling, scaled_conf_int, scaled_estimate, rel_error):
    print("Exact value:    \t%.10f" % exact_mean)
    print("Estimated value:\t%.10f" % (bounds[0]+result.estimation_processed*scaling))
    print("Confidence interval: \t[%.10f, %.10f]" % tuple(scaled_conf_int))
    print("Relative error:      \t%.10f" %rel_error)
    print("Oracle queries:      \t%.10f" %result.num_oracle_queries)
    print("Error:               \t%.10f" %abs(scaled_estimate-exact_mean))
    print("Estimate +- :              \t%.10f" %(abs(scaled_conf_int[1]-scaled_conf_int[0])/2))
    

def get_depth_for_Grover_operators(result, ae, problem, optimization_level):
    print("Powers =", result.powers)
    oracle_queries = result.num_oracle_queries
    print("Oracle queries = ", result.num_oracle_queries)
    backend = Aer.get_backend("aer_simulator")
    depth = []
    for i in range(len(result.powers)):
        depth_k =transpile(ae.construct_circuit(problem, k=i), backend = backend, basis_gates = ['id', 'rz', 'sx', 'x', 'cx'], optimization_level = optimization_level)
        depth.append(depth_k.depth())
        print(f'Circuit depth k = {i}: (simulator)', depth_k.depth())

    depth_Grover_operators = depth
    return depth_Grover_operators, oracle_queries

def run_IAE_iter(problem, alpha, epsilon, shots, seed, scaling, bounds, exact_mean, num_iterations, qc=False, computer=None):
    
    result = []
    ae = []
    scaled_estimate = []
    conf_int = []
    scaled_conf_int = []
    rel_error = []
    oracles = []

    for i in range(num_iterations):
        print(f'Iteration {i}')
        rnd_seed = random.seed(seed)
        result_i, ae_i, scaled_estimate_i, conf_int_i, scaled_conf_int_i, rel_error_i, oracles_i = perform_IAE(problem, alpha, epsilon, shots, rnd_seed, scaling, bounds, exact_mean, qc=False, computer=None)
        result.append(result_i)
        ae.append(ae_i)
        scaled_estimate.append(scaled_estimate_i)
        conf_int.append(conf_int_i)
        scaled_conf_int.append(scaled_conf_int_i)
        rel_error.append(rel_error_i)
        oracles.append(oracles_i)
   
    return result, ae, scaled_estimate, conf_int, scaled_conf_int, rel_error, oracles

from math import ceil
    
def stats_iter(result, rel_error, scaled_estimate, scaled_conf_int, exact_mean, oracles, bounds, scaling):

    est = [result[i].estimation_processed for i in range(len(result))]
    avg_est = np.mean(est)
    avg_est_scaled = (bounds[0]+avg_est*scaling)
    relative_error = np.mean(rel_error)
    absolute_error = np.mean(abs(np.array(scaled_estimate)-exact_mean))
    avg_conf_int = [np.mean(np.array(scaled_conf_int)[:,0]), np.mean(np.array(scaled_conf_int)[:,1])]
    print("Average estimate = %.10f" % avg_est_scaled)
    print("Average relative error = %.10f " % relative_error)
    print("Average absolute error = %.10f " % absolute_error)
    print("Average number of oracle calls = %.5f" % ceil(np.mean(oracles)))
    print("confidence interval: [%.10f, %.10f]" %tuple(avg_conf_int))
    print('mu +- %.10f' % ((avg_conf_int[1]-avg_conf_int[0])/2))
    print('Max estimate', max(scaled_estimate))
    print('Min estimate', min(scaled_estimate))
    print('Max rel_error', max(rel_error))
    print('Min rel_error', min(rel_error))
    print('Max oracles', max(oracles))  
    print('Min oracles', min(oracles))

def perform_MLAE(problem, eval, shots, seed, scaling, bounds, exact_mean, qc=False, computer=None):
    if (qc==True):
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.get_backend(computer)
        qi = QuantumInstance(backend, shots=shots)
    else:
        qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=shots, seed_simulator=seed)
    mlae = MaximumLikelihoodAmplitudeEstimation(
    evaluation_schedule=eval,
    quantum_instance=qi)
    result = mlae.estimate(problem)
    scaled_estimate = bounds[0] + result.estimation_processed*scaling
    conf_int = np.array(result.confidence_interval_processed)
    scaled_conf_int = bounds[0]+np.array(conf_int)*scaling
    rel_error = abs(scaled_estimate-exact_mean)/exact_mean*100
    oracles = result.num_oracle_queries

    return result, mlae, scaled_estimate, conf_int, scaled_conf_int, rel_error, oracles

def run_MLAE_iter(problem, eval, shots, seed, scaling, bounds, exact_mean, num_iterations):
    
    result = []
    ae = []
    scaled_estimate = []
    conf_int = []
    scaled_conf_int = []
    rel_error = []
    oracles = []

    for i in range(num_iterations):
        print(f'Iteration {i}')
        rnd_seed = random.seed(seed)
        result_i, ae_i, scaled_estimate_i, conf_int_i, scaled_conf_int_i, rel_error_i, oracles_i = perform_MLAE(problem, eval, shots, rnd_seed, scaling, bounds, exact_mean)
        result.append(result_i)
        ae.append(ae_i)
        scaled_estimate.append(scaled_estimate_i)
        conf_int.append(conf_int_i)
        scaled_conf_int.append(scaled_conf_int_i)
        rel_error.append(rel_error_i)
        oracles.append(oracles_i)
   
    return result, ae, scaled_estimate, conf_int, scaled_conf_int, rel_error, oracles


def perform_FAE(problem, alpha, maxiter, shots, seed, scaling, bounds, exact_mean, qc=False, computer=None):
    if (qc==True):
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.get_backend(computer)
        qi = QuantumInstance(backend, shots=shots)
    else:
        qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=shots, seed_simulator=seed)
    qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=shots, seed_simulator=seed)
    fae = FasterAmplitudeEstimation(
    delta=alpha,  # target accuracy
    maxiter=maxiter,  # determines the maximal power of the Grover operator
    quantum_instance=qi)
    result = fae.estimate(problem)
    scaled_estimate = bounds[0] + result.estimation_processed*scaling
    conf_int = np.array(result.confidence_interval_processed)
    scaled_conf_int = bounds[0]+np.array(conf_int)*scaling
    rel_error = abs(scaled_estimate-exact_mean)/exact_mean*100
    oracles = result.num_oracle_queries

    return result, fae, scaled_estimate, conf_int, scaled_conf_int, rel_error, oracles

def run_FAE_iter(problem, alpha, maxiter, shots, seed, scaling, bounds, exact_mean, num_iterations):
    
    result = []
    ae = []
    scaled_estimate = []
    conf_int = []
    scaled_conf_int = []
    rel_error = []
    oracles = []

    for i in range(num_iterations):
        print(f'Iteration {i}')
        rnd_seed = random.seed(seed)
        result_i, ae_i, scaled_estimate_i, conf_int_i, scaled_conf_int_i, rel_error_i, oracles_i = perform_FAE(problem, alpha, maxiter, shots, rnd_seed, scaling, bounds, exact_mean)
        result.append(result_i)
        ae.append(ae_i)
        scaled_estimate.append(scaled_estimate_i)
        conf_int.append(conf_int_i)
        scaled_conf_int.append(scaled_conf_int_i)
        rel_error.append(rel_error_i)
        oracles.append(oracles_i)
   
    return result, ae, scaled_estimate, conf_int, scaled_conf_int, rel_error, oracles


##Functions needed for Estimating value at risk
from qiskit.circuit.library import IntegerComparator

def get_cdf_circuit(x_eval, u):
    qr_state = QuantumRegister(u.num_qubits, 'state')
    qr_obj = QuantumRegister(1, 'obj')
    qr_comp = QuantumRegister(u.num_qubits-1, 'compare')
    state_preparation = QuantumCircuit(qr_state, qr_obj, qr_comp)
    
    state_preparation.append(u, qr_state)
    comparator = IntegerComparator(u.num_qubits, x_eval, geq=False)
    
    state_preparation.append(comparator, qr_state[:]+qr_obj[:]+qr_comp[:])
    
    return state_preparation

def run_ae_for_cdf(x_eval, u, epsilon=10**(-3), alpha=0.05, simulator="aer_simulator"):

    # construct amplitude estimation
    state_preparation = get_cdf_circuit(x_eval, u)
    qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=100)
    problem = EstimationProblem(
        state_preparation=state_preparation, objective_qubits=[u.num_qubits]
    )
    ae_var = IterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)
    result_var = ae_var.estimate(problem)

    return result_var

def run_mlae_for_cdf(x_eval, u, simulator="aer_simulator", eval_schedule=3):

    # construct amplitude estimation
    state_preparation = get_cdf_circuit(x_eval, u)
    qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=100)
    problem = EstimationProblem(
        state_preparation=state_preparation, objective_qubits=[u.num_qubits]
    )
    mlae_var = MaximumLikelihoodAmplitudeEstimation(
    evaluation_schedule=eval_schedule,
    quantum_instance=qi)
    result_mlae_var = mlae_var.estimate(problem)
    #num_oracles = result_mlae_var._num_oracle_queries

    return result_mlae_var

def run_fae_for_cdf(x_eval, u, alpha=0.05, simulator="aer_simulator", max_iter = 3):

    # construct amplitude estimation
    state_preparation = get_cdf_circuit(x_eval, u)
    qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=100)
    problem = EstimationProblem(
        state_preparation=state_preparation, objective_qubits=[u.num_qubits]
    )
    fae_var = FasterAmplitudeEstimation(delta=alpha,  # target accuracy
    maxiter=max_iter,  # determines the maximal power of the Grover operator
    quantum_instance=qi)
    result_fae_var = fae_var.estimate(problem)

    return result_fae_var


def bisection_search(
    objective, num_oracles, target_value, low_level, high_level, low_value=None, high_value=None
):
    """
    Determines the smallest level such that the objective value is still larger than the target
    :param objective: objective function
    :param target: target value
    :param low_level: lowest level to be considered
    :param high_level: highest level to be considered
    :param low_value: value of lowest level (will be evaluated if set to None)
    :param high_value: value of highest level (will be evaluated if set to None)
    :return: dictionary with level, value, num_eval
    """

    num_eval = 0
    oracles = []
    if low_value is None:
        low_value = objective(low_level)
        num_eval += 1
    if high_value is None:
        high_value = objective(high_level)
        num_eval += 1

    # check if low_value already satisfies the condition
    if low_value > target_value:
        return {
            "level": low_level,
            "value": low_value,
            "num_eval": num_eval,
            "comment": "returned low value",
        }
    elif low_value == target_value:
        return {"level": low_level, "value": low_value, "num_eval": num_eval, "comment": "success"}

    # check if high_value is above target
    if high_value < target_value:
        return {
            "level": high_level,
            "value": high_value,
            "num_eval": num_eval,
            "comment": "returned low value",
        }
    elif high_value == target_value:
        return {
            "level": high_level,
            "value": high_value,
            "num_eval": num_eval,
            "comment": "success",
        }
    while high_level - low_level > 1:

        level = int(np.round((high_level + low_level) / 2.0))
        num_eval += 1
        value = objective(level)
        oracles.append(num_oracles(level))
        if value >= target_value:
            high_level = level
            high_value = value
        else:
            low_level = level
            low_value = value

    return {"level": high_level, "value": high_value, "num_eval": num_eval, "comment": "success", "oracles": oracles}



def bisection_search_IAE_iter(u, alpha, num_iterations, low_level, high_level):
    var = []
    val = []
    oracles = []
    for i in range(num_iterations):
        print(i)
        objective = lambda x: run_ae_for_cdf(x, u)
        #no.append(lambda x: objective(x)._num_oracle_queries)
        objective_est = lambda x: objective(x).estimation
        num_oracles = lambda x: objective(x).num_oracle_queries
        # print('oracles=', num_oracles)
        bisection_result = bisection_search(
        objective_est, num_oracles, 1 - alpha, low_level = low_level, high_level = high_level, low_value = 0, high_value= 1)
        var.append(bisection_result["level"])
        val.append(bisection_result["value"])
        oracles.append(bisection_result["oracles"])
        print(bisection_result["comment"])
        # num_oracles = lambda x: objective_est._num_oracle_queries
    return var, val, oracles

def bisection_search_MLAE_iter(u, alpha, num_iterations, low_level, high_level):
    var = []
    val = []
    oracles = []
    for i in range(num_iterations):
        print(i)
        objective = lambda x: run_mlae_for_cdf(x, u)
        #no.append(lambda x: objective(x)._num_oracle_queries)
        objective_est = lambda x: objective(x).estimation
        num_oracles = lambda x: objective(x).num_oracle_queries
        # print('oracles=', num_oracles)
        bisection_result = bisection_search(
        objective_est, num_oracles, 1 - alpha, low_level = low_level, high_level = high_level, low_value = 0, high_value= 1)
        var.append(bisection_result["level"])
        val.append(bisection_result["value"])
        oracles.append(bisection_result["oracles"])
        print(bisection_result["comment"])
        # num_oracles = lambda x: objective_est._num_oracle_queries
    return var, val, oracles

def bisection_search_FAE_iter(u, alpha, num_iterations, low_level, high_level):
    var = []
    val = []
    oracles = []
    for i in range(num_iterations):
        print(i)
        objective = lambda x: run_mlae_for_cdf(x, u)
        #no.append(lambda x: objective(x)._num_oracle_queries)
        objective_est = lambda x: objective(x).estimation
        num_oracles = lambda x: objective(x).num_oracle_queries
        # print('oracles=', num_oracles)
        bisection_result = bisection_search(
        objective_est, num_oracles, 1 - alpha, low_level = low_level, high_level = high_level, low_value = 0, high_value= 1)
        var.append(bisection_result["level"])
        val.append(bisection_result["value"])
        oracles.append(bisection_result["oracles"])
        print(bisection_result["comment"])
        # num_oracles = lambda x: objective_est._num_oracle_queries
    return var, val, oracles

def prepare_state_cvar(var, u):
    ## Computing CVaR
    # define linear objective
    breakpoints = [0, var]
    slopes = [0,1]
    offsets = [0,0]  # subtract VaR and add it later to the estimate
    f_min = 0
    f_max = 2**u.num_qubits-1-var
    c_approx = 0.25

    cvar_objective = LinearAmplitudeFunction(
        u.num_qubits,
        slopes,
        offsets,
        domain=(0, 2**u.num_qubits - 1),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints,
    )

    cvar_objective.draw()

    qr_state = QuantumRegister(u.num_qubits, "state")
    qr_obj = QuantumRegister(1, "objective")
    qr_work = QuantumRegister(cvar_objective.num_ancillas, "work")

    # define the circuit
    state_preparation = QuantumCircuit(qr_state, qr_obj, qr_work, name="A")

    state_preparation.append(u, qr_state)

    # linear objective function
    state_preparation.append(cvar_objective, qr_state[:]+qr_obj[:] + qr_work[:])

    state_preparation.draw(output='mpl')

    return state_preparation, qr_state, cvar_objective


