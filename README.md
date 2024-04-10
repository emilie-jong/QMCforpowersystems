
# README
# Quantum Monte Carlo for Power Systems

This project focuses on comparing the performance of different QAE algorithms: IterativeAmplitudeEstimation,
MaximumLikelihoodAmplitudeEstimation and
FasterAmplitudeEstimation. These functions are integrated in Qiskit. In the code the mean, Value at Risk (VaR) and Conditional Value at Risk (CVaR) are estimated. This is done for different probability distributions. Furthermore, these algorithms are compared with classical Monte Carlo simulation. 

This project is part of a MSc thesis project: Quantum Computing for probabilistic flow in Power Systems. 

The file 'ClassicalMonteCarlo' provides a comparison for estimating the mean, VaR and CVaR of different probability distributions using Monte Carlo simulation.

In the file 'QAE_mean_VaR_CVaR' these statistical parameters are estimated using the mentioned functions in Qiskit for different probability distributions. The files 'Wbl' and 'Uniform' are separate files to load probability distributions onto the qubits. 

Author: Emilie Jong

Technical University of Denmark

