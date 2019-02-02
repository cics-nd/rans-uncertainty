"""
Main run file for svgd invariant nn.
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
"""

from utils.dataManager import DataManager
from utils.log import Log
from nn.foamSVGD import FoamSVGD

import os

if __name__ == '__main__':

    # Initialize logger
    lg = Log()

    # Define data location and timesteps
    trainingDir = ['../training-data/converge-diverge','../training-data/periodic-hills', \
                '../training-data/square-cylinder', '../training-data/square-duct', \
		'../training-data/tandem-cylinders']
    trainingDir = [os.path.join(os.getcwd(), dir0) for dir0 in trainingDir]
    ransTimes = [60, 90, 60, 60, 60]
    lesTimes = [200, 1000, 250, 1700, 170]
    dataManager = DataManager(trainingDir, ransTimes, lesTimes)
    
    foamNN = FoamSVGD(20) # Number of SVGD particles
    # Load pre-trained neural networks
    #foamNN.loadNeuralNet('./torchNets/foamNet')
    
    # First set up validation dataset
    foamNN.getTestingPoints(dataManager, n_data=500, n_mb=50)

    n = 10 # Number of training sets
    n_data = [10000 for i in range(n)] # Number of data per training set
    n_mb = [20 for i in range(n)] # Mini-batch size
    n_epoch = [10 for i in range(n)] # Number of epochs per training set

    # Training loop
    for i in range(n):
        # Parse data and create data loaders
        foamNN.getTrainingPoints(dataManager, n_data = n_data[i], n_mb = n_mb[i])

        lg.log('Training data-set number: '+str(i+1))
        foamNN.train(n_epoch[i], gpu=True)
        # Save neural networks
        foamNN.saveNeuralNet('foamNet')



