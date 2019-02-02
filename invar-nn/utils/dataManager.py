"""
Data manager file is used for reading and parsing flow data for training
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
"""
from utils.log import Log
from utils.foamReader import FoamReader
from utils.torchReader import TorchReader
from fluid.invariant import Invariant

import sys, random, re, os
import torch as th
import numpy as np

# Default tensor type
dtype = th.DoubleTensor

class DataManager():

    def __init__(self, directories, ransTimes, lesTimes):
        """
        Manages the training data, training data should be organized in the following structure:
        >trainingData
        **|- Flow name
        ****|- RANS
        ******|- Timestep
        ********|- S (symmetric tensor, see Eq. 12)
        ********|- R (rotational tensor, see Eq. 12 )
        ********|- k (RANS TKE)
        ****|- LES
        ******|- Timestep
        ********|- UPrime2Mean (Reynolds stress)
        Args:
            directories (string list): list of strings that contain the directory of training data
            ransTimes (int list): list of RANS training data
            lesTimes (int list): list of LES training data 
        """
        self.lg = Log()
        self.torchReader = TorchReader()
        self.foamReader = FoamReader()
        self.invar = Invariant()

        self.flowDict = []
        for i, dir0 in enumerate(directories):
            # Check to make sure there is a RANS and LES folder
            if(not os.path.isdir(dir0+"/RANS") or not os.path.isdir(dir0+"/LES")):
                self.lg.error('Cannot find RANS or LES folder in flow directory: '+dir0)
                self.lg.warning('Skipping flow')
                continue
            
            # Now check for RANS timestep folder
            if(not os.path.isdir(dir0+"/RANS/"+str(ransTimes[i]))):
                self.lg.error('Incorrect RANS timestep provided for: '+dir0)
                self.lg.warning('Skipping flow')
                continue

            # Now check for LES timestep folder
            if(not os.path.isdir(dir0+"/LES/"+str(lesTimes[i]))):
                self.lg.error('Incorrect LES timestep provided for: '+dir0)
                self.lg.warning('Skipping flow')
                continue
            # If prelim checks pass add the flow to dictionary
            mydict = {}
            mydict.update({'dir': dir0})
            mydict.update({'tRANS': ransTimes[i]})
            mydict.update({'tLES': lesTimes[i]})
            mydict.update({'idxmask': np.array([])})
            self.flowDict.append(mydict)

    def getDataPoints(self, svgdNN, nData, partition=None, mask=False, gpu=True):
        """
        Reads in flow data for training
        Args:
            svgdNN (foamSVGD) : Neural network model. Not used when randomly sampling 
                                but can be used for other methods when selecting training points
            nData (int): Number of training data
            partition (IntList): List of flow partitions, default evenly distributions
            mask (boolean): True if 
        Returns:
            x_train (DoubleTensor): [nCellx5] Tensor containing the 5 invariant NN inputs
            t_train (DoubleTensor): [nCellx5] Tensor containing the 10 tensor basis
            k_train (DoubleTensor): [nCellx5] RANS TKE
            y_train (DoubleTensor): [nCellx5] Target anisotropic tensor outputs
        """
        x_train = th.Tensor(nData,5).type(dtype) # Invariant inputs
        t_train = th.Tensor(nData,10,3,3).type(dtype) # Tensor basis
        k_train = th.Tensor(nData).type(dtype) # RANS TKE
        y_train = th.Tensor(nData,9).type(dtype) # Target output
    
        # Partition training points between the provided flows
        if(not partition):
            self.lg.warning('No partition array provided...')
            self.lg.log('Evenly distributing points between provided flows')
            partition = []
            for fDict in self.flowDict[:-1]:
                partition.append(int(nData/len(self.flowDict)))
            partition.append(int(nData-sum(partition)))
        else:
            self.lg.log('Using user defined partition')

        indexes = []
        # Randomly select points from the training flows
        self.lg.info('Sampling points based off random selection') 
        for i, fDict in enumerate(self.flowDict):
            # Read in RANS and LES data
            # Can read pre-processed data (e.g. readTensorTh in torchReader.py)
            s = self.torchReader.readTensorTh(fDict['tRANS'], 'S', dirPath=fDict['dir']+'/RANS')
            r = self.torchReader.readTensorTh(fDict['tRANS'], 'R', dirPath=fDict['dir']+'/RANS')
            k0 = self.torchReader.readScalarTh(fDict['tRANS'], 'k', dirPath=fDict['dir']+'/RANS')
            rs_avg = self.torchReader.readSymTensorTh(fDict['tLES'], 'UPrime2Mean', dirPath=fDict['dir']+'/LES')

            # Can read raw openFOAM data if one desires (e.g. readTensorData in foamReader.py)
            # s = self.foamReader.readTensorData(fDict['tRANS'], 'S', dirPath=fDict['dir']+'/RANS')
            # r = self.foamReader.readTensorData(fDict['tRANS'], 'R', dirPath=fDict['dir']+'/RANS')
            # k0 = self.foamReader.readScalarData(fDict['tRANS'], 'k', dirPath=fDict['dir']+'/RANS')
            # rs_avg = self.foamReader.readSymTensorData(fDict['tLES'], 'UPrime2Mean', dirPath=fDict['dir']+'/LES')

            k = k0.unsqueeze(0).unsqueeze(0).expand(3,3,k0.size()[0])
            k = k.permute(2, 0, 1)

            # Calculate the target scaled anisotropis tensor 
            # Note: scale by RANS TKE so it is consistent during the forward simulation
            # R-S = 2/3k + k*b
            b_avg = rs_avg/k - (2.0/3.0)*th.eye(3).type(dtype)

            # Randomly select the data points
            indx0 = np.arange(s.size()[0]).astype(int)
            indx0 = np.delete(indx0, fDict['idxmask'])
            np.random.shuffle(indx0)
            meshInd = th.from_numpy(indx0).type(th.LongTensor)[:partition[i]]
            # If add these indexes to the flows mask (used for validation data)
            if(mask):
                indxs = meshInd.numpy()
                fDict['idxmask'] = np.append(fDict['idxmask'], indxs.astype(int))

            # Start end indexes for current flow
            start0 = sum(partition[:i])
            end0 = sum(partition[:i+1])
            x_train[start0:end0] = self.invar.getInvariants(th.index_select(s,0,meshInd), th.index_select(r,0,meshInd))
            t_train[start0:end0] = self.invar.getTensorFunctions(th.index_select(s,0,meshInd), th.index_select(r,0,meshInd))
            k_train[start0:end0] = th.index_select(k0, 0, meshInd)
            y_train[start0:end0] = th.index_select(b_avg.view(-1,9), 0, meshInd)

        return x_train, t_train, k_train, y_train
