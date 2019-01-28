"""
Class for reading pre-processed OpenFOAM data in the PyTorch tensor format
Alternatively once can use foamReader for directly readin OpenFOAM output files.
However the latter is slower, thus pre-processing is encouraged.
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: 
doi: 
github: https://github.com/cics-nd/rans-uncertainty
===
"""
from utils.log import Log
import sys, random, re, os
import torch as th

# Default tensor type
dtype = th.DoubleTensor

class TorchReader():
    """
    Utility for reading in pre-processed OpenFoam tensor files.
    """
    def __init__(self):
        self.lg = Log()

    def loadTensor(self, fileName):
        """
        Read in tensor
        """
        try:
            self.lg.log('Attempting to read file: '+str(fileName))
            self.lg.log('Parsing file...')
            t0 = th.load(fileName)
            self.lg.success('Data field file successfully read.')

        except OSError as err:
            print("OS error: {0}".format(err))
            return
        except IOError as err:
            print("File read error: {0}".format(err))
            return
        except:
            print("Unexpected error:{0}".format(sys.exc_info()[0]))
            return

        return t0

    def readScalarTh(self, timeStep, fieldName, dirPath = '.'):
        data0 = self.loadTensor('{}/{}/{}-torch.th'.format(str(dirPath),str(timeStep),fieldName))
        try:
            data = data0.squeeze(1)
        except:
            data = data0
        return data

    def readVectorTh(self, timeStep, fieldName, dirPath = '.'):
        return self.loadTensor('{}/{}/{}-torch.th'.format(str(dirPath),str(timeStep),fieldName)).type(dtype)

    def readTensorTh(self, timeStep, fieldName, dirPath = '.'):
        data0 = self.loadTensor('{}/{}/{}-torch.th'.format(str(dirPath),str(timeStep),fieldName)).type(dtype)
        #Reshape into [nCells,3,3] Tensor
        return data0.view(data0.size()[0],3,-1)

    def readSymTensorTh(self, timeStep, fieldName, dirPath = '.'):
        data0 = self.loadTensor('{}/{}/{}-torch.th'.format(str(dirPath),str(timeStep),fieldName)).type(dtype)
        #Reshape into [nCells,3,3] Tensor
        return data0.view(data0.size()[0],3,-1)

    def readCellCenters(self, timeStep, dirPath='.'):
        return self.loadTensor('{}/{}/cellCenters-torch.th'.format(str(dirPath),str(timeStep))).type(dtype)