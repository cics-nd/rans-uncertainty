"""
Class for reading raw OpenFOAM data.
The encouraged method is to pre-process the OpenFOAM files
into PyTorch tensors and load them using torchReader.
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
"""
from utils.log import Log
import sys, random, re, os
import torch as th

# Default tensor type
dtype = th.DoubleTensor

class FoamReader():
    """
    Utility for reading in OpenFoam data files.
    Functions for reading in mesh data need to be updated if needed
    """
    def __init__(self):
        self.lg = Log()

    def readFieldData(self, fileName):
        """
        Reads in openFoam field (vector, or tensor)
        Args:
            fileName(string): File name
        Returns:
            data (FloatTensor): tensor of data read from file
        """
        #Attempt to read text file and extact data into a list
        try:
            self.lg.log('Attempting to read file: '+str(fileName))
            rgx = re.compile('[%s]' % '(){}<>')
            rgx2 = re.compile('\((.*?)\)') #regex to get stuff in parenthesis
            file_object  = open(str(fileName), "r").read().splitlines()
            
            #Find line where the internal field starts
            self.lg.log('Parsing file...')
            fStart = [file_object.index(i) for i in file_object if 'internalField' in i][-1] + 1
            fEnd = [file_object.index(i) for i in file_object[fStart:] if ';' in i][0]
            
            data_list = [[float(rgx.sub('',elem)) for elem in vector.split()] for vector in file_object[fStart+1:fEnd] if not rgx2.search(vector) is None]
            #For scalar fields
            if(len(data_list) == 0):
                data_list = [float(rgx.sub('',elem)) for elem in file_object[fStart+1:fEnd] if not len(rgx.sub('',elem)) is 0]
        except OSError as err:
            print("OS error: {0}".format(err))
            return
        except IOError as err:
            print("File read error: {0}".format(err))
            return
        except:
            print("Unexpected error:{0}".format(sys.exc_info()[0]))
            return

        self.lg.success('Data field file successfully read.')
        data = th.DoubleTensor(data_list)
        return data

    def readScalarData(self, timeStep, fileName, dirPath = ''):
        return self.readFieldData(str(dirPath)+'/'+str(timeStep)+'/'+fileName).type(dtype)

    def readVectorData(self, timeStep, fileName, dirPath = ''):
        return self.readFieldData(str(dirPath)+'/'+str(timeStep)+'/'+fileName).type(dtype)

    def readTensorData(self, timeStep, fileName, dirPath = ''):
        data0 = self.readFieldData(str(dirPath)+'/'+str(timeStep)+'/'+fileName).type(dtype)
        #Reshape into [nCells,3,3] Tensor
        return data0.view(data0.size()[0],3,-1)

    def readSymTensorData(self, timeStep, fileName, dirPath = ''):
        data0 = self.readFieldData(str(dirPath)+'/'+str(timeStep)+'/'+fileName).type(dtype)
        #Reshape into [nCells,3,3] Tensor
        data = th.DoubleTensor(data0.size()[0], 3, 3)
        data[:,0,:] = data0[:,0:3] #First Row is consistent
        data[:,1,0] = data0[:,1] #YX = XY
        data[:,1,1] = data0[:,3] #YY
        data[:,1,2] = data0[:,4] #YZ
        data[:,2,0] = data0[:,2] #ZX = XZ
        data[:,2,1] = data0[:,4] #ZY = YZ
        data[:,2,2] = data0[:,5]

        return data

    def readCellCenters(self, timeStep, dirPath=''):
        """
        Reads in openFoam vector field for the specified timestep
        Args:
            timeStep (float): Time value to read in at
            fileName(string): File name
        Returns:
            data (DoubleTensor): array of data read from file
        """
        #Attempt to read text file and extact data into a list
        try:
            file_path = dir+"/"+str(timeStep)+"/cellCenters"
            self.lg.log('Reading mesh cell centers '+file_path)

            rgx = re.compile('\((.*?)\)') #regex to get stuff in parenthesis
            file_object  = open(file_path, "r").read().splitlines()
            #Find line where the internal field starts
            commentLines = [file_object.index(line) for line in file_object if "//*****" in line.replace(" ", "")]
            fStart = [file_object.index(i) for i in file_object if 'internalField' in i][-1] + 1
            fEnd = [file_object.index(i) for i in file_object[fStart:] if ';' in i][0]
            
            cell_list0 = [rgx.search(center).group(1) for center in file_object[fStart+1:fEnd] if not rgx.search(center) is None]
            cell_list = [[float(elem) for elem in c0.split()] for c0 in cell_list0]
        except OSError as err:
            print("OS error: {0}".format(err))
            return
        except IOError as err:
            print("File read error: {0}".format(err))
            return
        except:
            print("Unexpected error:{0}".format(sys.exc_info()[0]))
            return

        return th.DoubleTensor(cell_list)