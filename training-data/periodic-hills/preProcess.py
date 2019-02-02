"""
A simple pre-processing file for converting raw OpenFOAM data to 
PyTorch tensors. This makes reading the data by the neural network
signifcantly faster. Additionally, depending on the flow, spacial
averages can be taken to increase smoothness of R-S fields.
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
"""
import sys, random, re, os
import numpy as np
import torch as th
import scipy as sc

def readFieldData(fileName):
    """
    Reads in openFoam field (vector, or tensor)
    Args:
        fileName(string): File name
    Returns:
        data (FloatTensor): tensor of data read from file
    """
    #Attempt to read text file and extact data into a list
    try:
        print('Attempting to read file: '+str(fileName))
        rgx = re.compile('[%s]' % '(){}<>')
        rgx2 = re.compile('\((.*?)\)') #regex to get stuff in parenthesis
        file_object  = open(str(fileName), "r").read().splitlines()
        
        #Find line where the internal field starts
        print('Parsing file...')
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

    print('Data field file successfully read.')
    data = th.DoubleTensor(data_list)
    return data

def readScalarData(timeStep, fileName, dir = ''):
    return readFieldData(str(dir)+'/'+str(timeStep)+'/'+fileName)

def readVectorData(timeStep, fileName, dir = ''):
    return readFieldData(str(dir)+'/'+str(timeStep)+'/'+fileName)

def readTensorData(timeStep, fileName, dir = ''):
    data0 = readFieldData(str(dir)+'/'+str(timeStep)+'/'+fileName)
    #Reshape into [nCells,3,3] Tensor
    return data0.view(data0.size()[0],3,-1)

def readSymTensorData(timeStep, fileName, dir = ''):
    data0 = readFieldData(str(dir)+'/'+str(timeStep)+'/'+fileName)
    # Reshape into [nCells,3,3] Tensor
    # Following symmTensor.H indexes since this is RAW openFOAM output
    data = th.DoubleTensor(data0.size()[0], 3, 3)
    data[:,0,:] = data0[:,0:3] #First Row is consistent
    data[:,1,0] = data0[:,1] #YX = XY
    data[:,1,1] = data0[:,3] #YY
    data[:,1,2] = data0[:,4] #YZ
    data[:,2,0] = data0[:,2] #ZX = XZ
    data[:,2,1] = data0[:,4] #ZY = YZ
    data[:,2,2] = data0[:,5]

    return data.view(-1,9)

def readCellCenters(timeStep, dir=''):
    """
    Reads in openFoam cellCenters field which contains a list of
    coordinates associated with each finite volume cell center.
    Generated using the following utility:
    https://bitbucket.org/peterjvonk/cellcenters
    Args:
        timeStep (float): Time value to read in at
        fileName(string): File name
    Returns:
        data (FloatTensor): array of data read from file
    """
    #Attempt to read text file and extact data into a list
    try:
        file_path = dir+"/"+str(timeStep)+"/cellCenters"
        print('Reading mesh cell centers '+file_path)

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

    return th.FloatTensor(cell_list)

def saveTensor(tensor, fieldName, timeStep, dir=''):
    """
    Save PyTorch field tensor
    """
    print('Saving tensor field: {}-torch.th'.format(fieldName))
    th.save(tensor, '{}/{}/{}-torch.th'.format(dir, timeStep, fieldName))

def fieldAverage(field, index_list):
    f0 = []
    for z_i in index_list:
        f0.append(th.sum(field[z_i],0)/len(z_i))

    return th.stack(f0)

if __name__ == '__main__':

    # LES
    les_dir = 'LES' # Directory
    les_time = 1000 # Time step

    # RANS
    rans_dir = 'RANS'
    rans_time = 90

    # Cell Centers
    cell_dir = 'RANS'
    cell_time = 90

    # First read cell centers
    # Cell centers field is generated using the following utility:
    # https://bitbucket.org/peterjvonk/cellcenters
    cell_centers = readCellCenters(cell_time, cell_dir)

    # Get unique x & y coords
    cell_n = cell_centers.numpy()
    cell_coord = np.array([cell_n[:,0], cell_n[:,1]])
    cell_xy = np.unique(cell_n[:,0:2], axis=0)
    saveTensor(cell_xy, 'cellCenters', rans_time, rans_dir)
    
    # Now get averaging indexes (where x & y are the same)
    avg_index = []
    for i in range(cell_xy.shape[0]):
        if(i%100 == 0):
            print('Finding average indexes {}/{}'.format(i, len(cell_xy)))
        avg_index.append(np.where(np.all(cell_n[:,0:2] == cell_xy[i], axis=1))[0])

    # Read in fields
    k = readScalarData(rans_time, 'k', dir=rans_dir)
    s = readTensorData(rans_time, 'S', dir=rans_dir)
    r = readTensorData(rans_time, 'R', dir=rans_dir)

    les_UPrime = readSymTensorData(les_time, 'UPrime2Mean', dir=les_dir)
    
    # Now average fields in Z direction
    k0 = fieldAverage(k, avg_index)
    r0 = fieldAverage(r, avg_index)
    s0 = fieldAverage(s, avg_index)

    les_UPrime0 = fieldAverage(les_UPrime, avg_index)
    
    #Save averaged fields
    saveTensor(k0, 'k', rans_time, rans_dir)
    saveTensor(s0, 'S', rans_time, rans_dir)
    saveTensor(r0, 'R', rans_time, rans_dir)

    saveTensor(les_UPrime0, 'UPrime2Mean', les_time, les_dir)
